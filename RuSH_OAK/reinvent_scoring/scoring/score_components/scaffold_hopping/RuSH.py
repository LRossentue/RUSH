#  Copyright (c) 2023, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
#
#     * Redistributions of source code must retain the above copyright 
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following 
#       disclaimer in the documentation and/or other materials provided 
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc.
#       nor the names of its contributors may be used to endorse or promote 
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Created by Luke Rossen, October 2023


"""
Custom scoring function for REINVENT, using OMEGA>ROCS to score generated samples by their combined
weighted Tanimoto (shape & color) similarity to a set of reference molecules, when input decorations
are present in the generated sample. this script essentially combines the code for rocs_score and 
rdkit based dcds score into one scoring function.

Luke Rossen, Novartis, October 2023. 
"""

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcNumAtomStereoCenters, CalcNumRotatableBonds
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as mfp
from rdkit.DataStructs import TanimotoSimilarity

from typing import List
import csv

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary

import os, shutil, sys, subprocess, time
from subprocess import Popen, PIPE
import multiprocessing
import pandas as pd
from pandas.core.groupby import SeriesGroupBy

import numpy as np 


class RuSHScore(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

        self.database_from_smiles = self.parameters.specific_parameters.get('database_from_smiles', True) 
        self.database_path   = self.parameters.specific_parameters.get('database_path'  , None) # oeb or sdf
        
        # ###### 2D parameters #########
        # in the order of [(mol, (fragments), linker),]
        self.reference_smiles = self.parameters.specific_parameters.get('reference_smiles',  # seeds 
                                                                        [('', ('', ''), '')])
        # unpack and reorder.
        self.reference_smiles, self.ref_frags, self.ref_linkers = list(zip(*self.reference_smiles))
        
        self.reference_molecules = [Chem.MolFromSmiles(smi) for smi in self.reference_smiles]
        self.reference_fragments = [tuple([Chem.MolFromSmiles(smi) for smi in fr]) for fr in self.ref_frags]
        self.reference_linkers   = [Chem.MolFromSmiles(smi) for smi in self.ref_linkers]
                
        self.n_fragments = len(self.reference_fragments[0])
        # allowance permits fuzzy fragment identification if <1.0. 
        self.allowance               = self.parameters.specific_parameters.get('allowance', 0.9)
        # We can partially reward a sample if some but not all fragments are included.
        self.partial_reward = self.parameters.specific_parameters.get('partial_reward', 0.2)
        
        # ###### 3D parameters #########          
        self.oeomega_CA   = self.parameters.specific_parameters.get('oeomega_CA'  ,'classic')
        self.oeomega_rms  = self.parameters.specific_parameters.get('oeomega_rms' , 0.5)

        self.n_conformers = self.parameters.specific_parameters.get('n_conformers', 20)
        self.max_centers  = self.parameters.specific_parameters.get('max_centers' , 3)
        self.max_molwt    = self.parameters.specific_parameters.get('max_molwt'   , 700)
        self.max_rotors   = self.parameters.specific_parameters.get('max_rotors'  , 18)

        # scores produced are equivalent, 2024 is faster due to better database/query setup. 
        self._set_ROCS_version(
                            self.parameters.specific_parameters.get('ROCS_version', '2024')
        )
        self.score_cutoff = self.parameters.specific_parameters.get('score_cutoff', 0.8) 
        self.roc_maxconfs = self.parameters.specific_parameters.get('roc_maxconfs', 100) # changed from 1 in 2024
        self.roc_besthits = self.parameters.specific_parameters.get('roc_besthits', 500) # changed from 1 in 2024
        self.roc_timeout  = self.parameters.specific_parameters.get('roc_timeout' , 1000)
        
        self.mcquery      = self.parameters.specific_parameters.get('mcquery'     , True) 
        self.nostructs    = self.parameters.specific_parameters.get('nostructs'   , True)
        
        # ###### score modulation parameters #########          
        self.shape_weight = self.parameters.specific_parameters.get('shape_weight', 1)
        self.color_weight = self.parameters.specific_parameters.get('color_weight', 1)
        self.rocs_weight  = self.parameters.specific_parameters.get('rocs_weight' , 1)
        self.jacc_weight  = self.parameters.specific_parameters.get('jacc_weight' , 1)
        
        self.round_by       = 3 # simply hardcoded for now
        self.score_operator = self.parameters.specific_parameters.get('score_operator', "max")
        # number of cpu cores rocs is allowed to use. 
        self.num_cores      = self.parameters.specific_parameters.get('num_cores', int(multiprocessing.cpu_count()/2))
        
        # specify where to create a temporary working directory, otherwise create it in the class' dir. 
        self.dir = self.parameters.specific_parameters.get('output_dir',
                                                           os.path.dirname(os.path.realpath(__file__)))
        # hidden parameter for now, can manually change this to inspect temporary files. 
        self.keep_files = False # can mess with checks if True during RL ! 
            
        # make a temporary directory for OMEGA and ROCS to operate from.
        self._create_set_temp_dir()
        
        # write each epoch's score breakdown to a csv log file in TEMP_DIR
        self.write_to_file = self.parameters.specific_parameters.get('write_to_file', False)
        
        if self.write_to_file:
            self._csv_log_path = os.path.join(self.TEMP_DIR, 'log.csv')
            headers = ["ID", "smiles", "linker_success", 
                       "reward_2d", "reward_3d", "score"]
            self._write_to_csv(headers, self._csv_log_path)
            
        # initialize the database to overlay onto the query using ROCS.
        if self.database_from_smiles: self._create_oeb_database()
        # or copy the database file to the temporary directory. 
        elif self.database_path:
            if   self.database_path.endswith('.sdf'): self._copy_sdf_database()
            elif self.database_path.endswith('.oeb'): self._copy_oeb_database()
        else: raise ValueError('No database information was passed.')
        
        # initialize and validate the reference molecules by running obtain_scaffold.py on them.
        self._initialize_references()
        
    
    def _write_to_csv(self, thing, csv_path):
        # create and otherwise append to a csv.
        with open(csv_path, 'a') as f:
            w = csv.writer(f)
            # pass a list to write a single row (or headers).
            if type(thing) is list:
                w.writerow(thing)
            # pass a dict assuming its coming from _score_molecules().
            elif type(thing) is dict:
                for k, v in zip(thing.keys(), thing.values()):
                    r = [k]
                    r.extend(v)
                    w.writerow(r)
            else:
                raise TypeError(f"invalid input type {type(thing)}")
    
    def _initialize_references(self):
        # write the reference molecules to .sdf file (if not done already).
        if self.DB_OEB_PATH.endswith('.sdf') and os.path.isfile(self.DB_OEB_PATH):
            # if we do ROCS from an .sdf, we recycle the copied .sdf.
            self.references_sdf_path = self.DB_OEB_PATH
        else:
            # otherwise we write the passed smiles to sdf.
            self.references_sdf_path = os.path.join(self.TEMP_DIR, 'references.sdf')
            self._mols_to_sdf(self._name_mols(
                self.reference_molecules, 
                [f'db_{i}' for i, _ in enumerate(self.reference_molecules)]
                ),
                self.references_sdf_path
            )
        # write the reference fragments to a seperate .sdf file for obtain_scaffold.py.
        self.fragments_sdf_path = os.path.join(self.TEMP_DIR, 'fragments.sdf')
        self._mols_to_sdf([frag for fragments in self.reference_fragments for frag in fragments],
                          self.fragments_sdf_path) # fragment list is linearized (!).
        # set up the output file.
        linker_file = os.path.join(self.TEMP_DIR, 'reference_links.csv')
        # run obtain_scaffold.py on the reference molecules.
        out, err, returncode = self._run_linker_finder(self.references_sdf_path, linker_file)
        # if the shell raised any errors, terminate here for troubleshooting.
        if returncode: raise RuntimeError(f"Problems running linkerfinder: {err}\nOutput: {out}")
        # fetch the output file produced by obtain_scaffold.py.
        linker_df = pd.read_csv(linker_file)
        # fetch relevant information.
        IDs     = linker_df.ID 
        success = linker_df.linker_success
        linkers = linker_df.linker
        
        # check each reference and report. 
        for ID, s, mol, linker, ref_linker in zip(IDs, success, self.reference_molecules, linkers, self.reference_linkers):
            # linker is a smiles string, make sure we can obtain a valid rdkit molecule from it. 
            linker = Chem.MolFromSmiles(linker, sanitize=True)
            # ensure the reference molecule can be properly processed, and a valid linker was obtained.
            # we do not care about stereochemistry in this case, so we validate with isomer-invariance. 
            assert int(s), f'Reference molecule could not be properly segmented:' \
                                   f'{Chem.MolToSmiles(mol)} | linker: {Chem.MolToSmiles(linker, isomericSmiles=False)}'
            # ensure the obtained linker is the same as the one provided by the user (processing works as intended).
            assert Chem.MolToSmiles(ref_linker) == Chem.MolToSmiles(linker, isomericSmiles=False), \
               f"{Chem.MolToSmiles(ref_linker)} =/= {Chem.MolToSmiles(linker, isomericSmiles=False)}."
            # now we're happy :) 
            print(f"reference {ID} {Chem.MolToSmiles(mol)} succesfully initialized and verified.")
    
    
    def _create_set_temp_dir(self):
        self.TEMP_DIR = os.path.join(self.dir, "TEMP_DIR/")
        try:
            os.mkdir(self.TEMP_DIR)
        except FileExistsError:
            pass
        os.chdir(self.TEMP_DIR)
    
    
    def _copy_oeb_database(self):
        assert os.path.isfile(self.database_path), "database file does not exist."
        # copy the specified database to the temporary directory if it does not exists there already.
        self.DB_OEB_PATH = os.path.join(self.TEMP_DIR, os.path.basename(self.database_path))
        
        if not os.path.isfile(self.DB_OEB_PATH):
            print(f"Copying .oeb database {self.database_path}")
            shutil.copy(self.database_path, self.TEMP_DIR)
        else:
            print(f"using existing .oeb database {self.DB_OEB_PATH}")

            
    def _copy_sdf_database(self):
        assert os.path.isfile(self.database_path), "database file does not exist."
        self.DB_OEB_PATH = os.path.join(self.TEMP_DIR,  os.path.basename(self.database_path))
        
        print(f"Copying .sdf database {self.database_path}")
        # ! here we overwrite self.reference_molecules with those from the .sdf.
        self.reference_molecules = []
        # we do some sanitization in case the .sdf is not properly curated. 
        for mol in Chem.SDMolSupplier(self.database_path, sanitize=True):
            # try to neutralize and clean up any charges.
            reference_mol = self._neutralize_atoms(mol)
            # kekulize the molecule for rdkit to be happy.
            Chem.rdmolops.Kekulize(reference_mol)
            # remove hydrogens (for 2D processing).
            reference_mol = Chem.RemoveHs(reference_mol)
            # store it to reduce overhead. 
            self.reference_molecules.append(reference_mol)
        
        self._mols_to_sdf(
            self._name_mols(
                self.reference_molecules, 
                [f'db_{i}' for i, _ in enumerate(self.reference_molecules)]
            ),
            self.DB_OEB_PATH # ! this is the rocs DB input, but its an .sdf, not .oeb
        )
            
        
    def _create_oeb_database(self):
        # create an .sdf file from the parsed list of smiles, and then run it through omega.
        self.DB_SDF_PATH = os.path.join(self.TEMP_DIR, 'db.sdf')
        self.DB_OEB_PATH = os.path.join(self.TEMP_DIR, 'db.oeb')
        
        self._mols_to_sdf(
            self._name_mols(
                self.reference_molecules, 
                [f'db_{i}' for i, _ in enumerate(self.reference_molecules)]
            ),
            self.DB_SDF_PATH
        )
        # create the database.oeb file to query sampled molecules against.
        self._run_omega(self.DB_SDF_PATH, self.DB_OEB_PATH)
    
    
    def _name_mols(self, molecules: List, names: List[str]):
        # method for providing molecule metadata.
        # Used to batch molecules in OMEGA/ROCS and retrieve the correct scores from report(s).
        for mol, name in zip(molecules, names):
            mol.SetProp("_Name", name)
        return molecules
    
    
    def _mols_to_sdf(self, molecules: List, output_file: str):
        # method for writing molecules to a local SDF file, to be read by OMEGA for conformer generation. 
        w = Chem.SDWriter(output_file) # overwrites existing filenames
        for mol in molecules:
            w.write(mol)
    
    
    def _filter_mols(self, molecules: List):
        # filter a list of molecules by a maximum number of stereocenters, molecular weight, and  rotatable bonds.
        # also filters bad input (i.e. SMILES that could not be translated to an rdkit mol (== None)).
        # reason for filtering large + flexible compounds is they give issues with conformer generation.
        # assumption is that significantly larger compounds are outside of the desired chemical space regardless.
        # always check these descriptors for your reference molecules(s) first and parameterize accordingly. 
        
        # add whatever filters here, can also be done through reinvent with smarts patterns as a seperate comp.
        # very rarely REINVENT can generate a molecule with no atoms, so we also check for mol and atoms.
        f_IDs, f_molecules = np.array(
                      [(mol.GetProp("_Name"), mol) for i, mol in enumerate(molecules) if mol if \
                       mol.GetNumAtoms() > 0 if \
                       CalcNumAtomStereoCenters(mol) <= self.max_centers if \
                       CalcExactMolWt(mol)           <= self.max_molwt if \
                       CalcNumRotatableBonds(mol)    <= self.max_rotors
                      ], dtype=object).transpose()
        
        print(f'{len(molecules) - len(f_molecules)} molecules filtered.')
        return np.array(f_IDs), np.array(f_molecules, dtype=object)

    
    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        score = self._score_molecules(molecules) 
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary    

    
    def _score_molecules(self, molecules: List, details=False):
        '''run a batch of sample molecules through the OMEGA>ROCS pipeline if a valid linker
        could be obtained, by identifying and removing reference fragments. Returns the 
        harmonic mean of the i) combined weighted Tanimoto (shape & color) score, and 
        ii) the jaccard distance of the linker and the refernce linkers, or 0.0 if not. 
        A partial fragment reward can be given if some but not all fragments are present. 
        If mulitple references are provided, scores are combined by self.score_operator.
        '''
        start = time.time() # track duration of a single round of scoring
        # name the molecules by enumeration.
        IDs = [f"sample_{_}" for _ in range(len(molecules))]
        molecules = self._name_mols(molecules, IDs)
        # initialize the scoring to 0.0, should any sample fail the pipeline. 
        scores = dict(zip(IDs, [0.0]*len(molecules)))        
        # additional dict for analysis, write_to_file. {ID:[smi, success, reward_2d, reward_3d, score],}
        detailed_scores = dict(zip(IDs, [[Chem.MolToSmiles(mol), .0, .0, .0, .0] for mol in molecules]))    
        # setup working files.
        sdf_path    = os.path.join(self.TEMP_DIR, 'query.sdf')
        oeb_path    = os.path.join(self.TEMP_DIR, 'query.oeb')
        log_path    = os.path.join(self.TEMP_DIR, 'rocs.log')
        linker_file = os.path.join(self.TEMP_DIR, 'links.csv')
        report_file = os.path.join(self.TEMP_DIR, 'query.rpt')

        # prefilter for molecules that are likely to score low/give trouble with the pipeline.
        f_IDs, f_molecules = self._filter_mols(molecules)
        if f_molecules.size > 0:
            # write the samples to sdf.
            self._mols_to_sdf(f_molecules, sdf_path)

            # identify fragments and try to obtain linkers from the filtered set of molecules.
            out, err, returncode = self._run_linker_finder(sdf_path, linker_file)
            if returncode: raise RuntimeError(f"Problems running linkerfinder: {err}\nOutput: {out}")

            # fetch the output file produced by obtain_scaffold.py.
            linker_df = pd.read_csv(linker_file)
            # we take the ID column from the df and not the original f_IDs set, ... 
            # as sometimes sdf write>read fails and a None object is read instead of a valid rdkit molecule.
            # because of this, obtain_scaffold.py does an additional check for <if mol: process>. 
            f_IDs     = linker_df.ID.to_numpy()
            f_success = linker_df.linker_success.to_numpy()
            f_linkers = np.array([Chem.MolFromSmiles(smi) for smi in linker_df.linker], dtype=object)
            # partial reward for molecules that contain some but not all fragments.
            for ID, success in zip(f_IDs, f_success):
                scores[ID] = min(success, self.partial_fragment_reward)
                detailed_scores[ID][1] = success
                detailed_scores[ID][4] = min(success, self.partial_fragment_reward)

            # filter for molecules that were sucessfully cleaved into a linker and fragments.
            f_IDs       = f_IDs[      np.where(f_success.astype(int))]
            f_molecules = f_molecules[np.where(f_success.astype(int))]
            f_linkers   = f_linkers[  np.where(f_success.astype(int))] 
            # if no molecules survived the filtering, we can skip this bit entirely.
            if f_molecules.size > 0:
                # rewrite the .sdf file to only contain molecules that satisfy all conditions.
                self._mols_to_sdf(f_molecules, sdf_path)

                # run omega on the filtered batch.
                out, err, returncode = self._run_omega(sdf_path, oeb_path)
                if returncode: raise RuntimeError(f"Problems running OMEGA: {err}\nOutput: {out}")

                # run rocs on the filtered batch.
                out, err, returncode = self._run_ROCS(oeb_path, report_file)
                # print error output if rocs had issues, but don't raise any errors.
                if returncode: 
                    print(f"Problems running ROCS: {err}\nOutput: {out}")
                    self._print_logs(log_path)
                # check if rocs did produce a report file. This means only a few failed rocs.
                if os.path.isfile(report_file):  
                    report_has_content, rewards_3d = self._read_report_rewards(report_file)
                    
                    if report_has_content:
                        # only process molecules that passed the OMEGA>ROCS pipeline.
                        indices = np.where(np.in1d(f_IDs, rewards_3d.index))
                        f_IDs = f_IDs[indices]
                        f_linkers = f_linkers[indices]

                        # now compute the jaccard distances between the linkers and reference linkers.
                        jaccard = self._compute_distances(f_IDs, f_linkers)
                        # combine the scores of each sample if multiple references are passed.
                        rewards_2d = self._score_method(jaccard.groupby('ShapeQuery').distance) 

                        # merge the rewards into a single df for easy processing. IDs will map correcty, also \w 2024
                        rewards = pd.DataFrame([rewards_2d, rewards_3d]).transpose() # rows = IDs
                        # update the scores for all samples in the report. 
                        for ID, (reward_2d, reward_3d) in rewards.iterrows():
                            # we take the harmonic mean of the rewards, so that optimizing one ... 
                            # does lead to minimizing the other. (e.g. linker diversity vs shape).
                            reward_combined = self._weighted_harmonic_mean(
                                                    scores=(reward_2d, reward_3d),
                                                    weights=(self.jacc_weight, self.rocs_weight))
                            # sometimes we get a nan here from issues with omega/rocs.
                            # so we just reward it 0.0 if this happens.
                            if self._is_valid(reward_combined):
                                scores[ID] = reward_combined
                                # extra logging of seperate rewards for analysis.
                                detailed_scores[ID][4] = reward_combined
                            detailed_scores[ID][2] = reward_2d
                            detailed_scores[ID][3] = reward_3d
                           

                    # some verbosity, not required but can be useful. 
                    else:
                        print("empty ROCS report file. No further scoring.")
                else:
                    print("no ROCS report file produced! No further scoring.")
            else:
                print("no linkers. No further scoring.")
        else:
            print("no filtered molecules left. No further scoring.")
        # delete this epoch's temorary files.
        if not self.keep_files: 
            self._delete_files([sdf_path, oeb_path, log_path, linker_file, report_file])
        # optionally write this round of scoring breakdown to a csv for analysis. 
        if self.write_to_file:
            self._write_to_csv(detailed_scores, self._csv_log_path)
            
        print(f'Finished ROCS scoring. {round(time.time() - start)} seconds.')
        # return only the list of scores
        if details:
            return np.array(list(scores.values()), dtype=np.float64).round(self.round_by), detailed_scores
        else:
            return np.array(list(scores.values()), dtype=np.float64).round(self.round_by) 
            
    
    
    def _is_valid(self, n):
         return False if n is None or np.isnan(n) else True
    
    
    def _weighted_harmonic_mean(self, scores, weights=None):
        if weights is None: weights = [1]*len(scores)
        # https://en.wikipedia.org/wiki/Harmonic_mean
        if 0.0 in scores:
            return 0.0
        else:
            return sum(weights)/sum(np.divide(weights, scores))

    
    def _score_method(self, x):
        # combine the scores of multiple reference molecules in a pandas groupby object. 
        # select a function by its corresponding key (string).
        return round({
            "min"  : SeriesGroupBy.min,
            "max"  : SeriesGroupBy.max,
            "mean" : SeriesGroupBy.mean,
        }[self.score_operator](x), self.round_by)

    
    def _compute_distances(self, IDs, linkers):
        # create a dataframe that is consistent with rocs report.
        results_df = pd.DataFrame(columns=['ShapeQuery', 'Name', 'distance']) 
        # compute the jaccard distance of all linker & reference linker pairs.
        for i, (ID, linker) in enumerate(zip(IDs, linkers)):
            for j, ref_linker in enumerate(self.reference_linkers):
                distance = self._jaccard_distance(linker, ref_linker)
                results_df.loc[(i * len(linkers)) + j] = [ID, f"db_{j}", distance]
        # return a long format dataframe.
        return results_df
    
    
    def _jaccard_distance(self, linker, reference_linker):
        distance = round(1 - TanimotoSimilarity(mfp(linker,           3, nBits=2048), 
                                                mfp(reference_linker, 3, nBits=2048)
                                               ), self.round_by)
        return distance
        
    
    def _print_logs(self, log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
        if lines: [print(line) for line in lines]
            
        
    def _delete_files(self, files):
        for file_path in files:
            if os.path.isfile(file_path): os.remove(file_path)
            
            
    def _run_linker_finder(self, query_sdf_path: str, output_file: str):
        commands = ["module purge",
                    "source  ~/anaconda3/etc/profile.d/conda.sh;",
                    "conda activate general;", # any environment with rdkit 2023.3+
                   f"python {os.path.dirname(os.path.realpath(__file__))}/obtain_scaffold.py " \
                   f"--query_sdf_path {query_sdf_path} " \
                   f"--deorations_sdf_path {self.fragments_sdf_path} " \
                   f"--n_decorations {self.n_fragments} " \
                   f"--allowance {self.allowance} " \
                   f"--output_csv_path {  output_file} "
        ]
        return self._run_cmd(commands)
    
    
    def _run_omega(self, input_file: str, output_file: str):
        # see https://docs.eyesopen.com/applications/omega/omega/omega_opt_params.html
        commands = [
            "source  ~/anaconda3/etc/profile.d/conda.sh;",
            "conda activate oepython;", # default OE venv setup. See documentation for installation and licensing. 
            ' '.join([
                    'oeomega', self.oeomega_CA, '-canonOrder', 'false', 
                    '-in', input_file, '-out', output_file,
                    '-maxconfs', str(self.n_conformers), 
                    '-rms', str(self.oeomega_rms), # see OMEGA documentation
                    # '-maxrot', str(self.max_rotors), # inconsistent with RDkit (counts amides)
                    '-flipper', 'true', '-flipper_warts', 'false', # *
                    '-flipper_maxcenters', str(self.max_centers),  # **
                    '-verbose', 'true', '-useGPU', 'false'])
        ]           # gpu disabled for now due to torch memory allocation conflict.
        # * flipper set to true because generated samples are stereo-invariant (greedy score)
        # ** in case any pass the max_center filter, do random flipping instead (should not happen).
        return self._run_cmd(commands)


    def _set_ROCS_version(self, version: str = '2024'):
        self.ROCS_version = version
        if version == '2023':
            self.rocs_groupby = 'ShapeQuery'
            self._run_ROCS = self._run_ROCS_2023
        elif version == '2024':
            self.rocs_groupby = 'Name'
            self._run_ROCS = self._run_ROCS_2024
        else:
            raise ValueError(f"version {version} not implemented.")


    def _read_report_rewards(self, report_file):
        # for old rocs implementation, name_col = 'ShapeQuery', for 2024 implementation name_col = 'Name'
        report_has_content, rewards_3d = False, []
        
        # fetch the rocs report.
        report = pd.read_table(report_file) 
        # check if the report contains any scores.
        if not report.empty:
            report_has_content = True
            # compute the weighted Tanimoto S&C score for all hits. 
            report['score'] = ((report.ShapeTanimoto * self.shape_weight + \
                                report.ColorTanimoto * self.color_weight
                                ) / (self.shape_weight + self.color_weight))
            # combine the scores of each sample if multiple references are passed.
            rewards_3d = self._score_method(report.groupby(self.rocs_groupby).score)
        
        return report_has_content, rewards_3d

        
    def _run_ROCS_2023(self, query_file, report_file):
        # see https://docs.eyesopen.com/applications/rocs/rocs/rocs_opt_params.html
        commands = [
            "source  ~/anaconda3/etc/profile.d/conda.sh;",
            "conda activate oepython;", # default OE venv setup. See documentation for installation and licensing. 
            ' '.join([
                     'rocs',
                     '-dbase', self.DB_OEB_PATH, 
                     '-query', query_file, '-mcquery', str(self.mcquery).lower(),
                     '-besthits', str(self.roc_besthits), 
                     '-rankby', 'TanimotoCombo',
                     '-cutoff', str(self.score_cutoff), # only report mols with reasonable combined scores.
                     '-nostructs', str(self.nostructs).lower(),      # dont report structures
                     '-report', 'one',  # output everything to one report file
                     '-status', 'none', # dont report status
                     '-stats', 'best',  # only report the best overlays for each isomer pair. 
                     '-reportfile', report_file,
                     '-mpi_np', str(self.num_cores), # could use work, because no config is passed for mp.
                     '-qconflabel', 'none', '-conflabel', 'none', # all confs of a mol have the same name. 
                     '-maxconfs', str(self.roc_maxconfs), '-subrocs'])
        ]        
        return self._run_cmd(commands, timeout=self.roc_timeout)
    

    def _run_ROCS_2024(self, query_file, report_file):
        # ROCS can be sped up if we switch around the db and query file, and handle the report differently. equivalent to:
        # rocs -dbase query.oeb -query lre001.sdf -report one -mcquery true -reportfile test.rpt -qconflabel none -conflabel none -nostructs true -maxconfs 100 -cutoff 0.8
        
        # original experiments were performed with ROCS_2023. unit tests show identical results for scoring, this implementation is just faster for our purpose.
        commands = [
            "source  ~/anaconda3/etc/profile.d/conda.sh;",
            "conda activate oepython;", # default OE venv setup. See documentation for installation and licensing. 
            ' '.join([
                     'rocs',
                     '-dbase', query_file, # switched! 
                     '-query', self.DB_OEB_PATH,  '-mcquery', str(self.mcquery).lower(),
                    '-besthits', str(self.roc_besthits), 
                     '-rankby', 'TanimotoCombo',
                     '-cutoff', str(self.score_cutoff), # only report mols with reasonable combined scores.
                     '-nostructs', str(self.nostructs).lower(),      # dont report structures
                     '-report', 'one',  # output everything to one report file
                     '-status', 'none', # dont report status
                     '-stats', 'best',  # only report the best overlays for each isomer pair. 
                     '-reportfile', report_file,
                     '-mpi_np', str(self.num_cores), # could use work, because no config is passed for mp.
                     '-qconflabel', 'none', '-conflabel', 'none', # all confs of a mol have the same name. 
                     '-maxconfs', str(self.roc_maxconfs)]) # subrocs removed
        ]
        
        return self._run_cmd(commands, timeout=self.roc_timeout)
    
    
    def _run_cmd(self, commands: List[str], timeout=20_000):
        """ 
        Run a named command in a subprocess. 
        Pass timeout to prevent extremely long epochs or deadlocks (will terminate the run!). 
        """
        process = Popen("/bin/bash", shell=True, universal_newlines=True,
                        stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding='utf8')      

        out, err = process.communicate(''.join([f'{cmd}\n' for cmd in commands]), timeout=timeout) 
        returncode = process.returncode
        
        process.kill()
        return out, err, returncode
    
    
    def _neutralize_atoms(self, mol):
        # obtained from http://www.rdkit.org/docs/Cookbook.html#neutralizing-charged-molecules
        # used to neutralize poorly made molecules from .sdf files (e.g. processed PBD files).
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        # mainly for simple cases like COO- and NH3+. 
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        return mol
    