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


import re
import os
import numpy as np
import pandas as pd

from typing import List

import rdkit
from rdkit import Chem
from rdkit import RDLogger     
from rdkit.Chem import AllChem, rdDistGeom, rdFMCS

from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdmolops import GetMolFrags, AddHs, FragmentOnBonds
from rdkit.Chem.rdFMCS import BondCompare, AtomCompare, RingCompare

from typing import List, Tuple


class ScaffoldFinder():
    def __init__(self, 
                 reference_decorations : List[Tuple[Chem.Mol]], 
                 allowance : float = 0.9, 
                 output_dir : str = os.getcwd(), 
                 name_mols : bool = False, 
                 write_results: bool = False, **kwds):
        
        self.output_file = os.path.join(output_dir, 'linkers.csv') # cd by default.
        self.allowance  = allowance      
        self.reference_decorations = reference_decorations
        self.name_mols = name_mols
        self.write_results = write_results
    
    def __call__(self, *args) -> pd.DataFrame:
        return self.process_molecules(*args)
    
    def process_molecules(self, molecules: List[Chem.Mol], names: List[str] = None) -> pd.DataFrame:
        # for each molecule, attempt to retrieve a scaffold by identifying and removing a set of reference decorations.
        # return (and write) all results to an output df/csv. 
        results_df = pd.DataFrame(columns=['ID', 'molecule', 'scaffold', 'scaffold_success', 
                                           'reference_ID', 'indices_scaffold', 'indices_decorations'])
        
        if self.name_mols: self._name_mols(molecules, names)
        
        for i, mol in enumerate(molecules):
            for j, decorations in enumerate(self.reference_decorations):
                # attempt to obtain a scaffold by identifying and removing the decorations. 
                scaffold_success, scaffold, indices_dict = self._obtain_scaffold(mol, decorations)   
                if int(scaffold_success):
                    break
            # if a set of decorations was positively identified, return the information.
            # also remember which of the reference molecules returned positive for the sample.
            results_df.loc[i] = [mol.GetProp("_Name"), Chem.MolToSmiles(mol), Chem.MolToSmiles(scaffold),
                                 scaffold_success, f"db_{j}", indices_dict['scaffold'], indices_dict['decorations']]
        
        if self.write_results: results_df.to_csv(self.output_file)
        return results_df 
    
    def _name_mols(self, molecules: List[Chem.Mol], names: List[str] = None) -> None:
        """
        Add _Name props to a list of RDKit molecules. 
        Enumerates by default if no names are passed.
        inplace operation.
        """
        if not names: names = [f"Mol_{_}" for _ in range(len(molecules))]
        [mol.SetProp("_Name", name) for mol, name in zip(molecules, names)]
        return None
        
    
    def _obtain_scaffold(self, mol, decorations):
        # try to obtain a scaffold from a mol by identifying and removing a set of decorations.
        # identified decorations are allowed to have minor variation from the reference, ...
        # determined by self.allowance, where 1.0 == exact match, <1.0 fuzzy match based on atom count.
        mol_indices = [a.GetIdx() for a in mol.GetAtoms()]
        # attempt to identify all decorations, and return relevant information for cleaving.
        success, decoration_indices, bonds_to_cleave, anchors = self._identify_decorations(mol, decorations)
        # if succesfully identified;
        if int(success):
            # try to cleave the relevant bonds and obtain a valid scaffold molecule.
            scaffold_success, scaffold = self._cleave_bonds(mol, bonds_to_cleave, anchors)
            # store the relevant indices of atoms belonging to mol, decorations (all), and scaffold for 3D scoring.
            indices_dict = {"mol"        : mol_indices,
                            # we do not include anchors in the decoration
                            "decorations": set(decoration_indices) - set([a.GetIdx() for a in anchors]),
                            # we do include the anchors in the scaffold
                            "scaffold"   : (set(mol_indices) - set(decoration_indices)) | set([a.GetIdx() for a in anchors]),
                           }
            # return the good stuff :)
            return scaffold_success, scaffold, indices_dict
        else:
            # otherwise make the return consistent, but shouldn't be used beyond troubleshooting.
            indices_dict = {"mol": mol_indices,
                            "decorations": decoration_indices, 
                            "scaffold"   : [],
                           }
            return success, mol, indices_dict
    
    
    def _fragment_molecule(self, mol, decoration, forbidden_indices):
        failure_step = 'success' 
        # initialize the global best choice statistics.
        best_atom = None
        best_indices = []
        best_R = []
        best_bond = None
        # find a valid MCS that contains the (*) atom. 
        res = self._get_MCS_res(mol, decoration, forbidden_indices)
        if res:
            # changes the smarts atom query in mol to !D1 (equivalent to the CustomAtomMatcher)
            querymol = self._fix_querymol(res.smartsString)
            # find all matches in mol for the query
            all_m_indices    =    mol.GetSubstructMatches(querymol)
            decoration_indices = decoration.GetSubstructMatch(querymol)
            # filter out substruct matches that contain forbidden indices.
            all_m_indices   = [m_ind for m_ind in all_m_indices if not any(
                                   ind in forbidden_indices for ind in m_ind)]
            # iterate over all matches, and select the best one.
            for m_indices in all_m_indices:
                # first find the anchor atom in mol. 
                mol_anchor = None
                # retrieve the anchor atom in mol through the decoration ...
                # ... and locate the potential bond to cleave.
                for i, f_i in enumerate(decoration_indices):
                    atom = decoration.GetAtomWithIdx(f_i)
                    if atom.GetAtomicNum() == 0: 
                        # MCS indices are ordered across the frag and mol.
                        mol_anchor = mol.GetAtomWithIdx(m_indices[i])
                        # find it's neighbors.
                        n_indices = [n.GetIdx() for n in mol_anchor.GetNeighbors()]
                        # ensure that only 1 of it's neighbors is in the MCS.
                        if sum([i in m_indices for i in n_indices]) == 1:
                            # find its sole neighbor in the MCS.
                            mol_anchor_n_idx = [i for i in n_indices if i in m_indices][0]
                            # find the bond between them, which is a candidate for the final bond to cleave.
                            new_cleave_bond = mol.GetBondBetweenAtoms(mol_anchor.GetIdx(), mol_anchor_n_idx)
                            break # only one exists so we can break here.     
                        else:
                            continue # otherwise finish the loop.
                else:
                    # if multiple neighbors are found, molecule is likely nonsense and non cleavable,
                    # so we break out of this set of indices, and dont store it as a valid MCS. 
                    break    

                # then obtain the number of atoms that exist past the anchor.
                walked, counted = self._walk_neighbors(mol, m_indices, begin_atom=mol_anchor)
                # select the best bond to cleave by maximizing the remainder of the molecule. 
                if len(counted) > len(best_R):
                    best_R = counted
                    best_indices = m_indices
                    best_bond = new_cleave_bond
                    best_atom = mol_anchor
                    
            # now that we identified which set of indices is the best MCS, ...
            # we attempt to remove it along with minor decorations that were not part of the MCS. 
            
            # this check should always be passed, but just in case its here. 
            if best_bond:
                # we attempt to cleave the molecule by fragmenting the best bond. 
                cleaved_mol, cleaved_decoration = self._cleave_mol(mol, best_bond, best_atom)
                # if the molecule was succesfully cleaved in two,
                if cleaved_decoration:  
                    # evaluate if the removed decoration is within the allowance of the reference decoration. 
                    if self.allowance  * (len(decoration.GetAtoms()) -1) <= (
                                  len(cleaved_decoration.GetAtoms()))    <= (
                    (1/self.allowance) * (len(decoration.GetAtoms()) -1)):
                        # i want to retrieve the exact indices of the cleaved decoration.
                        walked, cleaved_indices = self._walk_neighbors(mol, best_R, begin_atom=best_atom)
                        assert cleaved_indices, "troubleshooting"
                        # all checks are passed, so we return succes, along with the mol, decoration and logs.
                        return True, cleaved_mol, cleaved_decoration, cleaved_indices, best_bond, best_atom, failure_step

                    else:
                        failure_step = f'Cleaved decoration size ({len(cleaved_decoration.GetAtoms())}) outside allowance.'
                else:
                    failure_step = 'Molecule not split in two by breaking the cleave bond.'
            else:
                failure_step = 'No cleave bond was identified.'
        else:
            failure_step = 'No valid MCS (allowance & anchor) was found.'
        # shouldn't use this output beyond troubleshooting.
        return False, mol, None, [], best_bond, best_atom, failure_step 
        
    
    def _cleave_mol(self, mol, bond_to_cleave, cleave_atom):
        # attempt to split a molecule into two decorations by cleaving a bond. 
        
        # mark the atom that will remain in the cleaved molecule, i.e. not the cleaved decoration.
        mol.GetAtomWithIdx(cleave_atom.GetIdx()).SetUnsignedProp("cleave_site", True)
        # decoration the molecule on the specified bond.
        mol_fragments = FragmentOnBonds(mol, bondIndices=[bond_to_cleave.GetIdx()], addDummies=False)
        # could fail sanitization or return errors here. Also possible that this does not fragment the molecule.
        try: # try to obtain sanitized fragments.
            mol_fragments = list(GetMolFrags(mol_fragments, asMols = True, sanitizeFrags=True))
        
        except Chem.rdchem.AtomKekulizeException as e:
            # return the orignial molecule and no decoration if cleaving the bond resulted in aromaticity issues.
            # print(f"Kekulization error: {e}. cleaving mol {Chem.MolToSmiles(mol)} failed.")
            return mol, None
        # return the orignial molecule and no decoration if cleaving the bond did not fragment the molecule.
        if len(mol_fragments) < 2:
            return mol, None
        # make sure that nothing weird happened, and that the downstream logic is valid.
        assert len(mol_fragments) == 2, f'mol not properly cleaved. Cleaved {Chem.MolToSmiles(mol)} into" \
        f"{len(mol_fragments)} piece(s) instead: {[Chem.MolToSmiles(mol) for mol in mol_fragments]}'
        # correctly identify which fragment is which after cleaving. 
        for i, frag in enumerate(mol_fragments):
            for atom in frag.GetAtoms():
                try: 
                    if atom.GetUnsignedProp("cleave_site"): 
                        # remove the property again from mol for future decoration(s)
                        mol.GetAtomWithIdx(cleave_atom.GetIdx()).SetUnsignedProp("cleave_site", False)
                        # assign the correct fragments (for readability).
                        cleaved_mol, cleaved_decoration = frag, mol_fragments[1-i]
                        # return the cleaved mol and the fragment that was removed.
                        return cleaved_mol, cleaved_decoration
                except KeyError as e: pass
        # if it failed in any other way, raise an error for troubleshooting.
        raise RuntimeError
        
        
    def _identify_decorations(self, mol, decorations):
        # identify if all decorations are present in mol, and store their information ...
        # for eventual removal and (partial) scoring.
        forbidden_indices = []
        bonds_to_cleave = []
        atoms_to_keep = []
        # track how many decorations could be identified.
        success_rate = 0
        
        for decoration in decorations:
            # go through the fragmenting algorithm for each decoration, and retrieve information.
            # we do not proceed with obtaining the scaffold unless al decorations are present.
            success, _, _, c_indices, c_bond, c_atom, fail = self._fragment_molecule(mol, decoration, forbidden_indices)
            if success:
                forbidden_indices.extend(c_indices)
                bonds_to_cleave.append(c_bond)
                atoms_to_keep.append(c_atom)
                success_rate += 1
            else:
                # print(fail)
                pass
        # int(success) == 1 if all decorations are identified and removable, 0 otherwise.
        success = success_rate/len(decorations)
        # return information for cleaving :) 
        return success, forbidden_indices, bonds_to_cleave, atoms_to_keep
                            
                          
    def _cleave_bonds(self, mol, bonds_to_cleave, cleave_atoms):
        # cleave all bonds at once that connect previously identified decorations to the scaffold.
        success = False
        # mark the atoms that will remain in the cleaved molecule.
        for a in cleave_atoms:
            mol.GetAtomWithIdx(a.GetIdx()).SetUnsignedProp("cleave_site", True)
        # fragment the molecule on the specified bonds, and add dummy atoms.
        mol_fragments = FragmentOnBonds(mol, bondIndices=[b.GetIdx() for b in bonds_to_cleave], 
                                        addDummies=True, dummyLabels=[(0,0)]*len(bonds_to_cleave))
        # could fail sanitization or return errors here. Also possible that this does not fragment the molecule.
        try: # try to obtain sanitized fragments.
            mol_fragments = list(GetMolFrags(mol_fragments, asMols = True, sanitizeFrags=True))
        except Chem.rdchem.AtomKekulizeException as e:
            # return the orignial molecule and no decorations if cleaving the bond resulted in aromaticity issues.
            # print(f"Kekulization error: {e}. cleaving mol {Chem.MolToSmiles(mol)} failed.")
            return success, mol
        # return the orignial molecule and no decorations if cleaving the bonds did not fragment the molecule.
        if len(mol_fragments) < 1 + len(bonds_to_cleave):
            return success, mol
        # make sure that nothing weird happened.
        assert len(mol_fragments) == 1 + len(bonds_to_cleave), f'mol not properly cleaved. Cleaved {Chem.MolToSmiles(mol)} into" \
        f"{len(mol_fragments)} piece(s) instead of {1 + len(bonds_to_cleave)}: {[Chem.MolToSmiles(mol) for mol in mol_fragments]}'
        # correctly identify which fragment is which after cleaving. 
        for frag in mol_fragments:
            # make sure all the marked atoms are present in the scaffold. 
            counts = 0
            for atom in frag.GetAtoms():
                try: 
                    if atom.GetUnsignedProp("cleave_site"): counts += 1
                except KeyError as e: pass
            
            success = counts/len(cleave_atoms)
            # everything is satisfied :) 
            if int(success):
                # remove the properties again after counting. 
                for a in cleave_atoms:
                    mol.GetAtomWithIdx(a.GetIdx()).SetUnsignedProp("cleave_site", False)
                # return the obtained scaffold.
                return success, frag
        # if it failed in any other way, raise an error for troubleshooting.
        raise RuntimeError 
        
        
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

    def _walk_neighbors(self, mol, false_indices, begin_atom, walked = None, counted = None):
        # method to recursively count the number of atoms past a certain begin atom, ... 
        # that are not in the list of false_indices. Used to decide which bond is ...
        # ...best to cleave, when multiple are found in a given mol by _get_MCS_res. 
        if not walked : walked = []
        if not counted : counted = []

        begin_atom = mol.GetAtomWithIdx(begin_atom.GetIdx())
        # for each of the atom's neighbors, count up if conditions are met and recurse.
        for n in begin_atom.GetNeighbors():
            n_i = n.GetIdx()
            if n_i not in false_indices and n_i not in walked and n_i not in counted:
                counted.append(n_i)
                walked, counted = self._walk_neighbors(mol, false_indices, n, walked, counted)
            walked.append(n_i)

        return walked, counted

    
    def _get_MCS_res(self, mol, decoration, avoid_indices=None):
        # must be initialized for each molecule to reset flags and global params. 
        params = rdFMCS.MCSParameters()
        # custom overwrites, requires rdkit-mcs-refactor kernel
        mcs_acceptance = CustomMCSAcceptance(allowance=self.allowance)
        params.ShouldAcceptMCS = mcs_acceptance
        params.AtomTyper = CustomCompareElements(avoid_indices=avoid_indices)

        params.BondTyper = BondCompare.CompareOrderExact
        params.RingTyper = RingCompare.PermissiveRingFusion
        params.AtomCompareParameters.RingMatchesRingOnly = False # actually true, in custom call
        params.AtomCompareParameters.CompleteRingsOnly = False
        params.AtomCompareParameters.MatchValences = True
        params.AtomCompareParameters.MatchChiralTag = False
        params.BondCompareParameters.MatchFusedRings = True
        params.BondCompareParameters.MatchFusedRingsStrict = False
        params.Timeout = 1
        params.MaximizeBonds = True

        # find the best MCS that contains (*) and has atleast allowance*n atoms.
        res = rdFMCS.FindMCS([mol, decoration], params)
        # if flag, set res to None instead.
        if not mcs_acceptance.mcs_found:
            res = None
        return res
    

    
    def _fix_querymol(self, smarts):
        return Chem.MolFromSmarts(''.join(
            "[#0,!D1]" if ('#0' in _) else f'[{_}]' if ('#' in _) else _ for _ in re.split(r"[\[\]]",  
                                                                                           smarts)
                                  ))
    
    
class CustomCompareElements(rdFMCS.MCSAtomCompare):
    def __init__(self, avoid_indices=None):
        super().__init__()
        # a list of atom indices that are not allowed to be in an MCS result. 
        # used to identify multiple MCSs in a row in a given molecule without overlap.
        self.avoid_indices = avoid_indices if avoid_indices else [] 
        
        
    # edited atom compare element for rdFMCS to also match the (*) dummy atom to any other.
    # (*) atom must have a degree of 1 (exit point of the fragment).
    # match must be greater than 1 (connection to remainder of the molecule).
    
    def __call__(self, p, mol1, atom1, mol2, atom2):
        # figure out which is which, assuming mol is larger than fragment.
        mol_atom, fragment_atom = (atom1, atom2) if (len(mol1.GetAtoms()) > len(mol2.GetAtoms())) else (atom2, atom1)
        # if the mol atom is not allowed, break out.
        if mol_atom in self.avoid_indices:
            return False
        
        a1 = mol1.GetAtomWithIdx(atom1)
        a2 = mol2.GetAtomWithIdx(atom2)
        a1_is_dummy = (a1.GetAtomicNum() == 0)
        a2_is_dummy = (a2.GetAtomicNum() == 0)
        if a1_is_dummy ^ a2_is_dummy:
            atoms = (a1, a2)
            dummy_atom_idx = 0 if a1_is_dummy else 1
            other_atom_idx = 1 - dummy_atom_idx
            return atoms[dummy_atom_idx].GetDegree() == 1 and atoms[other_atom_idx].GetDegree() > 1
        if (a1.GetAtomicNum() != a2.GetAtomicNum()):
            return False
        return self.CheckAtomRingMatch(p, mol1, atom1, mol2, atom2)
    

    
class CustomMCSAcceptance(rdFMCS.MCSAcceptance):
    # edited MCS acceptance element for rdFMCS to only accept an MCS if it:
    # 1) contains the (*) dummy atom in the match (see CustomCompareElements above).
    # 2) is at least self.allowance * atom count in size. 
    # this, because we want to identify (sufficiently) complete fragments in mol only.
    # we assume the provded mol is always larger in atom count than the fragment. 
    def __init__(self, allowance):
        super().__init__()
        
        self._allowance = allowance
        self._mcs_found = False
        
    @property
    def mcs_found(self):
        return self._mcs_found  
    
    def __call__(self, query, target, match, match_bonds, params):
        # resolve which is which, assuming mol is larger.
        query_indices, target_indices = tuple(zip(*match))
        mol, fragment, mol_indices, fragment_indices = (query, 
                                                        target,
                                                        query_indices,
                                                        target_indices) if (
            
                      len(query.GetAtoms()) > len(target.GetAtoms())) else (
            
                                                        target, 
                                                        query, 
                                                        target_indices, 
                                                        query_indices)
        
        mol_anchor = None
        # evaluate if anchor in the match, and retrieve the equivalent atom in mol.                           
        for m_i, f_i in zip(mol_indices, fragment_indices):
            f_atom = fragment.GetAtomWithIdx(int(f_i)) # numpy integers raise an error here
            if f_atom.GetAtomicNum() == 0:
                mol_anchor = mol.GetAtomWithIdx(int(m_i))
                break
        
        # only consider matches that contain the anchor and have not been previously cleaved. 
        if mol_anchor:
            # only consider matches that are larger than some allowed fraction * fragment.
            if len(mol_indices) >= self._allowance * len(fragment.GetAtoms()):
                self._mcs_found = True    
                return True
            
        return False
    
