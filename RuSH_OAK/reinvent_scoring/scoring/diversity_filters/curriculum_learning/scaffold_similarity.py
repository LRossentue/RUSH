from copy import deepcopy

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Scaffolds import MurckoScaffold

from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.memory_record_dto import MemoryRecordDTO
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO


class ScaffoldSimilarity(BaseDiversityFilter):
    """Penalizes compounds based on atom pair Tanimoto similarity to previously generated Murcko Scaffolds."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)
        self._scaffold_fingerprints = {}

    def update_score(self, dto: UpdateDiversityFilterDTO) -> np.array:
        score_summary = deepcopy(dto.score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles

        for i in score_summary.valid_idxs:
            smile = self._chemistry.convert_to_rdkit_smiles(smiles[i])
            scaffold = self._calculate_scaffold(smile)

            # check, if another scaffold should be used as "bucket", because it is very similar as defined by the
            # "minsimilarity" threshold; if not, this call is a no-op and the smiles' normal Murcko scaffold will be used in case
            # -> usage of the "murcko scaffold filter" is actually a special case, where "minsimilarity" is 1.0
            scaffold = self._find_similar_scaffold(scaffold)
            scores[i] = 0 if self._smiles_exists(smile) else scores[i]

            if scores[i] >= self.parameters.minscore:
                loggable_data = self._compose_loggable_data(dto.loggable_data[i]) if dto.loggable_data else ''
                memory_dto = MemoryRecordDTO(i, dto.step, scores[i], smile, scaffold, loggable_data,
                                             score_summary.scaffold_log)
                self._add_to_memory(memory_dto)
                scores[i] = self._penalize_score(scaffold, scores[i])
        return scores

    def _calculate_scaffold(self, smile):
        mol = self._chemistry.smile_to_mol(smile)
        if mol:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = self._chemistry.mol_to_smiles(scaffold)
            except ValueError:
                scaffold_smiles = ''
        else:
            scaffold_smiles = ''
        return scaffold_smiles

    def _find_similar_scaffold(self, scaffold):
        """
        this function tries to find a "similar" scaffold (according to the threshold set by parameter "minsimilarity") and if at least one
        scaffold satisfies this criteria, it will replace the smiles' scaffold with the most similar one
        -> in effect, this reduces the number of scaffold buckets in the memory (the lower parameter "minsimilarity", the more
           pronounced the reduction)
        generate a "mol" scaffold from the smile and calculate an atom pair fingerprint

        :param scaffold: scaffold represented by a smiles string
        :return: closest scaffold given a certain similarity threshold 
        """
        if scaffold is not '':
            fp = Pairs.GetAtomPairFingerprint(Chem.MolFromSmiles(scaffold))

            # make a list of the stored fingerprints for similarity calculations
            fps = list(self._scaffold_fingerprints.values())

            # check, if a similar scaffold entry already exists and if so, use this one instead
            if len(fps) > 0:
                similarity_scores = DataStructs.BulkDiceSimilarity(fp, fps)
                closest = np.argmax(similarity_scores)
                if similarity_scores[closest] >= self.parameters.minsimilarity:
                    scaffold = list(self._scaffold_fingerprints.keys())[closest]
                    fp = self._scaffold_fingerprints[scaffold]

            self._scaffold_fingerprints[scaffold] = fp
        return scaffold

