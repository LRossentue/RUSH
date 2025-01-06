__all__ = ["RushSCORE"]

from dataclasses import dataclass, asdict, field
from typing import List, Tuple
import logging

import rdkit 
from rdkit import Chem

from RUSH.scoring_plugins.REINVENT4.reinvent_plugins.decorators import ComponentResults, molcache, add_tag
from RUSH.scoring_plugins.REINVENT4.reinvent_plugins.decorators import BaseParameters

from RUSH.scripts.RuSH import RuSHScorer

logger = logging.getLogger("reinvent")

@add_tag("__parameters")
@dataclass
class Parameters(BaseParameters):    
    output_dir : List[str] =  "~/RUSH"
    # the order of these molecules should be consistent.
    database_from_smiles : List[bool] = False
    reference_smiles : List[List[Tuple[str, Tuple[str], str]]] = field(default_factory=lambda: list(zip('', ('', ''), '')))
    database_path : List[str] = "~/RUSH/data/PDB_structures/pim447.sdf"

    partial_reward : List[float] = 0.3
    allowance      : List[float] = 0.9 
    
    oeomega_CA     : List[str] = 'classic'        
    oeomega_rms    : List[float] = 0.5
    n_conformers   : List[int] = 32   

    max_centers    : List[int] = 6
    max_molwt      : List[int] = 500
    max_rotors     : List[int] = 10
    
    roc_maxconfs   : List[int] = 100
    roc_besthits   : List[int] = 500
    roc_timeout    : List[int] = 1000
    score_cutoff   : List[float] = 0.8
    
    mcquery        : List[float] = True
    nostructs      : List[float] = True

    shape_weight   : List[float] = 1.0
    color_weight   : List[float] = 1.0
    jacc_weight    : List[float] = 1.0
    rocs_weight    : List[float] = 1.0
    score_operator : List[str] = 'mean'
    num_cores      : List[int] = 10
    
@add_tag("__component")
class RuSHScore:
    def __init__(self, params: Parameters):        
        self.endpoints = params.get_endpoints()

        for endpoint in self.endpoints:
            endpoint.RuSHScorer = RuSHScorer(**asdict(endpoint))

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> ComponentResults:
        scores = []
        for endpoint in self.endpoints:
            endpoint_mols = mols.copy()
            endpoint_scores = endpoint.RuSHScorer(endpoint_mols)
            
            scores.append(endpoint_scores)
        
        return ComponentResults(scores=scores)
    