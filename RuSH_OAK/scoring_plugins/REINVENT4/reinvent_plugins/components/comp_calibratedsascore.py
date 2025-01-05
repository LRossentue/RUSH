
__all__ = ["CalibratedSAScore"]

from dataclasses import dataclass, asdict
from typing import List
import logging
import numpy as np

import os
import sys

import rdkit 
from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig, Draw

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
sys.path.append(os.path.join(RDConfig.RDContribDir, 'NP_Score'))

import sascorer
import npscorer

from HARVARD.REINVENT.reinvent_plugins.decorators import ComponentResults, molcache, add_tag
from HARVARD.REINVENT.reinvent_plugins.decorators import BaseParameters
from HARVARD.REINVENT.reinvent_plugins.utils import Sigmoid, smarts_aa

logger = logging.getLogger("reinvent")

@add_tag("__parameters")
@dataclass
class Parameters(BaseParameters):    
    sigmoid_k: List[float] = -8.0
    sigmoid_dx : List[float] = 3.0
    
    # based on AZ seperation power using BCE (-8.0, 2.95) (would approach infinite k, otherwise k=max(set(k_paramspace)))
    
@add_tag("__component")
class CalibratedSAScore:
    def __init__(self, params: Parameters):        
        self.endpoints = params.get_endpoints()

        for endpoint in self.endpoints:
            endpoint.sigmoid = Sigmoid(**asdict(endpoint))

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> ComponentResults:
        scores = []
        for endpoint in self.endpoints:
            raw_scores = [sascorer.calculateScore(mol) for mol in mols]  # 36s/100k
            scores.append(np.array([endpoint.sigmoid(score) for score in raw_scores], dtype=float))
        
        return ComponentResults(scores=scores)
    