__all__ = ["molcache", "add_tag", "ComponentResults"]

from dataclasses import dataclass, asdict, field, replace
from typing import List, Dict, Optional, Callable, Any
from rdkit import Chem
import numpy as np
import logging

logger = logging.getLogger("reinvent")
cache = {}

# QoL additions to REINVENT for scoring plugins.

@dataclass
class BaseParameters:
    """
    Base class for REINVENT scoring component Parameters dataclass.
    Mostly QoL for dealing with multiple endpoints and default parameters.
    
    simply inherent and pass default parameters like so:
        class Parameters(BaseParameters):
            var: List[bool] = True
            
    use get_endpoints() to unpack into a list of endpoints:
            self.endpoints = params.get_endpoints()
            for endpoint in self.endpoints:
                endpoint.object = Object(**asdict(endpoint))
                
    """
    do_post : bool = True          
    
    def __post_init__(self):
        if self.do_post:
            for name, field in self.__class__.__dataclass_fields__.items():
                value = getattr(self, name)
                if not isinstance(value, list):
                    setattr(self, name, [value])
            
            max_len = max([len(getattr(self, name)) for name in self.__dataclass_fields__])
            for name in self.__dataclass_fields__:
                current = getattr(self, name)
                if len(current) == 1:
                    setattr(self, name, current * max_len)
        
        
    def get_endpoints(self, ) -> List[Any]:
        """
        QoL function to remap Parameters dataclass into a list of Parameters, one per endpoint. 
        No nested param lists.
        """
        params_dict = {k:v for k,v in asdict(self).items() if k != 'do_post'}
        num_endpoints = len(next(iter(params_dict.values())))
        
        return [
            self.__class__(do_post=False, **{
                key: values[i] if isinstance(values, list) else values 
                for key, values in params_dict.items()
            })
            for i in range(num_endpoints)
        ]


"""
Copy pasta from https://github.com/MolecularAI/REINVENT4/tree/main/reinvent_plugins 
So we don't need to deal with relative import issues if using components outside of the main reinvent loop.
"""

def molcache(func: Callable):
    def wrapper(self, smilies: List[str]):
        mols = []

        for smiles in smilies:
            if smiles in cache:
                mol = cache[smiles]
            else:
                mol = Chem.MolFromSmiles(smiles)
                cache[smiles] = mol

                if not mol:
                    logger.warning(f"{__name__}: {smiles} could not be converted")

            mols.append(mol)

        return func(self, mols)

    return wrapper


def add_tag(label: str, text: str = "True"):
    def wrapper(cls):
        setattr(cls, label, text)
        return cls

    return wrapper


@dataclass
class ComponentResults:
    scores: List[np.ndarray]
    scores_properties: Optional[List[Dict]] = None
    uncertainty: Optional[List[np.ndarray]] = None
    uncertainty_type: Optional[str] = None
    uncertainty_properties: Optional[List[Dict]] = None
    failures_properties: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None