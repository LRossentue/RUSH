# RuSH - Scaffold Hopping with Generative Reinforcement Learning
## Summary

![Test](https://github.com/LRossentue/RUSH/blob/main/Abstract.png)

This repository contains the code to reproduce the transfer learning (RE_TL), reinforcement learning (RE_RL), and baseline experiments (DL, LI_RL, LI_SF) published in our work 'Scaffold Hopping with Generative Reinforcement Learning'. This repository further contains two RuSH scoring plugins adapted for REINVENT3.2 (Link-INVENT) and REINVENT4, as well as standalone scripts to use our ScaffoldFinder and RuSH algorithms. 

A pre-print of the publication is available at:

[Rossen L, Sirockin F, Schneider N, Grisoni F. Scaffold Hopping with Generative Reinforcement Learning.](https://doi.org/10.26434/chemrxiv-2024-gd3j4)

## Features

- Notebooks for reproducing experiments. An example for PIM1 is provided for all published results under RUSH/notebooks.
- Input data for reproducing experiments. Reference structures are obtained from PDB (See Fig. 3) and provided under RUSH/data.
- Scoring plugins for REINVENT 3.2 and 4 are provided under RUSH/scoring_plugins.
- Standalone scripts for using RuSH and ScaffoldFinder are provided under RUSH/scripts.

## Demo

A notebook (RUSH/using_scaffoldfinder.ipynb) is provided to demonstrate ScaffoldFinder/RuSH using the PIM1 case study as an example. We provide a semi-curated list of known PIM1 inhibitors (RUSH/data/PIM1_CHEMBL2147_ligands.csv) retrieved with the ChEMBL REST API for demonstration only.

## Installation

```
# For REINVENT3.2:
1. follow instructions from: https://github.com/MolecularAI/Reinvent
2. replace default reinvent-scoring installation with RUSH/scoring_plugins/REINVENT3.2/reinvent_scoring
   or copy* the components in ~/scoring/score_components/scaffold_hopping to an existing reinvent-scoring.

# For REINVENT4:
1. follow instructions from: https://github.com/MolecularAI/REINVENT4
2. activate reinvent4 and run init_setup.py
3. append RUSH/scoring_plugins/REINVENT4/reinvent_plugins to PYTHONPATH
   or copy the components over to an existing reinvent_plugins.

# For standalone scripts:
follow REINVENT4 instructions, or
conda env create -f rush.yml
```

* *If copying, one must also update the import statement and enumerations where applicable.

## Usage

ScaffoldFinder
```javascript
from RUSH.scripts.scaffoldfinder import ScaffoldFinder
from rdkit import Chem

# very simple molecules as an example.
mols = [Chem.MolFromSmiles(smi) for smi in [
    "CCC(O)CCCCNCO",
    "CCCc1ccccc1CNCO",
    "CC1CCCCC1NCO",
    "C1CCC2CCC1NC2O",
    ]
]
# a single pair of very simple decorations as an example.
decorations = [tuple(Chem.MolFromSmiles(f) for f in reference_decoration_tuple) for reference_decoration_tuple in [('*C', "*O"),]]

SF = ScaffoldFinder(reference_decorations=decorations, name_mols=True)
SF.process_molecules(mols)
```

RuSH (REINVENT4)
```javascript
from RUSH.scoring_plugins.REINVENT4.reinvent_plugins.components.comp_RuSHscore import RuSHScore, Parameters

reference_mols = ['CC1CC(N)CC(c2ccncc2NC(=O)c2ccc(F)c(-c3c(F)cccc3F)n2)C1'] #pim447
reference_decorations = [('*c1cnccc1C1CC(C)CC(N)C1', '*c1c(F)cccc1F'),]
reference_scaffolds = [ "O=C(N[*])c1nc([*])c(F)cc1" ]

# 2 endpoints
parameters = Parameters(database_from_smiles=[True, False],
                        reference_smiles=[list(zip(reference_mols, reference_decorations, reference_scaffolds)),],
                        database_path=f"{RUSH}/data/PDB_structures/pim447.sdf",
                        output_dir=f"{RUSH}",
                        allowance=0.9)

scorer = RuSHScore(parameters)

smiles = [
    "CCC(O)CCCCNCO",
    "CCCc1ccccc1CNCO",
    "CC1CCCCC1NCO",
    "C1CCC2CCC1NC2O",
]

scores = scorer(smiles)
```

## REINVENT4 Scoring Component Reference

The parameterspace below is identical in the standalone script (RuSH.py) and REINVENT3.2 implementation.

### RuSHScore(Parameters)

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `output_dir` | `string` | Where TEMP_DIR is created for temporary files produced by OMEGA/ROCS. |
| `database_from_smiles` | `bool` | If reference poses should be generated (with OMEGA) from SMILES instead. |
| `reference_smiles` | `list` | List of reference molecule tuples to 'hop' from. In the order of [(mol, (fragments), linker),]. |
| `database_path` | `string` | Path to conformer database (SDF or OEB) to use as reference poses in ROCS. |
| `partial_reward` | `float` | Reward given if some, but not all decorations are correctly included in the scored design. |
| `allowance` | `float` | Allowance permits fuzzy fragment identification if <1.0. Formulated as the permitted ratio of number of atoms between the identified decoration and the specified reference decoration. |
| `oeomega_CA` | `string` | See https://docs.eyesopen.com/applications/omega/omega/omega_opt_params.html |
| `oeomega_rms` | `float` | - |
| `n_conformers` | `int` | - |
| `max_centers` | `int` | Filter parameter. Molecules above threshold are not scored. |
| `max_molwt` | `int` | Filter parameter. Molecules above threshold are not scored. |
| `max_rotors` | `int` | Filter parameter. Molecules above threshold are not scored. |
| `roc_maxconfs` | `int` | See https://docs.eyesopen.com/applications/rocs/rocs/rocs_opt_params.html |
| `roc_besthits` | `int` | - |
| `score_cutoff` | `float` | - |
| `roc_timeout` | `int` | Time limit in seconds before ROCS subroutine raises a TimeoutError. |
| `mcquery` | `string` | Set to False if you want to fetch individual conformer hits. |
| `nostructs` | `string` | Set to False for ROCS to also write poses to a file for visual inspection/further handling. |
| `shape_weight` | `float` | Weight to scale the Shape scoring in ROCS. |
| `color_weight` | `float` | Weight to scale the Color scoring in ROCS. |
| `jacc_weight` | `float` | Weight to scale the Jaccard distance scoring (as Harmonic mean). |
| `rocs_weight` | `float` | Weight to scale the Tanimoto Shape & Color scoring (as Harmonic mean). |
| `score_operator` | `string` | How to combine scores if multiple reference molecules are provided. |
| `num_cores` | `int` | Multithreading for ROCS. |

Returns: ComponentResults(scores)

## ScaffoldFinder

### RuSHScore(Parameters)

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `reference_decorations` | `list` | List of tuples containing sets of reference decoration to identify. In the order of List[Tuple[Chem.Mol]]. |
| `allowance` | `float` | Allowance permits fuzzy fragment identification if <1.0. Formulated as the permitted ratio of number of atoms 
| `output_dir` | `string` | Where results (csv) is written if write_results. |
| `name_mols` | `bool` | If molecules should be named by ScaffoldFinder during call. Named molecules are required for processing. |
| `write_results` | `bool` | If results should also be written to a formatted (csv) file during call. |

Returns: pd.DataFrame

## OpenEye License

For using RuSH, an OpenEye software license is required:
```bash
#!/bin/sh
export OE_LICENSE='</path/to/your/oe_license/file>'
```
or consult https://docs.eyesopen.com/applications/common/license.html.


## Authors

- [@LRossentue](https://github.com/LRossentue)
