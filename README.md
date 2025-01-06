# RuSH
## Scaffold Hopping with Generative Reinforcement Learning

This repository contains the code to reproduce the transfer learning, reinforcement learning, and baseline experiments published in our work 'Scaffold Hopping with Generative Reinforcement Learning'. This repository further contains scoring plugins for RUSH adapted for REINVENT3.2 and REINVENT4, as well as standalone scripts to use our ScaffoldFinder and RuSHScorer algorithms. 

A pre-print of the publication is available at:

Rossen L, Sirockin F, Schneider N, Grisoni F. Scaffold Hopping with Generative Reinforcement Learning. ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-gd3j4 This content is a preprint and has not been peer-reviewed.

## Features

- Notebooks for reproducing experiments. An example for PIM1 is provided for all published results.
- Scoring plugins for REINVENT 3.2 and 4.
- Standalone scripts for using RuSH and ScaffoldFinder.

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

* If copying, one must also update the import statement and enumerations where applicable.

## Usage

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



## OpenEye License

For using RuSH, an OpenEye software license is required:
```bash
#!/bin/sh
export OE_LICENSE='</path/to/your/oe_license/file>'
or, 
https://docs.eyesopen.com/applications/common/license.html
```

## Authors

- [@username](https://github.com/LRossentue)
