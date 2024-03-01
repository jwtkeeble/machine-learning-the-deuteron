## Machine Learning the Deuteron

This repository contains the associated code for '[Machine learning the deuteron](https://doi.org/10.1016/j.physletb.2020.135743)' and focuses on solving the ground-state wavefunction of the deuteron with a neural-network quantum state (NQS) in momentum space.

## Installation

You can reproduce the results of the paper by cloning the repository with,

`git clone https://github.com/jwtkeeble/machine-learning-the-deuteron`

and running the `run.py` script as shown below in the [Usage](#usage) section.

## Requirements

The requirements in order to run this script can be found in `requirements.txt` and can be installed via `pip` or `conda`.

## Usage

The arguments for the `run.py` script are as follows:

| Argument                      | Type    | Default      | Description                                               |
|-------------------------------|---------|--------------|-----------------------------------------------------------|
| `-H`/`--hidden_nodes`         | `int`   | 10           | Number of hidden neurons in the model                     |
| `--preepochs`                 | `int`   | 10000        | Number of pre-epochs for the pretraining phase            |
| `--epochs`                    | `int`   | 250000       | Number of epochs for the energy minimisation phase        |

## License 

The license of this repositry is [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/).
