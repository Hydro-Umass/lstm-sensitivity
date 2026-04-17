# LSTM Sensitivity

This project implements an Entity-Aware LSTM (EALSTM) model for hydrologic prediction and sensitivity analysis, using the CAMELS dataset.

## Project structure

``` dircolors
├── data # raw data storage
│   ├── basin_list.txt # list of basin IDs
│   ├── bids.csv # basin IDs for training
│   ├── camels_*.txt # CAMELS attributes files
│   ├── daymet # Daymet CAMELS meteorological data
│   ├── maurer # Maurer CAMELS meteorological data
│   ├── nldas # NLDAS CAMELS meteorological data
│   ├── usgs # USGS streamflow observations
│   └── vic # VIC parameter files
├── flake.lock
├── flake.nix
├── experiments # experiment definitions and configuration
│   ├── exp1_*_forcings.py # comparing forcing datasets
|   ├── exp2_*_reproducibility.py # hydrologic reproducibility
|   ├── exp3_*_nonphysical.py # non-physical inputs
|   ├── exp4_*_random_error.py # random error in forcings
|   ├── exp5_*_bias.py # bias in foricngs
|   ├── exp6_*_splits.py # different sample splits
├── lstm # source code
│   ├── __init__.py
│   ├── camels.py
│   ├── config.py
│   ├── evals.py
│   ├── models.py
│   └── train.py
├── pyproject.toml
├── README.md
└── uv.lock
```

## Usage

The data from CAMELS to train and evaluate the models need to be downloaded from [here](https://umass-my.sharepoint.com/:u:/g/personal/kandread_umass_edu/IQCdxoqPrrMpSYlrOz6fo5sWAS0gCCKd0nM46Y2acMZ7vz4?e=G0Je9z) and uncompressed in the root directory.

### Running experiments

Each of the experiments can be run with the corresponding file under the `experiments` directory, with command-line arguments used to override any of the default parameters. When an experiment completes output files are saved along with a TOML file that contains the configuration and parameters for that run to ensure reproducibility.

As an example, to run the first experiment with different forcings you would do
```bash
python experiments/exp01_lstm_forcings.py --preload --dropout-rate 0.4 daymet
```

The `--preload` argument is important as the CAMELS data are large enough that they need about 50GB of RAM to be able to run. If you're running this on a smaller machine do not use the preload argument and the code will read the data from an HDF5 file. Training will be slower but the memory footprint will be small.
