# Estimating high-resolution profiles of wind speeds from a global reanalysis dataset using TabNet
## Introduction 

This repository contains the code and resources for the "Estimating high-resolution profiles of wind speeds from a global reanalysis dataset using TabNet" work, which is submitted as a full paper to the Climate Informatics 2024 conference, and will be published as an article in the Environmental Data Science jourbal. The objective of this work is to predict vertical wind speed profiles provided by the meteorological features from ERA5 reanalysis dataset. In doing so, 25 meteorological variables from the ERA5 data are obtained for input features. On the otherhand, the CERRA analysis wind speeds at 12 vertical levels are obtained for targets. Our innovation lies in the idea to make the methodology generic across diverse target datasets. For that, the target wind speeds at 12 vertical levels are transformed into 5 coefficients by Chebyshev polynomial approximation. Now, the DL models trained for predicting 5 coefficients, instead of 12 wind speeds, making the models more prone to learning the physics. Further details of the methodology are presented in the main manuscript. 

## Hardware requirements
1. Operating system 
    -   PRETTY_NAME="Ubuntu 22.04.4 LTS"
    -   NAME="Ubuntu"
    -   VERSION_ID="22.04"
    -   VERSION="22.04.4 LTS (Jammy Jellyfish)"
    -   VERSION_CODENAME=jammy
    -   ID=ubuntu
    -   ID_LIKE=debian
2. GPUs
    - At least one NVIDIA RTX A6000 (or) Quadro P5000 GPU, with 16GB memory
    - Build cuda_11.5.r11.5/compiler.30672275_0
3. CPUs
    - At least 8 cores
4. Disk space
    - At least 5 GB

## Getting strated guide

To run this project locally, follow these steps:

1. Clone the repository: `git clone git@github.com:HarishBaki/CI2024.git`
2. Install the required conda dependencies through the `TabNet.yml` conda environment file

    ` conda env create -f TabNet.yml `  

## Step-by-step instructions for usage
### Data
This data 3.1.1 and 3.1.2 The necessary data for this work can be obtained and setup in the pipline as follows:
1. Create `data` folder within the repository. 
2. Download the data from zenodo repository: `https://doi.org/10.5281/zenodo.13855454` and place it within the `data` directory. It will consist of ERA5.nc, 2000.nc and 2001.nc files.
3. Create `data/CERRA_height_level` folder and move `data/2000.nc and data/2001.nc` file into `data/CERRA_height_level/`. This data consists of CERRA vertical wind speed profiles, at 12 vertical levels, as discribed in section 3.1.1 in the main article.
4. The ERA5.nc file consists of several meteorological variables (both original and derived).  Among the variables, 24 are utilized in training the TabNet models, as described in section 3.1.2 of the main article.

### Chebyshev  coefficient estimation
-   As mentioned, the innovation of our idea lies in the fact that the methodology is generic across diverse target wind speed profile datasets. 
-   To compute the Chebyshev polynomials of CERRA wind speed profiles, during 2000 and 2001 years, execute the `Chebyshev_coefficient_estimation.py` as:

    `python Chebyshev_coefficient_estimation.py`

- This will create `Chebyshev_Coefficnents.nc` file in data directory. 
- The core function is well documented in `libraries.py` file, which is detailed in section 3.2 of the main article.

### Experimental setup: TabNet model training
To train the TabNet model for predicting the Chebyshev coefficients of vertical wind speed profiles proided by ERA5 input features, follow these steps:

1. The TabNet_multioutput.py file contails all the necessary script to run tabnet.
2. This file requires three inputs, 
    -   A config_file, describing the details of
        -   Input meteorological file, 
        -   Input Chebyshev coefficient file,
        -   Input variables (25 in our case), 
        -   Train dates range,
        -   Train location (among the 11 locations of the dataset, which were initially designed for an elaborate experiments, though only used one location in this work),
        -   Options for n_d, 
        -   Options for n_steps,
        -   Options for n_independent,
        -   Options for n_shared,
        -   Options for gamma,
        -   Options for nTrials (number of trials for hyperparameter tuning via random search)
        -   Test dates range, 
        -   Test location,
        -   Experimental indice (8 in our case).
    -   The indices of target variables you want to train, in squre brackets seperated by comma
    -   The ensemble number
3. An example of execution is 

    ` python TabNet_multioutput.py "config.yaml" "[0,1,2,3,4]" "0" `
    - Here, the training instructions are provided through config.yaml. Since the nTrials is set 100 within config.yaml, it will run for 100 trials
    - It trains for all the target variables, that are the Chebyshev coefficients from C0 to C4
    - The training ensemble indice is 0 

4. Once the training finishes for the ensemble run, the best trained model out of the trials and the corresponding min_max_scaler are saved in trained_models/models_8th_set/Ens_0/
5. To run all the 10 ensembles with single file, and utilize the computational facility by running trials in parallel, a bash script is provided `run_all_TabNet.sh`, which can run 10 ensemble runs (from 0 to 9).

### Reproducing figures
5. The CI2024_plots notebook is used to create illustrations for the publication.

## Contact

If you have any questions or suggestions, feel free to contact [Harish Baki, h.baki@tudelft.nl].
