# Crime analysis project

## **Contents**
```text
crime_analysis/
├── old/                                    # Containts the old version of the project
    ├── Notebooks/                          # Contains notebooks with trainings/experiments
    ├── pcaautoencoder/                     # Contains the modules used to build the models
                                            # in the newer trainings
        ├── __init__.py                     # Init file
        ├── callbacks.py                    # Contains the callbacks for ModelWithTrainer
        ├── trainer.py                      # Class definitions for Trainer and ModelWithTrainer
        ├── model.py                        # Class definitions for PCAAutoencoder and AutoEncoder
        └── utils.py                        # Contains utilities for the training of autoencoders
    ├── try_pcaautoencoder.ipynb            # Trial training script with the pcaautoencoder
    ├── try_autoencoder.ipynb               # Trial training script with normal autoencoders
    ├── AE_multiple_trainings/              # ...
    └── ...                                 # Notebooks used for experiments -> to be cleaned up
├── new/                                    # Contains the current version of the project
    ├── modules/                            # Contains the modules used during the training
        ├── modelwithtrainer/               # Class  definitons for modelwithtrainer and its callbacks
            ├── __init__.py                 # Init file
            ├── callbacks.py                # Class definitions for the callbacks
            └── trainer.py                  # Class definitions for modelwtithtrainer
        ├── auteoncoders/                   # Model defintions
            ├── __init__.py                 # Init file
            ├── ae_utils.py                 # utility functions for training autoencoders on tabular data
            └── models.py                   # Class definitons for AutoEncoder, PCAAutoEncoder, MixedActivation
                                            # and PCAAELoss
    ├── notebooks/                          # Contains notebooks with the trainings/experiments
        ├── test_modules/                   # Testing if importing modules works with the current folder structure
            ├── try_autoencoder.ipynb       # Testing autoencoders
            └── try_pcaautoencoder.ipynb    # Testing pcaautoencoders
    ├── scripts/                            # Contains scripts with trainings/experiments
        └── experiments/                    # Contains experiments
            └── autoencoders_experiments.py # Contains the script to rerun the original experiments 
├── .gitignore                              # Contains files to be ignored by git
├── Dockerfile                              # Builds the docker image
├── preprocess.py                           # Creates the process data used for training
├── README.md                               # Project description and instructions
└── requirements.txt                        # Dependencies of the project
#################### Created only after running the docker image ####################
└── Data/                                   # Contains the data used for the project
    ├── original/                           # Contains the original .csv files as 
                                            # present at NIBRS website
    ├── processed/                          # Contains the preprocessed version 
                                            # of the data
        ├── input.csv                       # Contains the the data in the format it
                                            # is used to train the autoencoders
        ├── cat_out.csv                     # Categorical output data, previously used
                                            # for predicting categorical features,
                                            # currently not in use
        └── num_out.csv                     # Numerical output data, previously used 
                                            # for predicting numerical features,
                                            # currently not in use
```
## **Usage**
Building the docker image:
```text
docker build -t pytorch-gpu-jupyter .
```
Running the container (automatically downloads and preprocesses the data) and launching jupyter notebook:
```text
docker run --rm -it --gpus all -p 8888:8888 -p 6006:6006 -v "%cd%:/workspace" pytorch-gpu-jupyter
```
Running the container with bash (running the preprocessing script manually):
```text
docker run --rm -it --gpus all -p 8888:8888 -p 6006:6006 -v "%cd%:/workspace" pytorch-gpu-jupyter bash
python preprocess.py
```


## **TODOs**
- [x] Add gitignore file
- [x] Implement covariance loss
- [x] Make training loop agnostic to the loss function
- [x] Add tracking and printing of separate parts of loss (reconstruction, covariance and complete loss) to the training loop
- [x] Before increasing the latent dimension, the model should load back the weigths from the best model that has been saved
- [x] Create the dockerfile
- [x] Clean up requirements.txt
- [x] Add tensorboard tracking to PCAAAutoencoder training loop
- [x] Add global epochs and global steps to loss logging
- [x] Make model saving monitor reconstruction loss in the training loop when PCAAE_Loss is applied
- [x] Create separate module with model definiton and modular training loop
- [x] Fix model checkpoints (so that they monitor the list in the logs)
- [x] Add way to remap original categories to the 1h converted, shuffled samples
- [x] Add encoding the validation set to validation loop and plotting the hidden space in tensorboard
- [x] Modify training loop for PCAAutoencoder to remap the categories
- [x] Add normal autoencoder definiton to models module
- [x] Add automatic downloading and preprocessing of the data to the dockerfile
- [x] Fix tensorboard session starting bug
- [X] Clean up the code, create a new folder for the tidy version of experiments
- [X] Make PCAAutoencoder child class of autoencoder
- [X] Make it optional to embed the hidden space of all of the training data along with the validation data
- [X] Create class for mixed activation function
- [X] Rerun experiments and save their logs in the new folder
- [ ] Add_metrics and log_metrics in Trainer should not be static methods
- [ ] Test out gpu access within docker image on another machine
- [ ] Add more detailed description to readme

This will be updated with the data sources and a more detailed description