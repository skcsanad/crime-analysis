# Crime analysis project

## **Contents**:
- Files in the Data folder contain the original and processed data used for training
- Files in the Notebooks folder contain the preprocess, and exploration of the data and trainings as well

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
- [ ] Clean up the code, create a new folder for the tidy version of experiments
- [ ] Rerun experiments and save their logs
- [ ] Test out gpu access within docker image on another machine
- [ ] Add more detailed description to readme

This will be updated with the data sources and a more detailed description