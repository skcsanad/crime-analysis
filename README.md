# Crime analysis project

## **Contents**:
- Files in the Data folder contain the original and processed data used for training
- Files in the Notebooks folder contain the preprocess, and exploration of the data and trainings as well

## **TODOs**
- [x] Add gitignore file
- [x] Implement covariance loss
- [x] Make training loop agnostic to the loss function
- [x] Add tracking and printing of separate parts of loss (reconstruction, covariance and complete loss) to the training loop
- [ ] Before increasing the latent dimension, the model should load back the weigths from the best model that has been saved
- [ ] Create the dockerfile, it should download the data and run preprocess.py upon building the container
- [ ] Add tensorboard tracking to PCAAAutoencoder training loop
- [ ] Clean up the code, create a new branch for this state (clean version will go to the main branch)
- [ ] Add more detailed description to readme

This will be updated with the data sources and a more detailed description