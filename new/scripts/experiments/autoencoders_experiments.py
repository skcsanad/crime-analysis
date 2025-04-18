import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import sys
sys.path.append("../..")
from modules.autoencoders import AutoEncoder, MixedActivation, MixedLoss, MyDataset
from modules.modelwithtrainer import EarlyStopping, ModelCheckPoint
import json
import os


# Loading data
inputFeature = pd.read_csv('../../../Data/NIBRS_ND_2021/processed/input.csv', index_col='Unnamed: 0')
# Separating numerical and categorical features
numerical_features=['population','victim_seq_num','age_num_victim','incident_hour','incident_month','incident_day','incident_dayofmonth','incident_weekofyear']
categorical_features = ['resident_status_code','race_desc_victim',
'ethnicity_name_victim','pub_agency_name','offense_name','location_name','weapon_name'
,'injury_name','relationship_name','incident_isweekend']
# Onehot-encoding categorical features
inputFeature_1h = pd.get_dummies(inputFeature, columns=categorical_features)


categorymapper = {
    "all_columns" : categorical_features,
    "all_indices" : [],
    "columns": {},
    "values": {}
}

for column in inputFeature_1h.columns:
    splitted_colname = column.split(" ")
    double_splitted_colname = splitted_colname[0].split("_")
    original_colname = "_".join(double_splitted_colname[:-1])
    if original_colname in categorical_features:
        first_value = double_splitted_colname[-1:]
        column_value = " ".join(first_value + splitted_colname[1:])
        column_index = inputFeature_1h.columns.get_loc(column)
        categorymapper["all_indices"].append(column_index)
        categorymapper["columns"].update({column_index:original_colname})
        categorymapper["values"].update({column_index:column_value})


# Convert object columns to numeric if they represent categories
for column in inputFeature_1h.select_dtypes(include=['object']):
    inputFeature_1h[column] = inputFeature_1h[column].astype('category').cat.codes

# Train-test split
train, test = train_test_split(inputFeature_1h, test_size=0.1, random_state=42)

# Scaling features
scaler = MinMaxScaler().fit(train[numerical_features])
train[numerical_features] = scaler.transform(train[numerical_features])
test[numerical_features] = scaler.transform(test[numerical_features])

# Converting data to tensors
X_train = torch.nan_to_num(torch.Tensor(train.values.astype(np.float32)))
y_train = torch.nan_to_num(torch.Tensor(train.values.astype(np.float32)))

X_test = torch.nan_to_num(torch.Tensor(test.values.astype(np.float32)))
y_test = torch.nan_to_num(torch.Tensor(test.values.astype(np.float32)))

# Dataloaders
trainset = MyDataset(X_train, y_train)
testset = MyDataset(X_test, y_test)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)


#List of model configurations
model_configs = [
    {
        "encoder": nn.ModuleList([nn.Linear(229, 128), nn.ReLU(), nn.BatchNorm1d(128),
                                  nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
                                  nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32),
                                  nn.Linear(32, 16), nn.ReLU(), nn.BatchNorm1d(16),
                                  nn.Linear(16, 2), nn.ReLU(), nn.BatchNorm1d(2)]),

        "decoder": nn.ModuleList([nn.Linear(2, 16), nn.ReLU(), nn.BatchNorm1d(16),
                                  nn.Linear(16, 32), nn.ReLU(), nn.BatchNorm1d(32),
                                  nn.Linear(32, 64), nn.ReLU(), nn.BatchNorm1d(64),
                                  nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
                                  nn.Linear(128, 229), MixedActivation(limits=[8, inputFeature_1h.shape[1]],
                                                                       activations=nn.ModuleList([nn.ReLU(), nn.Sigmoid()]),
                                                                       dim=1)]),

        "save_filename": "ld_2_mse_bce"
    },

    {
        "encoder": nn.ModuleList([nn.Linear(229, 128), nn.ReLU(), nn.BatchNorm1d(128),
                                  nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
                                  nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32),
                                  nn.Linear(32, 16), nn.ReLU(), nn.BatchNorm1d(16)]),

        "decoder": nn.ModuleList([nn.Linear(16, 32), nn.ReLU(), nn.BatchNorm1d(32),
                                  nn.Linear(32, 64), nn.ReLU(), nn.BatchNorm1d(64),
                                  nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
                                  nn.Linear(128, 229), MixedActivation(limits=[8, inputFeature_1h.shape[1]],
                                                                       activations=nn.ModuleList([nn.ReLU(), nn.Sigmoid()]),
                                                                       dim=1)]),

        "save_filename": "ld_16_mse_bce"
    },


    {
        "encoder": nn.ModuleList([nn.Linear(229, 128), nn.ReLU(), nn.BatchNorm1d(128),
                                  nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
                                  nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32)]),

        "decoder": nn.ModuleList([nn.Linear(32, 64), nn.ReLU(), nn.BatchNorm1d(64),
                                  nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
                                  nn.Linear(128, 229), MixedActivation(limits=[8, inputFeature_1h.shape[1]],
                                                                       activations=nn.ModuleList([nn.ReLU(), nn.Sigmoid()]),
                                                                       dim=1)]),

        "save_filename": "ld_32_mse_bce"
    },

    {
        "encoder": nn.ModuleList([nn.Linear(229, 128), nn.ReLU(), nn.BatchNorm1d(128),
                                  nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64)]),

        "decoder": nn.ModuleList([nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
                                  nn.Linear(128, 229), MixedActivation(limits=[8, inputFeature_1h.shape[1]],
                                                                       activations=nn.ModuleList([nn.ReLU(), nn.Sigmoid()]),
                                                                       dim=1)]),

        "save_filename": "ld_64_mse_bce"
    },

]

all_logs = {}
os.makedirs("complete_models", exist_ok=True)

for config in model_configs:
    model = AutoEncoder(encoder=config["encoder"], decoder=config["decoder"])
    # Mixed loss function
    criterion = MixedLoss(limits=[8, inputFeature_1h.shape[1]],
                            losses=nn.ModuleList([nn.MSELoss(), nn.BCELoss()]),
                            dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 40
    print_every = 40
    callbacks = [ModelCheckPoint(monitored_metric="val loss epoch",
                                 minimize_metric=True),
                EarlyStopping(monitored_metric="val loss epoch",
                              minimize_metric=True,
                              patience=5)]
    
    logs = model.fit(optimizer, criterion, epochs, trainloader, testloader, callbacks=callbacks,
                     save_filename=config["save_filename"],
                     embed_hidden=[trainloader, testloader], categorymapper=categorymapper)
        
    all_logs[config["save_filename"]] = logs
    torch.save(model, f"complete_models/{config['save_filename']}.pt")

with open("logs.json", "w") as file:
    json.dump(all_logs, file)


    
    
    





