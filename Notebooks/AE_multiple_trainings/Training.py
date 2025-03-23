from training_building_blocks import *
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

# Dataset for tabular data
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Loading data
inputFeature = pd.read_csv('../../Data/NIBRS_ND_2021/processed/input.csv', index_col='Unnamed: 0')
# Separating numerical and categorical features
numerical_features=['population','victim_seq_num','age_num_victim','incident_hour','incident_month','incident_day','incident_dayofmonth','incident_weekofyear']
categorical_features = ['resident_status_code','race_desc_victim',
'ethnicity_name_victim','pub_agency_name','offense_name','location_name','weapon_name'
,'injury_name','relationship_name','incident_isweekend']
# Onehot-encoding categorical features
inputFeature_1h = pd.get_dummies(inputFeature, columns=categorical_features)

# Convert object columns to numeric if they represent categories
for column in inputFeature_1h.select_dtypes(include=['object']):
    inputFeature_1h[column] = inputFeature_1h[column].astype('category').cat.codes

# Train-test split
train, test = train_test_split(inputFeature_1h, test_size=0.1, random_state=42)

# For latent space generation
all_data = pd.concat([train, test], axis=0).sort_index()

# Normalizing numerical features
for feature in numerical_features:
  train[feature] = (train[feature] - train[feature].min()) / (train[feature].max() - train[feature].min())
  test[feature] = (test[feature] - test[feature].min()) / (test[feature].max() - test[feature].min())

# Converting data to tensors
X_train = torch.nan_to_num(torch.Tensor(train.values.astype(np.float32)))
y_train = torch.nan_to_num(torch.Tensor(train.values.astype(np.float32)))

X_test = torch.nan_to_num(torch.Tensor(test.values.astype(np.float32)))
y_test = torch.nan_to_num(torch.Tensor(test.values.astype(np.float32)))

X = torch.nan_to_num(torch.Tensor(all_data.values.astype('float32')))
y = torch.nan_to_num(torch.Tensor(all_data.values.astype('float32')))

# Dataloaders
trainset = MyDataset(X_train, y_train)
testset = MyDataset(X_test, y_test)
dataset = MyDataset(X, y)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Preparing data for the model evaluation
ages = pd.read_csv('../../Data/NIBRS_ND_2021/processed/num_out.csv', index_col='Unnamed: 0').sort_index()
race = pd.read_csv('../../Data/NIBRS_ND_2021/processed/cat_out.csv', index_col='Unnamed: 0')
inputFeature = pd.read_csv('../Data/NIBRS_ND_2021/processed/input.csv', index_col='Unnamed: 0')
inputFeature = inputFeature.join(ages)
inputFeature = inputFeature.join(race)
categorical = inputFeature[['resident_status_code','race_desc_victim',
'ethnicity_name_victim','pub_agency_name','offense_name','location_name','weapon_name'
,'injury_name','relationship_name','incident_isweekend', 'race_desc_offender']]
numerical = inputFeature[['population','victim_seq_num','age_num_victim','incident_hour','incident_month','incident_day','incident_dayofmonth','incident_weekofyear', 'age_num_offender']]

# Specifying model architectures
model_architectures = [{'input': 227,
                      'hidden': [128, 64],
                      'out_act': 'linear'}, 
                      {'input': 227,
                      'hidden': [128, 64, 32],
                      'out_act': 'linear'},
                      {'input': 227,
                      'hidden': [128, 64, 32],
                      'out_act': F.sigmoid},
                      {'input': 227,
                      'hidden': [128, 64, 32, 16],
                      'out_act': 'linear'},
                      {'input': 227,
                      'hidden': [128, 64, 32, 8],
                      'out_act': 'linear'},
                      {'input': 227,
                      'hidden': [128, 64, 32, 16, 2],
                      'out_act': 'linear'},]

locations = [f'model{n+1}' for n in range(len(model_architectures))]

for i, model_architecture in enumerate(model_architectures):
    if not os.path.exists(locations[i]):
        os.mkdir(locations[i])

    model = AutoEncoder(model_architecture['input'], model_architecture['hidden'], model_architecture['out_act'])
    location = locations[i]

    train_losses, test_losses = train_model(model, epochs=20, print_every=40, optimizer=optim.Adam, 
                                        lr=0.001, loss_fun='mixed', trainloader=trainloader, testloader=testloader, location=location, num_features=8, loss_fun1=nn.MSELoss(), loss_fun2=nn.BCELoss(), model_architecture=model_architecture)


    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend()
    plt.savefig(os.path.join(location, 'training.png'))


    model = model.to('cpu')
    model.load_state_dict(torch.load(os.path.join(location, 'model.pt')))
    num_clusters = full_eval(model, dataloader, 20, categorical, numerical, True, location)

    model_data = model_architecture
    model_data['min_trainloss'] = min(train_losses)
    model_data['min_testloss'] = min(test_losses)
    model_data['num_clusters'] = num_clusters

    with open('model_data.jsos', 'w') as file:
        json.dump(model_data, file)



