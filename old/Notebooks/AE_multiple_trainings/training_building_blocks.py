import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.cm as cm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
import os
import json


# Model
class AutoEncoder(nn.Module):
  def __init__(self, input_size, hidden_layers, out_act, **kwargs):
    super().__init__()
    # Layers
    self.layers  = nn.ModuleList()
    self.out_act = out_act
    self.out_params = kwargs
    # Encoder
    # Creating first layer
    self.num_encoding_layers = 0
    self.layers.append(nn.Linear(input_size, hidden_layers[0]))
    self.num_encoding_layers += 1
    self.layers.append(nn.BatchNorm1d(hidden_layers[0]))
    self.num_encoding_layers += 1
    # Creating other encoder layers specified in hidden_layers
    for i in range(1, len(hidden_layers)):
        self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        self.num_encoding_layers += 1
        self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
        self.num_encoding_layers += 1
      
    # Decoder
    self.num_decoding_layers = 0
    for i in reversed(range(1, len(hidden_layers))):
        self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i-1]))
        self.num_decoding_layers += 1
        self.layers.append(nn.BatchNorm1d(hidden_layers[i-1]))
        self.num_decoding_layers += 1
    self.layers.append(nn.Linear(hidden_layers[0], input_size))
    self.num_decoding_layers += 1

  # Encoding
  def encode(self, x):
    for i in range(1, self.num_encoding_layers, 2):
        #print(self.layers[i-1])
        x = self.layers[i-1](x)
        #print(self.layers[i])
        x = self.layers[i](x)
        x = F.relu(x)
    return x

  # Decoding
  def decode(self, x):
    for i in range(1, self.num_decoding_layers-1, 2):
        #print(self.layers[i + self.num_encoding_layers - 1])
        x = self.layers[i + self.num_encoding_layers - 1](x)
        #print(self.layers[i + self.num_encoding_layers])
        x = self.layers[i + self.num_encoding_layers](x)
        x = F.relu(x)
    x = self.layers[-1](x)
    if self.out_act == 'linear':
        return x
    elif self.out_act == 'mixed' :
        x_num = self.out_params['num_act'](x[:, :self.out_params['num_features']])
        x_cat = self.out_params['cat_act'](x[:, self.out_params['num_features']:])
        x = torch.cat((x_num, x_cat), dim=1)
    else:
        return self.out_act(x)
    
  def forward(self, x):
    enc = self.encode(x)
    dec = self.decode(enc)
    return dec


def train_model(model, epochs, print_every, optimizer, lr, loss_fun, trainloader, testloader, location, **kwargs):
    # Saving model architecture if specified
    if 'model_architecture' in kwargs.keys():
        model_architecture = kwargs['model_architecture']
        model_architecture['out_act'] = model_architecture['out_act'].__name__
        with open(os.path.join(location, 'model_architecture.json'), 'w') as file:
            json.dump(kwargs['model_architecture'], file)
    
    # Training on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Setting optimizer
    optimizer = optimizer(model.parameters(), lr=lr)

    # Defining loss function(s)
    if not loss_fun == 'mixed':
        criterion = loss_fun
    else:
        criterion1 = kwargs['loss_fun1']
        criterion2 = kwargs['loss_fun2']

    steps = 0
    train_losses, test_losses = [], []
    test_loss_min = np.Inf

    # Training loop
    for e in range(epochs):
      running_loss = 0
      running_loss_ = 0
      for X, y in trainloader:
        steps += 1
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        if loss_fun != 'mixed':
            loss = criterion(output)
        else:
            loss1 = criterion1(output[:, :kwargs['num_features']], y[:, :kwargs['num_features']])
            loss2 = criterion2(output[:, kwargs['num_features']:], y[:, kwargs['num_features']:])
            loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_loss_ += loss.item()
    
        if steps % print_every == 0:
          print(f'Epoch {e+1}/{epochs}, Step {steps}, Train Loss: {running_loss_/print_every:.3f}')
          running_loss_ = 0
    
    
      # Model evaluation on the test data
      else:
        running_testloss = 0
        with torch.no_grad():
          model.eval()
          for X, y in testloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            if loss_fun != 'mixed':
                loss = criterion(output)
            else:
                loss1 = criterion1(output[:, :kwargs['num_features']], y[:, :kwargs['num_features']])
                loss2 = criterion2(output[:, kwargs['num_features']:], y[:, kwargs['num_features']:])
                loss = loss1 + loss2
            running_testloss += loss.item()
        model.train()
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(running_testloss/len(testloader))
        
        # Saving model if results improve
        if test_losses[-1] <= test_loss_min:
            print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            test_loss_min,
            test_losses[-1]))
            torch.save(model.state_dict(), os.path.join(location, 'model.pt'))
                    
        print(f'Epoch {e+1}/{epochs}, Train Loss: {running_loss/len(trainloader):.3f}, Test Loss: {running_testloss/len(testloader):.3f}')
        
        # Early stopping with a patience of five
        if e >= 5:
          if len([*filter(lambda x: x < test_loss_min, test_losses[-5:])]) == 0:
             return (train_losses, test_losses)
          
    return (train_losses, test_losses)
    

# Creating latent space
def create_latent_space(model, dataloader):
    with torch.no_grad():
        model.eval()
        full_latent_space = []
        for X, _ in tqdm(dataloader):
            latent_space = model.encode(X)
            full_latent_space.append(latent_space)
    
        latent_space = torch.cat(full_latent_space, dim=0)
        latent_space = latent_space.numpy()
    return latent_space

# Further reducing latent space dimensinality to 2 variables if needed
def reduce_2_2var(latent_space):
    # Dimension reduction to 2 with PCA if neccesary
    if latent_space.shape[1] != 2:
        pca_2var = PCA(n_components=2)
        latent_space_2var = pca_2var.fit_transform(latent_space)
        return latent_space_2var
    else:
        return latent_space

# Clustering
def clustering(X, max_clusters, create_fig, location):
    # Function to calculate WCSS
    def calculate_wcss(X, max_clusters):
        wcss = []
        for i in range(1, max_clusters):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        return wcss

    # Calculate WCSS for different values of k
    wcss = calculate_wcss(X, max_clusters)
    # Use KneeLocator to find the elbow point
    kneedle = KneeLocator(range(1, max_clusters), wcss, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow
    # Kmeans with the optimal number of clusters
    kmeans_actual = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    kmeans_actual.fit(X)

    if create_fig:
        # Colormap for clustering
        cmap = cm.get_cmap('viridis', optimal_k)
        # Create figure
        f, ax = plt.subplots(2, 1, figsize=(6, 8))
        # Plot the Elbow Method graph
        ax[0].plot(range(1, 20), wcss, marker='o')
        ax[0].vlines(optimal_k, ax[0].get_ylim()[0], ax[0].get_ylim()[1], linestyles='dashed')
        ax[0].set_title('Elbow Method')
        ax[0].set_xlabel('Number of clusters')
        ax[0].set_ylabel('WCSS')
        ax[1].scatter(X[:, 0], X[:, 1], c=kmeans_actual.labels_, cmap=cmap)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10) for i in range(optimal_k)]
        labels = [f'Cluster {cluster}' for cluster in range(optimal_k)]
        ax[1].legend(handles, labels, title='Cluster')
        plt.savefig(os.path.join(location, 'clustering_fig.png'))
    
    return kmeans_actual

def eval_clustering(clustering, categorical, numerical, create_fig, location):
    categorical.fillna('unknown')
    categorical['cluster'] = clustering.labels_
    numerical['cluster'] = clustering.labels_
    # Label encoding
    le = LabelEncoder()
    for col in categorical.columns:
        categorical[col] = le.fit_transform(categorical[col])

    # Calculating chi-scores on the categorical data
    chi_scores = chi2(categorical, categorical['cluster'])
    chi_values = pd.Series(chi_scores[0], index=categorical.columns)
    chi_values_sorted = chi_values.sort_values(ascending=False)
    p_values = pd.Series(chi_scores[0], index=categorical.columns)
    p_values_sorted = p_values.sort_values(ascending=False)
    chi_out = pd.DataFrame({'chi': chi_values, 'p': p_values})

    # Comparing the numerical features
    numerical_sigmas = numerical.drop(columns='cluster').var()
    numerical_relative_sigmas = [pd.DataFrame({cluster: numerical.loc[numerical['cluster'] == cluster].drop(columns=['cluster']).var() / numerical_sigmas}) 
                                 for cluster in sorted(numerical['cluster'].unique())]
    numerical_relative_sigmas = pd.concat(numerical_relative_sigmas, axis=1)
    numerical_relative_sigmas['mean'] = numerical_relative_sigmas.mean(axis=1)
    sorted_sigmas = numerical_relative_sigmas.sort_values(ascending=False, by='mean')
    
    
    if create_fig:
        f, ax = plt.subplots(1, 3, figsize=(12, 4))
        chi_values_sorted.plot.bar(ax=ax[0])
        p_values_sorted.plot.bar(ax=ax[1])
        ax[2].bar(sorted_sigmas.index, sorted_sigmas['mean'])
        ax[2].set_xticks(sorted_sigmas.index)
        ax[2].set_xticklabels(sorted_sigmas.index, rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(location, 'chi_square.png'))

    return chi_out, numerical_relative_sigmas
    


def full_eval(model, dataloader, max_clusters, categorical, numerical, create_fig, location):
    latent_space = create_latent_space(model, dataloader)
    latent_space = reduce_2_2var(latent_space)
    clusters = clustering(latent_space, max_clusters, create_fig, location)
    num_clusters = np.unique(clusters.labels_)
    chi, sigmas = eval_clustering(clusters, categorical, numerical, create_fig, location)
    chi.to_csv(os.path.join(location, 'chi.csv'))
    sigmas.to_csv(os.path.join(location, 'sigmas.csv'))
    return num_clusters

