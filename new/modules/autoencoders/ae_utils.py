import torch
from torch.utils.data import Dataset
import numpy as np


# Dataset for tabular data
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def remap_metadata(input_array: np.ndarray, categorymapper: dict) -> dict:
    remapped = {col: ["N/A"] * input_array.shape[0] for col in categorymapper["all_columns"]}
    mask = np.zeros(input_array.shape[1], dtype=bool)
    mask[categorymapper["all_indices"]] = True
    input_array[:, ~mask] = 0
    for row in range(input_array.shape[0]):
        true_indices = np.where(input_array[row, :] == 1)[0]
        for index in true_indices:
            orig_col = categorymapper["columns"][index]
            orig_val = categorymapper["values"][index]
            remapped[orig_col][row] = orig_val
    return remapped


def remap_metadata_pt(input_tensor: torch.Tensor, categorymapper: dict) -> dict:
    remapped = {col: ["N/A"] * input_tensor.shape[0] for col in categorymapper["all_columns"]}
    mask = torch.zeros(input_tensor.shape[1], dtype=torch.bool)
    mask[categorymapper["all_indices"]] = True
    input_tensor[:, ~mask] = 0
    for row in range(input_tensor.shape[0]):
        true_indices = torch.where(input_tensor[row, :] == 1)[0]
        for index in true_indices:
            orig_col = categorymapper["columns"][index.item()]
            orig_val = categorymapper["values"][index.item()]
            remapped[orig_col][row] = orig_val
    return remapped