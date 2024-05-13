import os

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from cleanlab.pruning import get_noise_indices
from model import GRU_CNN_Model_p
import pickle
from tqdm import tqdm
from parameters import get_parameters

obj = get_parameters()

class CSVDataset(Dataset):
    def __init__(self, root_dir, label):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        data = pd.read_csv(file_path, header=None)
        data_tensor = torch.tensor(data.values, dtype=torch.float32)
        label_tensor = torch.tensor(self.label, dtype=torch.float32)
        return data_tensor, label_tensor

def LoadData(root_dir_pos, root_dir_neg):
    csv_dataset_pos = CSVDataset(root_dir_pos, label=1)
    csv_dataset_neg = CSVDataset(root_dir_neg, label=0)
    csv_dataset = torch.utils.data.ConcatDataset([csv_dataset_pos, csv_dataset_neg])
    return csv_dataset

def split_data(device):
    with open('train_left_clean.pkl', 'rb') as f:
        train_left_dataset = pickle.load(f)

    with open('test_left.pkl', 'rb') as f:
        test_left_dataset = pickle.load(f)

    with open('train_right_clean.pkl', 'rb') as f:
        train_right_dataset = pickle.load(f)

    with open('test_right.pkl', 'rb') as f:
        test_right_dataset = pickle.load(f)

    with open('train_all_clean.pkl', 'rb') as f:
        train_all_dataset = pickle.load(f)

    with open('test_all.pkl', 'rb') as f:
        test_all_dataset = pickle.load(f)

    return train_left_dataset,train_right_dataset,train_all_dataset, test_left_dataset, test_right_dataset, test_all_dataset

