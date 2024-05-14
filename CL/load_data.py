import os

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from cleanlab.pruning import get_noise_indices
from model import GRU_CNN_Model
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
    left_dataset = LoadData('../RLL/output_left_pos_csv', '../RLL/output_left_neg_csv')
    train_left_dataset, test_left_dataset = train_test_split(left_dataset, test_size=0.2, random_state=42)
    right_dataset = LoadData('../RLL/output_right_pos_csv', '../RLL/output_right_neg_csv')
    train_right_dataset, test_right_dataset = train_test_split(right_dataset, test_size=0.2, random_state=42)
    all_dataset = LoadData('../RLL/output_pos_csv', '../RLL/output_neg_csv')
    train_all_dataset, test_all_dataset = train_test_split(all_dataset, test_size=0.2, random_state=42)

    with open('train_left.pkl', 'wb') as f:
        pickle.dump(train_left_dataset, f)

    with open('train_right.pkl', 'wb') as f:
        pickle.dump(train_right_dataset, f)

    with open('train_all.pkl', 'wb') as f:
        pickle.dump(train_all_dataset, f)

    with open('test_left.pkl', 'wb') as f:
        pickle.dump(test_left_dataset, f)

    with open('test_right.pkl', 'wb') as f:
        pickle.dump(test_right_dataset, f)

    with open('test_all.pkl', 'wb') as f:
        pickle.dump(test_all_dataset, f)

    with open('train_left.pkl', 'rb') as f:
        train_left_dataset = pickle.load(f)

    with open('test_left.pkl', 'rb') as f:
        test_left_dataset = pickle.load(f)

    with open('train_right.pkl', 'rb') as f:
        train_right_dataset = pickle.load(f)

    with open('test_right.pkl', 'rb') as f:
        test_right_dataset = pickle.load(f)

    with open('train_all.pkl', 'rb') as f:
        train_all_dataset = pickle.load(f)

    with open('test_all.pkl', 'rb') as f:
        test_all_dataset = pickle.load(f)

    # datasets = [train_left_dataset, train_right_dataset, train_all_dataset]
    # cleaned_datasets = []
    # model=GRU_CNN_Model(gru_input_size=obj.input_size, gru_hidden_size=obj.hidden_size,
    #                            gru_num_layers=obj.num_layers,
    #                            gru_num_classes=obj.num_classes,
    #                            cnn_input_channels=obj.input_channels, cnn_output_size=obj.output_size).to(device)
    #
    # model.load_state_dict(torch.load('best_model.pth'))
    # left_train_loader = torch.utils.data.DataLoader(train_left_dataset,batch_size=64, shuffle=False, drop_last=False)
    # right_train_loader = torch.utils.data.DataLoader(train_right_dataset,batch_size=64, shuffle=False, drop_last=False)
    # all_train_loader = torch.utils.data.DataLoader(train_all_dataset,batch_size=64, shuffle=False, drop_last=False)
    # total=0
    # noise_data_orders=[]
    # for left_data, right_data, all_data in tqdm(zip(left_train_loader, right_train_loader, all_train_loader)):
    #     left_data[0] = left_data[0].to(device)
    #     left_data[1] = left_data[1].to(device)
    #     right_data[0] = right_data[0].to(device)
    #     right_data[1] = right_data[1].to(device)
    #     all_data[0] = all_data[0].to(device)
    #     all_data[1] = all_data[1].to(device)
    #     left_samples, left_labels = left_data[0], left_data[1]
    #     right_samples, right_labels = right_data[0], right_data[1]
    #     all_samples, all_labels = all_data[0], all_data[1]
    #     # 将三个数据批次作为元组传递给模型进行训练
    #     with torch.no_grad():
    #         _, psx = model(left_samples, right_samples, all_samples)
    #         labels = left_labels.cpu().numpy().astype(int)
    #         psxs = psx.cpu().numpy()
    #
    #         # 只清洗负样本
    #         neg_indices = [i for i in range(len(labels)) if labels[i] == 0]
    #         ordered_label_errors = get_noise_indices(s=labels, psx=psxs, sorted_index_method='normalized_margin')
    #
    #         # 设置清洗的置信度阈值
    #         confidence_threshold = 0.15
    #         # 仅保留高于阈值的负样本
    #         ordered_label_errors = [i for i in ordered_label_errors if
    #                                 labels[i] == 0 and psxs[i, 0] > confidence_threshold]
    #
    #         print(ordered_label_errors)
    #         for k in ordered_label_errors:
    #             noise_data_orders.append(k + total)
    #         total = total + left_samples.size(0)
    #
    # clean_order = list(range(len(train_left_dataset)))
    # for k in noise_data_orders:
    #     clean_order.pop(clean_order.index(k))
    # train_left_dataset_clean=[]
    # train_right_dataset_clean=[]
    # train_all_dataset_clean=[]
    #
    # for i in clean_order:
    #     train_left_dataset_clean.append(train_left_dataset[i])
    #     train_right_dataset_clean.append(train_right_dataset[i])
    #     train_all_dataset_clean.append(train_all_dataset[i])
    return train_left_dataset,train_right_dataset,train_all_dataset, test_left_dataset, test_right_dataset, test_all_dataset

