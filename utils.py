import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

def get_weight_matrix(batch_size, window_size):
    win_index = list(range(0,window_size))
    win_column = list(range(0,window_size))

    weight_df = pd.DataFrame(index = win_index, columns = win_column)
    for idx, row in weight_df.iterrows():
        for column in win_column:
            if idx > column:
                row[column] = np.nan
            elif idx <= column:
                row[column] = np.log1p(window_size - (column - idx))/np.log1p(window_size)

    weight_df = weight_df.fillna(0)

    weight_array = weight_df.to_numpy()
    list_weight = []

    for batch_num in range(1, batch_size+1):
        list_weight.append(weight_array)

    weight_matrix = np.array(list_weight, dtype=np.float64)
    return weight_matrix

def get_target_dims(dataset):
    if dataset == "SWAT":
        return None
    elif dataset == "SMAP":
        return [0]
    elif dataset == "MSL":
        return [0]
    elif dataset == "WADI":
        return None
    else:
        raise ValueError("unknown dataset " + str(dataset))

def data_get(dataset, dif_n):
    train_orig = pd.read_csv(f"./datasets/{dataset}/preprocess/train_normalization.csv",index_col=0)
    train = train_orig.to_numpy()

    test_orig = pd.read_csv(f"./datasets/{dataset}/preprocess/test_normalization.csv",index_col=0)
    test = test_orig.to_numpy()

    label_orig = pd.read_csv(f"./datasets/{dataset}/preprocess/label.csv",index_col=0)
    label = label_orig.values
    label = label[dif_n:]
    label_bool = label.astype(bool)

    # perform differencing
    dif_train = train.copy()
    dif_test = test.copy()
    for _ in range(dif_n):
        dif_train= dif_train[1:]-dif_train[:-1]
        dif_test= dif_test[1:]-dif_test[:-1]

    # match length between origin & differencing
    train=train[dif_n:]
    test=test[dif_n:]

    return (train, test, label_bool, dif_train, dif_test)

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window

def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def load(model, PATH, device="cpu"):
    model.load_state_dict(torch.load(PATH, map_location=device), strict=False)