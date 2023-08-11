from os import makedirs, path
from csv import reader
from ast import literal_eval
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
import pickle

# SMAP & MSL
dataset_list = ['SMAP','MSL']
for dataset in dataset_list:
    dataset_folder = "datasets/data"
    output_folder = "datasets/data/processed"
    makedirs(output_folder, exist_ok=True)
    with open(path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
        csv_reader = reader(file, delimiter=",")
        res = [row for row in csv_reader][1:]
    res = sorted(res, key=lambda k: k[0])
    data_info = [row for row in res if row[1] == dataset and row[0] != "P-2"]
    labels = []
    for row in data_info:
        anomalies = literal_eval(row[2])
        length = int(row[-1])
        label = np.zeros([length], dtype=np.bool_)
        for anomaly in anomalies:
            label[anomaly[0] : anomaly[1] + 1] = True
        labels.extend(label)

    labels = np.asarray(labels)
    print(dataset, "test_label", labels.shape)

    with open(path.join(output_folder, dataset + "_" + "test_label" + ".pkl"), "wb") as file:
        dump(labels, file)

    def concatenate_and_save(category):
        data = []
        for row in data_info:
            filename = row[0]
            temp = np.load(path.join(dataset_folder, category, filename + ".npy"))
            data.extend(temp)
        data = np.asarray(data)
        print(dataset, category, data.shape)
        with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
            dump(data, file)

    for c in ["train", "test"]:
        concatenate_and_save(c)

    with open(f'datasets/data/processed/{dataset}_train.pkl', 'rb') as f:
        train = pickle.load(f)
    with open(f'datasets/data/processed/{dataset}_test.pkl', 'rb') as f:
        test = pickle.load(f)
    with open(f'datasets/data/processed/{dataset}_test_label.pkl', 'rb') as f:
        labels = pickle.load(f) 
    scaler=MinMaxScaler()
    scaler.fit(train)
    train_preprocessed=pd.DataFrame(scaler.transform(train))
    test_preprocessed=pd.DataFrame(scaler.transform(test))
    labels=pd.Series(labels)
    makedirs(f'./datasets/{dataset}_NSR/preprocess', exist_ok=True)
    train_preprocessed.to_csv(f"./datasets/{dataset}_NSR/preprocess/train_normalization.csv")
    test_preprocessed.to_csv(f"./datasets/{dataset}_NSR/preprocess/test_normalization.csv")
    labels.to_csv(f"./datasets/{dataset}_NSR/preprocess/smd_label.csv")

# SWaT
train_orig = pd.read_csv("./datasets/SWAT_NSR/train.csv")
test_orig = pd.read_csv("./datasets/SWAT_NSR/test.csv")

train_orig = train_orig.drop(columns=['Timestamp', 'Normal/Attack'])
test_orig = test_orig.drop(columns=['Timestamp'])
test_orig = test_orig.rename(columns={'Normal/Attack' : 'Attack'})

test_orig.loc[test_orig.Attack=='Normal', 'Attack']=0
test_orig.loc[test_orig.Attack=='Attack', 'Attack']=1.0

for i in list(train_orig): 
    train_orig[i]=train_orig[i].apply(lambda x: str(x).replace("," , "."))
train_orig = train_orig.astype(float)

labels = test_orig['Attack']
test_orig = test_orig.drop(columns=['Attack'])

for i in list(test_orig):
    test_orig[i]=test_orig[i].apply(lambda x: str(x).replace("," , "."))
test_orig = test_orig.astype(float)

data_train = np.asarray(train_orig, dtype=np.float32)
data_test = np.asarray(test_orig, dtype=np.float32)
scaler = MinMaxScaler()
scaler.fit(data_train)
train_scaled = scaler.transform(data_train)
test_scaled = scaler.transform(data_test)
test = pd.DataFrame(test_scaled, columns = train_orig.columns, index=test_orig.index)
train = pd.DataFrame(train_scaled, columns = train_orig.columns, index= train_orig.index)

test.to_csv("./datasets/SWAT_NSR/preprocess/test_normalization.csv")
train.to_csv("./datasets/SWAT_NSR/preprocess/train_normalization.csv")
labels.to_csv("./datasets/SWAT_NSR/preprocess/swat_label.csv")

# WADI Preprocessing
train_orig = pd.read_csv("./datasets/WADI/train.csv", index_col=0)
test_orig = pd.read_csv("./datasets/WADI/test.csv", index_col=0)

train_orig = train_orig.drop(columns=['Date ','Time','2_LS_001_AL','2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS'])
test_orig = test_orig.drop(columns=['Date ','Time','2_LS_001_AL','2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS'])
test_orig = test_orig.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)' : 'Attack'})

if 'Attack' in train_orig.columns:
    train_orig = train_orig.drop(columns=['Attack'])

test_orig.loc[test_orig.Attack==1.0, 'Attack']=0
test_orig.loc[test_orig.Attack==-1.0, 'Attack']=1.0

labels = test_orig['Attack']
test_orig = test_orig.drop(columns=['Attack'])
train_orig = train_orig.fillna(method='ffill')

min_max_scaler = MinMaxScaler()

train_x = train_orig.values
min_max_scaler.fit(train_x)
train_scaled = min_max_scaler.fit_transform(train_x)
train = pd.DataFrame(train_scaled, columns = train_orig.columns, index= train_orig.index)

test_x = test_orig.values
test_scaled = min_max_scaler.transform(test_x)
test = pd.DataFrame(test_scaled, columns = train_orig.columns, index=test_orig.index)

if 'Attack' in train.columns:
    train = train.drop(columns=['Attack'])

test = test.reset_index(drop=True)
train = train.reset_index(drop=True)

test.to_csv("./datasets/WADI/preprocess/test_normalization.csv")
train.to_csv("./datasets/WADI/preprocess/train_normalization.csv")
labels.to_csv("./datasets/WADI/preprocess/label.csv")