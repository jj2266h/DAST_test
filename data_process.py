# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:31:47 2022

@author: HD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import preprocessing, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from scipy import interpolate
import scipy.io as sio

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

_cfg = config["cmapss"]
DATASET      = _cfg["dataset"]
dataset_path = _cfg["output_path"]
RUL_max      = _cfg["rul_max"]
window_Size  = _cfg["window_size"]
_data_path   = _cfg["data_path"]

min_max_scaler = preprocessing.MinMaxScaler()

#Import dataset
RUL_DATASET   = np.loadtxt(f'{_data_path}/RUL_{DATASET}.txt')
train_dataset = np.loadtxt(f'{_data_path}/train_{DATASET}.txt')
test_dataset  = np.loadtxt(f'{_data_path}/test_{DATASET}.txt')
train_dataset[:, 2:] = min_max_scaler.fit_transform(train_dataset[:,2:])
test_dataset[:, 2:] = min_max_scaler.transform(test_dataset[:,2:])
train_01_nor = train_dataset
test_01_nor = test_dataset

# Remove op conditions (col 2,3,4) + 7 worthless sensors (s1,s5,s6,s10,s16,s18,s19)
# Applied once to original 26-col array → 16 cols (unit, cycle, 14 sensors)
cols_to_remove = [2, 3, 4, 5, 9, 10, 14, 20, 22, 23]
train_01_nor = np.delete(train_01_nor, cols_to_remove, axis=1)
test_01_nor = np.delete(test_01_nor, cols_to_remove, axis=1)



trainX = []
trainY = []
trainY_bu = []
testX = []
testY = []
testY_bu = []
testInd = []
testLen = []
testX_all = []
testY_all = []
test_len = []

#Training set sliding time window processing
for i in range(1, int(np.max(train_01_nor[:, 0])) + 1):
    ind = np.where(train_01_nor[:, 0] == i)
    ind = ind[0]
    data_temp = train_01_nor[ind, :]
    for j in range(len(data_temp) - window_Size + 1):
        trainX.append(data_temp[j:j + window_Size, 2:].tolist())
        train_RUL = len(data_temp) - window_Size - j
        train_bu = RUL_max - train_RUL
        if train_RUL > RUL_max:
            train_RUL = RUL_max
            train_bu = 0.0
        trainY.append(train_RUL)
        trainY_bu.append(train_bu)


#Test set sliding time window processing
for i in range(1, int(np.max(test_01_nor[:, 0])) + 1):
    ind = np.where(test_01_nor[:, 0] == i)
    ind = ind[0]
    testLen.append(float(len(ind)))
    data_temp = test_01_nor[ind, :]
    testY_bu.append(data_temp[-1, 1])
    if len(data_temp) < window_Size:
        data_temp_a = []
        for myi in range(data_temp.shape[1]):
            x1 = np.linspace(0, window_Size - 1, len(data_temp))
            x_new = np.linspace(0, window_Size - 1, window_Size)
            tck = interpolate.splrep(x1, data_temp[:, myi])
            a = interpolate.splev(x_new, tck)
            data_temp_a.append(a.tolist())
        data_temp_a = np.array(data_temp_a)
        data_temp = data_temp_a.T
        data_temp = data_temp[:, 2:]
    else:
        data_temp = data_temp[-window_Size:, 2:]

    data_temp = np.reshape(data_temp, (1, data_temp.shape[0], data_temp.shape[1]))

    if i == 1:
        testX = data_temp
    else:
        testX = np.concatenate((testX, data_temp), axis=0)
    if RUL_DATASET[i - 1] > RUL_max:
        testY.append(RUL_max)
    else:
        testY.append(RUL_DATASET[i - 1])


#All data processing of test set
#the sliding stride=1.
for i in range(1, int(np.max(test_01_nor[:, 0])) + 1):
    ind = np.where(test_01_nor[:, 0] == i)
    ind = ind[0]
    data_temp = test_01_nor[ind, :]
    data_RUL = RUL_DATASET[i - 1]
    test_len.append(len(data_temp) - window_Size + 1)
    for j in range(len(data_temp) - window_Size + 1):
        testX_all.append(data_temp[j:j + window_Size, 2:].tolist())
        test_RUL = len(data_temp) + data_RUL - window_Size - j
        if test_RUL > RUL_max:
            test_RUL = RUL_max
        testY_all.append(test_RUL)


trainX = np.array(trainX)
testX = np.array(testX)
trainY = np.array(trainY)/RUL_max
trainY_bu = np.array(trainY_bu)/RUL_max
testY = np.array(testY)/RUL_max
testY_bu = np.array(testY_bu)/RUL_max

testX_all = np.array(testX_all)
testY_all = np.array(testY_all)

sio.savemat(f'{dataset_path}/{DATASET}_window_size_trainX.mat', {"train1X": trainX})
sio.savemat(f'{dataset_path}/{DATASET}_window_size_trainY.mat', {"train1Y": trainY})
sio.savemat(f'{dataset_path}/{DATASET}_window_size_testX.mat', {"test1X": testX})
sio.savemat(f'{dataset_path}/{DATASET}_window_size_testY.mat', {"test1Y": testY})


# --- Statistical features (論文 Section IV-B-1c) ---
# 對每個 sliding window 計算 regression coef 和 mean，各作為一列 appended
# window: (T, 14) → (T+2, 14)
_regr = linear_model.LinearRegression()

def _fea_extract1(window):
    """每個感測器對時間步的線性回歸係數"""
    t_idx = np.arange(window.shape[0]).reshape(-1, 1)
    return [float(_regr.fit(t_idx, window[:, s]).coef_[0]) for s in range(window.shape[1])]

def _fea_extract2(window):
    """每個感測器的平均值"""
    return list(np.mean(window, axis=0))

# 計算 train 統計特徵
train_fea1 = [_fea_extract1(w) for w in trainX]   # (N, 14)
train_fea2 = [_fea_extract2(w) for w in trainX]   # (N, 14)

# 用 train 的分布做 MinMaxScaler，再 transform test
scale1 = preprocessing.MinMaxScaler().fit(train_fea1)
scale2 = preprocessing.MinMaxScaler().fit(train_fea2)

train_fea1 = scale1.transform(train_fea1)  # (N, 14)
train_fea2 = scale2.transform(train_fea2)  # (N, 14)

test_fea1 = scale1.transform([_fea_extract1(w) for w in testX])  # (M, 14)
test_fea2 = scale2.transform([_fea_extract2(w) for w in testX])  # (M, 14)

# 在 time 維度 append 兩列統計特徵：(N, T, 14) → (N, T+2, 14)
trainX_new = np.concatenate([
    trainX,
    train_fea1[:, np.newaxis, :],
    train_fea2[:, np.newaxis, :],
], axis=1).astype(np.float32)

testX_new = np.concatenate([
    testX,
    test_fea1[:, np.newaxis, :],
    test_fea2[:, np.newaxis, :],
], axis=1).astype(np.float32)

print(f"trainX_new shape: {trainX_new.shape}")  # (N, 62, 14) for FD002
print(f"testX_new shape:  {testX_new.shape}")

sio.savemat(f'{dataset_path}/{DATASET}_window_size_trainX_new.mat', {"train1X_new": trainX_new})
sio.savemat(f'{dataset_path}/{DATASET}_window_size_testX_new.mat',  {"test1X_new": testX_new})
