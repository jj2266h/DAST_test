# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:31:47 2022

@author: HD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from scipy import interpolate
import scipy.io as sio
from numpy import *

min_max_scaler = preprocessing.MinMaxScaler()


#Import dataset
RUL_F001 = np.loadtxt('Cmapss_data/RUL_FD001.txt')
train_F001 = np.loadtxt('Cmapss_data/train_FD001.txt')
test_F001 = np.loadtxt('Cmapss_data/test_FD001.txt')
train_F001[:, 2:] = min_max_scaler.fit_transform(train_F001[:, 2:])
test_F001[:, 2:] = min_max_scaler.transform(test_F001[:, 2:])
train_01_nor = train_F001
test_01_nor = test_F001

#Delete worthless sensors
# Here, 1 means the original sensor index 0 in the sensor block [s1, ..., s21].
cols_to_delete = [1,5,6,10,16,18,19]
sensor_cols_to_delete = [col + 4 for col in cols_to_delete]
train_01_nor = np.delete(train_01_nor, sensor_cols_to_delete, axis=1)
test_01_nor = np.delete(test_01_nor, sensor_cols_to_delete, axis=1)

#parameters of data process
RUL_max = 125.0  
window_Size = 40 

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
    if RUL_F001[i - 1] > RUL_max:
        testY.append(RUL_max)
        #testY_bu.append(0.0)
    else:
        testY.append(RUL_F001[i - 1])    
        
        
#All data processing of test set
for i in range(1, int(np.max(test_01_nor[:, 0])) + 1):
    ind = np.where(test_01_nor[:, 0] == i)
    ind = ind[0]
    data_temp = test_01_nor[ind, :]
    data_RUL = RUL_F001[i - 1] 
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


sio.savemat('F001_window_size_trainX.mat', {"train1X": trainX})
sio.savemat('F001_window_size_trainY.mat', {"train1Y": trainY})
sio.savemat('F001_window_size_testX.mat', {"test1X": testX})
sio.savemat('F001_window_size_testY.mat', {"test1Y": testY})
