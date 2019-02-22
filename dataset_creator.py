#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 00:28:07 2018

@author: shanmukha
"""

import arff
import numpy as np
import pandas as pd
from imblearn.over_sampling import random_over_sampler,SMOTE
from sklearn import preprocessing

df_1 = arff.load(open('data/1year.arff'))
ds_1 = pd.DataFrame(df_1['data'])
#df_2 = arff.load(open('data/2year.arff'))
#ds_2 = pd.DataFrame(df_2['data'])
#df_3 = arff.load(open('data/3year.arff'))
#ds_3 = pd.DataFrame(df_3['data'])
#df_4 = arff.load(open('data/4year.arff'))
#ds_4 = pd.DataFrame(df_4['data'])
#df_5 = arff.load(open('data/5year.arff'))
#ds_5 = pd.DataFrame(df_5['data'])

def preprocess_data(ds):
    print('total rows before replacement:',len(ds))
    print('total columns before replacement:',len(ds.count()))
    ds = ds.replace('?',np.nan)
    ds = ds.fillna(ds.mean())
    ds = ds.dropna()
    ds = ds.sample(frac=1).reset_index(drop=True)
    train_x = ds.drop([64],1)
    train_y = ds[64]
#    train_x =[]
#    train_y = []
#    for row in ds.itertuples():
#        train_x.append(row[1:65])
#        train_y.append(row[65])
#    print('total rows in train set:',len(train_x))
#    print('total columns in train set:',len(train_x.columns))
#    print('total rows in train set: ',len(train_y))
#    print('total rows of class 0: ',(train_y.isin(['0'])).sum()) 
#    print('total rows of class 1: ',(train_y.isin(['1'])).sum())
#    train_x,train_y = SMOTE().fit_sample(train_x,train_y)
#    train_x = pd.DataFrame(train_x)
#    train_y = pd.DataFrame(train_y)
#    print('total rows in train set: ',len(train_y))
#    print('total rows of class 0: ',(train_y.isin(['0'])).sum())
#    print('total rows of class 1: ',(train_y.isin(['1'])).sum())
    train_l = len(train_x)
    mean = np.mean(train_x,0)
    std = np.std(train_x,0)
    mini = np.min(train_x,0)
    maxi = np.max(train_x,0)
    prepro = preprocessing.StandardScaler()
    train_x = prepro.fit_transform(train_x)
#    for i in range(64):
#        for j in range(train_l):
#            train_x[i][j] = (train_x[i][j] - mini[i])/(maxi[i] - mini[i])
            #train_x[i][j] = (train_x[i][j] - mean[i])/(std[i])
    
    #print(ds[[0,1,2]])
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    #print(len(train_x))
   #print(train_x[0])
#    test_l = l - train_l
#    train_x =[]
#    train_y = []
#    test_x =[]
#    test_y =[]
#    count = 0
#    for row in ds.itertuples():
#        count += 1
#        if count <= train_l:
#            train_x.append(row[1:65])
#            train_y.append(row[65])
#        elif train_l < count < train_l + test_l + 1:
#            test_x.append(row[1:65])
#            test_y.append(row[65])
    #print(train_l,test_l)
    
#    print('total rows in train set after smote:',len(train_x))
#    print('total columns in train set after smote:',len(train_x.columns))
#    print('total rows in test set:',len(test_x))
#    print('total columns in test set:',len(test_x[0]))
    return train_x,train_y,train_l

#print(ds_1.drop([63],1))
#preprocess_data(ds_1)
#e = []    
#for row in (ds_1.itertuples())  :
#    e_x = [row[1:65]]
#    e.append(e_x)
#    break
#print(e)
#preprocess_data(ds_1)
#preprocess_data(ds_2)
#preprocess_data(ds_3)
#preprocess_data(ds_4)
#preprocess_data(ds_5)
