#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 00:07:26 2018

@author: shanmukha
"""
import arff
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from dataset_creator import preprocess_data

df_1 = arff.load(open('data/1year.arff'))
ds_1 = pd.DataFrame(df_1['data'])

df_2 = arff.load(open('data/2year.arff'))
ds_2 = pd.DataFrame(df_2['data'])

df_3 = arff.load(open('data/3year.arff'))
ds_3 = pd.DataFrame(df_3['data'])

df_4 = arff.load(open('data/4year.arff'))
ds_4 = pd.DataFrame(df_4['data'])

df_5 = arff.load(open('data/5year.arff'))
ds_5 = pd.DataFrame(df_5['data'])

ds1_train_x,ds1_train_y,ds1_train_l = preprocess_data(ds_1)
#ds2_train_x,ds2_train_y,ds2_train_l = preprocess_data(ds_2)
#ds3_train_x,ds3_train_y,ds3_train_l = preprocess_data(ds_3)
#ds4_train_x,ds4_train_y,ds4_train_l = preprocess_data(ds_4)
#ds5_train_x,ds5_train_y,ds5_train_l = preprocess_data(ds_5)

x = tf.placeholder('float',[None,64])
y = tf.placeholder(tf.int64)

l_rate = 0.005
tot_epochs = 1000
batch_size = 256
dropout = 0.5
cv_splits = 10

def nn_model(x):
    x = tf.reshape(x,[-1,64])
    l_1 = tf.contrib.layers.fully_connected(x,64,activation_fn=tf.nn.relu)
    l_1 = tf.layers.dropout(l_1,dropout)
    l_2 = tf.contrib.layers.fully_connected(l_1,256,activation_fn=tf.nn.relu)
    l_2 = tf.layers.dropout(l_2,dropout)
    l_3 = tf.contrib.layers.fully_connected(l_2,192,activation_fn=tf.nn.relu)
    l_3 = tf.layers.dropout(l_3,dropout)
    l_4 = tf.contrib.layers.fully_connected(l_3,128,activation_fn=tf.nn.relu)
    l_4 = tf.layers.dropout(l_4,dropout)
    l_5 = tf.contrib.layers.fully_connected(l_4,64,activation_fn=tf.nn.relu)
    l_5 = tf.layers.dropout(l_5,dropout)
    l_6 = tf.contrib.layers.fully_connected(l_5,32,activation_fn=tf.nn.relu)
    l_6 = tf.layers.dropout(l_6,dropout)
    l_7 = tf.contrib.layers.fully_connected(l_6,16,activation_fn=tf.nn.relu)
    l_7 = tf.layers.dropout(l_7,dropout)
    l_8 = tf.contrib.layers.fully_connected(l_7,8,activation_fn=tf.nn.relu)
    l_8 = tf.layers.dropout(l_8,dropout)
    l_9 = tf.contrib.layers.fully_connected(l_8,4,activation_fn=tf.nn.relu)
    l_9 = tf.layers.dropout(l_9,dropout)
    output = tf.contrib.layers.fully_connected(l_9,2,activation_fn=None)
    return output

def nn_trainer(x,ds_train_x,ds_train_y,ds_train_l):
    prediction = nn_model(x)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    optimizer = tf.train.AdamOptimizer(l_rate).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        kf = KFold(n_splits=cv_splits)
        auc_cv = []
        accuracy_cv = []
        for train_id,val_id in kf.split(ds_train_x,ds_train_y):
            t_x = [ds_train_x[i] for i in train_id]
            t_y = [ds_train_y[i] for i in train_id]
            v_x = [ds_train_x[i] for i in val_id]
            v_y = [ds_train_y[i] for i in val_id]
            for epoch in range(tot_epochs):
                print('Started epoch :',epoch)
                epoch_loss = 0
                j,c = sess.run([optimizer,loss],feed_dict={x:t_x,y:t_y})
                epoch_loss += c
                print('Completed epoch :',epoch)
                print('Total loss in epoch ',epoch,' is :',epoch_loss)    
            correct = tf.equal(tf.argmax(prediction,1),y)
            auc,_ = tf.contrib.metrics.streaming_auc(tf.argmax(prediction,1),y)
            sess.run(tf.local_variables_initializer())
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            auc_cv.append(_.eval({x:v_x,y:v_y}))
            accuracy_cv.append(accuracy.eval({x:v_x,y:v_y}))
        auc_mean = tf.reduce_mean(auc_cv)
        accuracy_mean = tf.reduce_mean(accuracy_cv)
        print('Accuracy on cv set :',accuracy_cv)
        print('Area under ROC curve of cv set :',auc_cv)
        print('Mean accuracy on cv set :',accuracy_mean.eval())
        print('Mean area under ROC curve of cv set :',auc_mean.eval())
#        print('Accuracy on test set :',accuracy.eval({x:ds_test_x,y:ds_test_y}))
#        print('Area under ROC curve of test :',_.eval({x:ds_test_x,y:ds_test_y}))
        

nn_trainer(x,ds1_train_x,ds1_train_y,ds1_train_l)
#nn_trainer(x,ds2_train_x,ds2_train_y,ds2_train_l,ds2_test_x,ds2_test_y)
#nn_trainer(x,ds3_train_x,ds3_train_y,ds3_train_l,ds3_test_x,ds3_test_y)
#nn_trainer(x,ds4_train_x,ds4_train_y,ds4_train_l,ds4_test_x,ds4_test_y)
#nn_trainer(x,ds5_train_x,ds5_train_y,ds5_train_l,ds5_test_x,ds5_test_y)        
#print(ds1_test_x[:2])