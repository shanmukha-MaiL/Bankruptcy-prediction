#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 23:30:54 2018

@author: shanmukha
"""

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
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

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

l_rate = 0.001
tot_epochs = 2200
dropout = 0.6
cv_splits = 10
input_n = 64
l_1_n = 256
l_2_n = 20
output_n = 2
hyperparams = {'l_rate':l_rate,'tot_epochs':tot_epochs,'dropout':dropout,'cv_splits':cv_splits,
               'input_n':input_n,'l_1_n':l_1_n,'l_2_n':l_2_n,'output_n':output_n}

def nn_model(x):
    x = tf.reshape(x,[-1,64])
    l_1 = tf.contrib.layers.fully_connected(x,input_n,activation_fn=tf.nn.relu)
    l_1 = tf.layers.dropout(l_1,dropout)
    l_2 = tf.contrib.layers.fully_connected(l_1,l_1_n,activation_fn=tf.nn.relu)
    l_2 = tf.layers.dropout(l_2,dropout)
    l_3 = tf.contrib.layers.fully_connected(l_2,l_2_n,activation_fn=tf.nn.relu)
    l_3 = tf.layers.dropout(l_3,dropout)
#    l_4 = tf.contrib.layers.fully_connected(l_3,10,activation_fn=tf.nn.relu)
#    l_4 = tf.layers.dropout(l_4,dropout)
    output = tf.contrib.layers.fully_connected(l_3,output_n,activation_fn=None)
    return output

def nn_trainer(x,ds_train_x,ds_train_y,ds_train_l,k):
    prediction = nn_model(x)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    optimizer = tf.train.AdamOptimizer(l_rate).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        kf = KFold(n_splits=cv_splits)
        auc_cv = []
        accuracy_cv = []
        kf_n = 1
        for train_id,val_id in kf.split(ds_train_x,ds_train_y):
            t_x = [ds_train_x[i] for i in train_id]
            t_y = [ds_train_y[i] for i in train_id]
            v_x = [ds_train_x[i] for i in val_id]
            v_y = [ds_train_y[i] for i in val_id]
            print('length of train x set before smote: ',len(t_x))
            print('length of train y set before smote: ',len(t_y))
            t_x,t_y = SMOTE().fit_sample(t_x,t_y)
            print('length of train x set after smote: ',len(t_x))
            print('length of train y set after smote: ',len(t_y))
            for epoch in range(tot_epochs):
                print('Started epoch :',epoch,' in cross validation :',kf_n)
                epoch_loss = 0
                j,c = sess.run([optimizer,loss],feed_dict={x:t_x,y:t_y})
                epoch_loss += c
                print('Completed epoch :',epoch,' in cross validation :',kf_n)
                print('Total loss in epoch ',epoch,' is :',epoch_loss,' in cross validation :',kf_n)    
            correct = tf.equal(tf.argmax(prediction,1),y)
            auc,_ = tf.contrib.metrics.streaming_auc(tf.argmax(prediction,1),y)
            sess.run(tf.local_variables_initializer())
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            auc_cv.append(_.eval({x:v_x,y:v_y}))
            accuracy_cv.append(accuracy.eval({x:v_x,y:v_y}))
            kf_n += 1
        auc_mean = tf.reduce_mean(auc_cv).eval()
        accuracy_mean = tf.reduce_mean(accuracy_cv).eval()
        print('Accuracy on cv set :',accuracy_cv)
        print('Area under ROC curve of cv set :',auc_cv)
        print('Mean accuracy on cv set :',accuracy_mean)
        print('Mean area under ROC curve of cv set :',auc_mean)
#        with open('outputs_ds_'+str(k)+'.txt','a') as f:
#            f.writelines('Hyperparameters :'+ repr(hyperparams)+'\n\n')
#            f.writelines('Accuracy on cv set :'+ repr(accuracy_cv)+'\n')
#            f.writelines('Area under ROC curve of cv set :'+ repr(auc_cv)+'\n')
#            f.writelines('Mean accuracy on cv set :'+ repr(accuracy_mean)+'\n')
#            f.writelines('Mean area under ROC curve of cv set :'+ repr(auc_mean)+'\n\n\n\n')
#        print('Accuracy on test set :',accuracy.eval({x:ds_test_x,y:ds_test_y}))
#        print('Area under ROC curve of test :',_.eval({x:ds_test_x,y:ds_test_y}))
        

nn_trainer(x,ds1_train_x,ds1_train_y,ds1_train_l,1)
#nn_trainer(x,ds2_train_x,ds2_train_y,ds2_train_l,2)
#nn_trainer(x,ds3_train_x,ds3_train_y,ds3_train_l,3)
#nn_trainer(x,ds4_train_x,ds4_train_y,ds4_train_l,4)
#nn_trainer(x,ds5_train_x,ds5_train_y,ds5_train_l,5)        
#print(ds1_test_x[:2])