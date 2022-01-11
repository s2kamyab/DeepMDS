# -*- coding: utf-8 -*-
"""
Created on Tue May 25 18:27:49 2021

@author: shima
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 20:25:48 2019

@author: Nezam Avaran
"""
import numpy as np
from random import seed
from random import random
import matplotlib.pyplot as plt
from keras.models import Model ,load_model
from keras import backend as K
from keras.layers import *
from keras.layers.merge import concatenate as concat
import numpy as np
import scipy.io as sio
import h5py
from keras.callbacks import EarlyStopping
from keras.losses import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import keras
from os import listdir,makedirs
from os.path import isfile, join
from keras.callbacks import ModelCheckpoint
from PIL import Image
import sys
import tensorflow as tf
from sklearn.metrics import mean_squared_error as msqe

seed(1)
# trdata_path_landmark = 'E:\\thesis_phd\MDS\\landmark_alpha_dataset\\x2d\\'
# trdata_path_alpha = 'E:\\thesis_phd\MDS\\landmark_alpha_dataset\\alpha\\'
trdata_path_x3d = '../data/ComA_lmks\\train\\x3d\\'
trdata_path_x2d = '../data/ComA_lmks\\train\\pred_x2d_gt\\'
data_path ='..\\DNN_weights\\'# 'F:\\Kamyab_data\\IR_BE\\deepLearning\\Linear_regression\\NN\\training\\'

file_list = [f for f in listdir(trdata_path_x3d) if isfile(join(trdata_path_x3d, f))]

def Generator_kernel():
    while True:
        landmark_files=np.random.choice(file_list, size=1, replace=False)
        
        x3d = [sio.loadmat(trdata_path_x3d+str(f))['x3d'] for f in landmark_files]
        x3d = np.squeeze(np.array(x3d))
        x3d = (x3d - np.min(x3d))/(np.max(x3d)-np.min(x3d))
        
        x2d = [sio.loadmat(trdata_path_x2d+str(f))['a'] for f in landmark_files]
        x2d = np.squeeze(np.array(x2d))
        x2d = (x2d - np.min(x2d))/(np.max(x2d)-np.min(x2d))
        x2d = np.reshape(x2d,(51,2))
        
        landmarks = x2d
        k = 0
        input_from_landmarks = np.zeros((51**2,5))
        output_from_x3d = np.zeros((51**2,1))
        for i in range(51):
            for j in range(51):
                t1 = landmarks[i]
                t2 = landmarks[j]
                dot1 = t1*t2
                exp1 = np.exp(-(abs(t1-t2)))
                norm1 = np.expand_dims(np.linalg.norm(t1-t2), axis=0)
                input_from_landmarks[k] = np.concatenate((dot1,exp1,norm1)) 
                # output_from_x3d[k] = np.exp(-0.5*np.linalg.norm(x3d[i]-x3d[j]))#np.exp(-np.linalg.norm(x3d[i]-x3d[j])**2)
                output_from_x3d[k] = np.linalg.norm(x3d[i]-x3d[j])#np.exp(-np.linalg.norm(x3d[i]-x3d[j])**2)

                k = k + 1
        
        yield (input_from_landmarks, output_from_x3d)


batch_size=51**2
input_shape = (5,)

statistical_test_run_num = 1
#MSE_z_n = np.zeros(statistical_test_run_num)
#MSE_z_o = np.zeros(statistical_test_run_num)
for j in range(statistical_test_run_num):

    input_landmark = Input(input_shape,name='landmark') 
    
    dense1 = Dense(4*5, name = 'ff_h1', activation = 'linear')
    d1_bn = BatchNormalization()
    
    dense2 = Dense(4*5, name = 'ff_h2', activation = 'linear')
    d2_bn = BatchNormalization()
    
    dense3 = Dense(4*5, name = 'ff_h3', activation = 'linear')
    d3_bn = BatchNormalization()
    
    dense4 = Dense(4*5, name = 'ff_h4', activation = 'linear')
    d4_bn = BatchNormalization()
    
    dense5 = Dense(4*5, name = 'ff_h5', activation = 'linear')
    d5_bn = BatchNormalization()
    
    dense6 = Dense(1, name = 'alpha', activation = 'linear')
    d6_bn = BatchNormalization()
    #x = BatchNormalization()(input_landmark)
    x = dense1(input_landmark)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = d1_bn(x)
    
    x = dense2(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x_d2 = d2_bn(x)
    
    x = dense3(x_d2)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = d3_bn(x)
#    
    x = dense4(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x_d4 = d4_bn(x)
#    
    x = concatenate([x_d2,x_d4])
    x = dense5(x)
    x =keras.layers.LeakyReLU(alpha=0.3)(x)
    x = d5_bn(x)
#    
    x_alpha = dense6(x)
#    x = BatchNormalization()(x)
    out_dist = d6_bn(x_alpha)
    #y = Lambda(forward, output_shape = (216,),name='forward')(x_alpha)
    
    
    deep_kernel = Model(inputs = input_landmark , outputs = out_dist)
    deep_kernel.summary()
        
    
    
    deep_kernel.compile(optimizer = 'Adam' , loss = 'mean_squared_error')#'binary_crossentropy')
    
    deep_kernel.load_weights(data_path+'deep_dissim_from_2D_dist_pers_rem_proj.h5')
    # history_deep_kernel_z = deep_kernel.fit_generator(Generator_kernel(), steps_per_epoch =1000, epochs =2)#,callbacks=[checkpointer])
    # deep_kernel.save_weights(data_path+'deep_dissim_from_2D_dist_pers_rem_proj.h5')
    
##########################################################################################################
#                                          TESTing
###########################################################################################################    
    
 
        
# tstdata_path = 'E:\\thesis_phd\\MDS\\FLAME\\results\\pred_2d_gt\\'
# file_list_test = [f for f in listdir(tstdata_path) if isfile(join(tstdata_path, f))]

# res_path = 'E:\\thesis_phd\\MDS\\FLAME\\results\\pred_dissim_ours\\'
# pred_distance=[]
# # test_faces = sio.loadmat('E:\\thesis_phd\\MDS\\code_BFM\\X_tst.mat')['X_tst']
# for k in range(len(file_list_test)):
#     print(k)
#     x3d = sio.loadmat(tstdata_path+file_list_test[k])['a']
#     x3d = np.squeeze(np.array(x3d))
#     x3d = np.reshape(x3d, [51,2])
#     # x3d=np.transpose(x3d)
#     x3d = (x3d - np.min(x3d))/(np.max(x3d)-np.min(x3d))
# #        landmarks = np.squeeze(landmarks)
#     landmarks = x3d[:,0:2]
#     # landmarks = (landmarks+ 85000)/(80000+85000)
#     h=0
#     input_from_landmarks = np.zeros((51**2,5))
#     for i in range(51):
#         for j in range(51):
#             t1 = landmarks[i]
#             t2 = landmarks[j]
#             dot1 = t1*t2
#             exp1 = np.exp(-(abs(t1-t2)))
#             norm1 = np.expand_dims(np.linalg.norm(t1-t2), axis=0)
#             input_from_landmarks[h] = np.concatenate((dot1,exp1,norm1)) 
#             h = h + 1
            
            
##########################################################################################################
#                                          Training
###########################################################################################################    
    
 
        
# tstdata_path = '../data/ComA_lmks\\train\\pred_x2d_gt\\'
# file_list_test = [f for f in listdir(tstdata_path) if isfile(join(tstdata_path, f))]

# res_path = '../data/ComA_lmks\\train\\pred_dissim_ours\\'
# pred_distance=[]
# # test_faces = sio.loadmat('E:\\thesis_phd\\MDS\\code_BFM\\X_tst.mat')['X_tst']
# for k in range(len(file_list_test)):
#     print(k)
#     x3d = sio.loadmat(tstdata_path+file_list_test[k])['a']
#     x3d = np.squeeze(np.array(x3d))
#     x3d = np.reshape(x3d, [51,2])
#     # x3d=np.transpose(x3d)
#     x3d = (x3d - np.min(x3d))/(np.max(x3d)-np.min(x3d))
# #        landmarks = np.squeeze(landmarks)
#     landmarks = x3d[:,0:2]
#     # landmarks = (landmarks+ 85000)/(80000+85000)
#     h=0
#     input_from_landmarks = np.zeros((51**2,5))
#     for i in range(51):
#         for j in range(51):
#             t1 = landmarks[i]
#             t2 = landmarks[j]
#             dot1 = t1*t2
#             exp1 = np.exp(-(abs(t1-t2)))
#             norm1 = np.expand_dims(np.linalg.norm(t1-t2), axis=0)
#             input_from_landmarks[h] = np.concatenate((dot1,exp1,norm1)) 
#             h = h + 1 
#     pred_distance=np.squeeze(deep_kernel.predict(input_from_landmarks))

#     sio.savemat(res_path+file_list_test[k],{'a': pred_distance})
    
    

#######################################################################################################
#                                  save data for viz results
###################################################################################################
data_path='E:\\thesis_phd\\MDS\\FLAME\\TF_FLAME-master\\data\\ours_pred_profile\\'
res_path = 'E:\\thesis_phd\\MDS\\FLAME\\TF_FLAME-master\\data\\ours_dissim\\'
cases=[]
cases.append('imgHQ00039')
cases.append('imgHQ00088')
cases.append( 'imgHQ00095')
cases.append('imgHQ01148')
for g in range(len(cases)):
    print(g)
    lmks=sio.loadmat(data_path+cases[g]+'.mat')['a']
    lmks = ( lmks- np.min( lmks))/(np.max( lmks)-np.min( lmks))
    lmks = np.max(lmks) - lmks
    lmks = np.reshape(lmks , (51,2))
    # lmks=np.transpose(lmks)
#        landmarks = np.squeeze(landmarks)
      # lmks = x3d[:,0:2]
    # landmarks = (landmarks+ 85000)/(80000+85000)
    h=0
    input_from_landmarks = np.zeros((51**2,5))
    for i in range(51):
        for j in range(51):
            t1 =  lmks[i]
            t2 =  lmks[j]
            dot1 = t1*t2
            exp1 = np.exp(-(abs(t1-t2)))
            norm1 = np.expand_dims(np.linalg.norm(t1-t2), axis=0)
            input_from_landmarks[h] = np.concatenate((dot1,exp1,norm1)) 
            h = h + 1
            
            
            
    pred_distance=np.squeeze(deep_kernel.predict(input_from_landmarks))

    sio.savemat(res_path+cases[g]+'.mat',{'a': pred_distance})