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
from tensorflow.keras.models import Model ,load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.layers import concatenate as concat
import numpy as np
import scipy.io as sio
import h5py
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tensorflow.keras
from os import listdir,makedirs
from os.path import isfile, join
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image
import sys
import tensorflow as tf
from sklearn.metrics import mean_squared_error as msqe

trdata_path_landmark = '../data/ComA_lmks\\train\\x2d\\'
# trdata_path_alpha = 'E:\\thesis_phd\MDS\\landmark_alpha_dataset\\alpha\\'
trdata_path_x3d = '../data/ComA_lmks\\train\\x3d\\'

data_path ='models\\'# 'F:\\Kamyab_data\\IR_BE\\deepLearning\\Linear_regression\\NN\\training\\'

file_list = [f for f in listdir(trdata_path_landmark) if isfile(join(trdata_path_landmark, f))]

def Generator_kernel():
    while True:
        landmark_files=np.random.choice(file_list, size=16, replace=False)
        landmarks = [sio.loadmat(trdata_path_landmark+str(f))['x2d'] for f in landmark_files]
        x3d = [sio.loadmat(trdata_path_x3d+str(f))['x3d'] for f in landmark_files]
        input_from_landmarks=np.zeros((16,51*2))
        output_from_depth=np.zeros((16,51))
        for i in range(len(landmarks)):
            landmarks2 = np.array(landmarks[i])
            landmarks2 = (landmarks2-np.tile(np.mean(landmarks2 , axis=0), [51,1]))/np.tile(np.std(landmarks2[:,0])+np.std(landmarks2[:,1]), landmarks2.shape)

            x3d2 = np.array(x3d[i])
            x3d22 = (x3d2-np.tile(np.mean(x3d2 , axis=0), [51,1]))/np.tile(np.std(x3d2[:,0])+np.std(x3d2[:,1]), x3d2.shape)

            input_from_landmarks[i] =np.reshape( landmarks2[:,0:2] , (51*2,))#np.zeros((31**2,5))
            output_from_depth[i] = x3d2[:,2]#np.zeros((31**2,1))
        yield (input_from_landmarks, output_from_depth)


batch_size=16
input_shape = (51*2,)

statistical_test_run_num = 1
#MSE_z_n = np.zeros(statistical_test_run_num)
#MSE_z_o = np.zeros(statistical_test_run_num)
for j in range(statistical_test_run_num):

    input_landmark = Input(input_shape,name='landmark') 
    
    dense1 = Dense(2*51, name = 'ff_h1', activation = 'tanh')
    d1_bn = BatchNormalization()
    
    dense2 = Dense(2*51, name = 'ff_h2', activation = 'tanh')
    d2_bn = BatchNormalization()
    
    dense3 = Dense(2*51, name = 'ff_h3', activation = 'tanh')
    d3_bn = BatchNormalization()
    
    dense4 = Dense(2*51, name = 'ff_h4', activation = 'tanh')
    d4_bn = BatchNormalization()
    
    dense5 = Dense(2*51, name = 'ff_h5', activation = 'tanh')
    d5_bn = BatchNormalization()
    
    dense6 = Dense(51, name = 'alpha', activation = 'tanh')
    # d6_bn = BatchNormalization()
    #x = BatchNormalization()(input_landmark)
    x = dense1(input_landmark)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = d1_bn(x)
    
    x = dense2(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)
    # x_d2 = d2_bn(x)
    
    x = dense3(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = d3_bn(x)
#    
    x = dense4(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)
    # x_d4 = d4_bn(x)
#    
    # x = concatenate([x_d2,x_d4])
    x = dense5(x)
    # x =keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = d5_bn(x)
#    
    x_alpha = dense6(x)
#    x = BatchNormalization()(x)
    # out_dist = d6_bn(x_alpha)
    #y = Lambda(forward, output_shape = (216,),name='forward')(x_alpha)
    
    
    deep_kernel = Model(inputs = input_landmark , outputs = x_alpha)
    deep_kernel.summary()
        
    
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
    deep_kernel.compile(optimizer = opt , loss = 'mean_squared_error')#'binary_crossentropy')#
    
    deep_kernel.load_weights(data_path+'deep_PAMI_depth_FLAME.h5')
    # history_deep_kernel_z = deep_kernel.fit_generator(Generator_kernel(), steps_per_epoch =1000, epochs =5)#,callbacks=[checkpointer])
    # deep_kernel.save_weights(data_path+'deep_PAMI_depth_FLAME.h5')
    
##########################################################################################################
#                                          TESTing
###########################################################################################################    
# # deep_kernel.load_weights(data_path+'deep_PAMI_depth_BFM.h5')    
# res_path = 'E:\\thesis_phd\\MDS\\FLAME\\results\\pred_pami_occ\\'
# tstdata_path_landmark = '../data/ComA_lmks\\test\\x2d\\'
# file_list_test = [f for f in listdir(tstdata_path_landmark) if isfile(join(tstdata_path_landmark, f))]
# pred_distance=[]
# X_tst = []
# # landmark_files=np.random.choice(file_list_test, size=1000, replace=False)
# # landmarks = [sio.loadmat(tstdata_path_landmark+str(f))['vals'] for f in landmark_files]
# # input_from_landmarks=np.zeros((225,31*2))
# # X_tst = landmarks
# # output_from_depth=np.zeros((225,31))
# for i in range(len(file_list_test)):
#     print(i)
#     landmarks = sio.loadmat(tstdata_path_landmark+str(file_list_test[i]))['x2d'] 
#     # input_from_landmarks=np.zeros((225,31*2))
#     landmarks2 = np.array(landmarks[:,0:2])
#     # landmarks2 = np.reshape(landmarks2, [31,3])
#     landmarks2 = (landmarks2-np.tile(np.mean(landmarks2 , axis=0), [51,1]))/np.tile(np.std(landmarks2[:,0])+np.std(landmarks2[:,1]), landmarks2.shape)
#     input_from_landmarks =np.reshape( landmarks2 , (51*2,))#np.zeros((31**2,5))
#     input_from_landmarks = np.expand_dims(input_from_landmarks , axis=0)
#     # output_from_depth[i] = landmarks2[:,2]#np.zeros((31**2,1))
#     # X_tst.append(landmarks2)
#     pred_distance=np.squeeze(deep_kernel.predict(input_from_landmarks))
#     sio.savemat(res_path+str(i)+'.mat',{'a': pred_distance})
# # sio.savemat(res_path+'X_tst.mat',{'a': X_tst})
##########################################################################################################
#                                          Training
###########################################################################################################    
# deep_kernel.load_weights(data_path+'deep_PAMI_depth_BFM.h5')    
res_path = '../data/ComA_lmks\\train\\pred_pami_occ\\'
tstdata_path_landmark = '../data/ComA_lmks\\train\\x2d\\'
file_list_test = [f for f in listdir(tstdata_path_landmark) if isfile(join(tstdata_path_landmark, f))]
pred_distance=[]
X_tst = []
# landmark_files=np.random.choice(file_list_test, size=1000, replace=False)
# landmarks = [sio.loadmat(tstdata_path_landmark+str(f))['vals'] for f in landmark_files]
# input_from_landmarks=np.zeros((225,31*2))
# X_tst = landmarks
# output_from_depth=np.zeros((225,31))
for i in range(len(file_list_test)):
    print(i)
    landmarks = sio.loadmat(tstdata_path_landmark+str(file_list_test[i]))['x2d'] 
    # input_from_landmarks=np.zeros((225,31*2))
    landmarks2 = np.array(landmarks[:,0:2])
    # landmarks2 = np.reshape(landmarks2, [31,3])
    landmarks2 = (landmarks2-np.tile(np.mean(landmarks2 , axis=0), [51,1]))/np.tile(np.std(landmarks2[:,0])+np.std(landmarks2[:,1]), landmarks2.shape)
    input_from_landmarks =np.reshape( landmarks2 , (51*2,))#np.zeros((31**2,5))
    input_from_landmarks = np.expand_dims(input_from_landmarks , axis=0)
    # output_from_depth[i] = landmarks2[:,2]#np.zeros((31**2,1))
    # X_tst.append(landmarks2)
    pred_distance=np.squeeze(deep_kernel.predict(input_from_landmarks))
    sio.savemat(res_path+str(i)+'.mat',{'a': pred_distance})
# sio.savemat(res_path+'X_tst.mat',{'a': X_tst})

# #######################################################################################################
# #                                  save data for viz results
# ###################################################################################################
# data_path='E:\\thesis_phd\\MDS\\FLAME\\TF_FLAME-master\\data\\ours_pred_profile\\'
# res_path = 'E:\\thesis_phd\\MDS\\FLAME\\TF_FLAME-master\\data\\pred_pami\\'
# cases=[]
# cases.append('imgHQ00039')
# cases.append('imgHQ00088')
# cases.append( 'imgHQ00095')
# cases.append('imgHQ01148')
# for g in range(len(cases)):
#     print(g)
#     lmks=sio.loadmat(data_path+cases[g]+'.mat')['a']
#     # lmks = ( lmks- np.min( lmks))/(np.max( lmks)-np.min( lmks))
#     # lmks = np.max(lmks) - lmks
#     landmarks2 = np.reshape(lmks , (51,2))
#    #     landmarks = sio.loadmat(tstdata_path_landmark+str(file_list_test[i]))['x2d'] 
# #     # input_from_landmarks=np.zeros((225,31*2))
#     # landmarks2 = np.array(landmarks2[:,0:2])
#     # landmarks2 = np.reshape(landmarks2, [31,3])
#     landmarks2 = (landmarks2-np.tile(np.mean(landmarks2 , axis=0), [51,1]))/np.tile(np.std(landmarks2[:,0])+np.std(landmarks2[:,1]), landmarks2.shape)
#     input_from_landmarks =np.reshape( landmarks2 , (51*2,))#np.zeros((31**2,5))
#     input_from_landmarks = np.expand_dims(input_from_landmarks , axis=0)
#     # output_from_depth[i] = landmarks2[:,2]#np.zeros((31**2,1))
#     # X_tst.append(landmarks2)
#     pred_distance=np.squeeze(deep_kernel.predict(input_from_landmarks))
#     sio.savemat(res_path+cases[g]+'.mat',{'a': pred_distance})
# # sio.savemat(res_path+'X_tst.mat',{'a': X_tst})
