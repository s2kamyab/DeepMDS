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

trdata_path_x3d = '..\\data\\Occluded\\train\\x3d\\'
trdata_path_x2d = '..\\data\\Occluded\\train\\x2d\\'
data_path ='..\\DNN_weight\\'# 'F:\\Kamyab_data\\IR_BE\\deepLearning\\Linear_regression\\NN\\training\\'

file_list = [f for f in listdir(trdata_path_x3d) if isfile(join(trdata_path_x3d, f))]

def Generator_kernel():
    while True:
        landmark_files=np.random.choice(file_list[0:3000], size=batch_size, replace=False)
        
        x3d = [sio.loadmat(trdata_path_x3d+str(f))['x3d'] for f in landmark_files]
        
        x2d = [sio.loadmat(trdata_path_x2d+str(f))['x2d'] for f in landmark_files]
        x2d = np.squeeze(np.array(x2d))
        x2d_gt = np.zeros((batch_size,144))
        x2d_p = np.zeros((batch_size,144))
        for i in range(batch_size):
            x2d_gt[i]  = np.reshape(x3d[i][:,0:2], (72*2,))
            x2d_p[i]  = np.reshape(x2d[i] ,(72*2,))
            
        yield (x2d_p, x2d_gt)


batch_size=16
input_shape = (72*2,)

statistical_test_run_num = 1
#MSE_z_n = np.zeros(statistical_test_run_num)
#MSE_z_o = np.zeros(statistical_test_run_num)
for j in range(statistical_test_run_num):

    input_landmark_p = Input(input_shape,name='landmark') 
    
    dense1 = Dense(64, name = 'ff_h1', activation = 'linear')
    d1_bn = BatchNormalization()
    
    dense2 = Dense(32, name = 'ff_h2', activation = 'linear')
    d2_bn = BatchNormalization()
    
    dense3 = Dense(16, name = 'ff_h3', activation = 'linear')
    d3_bn = BatchNormalization()
    
    dense4 = Dense(16, name = 'ff_h4', activation = 'linear')
    d4_bn = BatchNormalization()
    
    dense5 = Dense(32, name = 'ff_h5', activation = 'linear')
    d5_bn = BatchNormalization()
    
    dense6 = Dense(64, name = 'ff_h6', activation = 'linear')
    d6_bn = BatchNormalization()
    
    dense7 = Dense(72*2, name = 'ff_h7', activation = 'linear')
    d7_bn = BatchNormalization()
    #x = BatchNormalization()(input_landmark)
    x = dense1(input_landmark_p)
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
    # x = concatenate([x_d2,x_d4])
    x = dense5(x)
    x =keras.layers.LeakyReLU(alpha=0.3)(x)
    x = d5_bn(x)
#    
    x = dense6(x)
    x =keras.layers.LeakyReLU(alpha=0.3)(x)
    x = d6_bn(x)
    
    x = dense7(x)
    x =keras.layers.LeakyReLU(alpha=0.3)(x)
    out_gt = d7_bn(x)
    #y = Lambda(forward, output_shape = (216,),name='forward')(x_alpha)
    
    
    deep_kernel = Model(inputs = input_landmark_p , outputs = out_gt)
    deep_kernel.summary()
        
    
    
    deep_kernel.compile(optimizer = 'Adam' , loss = 'mean_squared_error')#'binary_crossentropy')
    
    deep_kernel.load_weights(data_path+'AE_occ2orth_BFM_3000.h5')
    history_deep_kernel_z = deep_kernel.fit_generator(Generator_kernel(), steps_per_epoch =1000, epochs =1)#,callbacks=[checkpointer])
    deep_kernel.save_weights(data_path+'AE_occ2orth_BFM_3000.h5')
    
##########################################################################################################
#                                          TESTing
###########################################################################################################    
    
 
        
tstdata_path = '..\\data\\Occluded\\test\\x2d\\'
file_list_test = [f for f in listdir(tstdata_path) if isfile(join(tstdata_path, f))]

res_path = '..\\results\\pred_x2d_orth_3000\\'
pred_distance=[]
# test_faces = sio.loadmat('E:\\thesis_phd\\MDS\\code_BFM\\X_tst.mat')['X_tst']
for k in range(len(file_list_test)):
    print(k)
    x2d_p = sio.loadmat(tstdata_path+file_list_test[k])['x2d']
    x2d_p = np.squeeze(np.array(x2d_p))
    x2d_p = (x2d_p - np.min(x2d_p))/(np.max(x2d_p)-np.min(x2d_p))
    landmarks = np.reshape(x2d_p,(72*2,))
    landmarks = np.expand_dims(landmarks, axis=0)
    pred_gt=np.squeeze(deep_kernel.predict(landmarks))

    sio.savemat(res_path+file_list_test[k],{'a': pred_gt})
    
##########################################################################################################
#                                          TRaining
###########################################################################################################    
    
 
        
tstdata_path = '..\\data\\Occluded\\train\\x2d\\'
file_list_test = [f for f in listdir(tstdata_path) if isfile(join(tstdata_path, f))]

res_path = '..\\data\\Occluded\\train\\pred_x2d_orth_3000\\'
pred_distance=[]
# test_faces = sio.loadmat('E:\\thesis_phd\\MDS\\code_BFM\\X_tst.mat')['X_tst']
for k in range(len(file_list_test)):
    print(k)
    x2d_p = sio.loadmat(tstdata_path+file_list_test[k])['x2d']
    x2d_p = np.squeeze(np.array(x2d_p))
    x2d_p = (x2d_p - np.min(x2d_p))/(np.max(x2d_p)-np.min(x2d_p))
    landmarks = np.reshape(x2d_p,(72*2,))
    landmarks = np.expand_dims(landmarks, axis=0)
    pred_gt=np.squeeze(deep_kernel.predict(landmarks))

    sio.savemat(res_path+file_list_test[k],{'a': pred_gt})
    
        

