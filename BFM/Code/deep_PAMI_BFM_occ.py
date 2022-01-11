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

seed(1)
# trdata_path_landmark = 'E:\\thesis_phd\MDS\\landmark_alpha_dataset\\x2d\\'
# trdata_path_alpha = 'E:\\thesis_phd\MDS\\landmark_alpha_dataset\\alpha\\'
trdata_path_x3d = '..\\data\\No_occlusion\\train\\x3d\\'
trdata_path_x2d = '..\\data\\No_occlusion\\train\\x2d\\'
data_path ='..\\DNN_weight\\'# 'F:\\Kamyab_data\\IR_BE\\deepLearning\\Linear_regression\\NN\\training\\'

file_list = [f for f in listdir(trdata_path_x3d) if isfile(join(trdata_path_x3d, f))]


def Generator_kernel():
    while True:
        landmark_files=np.random.choice(file_list[0:3000], size=16, replace=False)
        landmarks_2d = [sio.loadmat(trdata_path_x2d+str(f))['x3d'] for f in landmark_files]
        landmarks_3d = [sio.loadmat(trdata_path_x3d+str(f))['x3d'] for f in landmark_files]
        input_from_landmarks=np.zeros((16,72*2))
        output_from_depth=np.zeros((16,72))
        for i in range(len(landmarks_2d)):
            landmarks2 = np.array(landmarks_2d[i])
            # landmarks2 = np.reshape(landmarks2, [3,72])
            # landmarks2 = np.transpose(landmarks2)
            landmarks2 = (landmarks2-np.tile(np.mean(landmarks2 , axis=0), [72,1]))/np.tile(np.std(landmarks2[:,0])+np.std(landmarks2[:,1]), landmarks2.shape)
            landmarks2 = landmarks2[:,0:2]
            # landmarks2 = (landmarks2 - np.min(landmarks2))/(np.max(landmarks2)-np.min(landmarks2))
            # landmarks2  = (landmarks2 + 85000)/(80000+85000)
            landmarks3= np.array(landmarks_3d[i])
            # landmarks2_no_noise = np.reshape(landmarks2_no_noise, [3,72])
            # landmarks2_no_noise = np.transpose(landmarks2_no_noise)
            landmarks3 = (landmarks3-np.tile(np.mean(landmarks3 , axis=0), [72,1]))/np.tile(np.std(landmarks3[:,0])+np.std(landmarks3[:,1]), landmarks3.shape)
            landmarks3 = landmarks3[:,2]
            input_from_landmarks[i] =np.reshape( landmarks2 , (72*2,))#np.zeros((31**2,5))
            output_from_depth[i] = landmarks3#np.zeros((31**2,1))
        yield (input_from_landmarks, output_from_depth)


batch_size=16
input_shape = (72*2,)

statistical_test_run_num = 1
#MSE_z_n = np.zeros(statistical_test_run_num)
#MSE_z_o = np.zeros(statistical_test_run_num)
for j in range(statistical_test_run_num):

    input_landmark = Input(input_shape,name='landmark') 
    
    dense1 = Dense(2*72, name = 'ff_h1', activation = 'tanh')
    d1_bn = BatchNormalization()
    
    dense2 = Dense(2*72, name = 'ff_h2', activation = 'tanh')
    d2_bn = BatchNormalization()
    
    dense3 = Dense(2*72, name = 'ff_h3', activation = 'tanh')
    d3_bn = BatchNormalization()
    
    dense4 = Dense(2*72, name = 'ff_h4', activation = 'tanh')
    d4_bn = BatchNormalization()
    
    dense5 = Dense(2*72, name = 'ff_h5', activation = 'tanh')
    d5_bn = BatchNormalization()
    
    dense6 = Dense(72, name = 'alpha', activation = 'tanh')
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
        
    
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    deep_kernel.compile(optimizer = opt , loss = 'mean_squared_error')#'binary_crossentropy')#
    
    # deep_kernel.load_weights(data_path+'deep_PAMI_occ1.h5')
    history_deep_kernel_z = deep_kernel.fit_generator(Generator_kernel(), steps_per_epoch =1000, epochs =10)#,callbacks=[checkpointer])
    deep_kernel.save_weights(data_path+'deep_PAMI_occ_3000.h5')
    
##########################################################################################################
#                                          TESTing
###########################################################################################################    
# deep_kernel.load_weights(data_path+'deep_PAMI_depth_BFM_noisy.h5')    
res_path = '..\\results\\pred_pami_occ_3000\\'
tstdata_path_landmark = '..\\data\\No_occlusion\\test\\x2d\\'
file_list_test = [f for f in listdir(tstdata_path_landmark) if isfile(join(tstdata_path_landmark, f))]
pred_distance=[]
for i in range(len(file_list_test)):
    print(i)
    landmarks = sio.loadmat(tstdata_path_landmark+str(file_list_test[i]))['x3d'] 
    # input_from_landmarks=np.zeros((225,31*2))
    landmarks2 = np.array(landmarks)
    # landmarks2 = np.reshape(landmarks2, [3,72])
    # landmarks2 = np.transpose(landmarks2)
    landmarks2 = (landmarks2-np.tile(np.mean(landmarks2 , axis=0), [72,1]))/np.tile(np.std(landmarks2[:,0])+np.std(landmarks2[:,1]), landmarks2.shape)
    
    landmarks2= landmarks2[:,0:2]# landmarks2 = (landmarks2 - np.min(landmarks2))/(np.max(landmarks2)-np.min(landmarks2))
    # landmarks2  = (landmarks2 + 85000)/(80000+85000)
    input_from_landmarks =np.reshape( landmarks2 , (72*2,))#np.zeros((31**2,5))
    input_from_landmarks = np.expand_dims(input_from_landmarks , axis=0)
    pred_distance=np.squeeze(deep_kernel.predict(input_from_landmarks))
    sio.savemat(res_path+file_list_test[i],{'a': pred_distance})