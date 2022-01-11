# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 23:17:33 2021

@author: shima
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

trdata_path_landmark = '..\\data\\Occluded\\train\\img\\'
trdata_path_alpha = '..\\data\\Occluded\\train\\alpha\\'
trdata_path_x3d = '..\\data\\Occluded\\train\\x3d\\'

data_path ='..\\DNN_weight\\'# 'F:\\Kamyab_data\\IR_BE\\deepLearning\\Linear_regression\\NN\\training\\'

file_list = [f for f in listdir(trdata_path_landmark) if isfile(join(trdata_path_landmark, f))]

def Generator_kernel():
    while True:
        img_files=np.random.choice(file_list[0:3000], size=16, replace=False)
        imgs = [sio.loadmat(trdata_path_landmark+str(f))['img'] for f in img_files]
        imgs = np.array(imgs)
        imgs = 2*imgs/255-1
#        imgs= imgs+np.random.normal(0.0, np.random.random( imgs.shape)/70, imgs.shape)
        # imgs = tf.keras.applications.mobilenet.preprocess_input(imgs)
        
        alpha = [sio.loadmat(trdata_path_alpha+str(f))['alpha'] for f in img_files]
        alpha = np.array(alpha)
        x3d = coef2object2(alpha)
        # x3d = []
        # for i in range(len(alpha)):
        #     x3d.append(coef2object2(alpha[i]))
        
        x3d = np.array(x3d)

        yield (imgs, x3d)


def coef2object2(coef):
    b =np.squeeze(coef)*np.squeeze(ev)#tf.keras.backend.multiply(coef ,ev)
    a=np.matmul(b,pc)
    obj = mu + a
    return obj

def coef2object(coef):
    b = tf.math.multiply(coef,ev2)#tf.keras.backend.multiply(coef ,ev)
    a=tf.matmul(b,pc2)
    obj = mu2 + a
    return obj

def to_standard(x):
    x2 = (x - tf.keras.backend.mean(x))/tf.keras.backend.std(x)
    return x2


batch_size=16
mu = sio.loadmat('PublicMM1/01_MorphableModel.mat')['shapeMU']
mu = np.tile(mu ,batch_size )
mu = np.transpose(mu)
mu2 = tf.keras.backend.constant(mu)
pc = sio.loadmat('PublicMM1/01_MorphableModel.mat')['shapePC']
pc = np.transpose(pc)
pc2 = tf.keras.backend.constant(pc)
ev = sio.loadmat('PublicMM1/01_MorphableModel.mat')['shapeEV']
ev = np.tile(ev ,batch_size )
ev = np.transpose(ev)
ev2 = tf.keras.backend.constant(ev)
tl = sio.loadmat('PublicMM1/01_MorphableModel.mat')['tl']
tl=tl-1

input_shape_MF = (96,96,3)





model = tf.keras.applications.MobileNet(
    input_shape=input_shape_MF,
    alpha=0.3125,
    depth_multiplier=1,
    dropout=0.001,
    include_top=False,
    weights=None,
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax"
)
# model.summary()

x = Flatten()(model.output)
x = Dense(199, activation='linear')(x)
# x = Dropout(0.5)(x)
# alpha = BatchNormalization()(x)
alpha = tf.keras.layers.Lambda(to_standard, output_shape=(199,))(x)
x3d = tf.keras.layers.Lambda(coef2object, output_shape=(160470,))(alpha)

#create graph of your new model
mobileFace = Model(inputs = model.input, outputs = x3d)
mobileFace_test = Model(inputs = model.input, outputs = alpha)

#compile the model
mobileFace.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

mobileFace.summary()
#mobileFace.load_weights(data_path+'deep_mobileface_occ.h5')  
history_mobileFace = mobileFace.fit_generator(Generator_kernel(), steps_per_epoch =500, epochs =200)#,callbacks=[checkpointer])
mobileFace.save_weights(data_path+'deep_mobileface_occ_3000.h5')
    
##########################################################################################################
#                                          TESTing
###########################################################################################################    
  
res_path = '..\\results\\pred_mobileface_occ_3000\\'
tstdata_path_img = '..\\data\\Occluded\\test\\img\\'

file_list_test = [f for f in listdir(tstdata_path_img) if isfile(join(tstdata_path_img, f))]

for i in range(len(file_list_test)):
    print(i)
    # img_files=np.random.choice(file_list_test, size=16, replace=False)
    imgs = sio.loadmat(tstdata_path_img +file_list_test[i])['img'] 
    imgs = np.array(imgs)
    imgs = 2*imgs/255-1
#    imgs= imgs+np.random.normal(0.0, np.random.random( imgs.shape)/70, imgs.shape)
    imgs = np.expand_dims(imgs , axis=0)
    # imgs = tf.keras.applications.mobilenet.preprocess_input(imgs)

    
    pred_alpha=np.squeeze(mobileFace_test.predict(imgs))
    sio.savemat(res_path+file_list_test[i]+'.mat',{'a': pred_alpha})
#    sio.savemat(tstdata_path_img_noisy+file_list_test[i],{'a': imgs})