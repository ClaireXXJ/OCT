#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""======================================================================"""
"""IDE used: Spyder
   Python 3.6.4 on Anaconda3"""
#============================================================================
#============================================================================
#----------------------------------------------------------------------------
"""
Step_1: Importing Required Packages or modules:
"""
#----------------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#import idx2numpy
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers
from keras import metrics
from keras.utils.np_utils import to_categorical
from keras import optimizers
import scipy.io
import scipy.misc
import urllib.request as urllib
#============================================================================
#----------------------------------------------------------------------------
"""
Step_2: Data Acquisition and preproccessing:
"""
#----------------------------------------------------------------------------
#response = urllib.urlopen('http://harvey.binghamton.edu/~cmille24/data_v2.0.mat')
images = images = scipy.io.loadmat("data_v1.0.mat")
Xtrain = images["Xtrain"]
Xtest = images["Xtest"]
Ytrain = images["Ytrain"]
Ytest = images["Ytest"]
X_train = np.array(Xtrain, 'float32')
Y_train = np.array(Ytrain, 'float32')
X_test = np.array(Xtest, 'float32')
Y_test = np.array(Ytest, 'float32')
X_train = X_train.reshape(400, 512, 512, 1).astype('float32')
X_test = X_test.reshape(120, 512, 512, 1).astype('float32')
#============================================================================
#----------------------------------------------------------------------------
"""
Step_3: Model Building and Definition:
"""
#----------------------------------------------------------------------------
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), 
                 kernel_regularizer=regularizers.l2(0.04), strides=(2,2), 
                 padding='SAME', activation='relu', 
                 data_format='channels_last', input_shape=(512,512,1)))
model.add(MaxPooling2D(pool_size=(4,4), strides=(4,4)))
model.add(Dropout(rate=0.05,seed=3))
model.add(Conv2D(filters=64, kernel_size=(2,2), 
                 kernel_regularizer=regularizers.l2(0.04), strides=(2,2), 
                 padding='SAME', activation='relu'))

model.add(MaxPooling2D(pool_size=(4,4), strides=(4,4)))
model.add(Dropout(rate=0.05,seed=8))
model.add(Flatten())
model.add(Dense(units=30, activation='tanh', 
                kernel_regularizer=regularizers.l2(0.04)))
model.add(Dense(units=4, activation='softmax', 
                kernel_regularizer=regularizers.l2(0.04)))
#============================================================================
#----------------------------------------------------------------------------
"""
Step_4: Model Compilation:
"""
#----------------------------------------------------------------------------
#Reduce the learning rate if training accuracy suddenly drops and keeps 
#  decreasing
sgd = optimizers.SGD(lr=0.003) # lr by default is 0.01 for SGD
model.compile(loss='categorical_crossentropy', optimizer=sgd, 
              metrics=[metrics.categorical_accuracy])

#============================================================================
#----------------------------------------------------------------------------
"""
Step_5: Model Fitting:
"""
#----------------------------------------------------------------------------
model.fit(X_train, Y_train, epochs=50, batch_size=50)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_6: Model Evaluation:
"""
#----------------------------------------------------------------------------
print("\nEvaluating the model on test data. This won't take long. Relax!")
test_loss, test_accuracy = model.evaluate(X_test, Y_test, batch_size=10)
print("\nAccuracy on test data : ", test_accuracy*100)
print("\nLoss on test data : ", test_loss)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_7: Ending the Session:
"""
#----------------------------------------------------------------------------
#Manually end the session to avoid occasional exceptions while running the 
#  ... program
from keras import backend as K
K.clear_session()
#============================================================================
#----------------------------------------------------------------------------
"""                            End of Program                             """
#----------------------------------------------------------------------------
