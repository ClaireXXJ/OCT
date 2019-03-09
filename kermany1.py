#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""======================================================================"""
"""IDE used: Spyder
   Python 3.6.4 on Anaconda3"""
#============================================================================
"""
Created on Wed Apr 19 12:33:13 2018

Due Date: Tuesday May 15 23:59 2018

@authors: Alem, Charlie, and Claire
Email:afitwi1@binghamton.edu
Neural Network & Deep Learning - EECE680C
Department of Electrical & Computer Engineering
Watson Graduate School of Engineering & Applied Science
The State University of New York @ Binghamton
"""
#============================================================================
"""
Projec:Terse Description:
-->
-->
-->
"""
#============================================================================
#----------------------------------------------------------------------------
"""
Step_1: Importing Required Packages or tensor
flow modules:
"""
#----------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import urllib.request as urllib
import scipy.io
import scipy.misc
import sys
import pdb
#============================================================================
#----------------------------------------------------------------------------
"""
Step_2: Extract Input Data:
"""
#---------------------------------------------------------------------------- 
 #load data from data.mat
#t_data.read_data_sets("MNIST_data/",one_hot=True)
#load data from data.mat
imgs="data_v1.0.mat"
images = scipy.io.loadmat("data_v1.0.mat")
Xtrain = images["Xtrain"]
Xtest = images["Xtest"]
Ytrain = images["Ytrain"]
Ytest = images["Ytest"]
x_train=np.array(Xtrain,tf.float32)
y_train=np.array(Ytrain,tf.float32)
x_test=np.array(Xtest,tf.float32)
y_test=np.array(Ytest,tf.float32)

Xtrain_flat = tf.reshape(Xtrain,[400,262144])
ex = Xtrain_flat[0]
rebuild = np.zeros((512,512))
f_idx = 0
for
pdb.set_trace()
sys.exit(1)
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_3: Initializing the weight parameters:
"""
#----------------------------------------------------------------------------
def init_weights(shape):
    """It initializes all the weights of the model"""
    init_weight_vals=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_weight_vals)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_4: Initializing the bias parameters:
"""
#----------------------------------------------------------------------------
def init_bias(shape):
    """It initializes all the biases of the model"""
    init_bias_vals=tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_5: Calling the CONV2D from tensor flow:
"""
#----------------------------------------------------------------------------
def conv2d(x,W):
    """x --> [batch,H,W,Channels]
       W --> [filter H, filter W, channels IN, Channels OUT]
    """
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#============================================================================
#----------------------------------------------------------------------------
"""
Step_6: Defining the Pooling Layer:
"""
#----------------------------------------------------------------------------
def max_pool_2by2(x):
    """x --> [batch,h,w,c]       
    """
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#============================================================================
#----------------------------------------------------------------------------
"""
Step_7: Defining the Convolutional Layer:
"""
#----------------------------------------------------------------------------
def convolutional_layer(input_x,shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])    
    return tf.nn.relu(conv2d(input_x,W)+b)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_8: Defining the Normal fully connected Layer:
"""
#----------------------------------------------------------------------------
def normal_full_layer(input_layer,size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size,size])
    b = init_bias([size])    
    return tf.matmul(input_layer,W)+b
#============================================================================
#----------------------------------------------------------------------------
"""
Step_9: Defining Placeholders:
"""
#----------------------------------------------------------------------------
x = tf.placeholder(tf.float32,shape=[None,784]) #784=28x28
y_true = tf.placeholder(tf.float32,shape=[None,10])#10 is # of classes/labels
#============================================================================
#----------------------------------------------------------------------------
"""
Step_10: The Layers, and calling the respective functions:
"""
#----------------------------------------------------------------------------
x_image = tf.reshape(x,[-1,28,28,1])#input size
convo_1 = convolutional_layer(x_image,shape=[5,5,1,32])
convo_1_pooling = max_pool_2by2(convo_1)
convo_2 = convolutional_layer(convo_1_pooling,shape=[5,5,32,64])
convo_2_pooling = max_pool_2by2(convo_2)
#Flattening the network
convo_2_flat = tf.reshape(convo_2_pooling, [-1,7*7*64])
#fully COnnected Layer
full_layer_one=tf.nn.relu(normal_full_layer(convo_2_flat,1024))
#============================================================================
#----------------------------------------------------------------------------
"""
Step_11: Dropout, regularization scheme:
"""
#----------------------------------------------------------------------------
hold_prob = tf.placeholder(tf.float32)
full_one_dropout=tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
y_pred = normal_full_layer(full_one_dropout,10)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_12: The Loss function:
"""
#----------------------------------------------------------------------------
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=y_true,logits=y_pred))
#============================================================================
#----------------------------------------------------------------------------
"""
Step_12: The Optimizer:
"""
#----------------------------------------------------------------------------
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)
init=tf.global_variables_initializer()
#============================================================================
#----------------------------------------------------------------------------
"""
Step_13: Batch creation function
"""
#----------------------------------------------------------------------------
def createBatch(Xdata,Ydata,size,max_idx):
    chosen_examples = []
    #select which samples will be used to train
    if(isRand):
        sample = randint(0,209)
        for i in range(size - 1):
            while(sample in chosen_examples):
                sample = randint(0,209)
            chosen_examples.append(sample)
    else:
        for i in range(size):
            chosen_examples.append(i)
#============================================================================
#----------------------------------------------------------------------------

"""
Step_14: Creating the Session to carry out the training
"""
#---------------------------------------------------------------------------- 
steps = 5000# iterations --> 100,000
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(50)
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        if i%100 == 0:
            print("On step:{}".format(i))
            print("Accuracy:")
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            print(sess.run(acc,feed_dict={x:mnist.test.images,
                           y_true:mnist.test.labels,hold_prob:1.0}))
            print('\n')
    
