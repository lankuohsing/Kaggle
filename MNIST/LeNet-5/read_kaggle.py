# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 00:25:07 2017

@author: lankuohsing
"""
# In[6]:
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import kaggle_mnist_input_data
from numpy import random as nr
# 加载mnist_inference.py中定义的常量和前向传播的函数
# In[]
NUM_CLASS=10
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# In[6]:
train_filename='../input/train.csv'
(train_images,train_labels,num_train,num_feature)=kaggle_mnist_input_data.read_train_data(train_filename)
training_images=train_images.as_matrix().astype('float64')/255.0
training_labels=train_labels.as_matrix()
training_labels_onehot=kaggle_mnist_input_data.dense_to_one_hot(training_labels,num_classes=NUM_CLASS)
batch_indices=nr.choice(range(num_train),BATCH_SIZE)
batch_x=training_images[batch_indices,:]
batch_y=training_labels_onehot[batch_indices,:]
# In[6]:
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
xs, ys = mnist.train.next_batch(BATCH_SIZE)

# In[6]: