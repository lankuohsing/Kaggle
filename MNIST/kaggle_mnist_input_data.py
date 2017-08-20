# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:38:49 2017

@author: lankuohsing
"""

# In[1]:

#packages
import pandas as pd
import numpy as np
from numpy import random as nr

# In[3]:

def read_train_data(filename):#用于读取数据的函数
    labeled_images = pd.read_csv(filename)
    [num_sample,num_feature]=labeled_images.shape
    images = labeled_images.iloc[:,1:]
    labels = labeled_images.iloc[:,:1]
    return (images,labels,num_sample,num_feature)

def read_test_data(filename):#用于读取数据的函数
    notlabeled_images = pd.read_csv(filename)
    [num_sample,num_feature]=notlabeled_images.shape
    images = notlabeled_images
    return (images,num_sample,num_feature)

#OneHot encoding
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot