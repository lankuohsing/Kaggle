# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 21:33:53 2017

@author: lankuohsing
"""
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
# matplotlib inline

labeled_images = pd.read_csv('./input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


'''
#viewing an image
i=1     #viewing the i-th image
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])
'''
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
accuracy=clf.score(test_images,test_labels)
print(accuracy)