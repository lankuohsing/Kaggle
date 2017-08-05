
# coding: utf-8

# 单层神经网络 MNIST

# In[89]:

#packages
import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import random as nr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# In[90]:

#一些常量的定义
img_size=28*28
learning_rate=1e-8
batch_size=1000
num_data=42000
train_size=0.8
display_step=50


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

# In[4]:

train_filename='/Dataset_for_dl/input/train.csv'
(train_images,train_labels,num_train,num_feature)=read_train_data(train_filename)

test_filename='/Dataset_for_dl/input/test.csv'
(test_images,num_test,num_feature)=read_test_data(test_filename)

# In[93]:
'''
#切割数据集
(train_images, test_images,train_labels, test_labels) = train_test_split(images, labels, train_size=train_size, random_state=0)
'''

# In[94]:

#将DataFrame转化为Matrix
training_images=train_images.as_matrix()
training_labels=train_labels.as_matrix()
testing_images=test_images.as_matrix()



# In[95]:

#OneHot encoding
enc = OneHotEncoder()
enc.fit(training_labels)
print("enc.n_values_ is:",enc.n_values_  )
num_class=enc.n_values_#类别数量
training_labels_onehot=enc.transform(training_labels).toarray()



# In[96]:

#构建图
x=tf.placeholder("float",[None,img_size])#[none,img_size]表示在tf运行时能输入任意形数量的images,每张image尺寸为img_size
#模型参数，初始值均为零
W = tf.Variable(tf.zeros([img_size,num_class]))#权重值
b = tf.Variable(tf.zeros([num_class]))#偏置量
y = tf.nn.softmax(tf.matmul(x,W) + b)


# In[97]:

#训练模型
y_ = tf.placeholder("float", [None,num_class])#正确的标签
cross_entropy = -tf.reduce_sum(y_*tf.log(y+1e-10))#
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
init = tf.global_variables_initializer()


# In[98]:

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        batch_indices=nr.choice(range(int(num_data*train_size)),batch_size)
        batch_xs=training_images[batch_indices,:]
        batch_ys=training_labels_onehot[batch_indices,:]

        _,c,W1,b1,y1,y2=sess.run([train_step,cross_entropy,W,b,y,y_], feed_dict={x: batch_xs, y_: batch_ys})
        if i % display_step == 0:
            print("Epoch:", '%04d' % (i+1), "cost=",                   "{:.9f}".format(c))
            #print("W:",W1)
            #print("b:",b1)
            #print("y:",y1)
            #print("y_:",y2)

    test_labels_onehot=sess.run(y,feed_dict={x:testing_images})
    results=np.argmax(test_labels_onehot, 1)
    df = pd.DataFrame(results)
    df.index.name='ImageId'
    df.index+=1
    df.columns=['Label']
    df.to_csv('./results1.csv', header=True)


