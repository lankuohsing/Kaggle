# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 19:53:52 2017

@author: lankuohsing
"""http://blog.csdn.net/blogdevteam/article/details/76092129

import tensorflow as tf

#定义神经网络的参数
INPUT_NODE=28*28#输入
OUTPUT_NODE=10#输出
LAYER1_NODE=500#?

IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10#标签数

#第一层卷积层的尺寸和深度
CONV1_DEEP=32
CONV1_SIZE=5
#第二层卷积层的尺寸和深度
CONV2_DEEP=64
CONV2_SIZE=5
#全连接层的节点个数
FC_SIZE=512


#定义神经网络的前向传播过程
#这里添加一个新的参数train，用于区别训练过程和测试过程
#这里将用到dropout方法，dropout可进一步提升模型可靠性并防止果泥和，只在训练时使用
def inference(input_tensor,train,regularizer):
    #实现第一层神经网络的变量并完成前向传播过程。
    with tf.variable_scope('layer1-conv'):
        conv1_weight=tf.get_variable(
            "weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
        conv1_biases=tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        #使用边长为5，深度为32的滤波器，滤波器移动的步长为1，且使用全0填充
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding'SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    #实现第二层池化曾的前向传播过程
    #选用最大池化层，池化层的滤波器边长为2，使用全0填充
    with tf.name_scope('layer2-pool1'):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides[1,2,2,1],padding='SAME')
    #第三层卷积层
    #输入为14*14*32，输出为14*14*64
    with tf.name_scope('layer3-conv2'):
        conv2_weights=tf.get_variable(
            "weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
        conv2_biases=tf.get_variable("bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2=tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    #实现第四层池化层的前向传播过程
    with tf.name_scope('layer-pool2'):
        pool2=tf.nn.max_pool(
            relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

