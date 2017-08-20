# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 21:14:36 2017

@author: lankuohsing
"""
# In[]
import pandas as pd
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import kaggle_mnist_input_data
# 加载mnist_inference.py 和 kaggle_mnist_train.py中定义的常量和函数
import kaggle_mnist_inference
import kaggle_mnist_train

# 每10秒加载一次最新的模型， 并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10


def evaluate(testing_images, num_test):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, shape=(num_test, 28, 28, 1), name='x1-input')
        #y_ = tf.placeholder(tf.float32, [None, kaggle_mnist_inference.OUTPUT_NODE], name='y-input')

        validate_feed = {x:np.reshape(testing_images, (num_test, 28, 28, 1))}
        # 直接通过调用封装好的函数来计算前向传播的结果。
        # 因为测试时不关注正则损失的值，所以这里用于计算正则化损失的函数被设置为None。
        y = kaggle_mnist_inference.inference(x, False, None)

        # 使用前向传播的结果计算正确率。
        # 如果需要对未知的样例进行分类，那么使用tf.argmax(y, 1)就可以得到输入样例的预测类别了。
        #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平局值了。
        # 这样就可以完全共用mnist_inference.py中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(kaggle_mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        #每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化


        with tf.Session() as sess:
            # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
            ckpt = tf.train.get_checkpoint_state(kaggle_mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名得到模型保存时迭代的轮数
                #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                results=sess.run(y, feed_dict = validate_feed)
                return results
                #print("After %s training step(s), validation accuracy = %f" % (global_step, accuracy_score))
            else:
                print("No checkpoint file found")
                return




# In[]

#train_filename='../input/train.csv'
#(train_images,train_labels,num_train,num_feature)=kaggle_mnist_input_data.read_train_data(train_filename)
test_filename='../input/test.csv'
(test_images,num_test,num_feature)=kaggle_mnist_input_data.read_test_data(test_filename)
#将DataFrame转化为Matrix
#training_images=train_images.as_matrix()
#training_labels=train_labels.as_matrix()
testing_images=test_images.as_matrix().astype('float32')/255.0
i=0
temp0_test_images=testing_images[700*i:700*(i+1),:]
temp0_y=evaluate(temp0_test_images, num_test/40)
for i in range(1,40):
    print(i)
    temp1_test_images=testing_images[700*i:700*(i+1),:]
    temp1_y=evaluate(temp1_test_images, num_test/40)
    temp0_y=np.vstack((temp0_y,temp1_y))
    
 
#results=tf.sparse_to_dense(y)
results=np.argmax(temp0_y,1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,len(results)+1),name = "ImageId"),results],axis = 1)
submission.to_csv("./tf_MNIST_conv.csv",index=False)
"""
testing_images2=testing_images[14000:28001,:]

#training_labels_onehot=kaggle_mnist_input_data.dense_to_one_hot(training_labels,num_classes=NUM_CLASS)

# In[]
y2=evaluate(testing_images2, num_test/2)
#results=tf.sparse_to_dense(y)
results2=argmax(y2,1)
results2 = pd.Series(results2,name="Label2")
submission2 = pd.concat([pd.Series(range(1,len(results2)+1),name = "ImageId"),results2],axis = 1)
submission2.to_csv("./tf_MNIST_conv2.csv",index=False)
"""
