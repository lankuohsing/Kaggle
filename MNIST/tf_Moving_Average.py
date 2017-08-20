# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 22:43:38 2017

@author: lankuohsing
"""
import tensorflow as tf


v1 =tf.Variable(dtype=tf.float32, initial_value=0.)
decay = .99
num_updates = tf.Variable(0, trainable=False)
ema = tf.train.ExponentialMovingAverage(decay=decay, num_updates=num_updates)

update_var_list = [v1]      # 定义更新变量列表
ema_apply = ema.apply(update_var_list)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([v1, ema.average(v1)]))
                                                # [0.0, 0.0]（此时 num_updates = 0 ⇒ decay = .1, ），shadow_variable = variable = 0.

    sess.run(tf.assign(v1, 5))
    sess.run(ema_apply)
    print(sess.run([v1, ema.average(v1)]))
                                                # 此时，num_updates = 0 ⇒ decay =.1,  v1 = 5;
                                                # shadow_variable = 0.1 * 0 + 0.9 * 5 = 4.5 ⇒ variable
    sess.run(tf.assign(num_updates, 10000))
    sess.run(tf.assign(v1, 10))
    sess.run(ema_apply)
    print(sess.run([v1, ema.average(v1)]))
                                                # decay = .99,
                                                # shadow_variable = 0.99 * 4.5 + .01*10 ⇒ 4.555
    sess.run(ema_apply)
    print(sess.run([v1, ema.average(v1)]))
                                                # decay = .99
                                                # shadow_variable = .99*4.555 + .01*10 = 4.609

