#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script test `BaseTrainer` by inherition.

XXX
C.f. https://www.tensorflow.org/get_started/mnist/beginners.
"""

from tftrainer import SimpleTrainer
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets


# Add `source_url` since the data cannot be downloaded via Python
mnist = read_data_sets(
    '../dat/mnist/', one_hot=True, source_url='../dat/mnist/')

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

t = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(t * tf.log(y), reduction_indices=[1]),
    name='cross_entropy')


trainer = SimpleTrainer(
    loss=cross_entropy,
    logdir='../dat/logs/',
    dir_to_ckpt='../dat/checkpoints/test_simple_trainer/')

def get_feed_dict_generator():
    while True:
        batch_xs, batch_ts = mnist.train.next_batch(100)
        feed_dict = {x: batch_xs, t: batch_ts}
        yield feed_dict
feed_dict_generator = get_feed_dict_generator()

trainer.train(1000, feed_dict_generator)
