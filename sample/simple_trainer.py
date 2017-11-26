#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script test `BaseTrainer` by inherition.

XXX
C.f. https://www.tensorflow.org/get_started/mnist/beginners.
"""


import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from .base_trainer import BaseTrainer


def get_sess(graph, sess_config, sess_target, debug):
  sess = tf.Session(graph=graph,
      config=sess_config, target=sess_target)
  if debug:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
  return sess


def get_train_ops(graph, loss, optimizer, global_step):

  with graph.as_default():

    with tf.name_scope('optimization'):

      train_op = optimizer.minimize(
          loss, global_step=global_step, name='train_op')

  return [train_op]


def get_summarizer(graph, log_vars):

  with graph.as_default():

    with tf.name_scope('summarization'):

      for v in log_vars:
        tf.summary.scalar(v.name, v)
        tf.summary.histogram(v.name, v)

      summarizer = tf.summary.merge_all()

  return summarizer


def get_writer(logdir, graph):
  writer = tf.summary.FileWriter(logdir, graph)
  return writer

def get_saver(graph):
  # Saver shall be initialized within the graph
  with graph.as_default():
    saver = tf.train.Saver()
  return saver

def save(saver, sess, dir_to_ckpt, global_step):
  saver.save(
      sess,
      os.path.join(dir_to_ckpt, 'checkpoint'),
      global_step=global_step)

def restore(dir_to_ckpt, saver, sess):
  # Get checkpoint
  # CAUTION that the arg of `get_checkpoint_state` is
  # `checkpoint_dir`, i.e. the directory of the `checkpoint`
  # to be restored from.
  ckpt = tf.train.get_checkpoint_state(dir_to_ckpt)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    print("INFO - There's been no ckpt yet.")



class SimpleTrainer(BaseTrainer):
  """With the basic implementation for each method."""

  def __init__(self, loss, optimizer=None, log_vars=None, debug=False,
               *args, **kwargs):

    self.loss = loss
    self.log_vars = log_vars if log_vars is not None \
                    else [self.loss]

    self.optimizer = optimizer if optimizer is not None \
                     else tf.train.AdamOptimizer(0.01)

    self.debug = debug

    super(SimpleTrainer, self).__init__(*args, **kwargs)


  def get_sess(self):
    return get_sess(self.graph, self.sess_config,
                    self.sess_target, self.debug)

  def get_train_ops(self):
    return get_train_ops(self.graph, self.loss,
                         self.optimizer, self.global_step)

  def get_summarizer(self):
    return get_summarizer(self.graph, self.log_vars)

  def get_writer(self):
    return get_writer(self.logdir, self.graph)

  def get_saver(self):
    return get_saver(self.graph)

  def save(self):
    return save(self.saver, self.sess, self.dir_to_ckpt, self.global_step)

  def restore(self):
    return restore(self.dir_to_ckpt, self.saver, self.sess)
