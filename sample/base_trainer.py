#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script involves `iterate` and `BaseTrainer`."""


import abc
import numpy as np
import tensorflow as tf
from tqdm import tqdm



def iterate(sess, train_ops, feed_dict_generator,
            summarizer=None, writer=None, global_step=None,
            options=None, run_metadata=None):
  """Iterates one step for optimizing the `train_op`.

  CAUTION:
    This "function" will change the state of the `sess` and the `global_step`
    (if not `None`).

  NOTE:
    This implementation abstracts all, and nothing else is essential. (That is,
    all args in all employed functions (methods) have been fullfilled.)

  Args:
    sess:
      An instance of `tf.Session()`, as the session this iteration works on.

    train_ops:
      List of [`Op`], as the train-op to be iterated. Ensure that it has been
      initialized.

    feed_dict_generator:
      Generator that emits a `feed_dict` associated to the `tf.placeholder`s
      needed by the `train_op`, at each calling of `next()`.

    summarizer:
      A "summary op" that summarizes the graph, e.g. `tf.summary.merge_all`,
      optional.

    writer:
      An instance of `tf.summary.FileWriter` that writes the summary into disk.
      If the `summarizer` is `None` (as default), then this argument is useless,
      optional.

    global_step:
      An un-trainalbe variable with a scalar shape and an integer dtype,
      optional.

    options:
      A `[RunOptions]` protocol buffer or `None`, as the associated argument of
      `tf.Session.run()`, optional.

    run_metadata:
      A `[RunMetadata]` protocol buffer or `None`, as the associated argument of
      `tf.Session.run()`, optional.

  Returns:
    List of the values of `train_ops`.

  Raises:
    `StopIteration` from `next(feed_dict_generator)`.
  """

  # Get `fetches`
  fetches = train_ops
  if summarizer is not None:
    fetches.append(summarizer)

  # Get `feed_dict`
  feed_dict = next(feed_dict_generator)

  # Iterate in one step and get values
  fetch_vals = sess.run(fetches,
                        feed_dict=feed_dict,
                        options=options,
                        run_metadata=run_metadata)

  # Update `global_step` value
  if global_step is not None:
    sess.run(global_step.assign_add(1))

  # Write to TensorBoard
  if summarizer is not None and writer is not None:
    _, summary = fetch_vals
    writer.add_summary(summary, global_step=global_step)

  # Return the values of `train_ops`
  if summarizer is not None:
    # The last element of `fetch_vals` will be the `summary`
    train_ops_vals = fetch_vals[:-1]
  else:
    train_ops_vals = fetch_vals
  return train_ops_vals



class BaseTrainer(object):
  """Abstract base class of trainer that supplements the `iterate`.

  Args:
    log_vars:
      List of instances of `tf.Variable`, as the variables to be logged into
      TensorBoard.
  """

  def __init__(self, loss, logdir=None, dir_to_save=None,
               graph=None, sess=None, sess_config=None, sess_target=''):

    self.loss = loss
    self.logdir = logdir
    self.dir_to_save = dir_to_save

    if graph is not None:
      self.graph = graph
    else:
      self.graph = tf.get_default_graph()

    self.optimizer = self.get_optimizer()
    self.train_op = self.build_optimization()

    if self.logdir is not None:
      self.summarizer = self.build_summarization()
      self.writer = self.get_writer()

    if self.dir_to_save is not None:
      self.saver = self.get_saver()

    if sess is not None:
      self.sess = sess
    else:
      self.sess = tf.Session(graph=self.graph,
          config=sess_config, target=sess_target)

    self.global_step = tf.Variable(0, trainable=False, name='global_step')


  @abc.abstractmethod
  def get_optimizer(self):
    """Returns an instance of `tf.train.Optimizer`."""
    pass


  @abc.abstractmethod
  def get_grad_and_var_list(self):
    """Retruns list of tuples of gradients and variables, that will be argument
    of `self.optimizer.apply_gradients()`."""
    pass


  def build_optimization(self):
    """Implements the scope `optimization`.

    Returns:
      Op for optimization in one iteration.
    """

    with self.graph.as_default():

      with tf.name_scope('optimization'):

        gvs = self.get_grad_and_var_list()
        train_op = self.optimizer.apply_gradients(gvs)

    return train_op


  @abc.abstractmethod
  def build_summarization(self):
    """Implements the scope `summarization`.

    Returns:
      Op for summarization (to `tensorboard`) in one iteration.
    """
    pass


  @abc.abstractmethod
  def get_writer(self):
    """Returns an instance of `tf.summary.FileWriter`. This method will be
    called only when `self.logdir` is not `None`."""
    pass


  @abc.abstractmethod
  def get_saver(self):
    """Returns an instance of `tf.Saver`. This method will be
    called only when `self.dir_to_save` is not `None`."""
    pass


  @abc.abstractmethod
  def save(self):
    """Save the checkpoint and anything else you want. This method will be
    called only when `self.dir_to_save` is not `None`."""
    pass


  @abc.abstractmethod
  def restore(self):
    """Restore the checkpoint and anything else you want. This method will be
    called only when `self.dir_to_save` is not `None`."""
    pass


  def train(self, n_iters, feed_dict_generator, initializer=None,
            saver_skip_step=100, writer_skip_step=10, options=None,
            run_metadata=None):
    """As the trainer trains.

    Args:
      n_iters:
        `int`, as the number of iterations.

      feed_dict_generator:
        A generator that emits a feed_dict at each calling of `next()`.

      initializer:
        XXX

      saver_skip_step:
        `int`, as the skip-step for calling `self.save()`, optional.

      writer_skip_step:
        `int`, as the skip-step for writing summary to TensorBoard, optional.

      options:
        A `[RunOptions]` protocol buffer or `None`, as the associated argument
        of `tf.Session.run()`, optional.

      run_metadata:
        A `[RunMetadata]` protocol buffer or `None`, as the associated argument
        of `tf.Session.run()`, optional.
    """

    if initializer is not None:
      self.sess.run(initializer)
    else:
      self.sess.run(tf.global_variables_initializer())

    # Restore
    if self.dir_to_save is not None:
      self.restore()

    # Iterations
    for i in tqdm(range(n_iters)):  # XXX

      if self.logdir is not None and i % writer_skip_step == 0:
        # Shall summarize and write summary
        summarizer = self.summarizer
        wirter = self.writer
      else:
        # Not summarize and write summary
        summarizer = None
        writer = None

      iterate(self.sess, self.train_op, feed_dict_generator,
              summarizer=summarizer, writer=writer,
              global_step=self.global_step, options=options,
              run_metadata=run_metadata)

      # Save at `skip_step`
      if self.dir_to_save is not None:
        if (i+1) % saver_skip_step == 0:
          self.save()
