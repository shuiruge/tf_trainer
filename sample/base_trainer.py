#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script involves `iterate` and `BaseTrainer`."""


import abc
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from .utils import ensure_directory



def iterate(sess, train_ops, feed_dict,
            summarizer=None, writer=None, global_step=None,
            options=None, run_metadata=None):
  """Iterates one step for training the `train_ops`.

  CAUTION:
    This "function" will change the state of the `sess`.

  NOTE:
    This implementation abstracts all, and nothing else is essential. (That is,
    all args in all employed functions (methods) have been fullfilled.)

  Args:
    sess:
      An instance of `tf.Session()`, as the session this iteration works on.

    train_ops:
      List of `Op`s, as the train-ops to be iterated. Ensure that it has been
      initialized.

    feed_dict:
      A `feed_dict` associated to the `tf.placeholder`s needed by the
      `train_ops`.

    summarizer:
      A "summary op" that summarizes the graph, e.g. `tf.summary.merge_all`,
      optional.

    writer:
      An instance of `tf.summary.FileWriter` that writes the summary into disk.
      If the `summarizer` is `None` (as default), then this argument is useless,
      optional.

    global_step:
      `int` or `None`, optional.

    options:
      A `[RunOptions]` protocol buffer or `None`, as the associated argument of
      `tf.Session.run()`, optional.

    run_metadata:
      A `[RunMetadata]` protocol buffer or `None`, as the associated argument of
      `tf.Session.run()`, optional.

  Returns:
    List of the values of `train_ops`.
  """

  # Get `fetches`
  fetches = train_ops.copy()  # shallow copy a list.
  if summarizer is not None:
    fetches.append(summarizer)

  # Iterate in one step and get values
  fetch_vals = sess.run(fetches,
                        feed_dict=feed_dict,
                        options=options,
                        run_metadata=run_metadata)

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
    graph:
      An instance of `tf.Graph` or `None`, as the graph to be trained,
      optional.

    sess:
      An instance of `tf.Session` or `None`, optional. If not `None`, the
      method `self.get_sess()` will not be called.

    sess_config:
      Optional. A ConfigProto protocol buffer with configuration options for
      the session.

    sess_target:
      Optional. The execution engine to connect to. Defaults to using an
      in-process engine. See Distributed TensorFlow for more examples.

    logdir:
      `str` or `None`, as the logdir of TensorBoard, optional.

    dir_to_ckpt:
      `str` or `None`, as the directory to checkpoints, optional.

    init_global_step:
      `int` as the initial value of global step, optional.

    initializer:
      A TenosrFlow initializer, or `None`, optional. If `None`, then using
      `tf.global_variables_initializer` as the employed initializer.

  Attributes:
    `graph`, `sess`, `logdir`, summarizer (if `logdir` is not `None`), writer 
    (if `logdir` is not `None`), `dir_to_ckpt`, `saver` (if `dir_to_ckpt` is
    not `None`), `global_step`, `train_ops`, `initializer`.

  Methods:
    get_sess:
      Abstract method. Returns an instance of `tf.Session()` as the argument
      `sess` of `iterate()`. Only when `sess` in `__init__()` is `None` will
      this method to be called.

    get_train_ops:
      Abstract method. Returns list of ops as the argument `train_ops` of
      `iterate()`.

    get_summarizer:
      Abstract method. Returns op as the argument `summarizer` of `iterate()`.

    get_writer:
      Abstract method. Returns an instance of `tf.summary.FileWriter`. This
      method will be called only when `self.logdir` is not `None`.

    get_saver:
      Abstract method. Returns an instance of `tf.Saver`. This method will be
      called only when `self.dir_to_ckpt` is not `None`.

    save:
      Abstract method. Save the checkpoint of `self.sess` to disk
      (`self.dir_to_ckpt`). This method will be called only when
      `self.dir_to_ckpt` is not `None`.

    restore:
      Abstract method. Restore the checkpoint to `self.sess` from disk
      (`self.dir_to_ckpt`). This method will be called only when
      `self.dir_to_ckpt` is not `None`.

    train:
      As the trainer trains.
  """


  def __init__(self, graph=None, sess=None, sess_config=None, sess_target='',
        logdir=None, dir_to_ckpt=None, init_global_step=0, initializer=None):

    self.logdir = logdir
    self.dir_to_ckpt = dir_to_ckpt

    # Notice that `tf.Graph.__init__()` needs no argument, so an abstract
    # `get_graph` method is not essential, thus directly define `self.graph`
    if graph is not None:
      self.graph = graph
    else:
      self.graph = tf.get_default_graph()

    # Added name-scope "auxillary_ops" into `self.graph`.
    # Building of `train_ops` may need `self.global_step`, which thus
    # shall be defined in front.
    with self.graph.as_default():
      with tf.name_scope('auxillary_ops'):
        with tf.name_scope('increase_global_step_op'):
          self.global_step = tf.Variable(
              init_global_step, trainable=False, name='global_step')
          self.increase_global_step_op = self.global_step.assign_add(1)

    self.train_ops = self.get_train_ops()

    # Initializer shall be placed in the end of the graph.
    # XXX: Why?
    with self.graph.as_default():
      if initializer is not None:
        self.initializer = initializer
      else:
        self.initializer = tf.global_variables_initializer()

    if self.logdir is not None:
      self.summarizer = self.get_summarizer()
      self.writer = self.get_writer()

    if self.dir_to_ckpt is not None:
      ensure_directory(self.dir_to_ckpt)
      self.saver = self.get_saver()

    if sess is not None:
      self.sess = sess
    else:
      self.sess_config = sess_config
      self.sess_target = sess_target
      self.sess = self.get_sess()

    # Restore checkpoint in `self.dir_to_ckpt` to `self.sess`
    if self.dir_to_ckpt is not None:
      self.restored = self.restore()

  @abc.abstractmethod
  def get_sess(self):
    """Returns an instance of `tf.Session()` as the argument `sess` of
    `iterate()`. Only when `sess` in `__init__()` is `None` will this method
    to be called."""
    sess = tf.Session(graph=self.graph,
        config=self.sess_config, target=self.sess_target)
    return sess


  @abc.abstractmethod
  def get_train_ops(self):
    """Returns list of ops as the argument `train_ops` of `iterate()`."""
    pass


  @abc.abstractmethod
  def get_summarizer(self):
    """Returns op as the argument `summarizer` of `iterate()`."""
    pass


  @abc.abstractmethod
  def get_writer(self):
    """Returns an instance of `tf.summary.FileWriter`. This method will be
    called only when `self.logdir` is not `None`."""
    pass


  @abc.abstractmethod
  def get_saver(self):
    """Returns an instance of `tf.Saver`. This method will be
    called only when `self.dir_to_ckpt` is not `None`."""
    pass


  @abc.abstractmethod
  def save(self):
    """Save the checkpoint of `self.sess` to disk (`self.dir_to_ckpt`). This
    method will be called only when `self.dir_to_ckpt` is not `None`."""
    pass


  @abc.abstractmethod
  def restore(self):
    """Restore the checkpoint to `self.sess` from disk (`self.dir_to_ckpt`). This
    method will be called only when `self.dir_to_ckpt` is not `None`.

    Returns:
      `bool`, being `True` if sucessfully restored from checkpoint; else `False`.
    """
    pass


  def initialize_if_not_restored(self):
    """Notice that re-initializing variables will cancel all have restored."""
    if self.dir_to_ckpt is None:
      self.sess.run(self.initializer)
    else:
      if not self.restored:
        self.sess.run(self.initializer)


  def get_global_step_val(self):
    """Returns an `int` as the temporal value of global step."""
    global_step_val = tf.train.global_step(self.sess, self.global_step)
    return global_step_val


  def get_to_save_generator(self, skip_step=100):
    """XXX"""
    i = 0
    while True:
      i += 1
      yield True if i % skip_step == 0 else False


  def get_to_summarize_generator(self, skip_step=100):
    """XXX"""
    i = 0
    while True:
      i += 1
      yield True if i % 100 == 0 else False


  def train(self, n_iters, feed_dict_generator,
            options=None, run_metadata=None, verbose=True):
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

      verbose:
        `bool`.
    """

    self.initialize_if_not_restored()

    if self.logdir is not None:
      to_summarize = self.get_to_summarize_generator()
    if self.dir_to_ckpt is not None:
      to_save = self.get_to_save_generator()

    if verbose:
      global_step_val = tf.train.global_step(self.sess, self.global_step)
      print('INFO - Start training at global step {}.'.format(global_step_val))

    # Iterations
    for i in tqdm(range(n_iters)):  # XXX

      try:
        # Not summarize and write summary if not ...
        summarizer = None
        writer = None

        if self.logdir is not None:
          if next(to_summarize) is True:
            # Shall summarize and write summary
            summarizer = self.summarizer
            writer = self.writer

        feed_dict = next(feed_dict_generator)
        global_step_val = self.get_global_step_val()
        iterate(self.sess, self.train_ops, feed_dict,
                summarizer=summarizer, writer=writer,
                global_step=global_step_val, options=options,
                run_metadata=run_metadata)
        self.sess.run(self.increase_global_step_op)

      except StopIteration:
        print('INFO - No more training data to iterate.')
        break

      # Save at `skip_step`
      if self.dir_to_ckpt is not None:
        if next(to_save) is True:
          self.save()

    # Finally
    if verbose and self.dir_to_ckpt is not None:
      print('INFO - Saved to {}.'.format(self.dir_to_ckpt))
