#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script involves `iterate` and `BaseTrainer`."""


import abc
import numpy as np
import tensorflow as tf
from tqdm import tqdm



def iterate(sess, iter_ops, feed_dict, summarizer=None, writer=None,
            global_step_val=None, options=None, run_metadata=None):
  """Iterates one step for the TensorFlow `Op`s in `iter_ops`.

  CAUTION:
    This "function" will change the state of the `sess`.

  NOTE:
    This implementation abstracts all, and nothing else is essential. (That is,
    all args in all employed functions (methods) have been fullfilled.)

    Since the saving process in TensorFlow is not achived by `Op` (as how the
    summarizing and writing to TensorBoard are done), it is not essential, thus
    will not also, be included herein.

  Args:
    sess:
      An instance of `tf.Session()`, as the session this iteration works on.

    iter_ops:
      List of `Op`s to be iterated. Ensure that it has been initialized.

    feed_dict:
      A `feed_dict` associated to the `tf.placeholder`s needed by the `iter_ops`.

    summarizer:
      A "summary op" that summarizes the graph, e.g. `tf.summary.merge_all`,
      optional.

    writer:
      An instance of `tf.summary.FileWriter` that writes the summary into disk.
      If the `summarizer` is `None` (as default), then this argument is useless,
      optional.

    global_step_val:
      `int` or `None`, as the value of global-step, optional.

    options:
      A `[RunOptions]` protocol buffer or `None`, as the associated argument of
      `tf.Session.run()`, optional.

    run_metadata:
      A `[RunMetadata]` protocol buffer or `None`, as the associated argument of
      `tf.Session.run()`, optional.

  Returns:
    List of numpy arraries, as the values of `iter_ops` in this iteration.
  """

  # Get `fetches`
  fetches = [op for op in iter_ops]
  if summarizer is not None:
    fetches.append(summarizer)

  # Iterate in one step and get values
  fetch_vals = sess.run(fetches,
                        feed_dict=feed_dict,
                        options=options,
                        run_metadata=run_metadata)

  # Write to TensorBoard
  if summarizer is not None and writer is not None:
    summary = fetch_vals[-1]
    writer.add_summary(summary, global_step=global_step_val)

  # Return the values of `iter_ops`
  if summarizer is not None:
    # The last element of `fetch_vals` will be the `summary`
    iter_op_vals = fetch_vals[:-1]
  else:
    iter_op_vals = fetch_vals
  return iter_op_vals



class BaseTrainer(object):
  """Abstract base class of trainer that supplements the `iterate`.
  
  In addition, it will:

    1. make initialization;

    2. the `save` and `restore`, as well as their essential instance of
       `tf.train.Saver`, will be defined, since initialization is entangled
       with them;

    3. a global-step-variable and its incrementation-`Op` will be defined.
    

  Args:
    init_global_step:
      `int` as the initial value of global step, optional.

    initializer:
      A TenosrFlow initializer, or `None`, optional. If `None`, then using
      `tf.global_variables_initializer` as the employed initializer.

  Attributes:
    saver:
      An instance of `tf.train.Saver`.

    global_step:
      An un-trainable instance of `tf.Veriable`.

    initializer:
      An instance of TensorFlow initializer.

    restored:
      `bool`, for labeling if method `restore()` has ever been called.

  Methods:
    get_sess:
      Returns an instance of `tf.Session()` as the argument `sess` of
      `iterate()`.

    get_iter_ops:
      Abstract method. Returns list of ops as the argument `iter_ops` of
      `iterate()`.

    get_summarizer:
      Abstract method. Returns TensorFlow `Op` or `None` as the argument
      `summarizer` of `iterate()`.

    get_writer:
      Abstract method. Returns an instance of `tf.summary.FileWriter` or `None`
      as the argument `writer` of `iterate()`.

    get_global_step_val:
      Returns an `int` as the temporal value of global step.

    iterate_body:
      The body of iteration. It gets the arguments needed by `iterate()` and
      runs `iterate()` once. Also, it increments the value of
      `self.global_step`.

    create_saver:
      Abstract method. Returns an instance `tf.train.Saver()`.

    save:
      Abstract method. Saves the checkpoint of `self.sess` to disk, or do
      nothing is none is to be saved.

    restore:
      Abstract method. Restores the checkpoint to `self.sess` from disk, or
      do nothing is none is to be saved. Calling this method SHALL modify the
      attribute `self.has_restored` to be `True`.

    train:
      As the trainer trains.
  """

  def __init__(self, init_global_step=0, verbose=True):
    """Session shall be created after the graph has been completely built up
    (thus will not be modified anymore). The reason is that the partition of
    the resources and the edges of the graph in the session is optimized based
    on the graph. Thus the graph shall not be modified after the session having
    been created.
    """

    # `tf.Session` shall be created AFTER `super().__init__()`, when finishing
    # building `self.graph`.

    self.verbose = verbose

    # Initialize
    self.has_restored = False

    # Building of `iter_ops` may need `self.global_step`, which thus shall be
    # defined in front.
    self.global_step, self.increase_global_step_op = \
        self.build_global_step(init_global_step)


  def train(self, n_iters, feed_dict_generator, initializer=None):
    """As the trainer trains.

    Args:
      n_iters:
        `int`, as the number of iterations.

      feed_dict_generator:
        A generator that emits a feed_dict at each calling of `next()`.
    """

    # Initialize
    self.initialize(initializer)

    # Iterations
    pbar = tqdm(range(n_iters))
    for i in pbar:

      try:
        self.iter_op_vals = self.iterate_body(feed_dict_generator)
        pbar.set_description(self.set_pbar_description())

      except StopIteration:
        # Meaning that the `feed_dict_generator` has been exhausted.
        print('INFO - No more training data to iterate.')
        break

    # Save the checkpoint to disk at the end of training
    self.save()


  def set_pbar_description(self):
    """To be overrided."""
    return None


  def initialize(self, initializer):
    """Shall be called after being restored and having created `self.sess`."""

    if self.has_restored:
      if self.verbose:
        print('INFO - Restored, thus without initialization.')

    else:
      if initializer is None:
        self.sess.run(tf.global_variables_initializer())
      else:
        self.sess.run(initializer)

      if self.verbose:
        print('INFO - Initialized without restoring.')

    if self.verbose:
      global_step_val = self.get_global_step_val()
      print('INFO - Start training at global step {}.'.format(global_step_val))


  def iterate_body(self, feed_dict_generator):
    """The body of iteration. It gets the arguments needed by `iterate()` and
    runs `iterate()` once. Also, it increments the value of `self.global_step`.

    Appending anything into this `iterate_body()` can be simply archived by
    re-implementing `iterate_body()` with `super().iterate_body(...)`.

    Returns:
      List of numpy arraries, as the values of `Op`s from `self.get_iter_ops()`
      in this iteration.
    """

    # Run `iterate()` once
    iter_op_vals = iterate(**self.get_iterate_kwargs(feed_dict_generator))

    # Also, increment the value of `self.global_step`
    self.sess.run(self.increase_global_step_op)

    return iter_op_vals


  def get_iterate_kwargs(self, feed_dict_generator):
    """Get the kwargs of `iterate()`."""

    iterate_kwargs = {
        'sess':
            self.get_sess(),
        'iter_ops':
            self.get_iter_ops(),
        'feed_dict':
            next(feed_dict_generator),
        'summarizer':
            self.get_summarizer(),
        'writer':
            self.get_writer(),
        'global_step_val':
            self.get_global_step_val(),
        'options':
            self.get_options(),
        'run_metadata':
            self.get_run_metadata(),
    }

    return iterate_kwargs


  @abc.abstractmethod
  def save(self):
    """Abstract method. Saves the checkpoint of `self.sess` to disk, or do
    nothing is none is to be saved."""
    pass


  @abc.abstractmethod
  def restore(self):
    """Abstract method. Restores the checkpoint to `self.sess` from disk, or
    do nothing is none is to be saved. Calling this method SHALL modify the
    attribute `restored` to be `True`.

    Returns:
      `bool`, being `True` if sucessfully restored from checkpoint; else
      `False`.
    """
    self.has_restored = True


  def build_global_step(self, init_global_step):
    """Builds and returns `global_step` and `increase_global_step_op` which
    increments the value of `global_step` at each call.

    Args:
      init_global_step:
        `int` as the initial value of global step, optional.

    Returns:
      `global_step` and `increase_global_step_op` which increments the value
      of `global_step` at each call.
    """

    with self.graph.as_default():

      with tf.name_scope('increase_global_step_op'):

        global_step = tf.Variable(
            init_global_step, trainable=False, name='global_step')

        increase_global_step_op = global_step.assign_add(1)

    return (global_step, increase_global_step_op)


  @abc.abstractmethod
  def create_saver(self):
    """Abstract method. Returns an instance `tf.train.Saver()`.
    
    CAUTION:
      This method shall be called after introducing all variables that are to
      be saved and restored. Otherwise, a `FailedPreconditionError` exception
      "Attempting to use uninitialized value ..." will raise.

    NOTE:
      Saver shall be initialized within `self.graph`.
    """
    pass


  # ---------------- Methods within `self.iterate_body()` ----------------

  def get_global_step_val(self):
    """Returns an `int` as the temporal value of global step."""
    global_step_val = tf.train.global_step(self.sess, self.global_step)
    return global_step_val


  @abc.abstractmethod
  def get_sess(self):
    """Returns an instance of `tf.Session()` as the argument `sess` of
    `iterate()`.

    Since there's only one session is needed through out the training process,
    we just return the `self.sess` created by method `self.create_sess`.
    """


  @abc.abstractmethod
  def get_iter_ops(self):
    """Abstract method. Returns list of ops as the argument `iter_ops` of
    `iterate()`."""
    pass


  @abc.abstractmethod
  def get_summarizer(self):
    """Abstract method. Returns TensorFlow `Op` or `None` as the argument
    `summarizer` of `iterate()`."""
    return None


  @abc.abstractmethod
  def get_writer(self):
    """Abstract method. Returns an instance of `tf.summary.FileWriter` or `None`
    as the argument `writer` of `iterate()`."""
    return None


  @abc.abstractmethod
  def get_options(self):
    """Abstract method. Returns a`[RunOptions]` protocol buffer or `None`, as
      the argument `options` of `iterate()`."""
    return None


  @abc.abstractmethod
  def get_run_metadata(self):
    """Abstract method. Returns a `[RunMetadata]` protocol buffer or `None`, as
    the argument `run_metadata` of `iterate()`."""
    return None
