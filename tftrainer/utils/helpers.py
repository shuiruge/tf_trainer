#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Description
---------
Helper functions for TensorFlow.

Focked from _TensorFlow for Machine Intelligence_, chapter
_8. Helper Functions, Code Structure, and Classes_.

`define_scope()` and its helper `doublewrap()` are focked from
[here](https://danijar.github.io/structuring-your-tensorflow-models).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
import errno
import os
import tensorflow as tf



def ensure_directory(directory):
    """
    Check whether the `directory` exists or not. If not, then create
    this directory.

    Args:
        directory: str
    Returns:
        None
    """

    directory = os.path.expanduser(directory)

    try:
        os.makedirs(directory)

    except OSError as e:

        if e.errno != errno.EEXIST:

            raise e
    
    return None



class AttrDict(dict):
    """
    This simple class just provides some convenince when working
    with configuration objects.
    """

    def __getattr__(self, key):

        if key not in self:

            raise AttributeError

        else:

            return self[key]


    def __setattr__(self, key, value):
        
        if key not in self:

            raise AttributeError
        
        else:
            self[key] = value




def lazy_property(function):
    """
    As a decorator. asdf

    Remark:
        When `define_scope()` is used as the decorator, there is no need
        to use `lazy_property` any more, since the first involves the later.

    Example:
        Within a class of model of Tensorflow: 

            @lazy_property
            def inputs(self):
                return tf.placeholder(tf.int32)            

        is equivalent to make the following in __init__:
            
            self.inputs = tf.placeholder(tf.int32)
    """

    attribute = '_lazy_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):

        if not hasattr(self, attribute):

            setattr(self, attribute, function(self))

        return getattr(self, attribute)
       
    return wrapper



def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):

        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        
            return function(args[0])

        else:

            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.

    Remark:
        This decorator has involved `lazy_property()`.

    Example:
        Within a class of model of Tensorflow: 

            @define_scope
            def inputs(self):
                return tf.placeholder(tf.int32)            

        is equivalent to make the following in __init__:
            
            with tf.name_scope('inputs'):
                self.inputs = tf.placeholder(tf.int32)
    """

    attribute = '_cache_' + function.__name__

    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        
        if not hasattr(self, attribute):

            with tf.variable_scope(name, *args, **kwargs):

                setattr(self, attribute, function(self))
                    
        else:
            return getattr(self, attribute)
         
    return decorator
