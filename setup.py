#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


NAME = 'tftrainer'
DESCRIPTION = 'TensorFlow Trainer"
AUTHOR = 'shuiruge'
AUTHOR_EMAIL = 'shuiruge@hotmail.com'
URL = 'https://github.com/shuiruge/tf_trainer/'
VERSION = '0.2.0'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
 
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license='MIT',
    url=URL,
    packages=find_packages(exclude=[
        'tests.*', 'tests',
        'examples.*', 'examples',
        'dat.*', 'dat',
    ]),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Programming :: TensorFlow'
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python 3+',
        'Framework :: TensorFlow',
    ],
)
