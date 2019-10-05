#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from pathlib import Path

from setuptools import find_packages, setup


# Package meta-data.
NAME = 'regression_model'
DESCRIPTION = 'Train and deploy regression model.'
URL = 'https://github.com/JoaquinRives/azure-package'
EMAIL = 'joaquin.rives01@email.com'
AUTHOR = 'Joaquin Rives'
REQUIRES_PYTHON = '>=3.6.0'



here = os.path.abspath(os.path.dirname(__file__))


# Requirements
def list_reqs(fname=os.path.join(here, 'requirements.txt')):
    with open(fname) as fd:
        return fd.read().splitlines()


try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Setup
setup(
    name=NAME,
    version='0.1',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={'regression_model': ['0.1']},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
