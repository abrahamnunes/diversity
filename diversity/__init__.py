# -*- coding: utf-8 -*-

"""Top-level package for Diversity."""

__author__ = """Abraham Nunes"""
__email__ = 'nunes@dal.ca'
__version__ = '0.1.0'


from diversity.indices import *
from diversity.decompositions import *
from diversity import base
from diversity import datasets
from diversity import estimators

__all__ = [
    'base',
    'datasets',
    'decompositions',
    'estimators',
    'indices']
