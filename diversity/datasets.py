# -*- coding: utf-8 -*-
import numpy as np

def skew_pmf(a=1, n=10):
    """ Creates a skewed discrete distribution

    Arguments:

        a: `float>0`. The degree of skew
        n: `int`. The number of partitions in the distribution

    Returns:

        `ndarray(n)`
    """
    if a == 1:
        out = np.ones(n)/n
    else:
        normalization_constant = (-1+a**(1/(n-1)))/(-1+a**(n/(n-1)))
        out = np.array([a**((i)/(n-1)) for i in range(n)])
        out = normalization_constant*out
    return out
