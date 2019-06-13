# -*- coding: utf-8 -*-
import numpy as np

def _normalize(x):
    """ Normalizes the input vector

    Arguments:

        x: `ndarray(n)`. Input vector

    Returns:

        `ndarray(n)`. Sums to 1

    """
    return x/x.sum()

def _berger_parker_diversity(x):
    """ Base function for Berger-Parker diversity index

    Arguments:

        x: `ndarray(n)`. Input vector

    Returns:

        `float`

    """
    p = _normalize(x)
    return 1/np.max(p)

def _gei(y, q=1):
    """ Base function for the Generalized Entropy Index

    Arguments:

        y: `ndarray(n)`. Vector

    Returns:

        `float`

    """
    if q == 0:
        out = _mean_logdev(y)
    elif q == 1:
        out = _theil(y)
    else:
        out = (_qsum(y/np.mean(y), q)/y.size - 1)/(q*(q-1))
    return out

def _gini_lorenz(L):
    """ Base function for computation of the Gini inequality coefficient based on the Lorenz curve

    Arguments:

        L: `ndarray(nclasses)`. Lorenz curve

    Returns:

        `float`
    """
    out = 2*np.mean(np.linspace(0, 1, L.size)-L)

def _hartley(x):
    """ Base function for Hartley heterogeneity

    Arguments:

        x: `ndarray(n)`. Input vector

    Returns:

        `float`

    """
    p = _normalize(x)
    return p[p.nonzero()].size

def _lorenz_curve(x):
    """ Base function to construct a Lorenz curve

    Arguments:

        x: `ndarray(nclasses)`. Vector of features

    Returns:

        `ndarray(nclasses)`

    References:

        - Lorenz MO. Methods of Measuring the Concentration of Wealth. Publ Am Stat Assoc 1905; 9: 202–19.
    """
    return np.cumsum(np.sort(x)/np.sum(x))

def _mean_logdev(y):
    """ Base function for the Mean log deviation

    Arguments:

        y: `ndarray(n)`. Vector

    Returns:

        `float`
    """
    n = y.size
    y_ = y/np.mean(y)
    return -(np.sum(np.ma.log(y_)))/n

def _pietra_lorenz(L):
    """ Base function for computation of the Pietra inequality coefficient based on the Lorenz curve

    Arguments:

        L: `ndarray(nclasses)`. Lorenz curve

    Returns:

        `float`
    """
    out = np.max(np.linspace(0, 1, L.size)-L)

def _qsum(x, q=1):
    """ Base function for the exponentiated sum

    Arguments:

        x: `ndarray(n)`. Vector
        q: `float>=0`. Order of the q-sum

    Returns:

        `float`
    """
    return np.sum(x[x.nonzero()]**q)

def _RanVR(x):
    """ Base function for Range around the Mode

    Arguments:

        x: `ndarray(n)`. Input vector

    Returns:

        `float`

    """
    p = _normalize(x)
    return np.min(p)/np.max(p)

def _renyi(x, q=1):
    """ Base function for Renyi heterogeneity

    Arguments:

        x: `ndarray(n)`. Input vector
        q: `float>=0`. Order of the Renyi heterogeneity measure

    Returns:

        `float`
    """
    p = _normalize(x)
    if q == 0:
        out = _hartley(p)
    elif q == 1:
        out = _shannon(p)
    elif q == np.inf:
        out = _berger_parker_diversity(p)
    else:
        out = _qsum(p, q)**(1/(1-q))
    return out

def _shannon(x):
    """ Base function for Shannon heterogeneity

    Arguments:

        x: `ndarray(n)`. Input vector

    Returns:

        `float`

    """
    p = _normalize(x)
    return np.exp(-np.sum(p*np.ma.log(p)))

def _theil(y):
    """ Base function for the Theil index

    Arguments:

        y: `ndarray(n)`. Vector

    Returns:

        `float`
    """
    n = y.size
    y_ = y/np.mean(y)
    return (np.sum(y_*np.ma.log(y_)))/n

def _tsallis_entropy(x, q=1):
    """ Base function for Tsallis entropy

    Arguments:

        x: `ndarray(n)`. Input vector
        q: `float>=0`. Order of the Tsallis entropy measure

    Returns:

        `float`
    """
    p = _normalize(x)
    if q == 1:
        out = -np.sum(p*np.ma.log(p))
    else:
        out = (1-_qsum(p, q))/(q-1)
    return out
