# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as ss
import diversity.base as div_base

def berger_parker_diversity(p, axis=0):
    """ Berger-Parker diversity index

    Arguments:

        p: `ndarray(n)`. Probability distribution
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`

    """
    return np.apply_along_axis(div_base._berger_parker_diversity, axis=axis, arr=p)

def berger_parker_dominance(p, axis=0):
    """ Berger-Parker dominance index

    Arguments:

        p: `ndarray(n)`. Probability distribution
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`

    """
    return 1/berger_parker_diversity(p, axis=axis)



def coef_variation(y, axis=0):
    """ Coefficient of variation

    Arguments:

        y: `ndarray(n)`. Vector
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`

    """
    out = np.std(y, axis=axis)/np.mean(y, axis=axis)

def freeman_index(p, axis=0):
    """ Freeman index

    Arguments:

        p: `ndarray(n)`. Probability distribution
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`

    """
    return 1-berger_parker_dominance(p, axis=axis)

def gei(y, q=1, axis=0):
    """ Generalized Entropy Index

    Arguments:

        y: `ndarray(n)`. Vector
        q: `float`. Order of the entropy index
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`

    """
    return np.apply_along_axis(div_base._gei, axis=axis, arr=y, q=q)

def gini_simpson(p, axis=0):
    """ Gini-Simpson index

    Arguments:

        p: `ndarray(n)`. Probability distribution
        axis: `int`. Axis along which to apply the measure

    Returns:

        `ndarra`
    """
    return 1 - (1/np.apply_along_axis(div_base._renyi, axis=axis, arr=p, q=2))


def hartley(p, axis=0):
    """ Hartley heterogeneity

    Arguments:

        p: `ndarray(n)`. Probability distribution
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`

    """
    return np.apply_along_axis(div_base._hartley, axis=axis, arr=p)

def hartley_entropy(p, axis=0):
    """ Hartley entropy

    Arguments:

        p: `ndarray(n)`. Probability distribution
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`

    """
    return np.log(np.apply_along_axis(div_base._hartley, axis=axis, arr=p))

def lincoln_index(s1, s2, method='chapman', alpha=0.05):
    """ Returns the Lincoln Index of a population size given two samples

    Arguments:

        s1: `ndarray(n1)`. Vector of categorical species ID's for the first sample
        s2: `ndarray(n2)`. Vector of categorical species ID's for the second sample
        method: `{'base', 'chapman', 'bayesian'}`. Method by which to compute the estimate
        alpha: `0<float<1`. Significance threshold (for use with Chapman estimator)

    Returns:

        `method='base'` returns a single `float`
        `method='chapman'` returns a tuple with the Chapman estimator and 95% confidence interval `(estimate, ci_low, ci_high)`
        `method='bayesian'` returns a tuple with the Bayesian estimate of the mean and mean+/-SD

    """
    n1, n2 = np.unique(s1).size, np.unique(s2).size
    n12 = np.intersect1d(s1, s2).size
    if method == 'base':
        out = (n1*n2)/n12
    elif method == 'chapman':
        nhat = ((n1 + 1)*(n2 + 1)/n12) - 1
        se_a = 1/(n12 + 0.5)
        se_b = 1/(n1 - n12 + 0.5)
        se_c = 1/(n2 + 0.5)
        se_d = (n12 + 0.5)/((n1-n12+0.5)*(n2+0.5))
        se = np.sqrt(se_a + se_b + se_c + se_d)
        ci_err_a = n1 + n2 - n12 - 0.5
        ci_err_b = ((n1-n12+0.5)*(n2 - n12 + 0.5)/(n12 + 0.5))
        ci_err_c_low = np.exp(-ss.norm().ppf(1-(alpha/2))*se)
        ci_err_c_high = np.exp(ss.norm().ppf(1-(alpha/2))*se)
        ci_low = ci_err_a + ci_err_b*ci_err_c_low
        ci_high = ci_err_a + ci_err_b*ci_err_c_high
        out = (nhat, ci_low, ci_high)
    elif method == 'bayesian':
        nhat  = ((n1-1)*(n2-1))/(n12 - 2)
        sdhat = np.sqrt(nhat*(((n1-n12 + 1)*(n2-n12 + 1))/((n12-2)*(n12-3))))
        out = (nhat, nhat-sdhat, nhat+sdhat)

    return out


def mean_logdev(y, axis=0):
    """ Mean Log Deviation

    Arguments:

        y: `ndarray(n)`. Vector
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`

    """
    return np.apply_along_axis(div_base._mean_logdev, axis=axis, arr=y)

def ModVR(p, axis=0):
    """ Variation around the mode

    Arguments:

        p: `ndarray(n)`. Probability distribution
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`

    """
    n = p.shape[axis]
    return (n/(n-1))*berger_parker_dominance(p, axis=axis)

def RanVR(p, axis=0):
    """ Variation around the mode

    Arguments:

        p: `ndarray(n)`. Probability distribution
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`

    """
    return np.apply_along_axis(div_base._RanVR, axis=axis, arr=p)

def range(x, axis=0):
    """ Range

    Arguments:

        x: `ndarray(n)`. Vector of values to test
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`
    """
    return np.max(x, axis=axis)-np.min(x, axis=axis)

def renyi(p, q=1, axis=0):
    """ Renyi heterogeneity

    Arguments:

        p: `ndarray(n)`. Probability distribution
        q: `float>=0`. Order of the Renyi heterogeneity measure
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`
    """
    return np.apply_along_axis(div_base._renyi, axis=axis, arr=p, q=q)

def renyi_alpha(P, w=None, q=1, axis=0):
    """ Within-group (Alpha) Renyi heterogeneity

    Arguments:

        p: `ndarray((nsamples, nclasses))`. Probability distribution
        q: `float>=0`. Order of the Renyi heterogeneity measure
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`
    """
    if w is None:
        w = np.array([1/P.shape[0]]*P.shape[0])

    if q == 1:
        out = np.exp(np.sum(-w*np.sum(P*np.ma.log(P), 1)))
    else:
        out = (np.sum((w**q)*np.sum(P**q, 1))/np.sum(w**q))**(1/(1-q))

    return out

def renyi_entropy(p, q=1, axis=0):
    """ Renyi entropy

    Arguments:

        p: `ndarray(n)`. Probability distribution
        q: `float>=0`. Order of the Renyi entropy measure
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`
    """
    return np.log(np.apply_along_axis(div_base._renyi, axis=axis, arr=p, q=q))

def shannon(p, axis=0):
    """ Shannon heterogeneity

    Arguments:

        p: `ndarray(n)`. Probability distribution
        axis: `int`. Axis along which to apply the measure


    Returns:

        `float`

    """
    return np.apply_along_axis(div_base._shannon, axis=axis, arr=p)

def shannon_entropy(p, axis=0):
    """ Shannon entropy

    Arguments:

        p: `ndarray(n)`. Probability distribution
        axis: `int`. Axis along which to apply the measure


    Returns:

        `float`

    """
    return np.log(shannon(p, axis=axis))


def simpson(p, axis=0):
    """ Simpson index

    Arguments:

        p: `ndarray(n)`. Probability distribution
        axis: `int`. Axis along which to apply the measure

    Returns:

        `ndarra`
    """
    return 1/simpson_dominance(p, axis=axis)


def simpson_dominance(p, axis=0):
    """ Simpson dominance index

    Arguments:

        p: `ndarray(n)`. Probability distribution
        axis: `int`. Axis along which to apply the measure

    Returns:

        `ndarra`
    """
    return np.apply_along_axis(div_base._renyi, axis=axis, arr=p, q=2)

def theil(y, axis=0):
    """ Theil Index

    Arguments:

        y: `ndarray(n)`. Vector
        axis: `int`. Axis along which to apply the measure

    Returns:

        `float`

    """
    return np.apply_along_axis(div_base._theil, axis=axis, arr=y)

def tsallis_entropy(p, q=1, axis=0):
    """ Tsallis entropy

    Arguments:

        p: `ndarray(n)`. Probability distribution
        q: `float`. Order of the Tsallis entropy
        axis: `int`. Axis along which to apply the measure

    Returns:

        `ndarra`
    """
    return np.apply_along_axis(div_base._tsallis_entropy, axis=axis, arr=p, q=q)
