# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as ss
from sklearn.model_selection import LeaveOneOut

def bootsamp(x, K=10000, rng=np.random.RandomState()):
    """ Generates indices for bootstrap sampling of a dataset's empirical distribution

    Arguments:

        x: `ndarray(nsamples)`. 1-D Dataset being resampled
        K: `int`. Number of bootstrap samples to be drawn
        rng: `numpy.random.RandomState`

    Returns:

        `ndarray((nsamples, K))`

    """
    nsamples = x.shape[0]
    out = rng.choice(x, size=(K, nsamples))
    return out

def bootci(x, estimator, K=1000, alpha=0.05, rng=np.random.RandomState(), **kwargs):
    """ Computes bootstrap bias-corrected and accelerated confidence intervals

    Arguments:

        x: `ndarray(nsamples,)`. Data
        estimator: `function`. Heterogeneity measurement function
        K: `int`. Number of bootstrap samples
        alpha: `0<float<1`. Significance threshold
        rng: `numpy.random.RandomState`
        **kwargs: Arguments for teh estimator

    Results:

        `ndarray(3)`. Mean, CILower, CIUpper.
    """
    n = x.size; loo = LeaveOneOut(); norm = ss.norm()
    Za = norm.ppf(alpha); Z_a= norm.ppf(1-alpha)

    # Get bootstrap samples and compute estimates
    Tbs = estimator(bootsamp(x, K, rng), axis=1, **kwargs)
    Mbs  = np.mean(Tbs)

    # Bias correction factor
    Zo = norm.ppf(np.sum(np.less(Tbs, Mbs))/K)

    # Acceleration factor (based on jackknife estimates)
    Tjk = np.array([estimator(x[xin], axis=0, **kwargs) for xin, _ in loo.split(x)])
    d_jk = np.mean(Tjk)-Tjk
    a_ = (np.sum(d_jk**3))/(6*np.sum(d_jk**2)**(3/2))

    # Compute and return confidence intervals
    Phi = lambda z0, Z, ahat: norm.cdf(z0+(z0+Z)/(1-ahat*(z0+Z)))

    return (m_bs, np.percentile(Tbs, [Phi(Zo,Za,a_)*100, Phi(Zo,Z_a,a_)*100]))
