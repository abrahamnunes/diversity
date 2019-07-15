# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as ss
from sklearn.model_selection import LeaveOneOut

def bootci(X,
           estimator,
           K=1000,
           alpha=0.05,
           rng = np.random.RandomState(),
           **kwargs):
    """ Computes bootstrap bias-corrected and accelerated confidence intervals

    Arguments:

        X: `ndarray((nsamples, features))`. Data
        estimator: `function`. Heterogeneity measurement function
        K: `int`. Number of bootstrap samples
        alpha: `0<float<1`. Significance threshold
        rng: `numpy.random.RandomState`
        **kwargs: Arguments for the estimator

    Results:

        `ndarray(3)`. Mean, CILower, CIUpper.

    """
    nsamples, nfeatures = X.shape; loo = LeaveOneOut(); norm = ss.norm()
    Za = norm.ppf(alpha); Z_a= norm.ppf(1-alpha)

    # Get bootstrap samples
    bs = rng.choice(np.arange(nsamples), size=(K, nsamples))

    # Compute estimates
    Tbs = np.stack(estimator(X[bs[i]], **kwargs) for i in range(K))
    Mbs  = np.mean(Tbs)

    if np.logical_not(np.all(np.equal(Tbs, Mbs))):
        # Bias correction factor
        Zo = norm.ppf(np.sum(np.less(Tbs, Mbs))/K)

        # Acceleration factor (based on jackknife estimates)
        Tjk = np.array([estimator(X[xin], **kwargs) for xin, _ in loo.split(X)])
        d_jk = np.mean(Tjk)-Tjk
        a_ = (np.sum(d_jk**3))/(6*np.sum(d_jk**2)**(3/2))

        # Compute and return confidence intervals
        Phi = lambda z0, Z, ahat: norm.cdf(z0+(z0+Z)/(1-ahat*(z0+Z)))
        pctile_low = Phi(Zo,Za,a_)*100
        pctile_high = Phi(Zo,Z_a,a_)*100
        out = (Mbs, np.percentile(Tbs, [pctile_low, pctile_high]))
    else:
        out = (Mbs, np.array([Mbs, Mbs]))

    return out
