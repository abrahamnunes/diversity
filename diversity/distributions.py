# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as ss

def pool_mvn_covariance(mean, cov):
    """ Pools the covariance matrices between several samples

    Arguments:

        mean: `ndarray((nsamples, n))`. Means for `nsamples` assemblages
        cov: `ndarray((nsamples, n, n))`. The covariance matrix for `nsamples` assemblages


    Returns:

        `float`
    """

    dM = (mean-np.tile(mean.mean(0), [mean.shape[0], 1]))
    return (np.sum(cov, 0) + dM.T@dM)/mean.shape[0]


def mvn_renyi(cov, q=1):
    """ Renyi heterogeneity of a multivariate Gaussian distribution

    Arguments:

        cov: `ndarray((n, n))`. The covariance matrix
        q: `float>0`. The order of the measure

    Returns:

        `float`
    """
    if q == 1:
        out = np.exp(cov.shape[0]/2)*np.sqrt(np.linalg.det(2*np.pi*cov))
    else:
        out = np.sqrt(np.linalg.det(q**(1/(q-1) * 2*np.pi*cov)))

def mvn_renyi_alpha(cov, q=1):
    """ Renyi heterogeneity of a multivariate Gaussian distribution

    Note: currently this only supports 1/nsamples weights for each of the component distributions

    Arguments:

        cov: `ndarray((nsamples, n, n))`. The covariance matrix for `nsamples` assemblages
        q: `float>0`. The order of the measure

    Returns:

        `float`
    """
    nsamples, n, _ = cov.shape
    detC = np.linalg.det(cov)
    if q == 1:
        Z = (2*np.pi*np.e)**(n/2)
        out = Z*np.prod(detC**(1/(2*nsamples)))
    else:
        Z = np.sqrt((2*np.pi*q**(1/(q-1)))**n)
        out = Z*(np.sum(detC**((1-q)/2))/nsamples)**(1/(1-q))
    return out

def mvn_renyi_decomp(mean, cov, w=None, q=1):
    """ Decomposition of Renyi heterogeneity of a Gaussian Mixture

    Arguments:

        mean: `ndarray((nsamples, n))`. Means for `nsamples` assemblages
        cov: `ndarray((nsamples, n, n))`. The covariance matrix for `nsamples` assemblages
        q: `float>0`. The order of the measure

    Returns:

        Tuple with the following
            gamma: `float`. Pooled Renyi heterogeneity
            alpha: `float`. Within group heterogeneity
            beta:  `float`. Between group heterogeneity

    """
    gamma = mvn_renyi(pool_mvn_covariance(mean, cov), q=q)
    alpha = mvn_renyi_alpha(mean, cov, q=q)
    return (gamma, alpha, gamma/alpha)
