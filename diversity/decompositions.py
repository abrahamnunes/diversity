# -*- coding: utf-8 -*-
import numpy as np
import diversity as div

def renyi_decomp(P, w=None, q=1):
    """ Performs multiplicative decomposition of the Renyi heterogeneity

    Arguments:

        P: `ndarray((nsamples, nclasses))`. Matrix where each row is a probability distribution over classes from a given assemblage
        w: `ndarray(nsamples)`. Vector of weights for each sample
        q: `float>0`. Order of the heterogeneity measure

    Returns:

        Tuple with the following:
            gamma: `float`. The pooled heterogeneity
            alpha: `float`. The within-group heterogeneity
            beta:  `float`. The between-group heterogeneity

    """
    # Compute gamma
    Pagg = P.mean(0)
    Pagg = Pagg/np.sum(Pagg)
    gamma = div.renyi(Pagg, q)

    # Compute alpha
    alpha = div.renyi_alpha(P, w, q)

    return (gamma, alpha, gamma/alpha)
