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

def func_cwm(X, p):
    """ Functional Community Weighted Mean

    Arguments:

        X: `ndarray((nclasses, ntraits))`. Trait values for each class
        p: `ndarray(nclasses)`. Discrete probability distribution over classes

    Returns:

        `ndarray(ntraits)` or `float` if `ntraits==1`

    References:

        - Mason NWH et al. An index of functional diversity. J Veg Sci 2003; 14: 571–8.
    """
    if np.ndim(X) == 1:
        X = X.reshape(p.size, 1)
    out = np.einsum('i,ij->j', p, X)
    return out

def func_div(X, p):
    """ Functional divergence measure

    Arguments:

        X: `ndarray((nclasses, ntraits))`. Trait values for each class
        p: `ndarray(nclasses)`. Discrete probability distribution over classes

    Returns:

        `ndarray(ntraits)` or `float` if `ntraits==1`

    References:

        - Mason NWH et al. An index of functional diversity. J Veg Sci 2003; 14: 571–8.
    """
    if np.ndim(X) == 1:
        X = X.reshape(p.size, 1)
    _logX = np.einsum('i,ij->j', p, np.ma.log(X))
    sq_diff = np.power(np.ma.log(X) - np.tile(_logX, [X.shape[0], 1]), 2)
    V = np.einsum('i,ij->j', p, sq_diff)
    out = (2/np.pi)*np.arctan(5*V)
    return out

def gei(y, q=1, axis=0):
    """ Generalized Entropy Index

    Arguments:

        y: `ndarray(n)`. Vector
        q: `float`. Order of the entropy index
        axis: `int`. Axis along which to apply the measure

    Returns:

        `ndarray`

    """
    return np.apply_along_axis(div_base._gei, axis=axis, arr=y, q=q)

def gini_lorenz(L, axis=0):
    """ Computation of the Gini inequality coefficient based on the Lorenz curve

    Arguments:

        L: `ndarray((nclasses,))`. Lorenz curve
        axis: `int`. Axis over which to apply the measure

    Returns:

        `ndarray`
    """
    return np.apply_along_axis(div_base._gini_lorenz, axis=axis, arr=L)

def gini_simpson(p, axis=0):
    """ Gini-Simpson index

    Arguments:

        p: `ndarray(n)`. Probability distribution
        axis: `int`. Axis along which to apply the measure

    Returns:

        `ndarray(n)`
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

def lorenz_curve(X, axis=0, return_prop=True):
    """ Constructs a Lorenz curve

    Arguments:

        X: `ndarray((nclasses, nsamples))` if `axis==0` or `ndarray((nsamples, nclasses))` if `axis==1`. Values over which Lorenz curve should be computed
        axis: `int`. Axis over which to compute Lorenz curves
        return_prop: `bool`. Whether to also return the cumulative proportion of the population.

    Returns:

        (`ndarray(nclasses)`, `ndarray(X.shape)`) if `return_prop==True`
        `ndarray(X.shape)` if `return_prop==False`

    References:

        - Lorenz MO. Methods of Measuring the Concentration of Wealth. Publ Am Stat Assoc 1905; 9: 202–19.
    """
    L = np.apply_along_axis(div_base._lorenz_curve, axis=axis, arr=X)
    if return_prop == True:
        p = np.linspace(0, 1, L.shape[axis])
        out = (p, L)
    else:
        out = L
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

def pietra_lorenz(L, axis=0):
    """ Computation of the Pietra inequality coefficient based on the Lorenz curve

    Arguments:

        L: `ndarray((nclasses,))`. Lorenz curve
        axis: `int`. Axis over which to apply the measure

    Returns:

        `ndarray`
    """
    return np.apply_along_axis(div_base._pietra_lorenz, axis=axis, arr=L)

def qrqe(D, p, q=1):
    """ Rao's generalized quadratic entropy

    Arguments:

        D: `ndarray((nclasses, nclasses))`. Dissimilarity matrix
        p: `ndarray(nclasses)`. Probability distribution
        q: `float>0`. Order of the RQE measure

    Returns:

        `float`

    References:

        - Chiu CH, Chao A. Distance-based functional diversity measures and their decomposition: A framework based on hill numbers. PLoS One 2014; 9.
        - Rao CR. Diversity and dissimilarity coefficients: A unified approach. Theoretical Population Biology. 1982;21(1):24-43


    """
    return np.sum(D*np.power(np.outer(p, p), q))

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

def rqe(D, a, method='nei-tajima'):
    """ Rao's quadratic entropy

    Arguments:

        D: `ndarray((nclasses, nclasses))`. Dissimilarity matrix
        a: `ndarray(nclasses)`. Vector of abundances. If `method='nei-tajima'`, this must be a vector of counts.
        method: `{'nei-tajima', 'base'}`. Which RQE is being calculated. The Nei-Tajima estimator is unbiased and returns a variance estimate

    Returns:

        `(estimate [float], variance [float])` if `method='nei-tajima'`
        `float` if `method='base'`.

    TODO:

        - Currently the variance estimates with the Nei-Tajima estimator behave strangely. I would not yet use these in production. I will do some investigation comparing these to bootstrap estimates.
        - Add capacity for bootstrap estimation.

    References:

        - Rao (1982). Diversity and dissimilarity coefficients: A unified approach. Theoretical Population Biology. 21(1):24-43
        - Nei & Tajima (1981). DNA Polymorphism Detectable by Restriction Endonucleases. Genetics. 97: 145-163


    """
    p = div_base._normalize(a)

    # Extract upper triangular elements
    if method == 'base':
        out = np.einsum('i,ij,j->', p, D, p)
    else:
        p = div_base._normalize(a)
        Nt = a.sum()
        Nc = p.size
        pDp = np.einsum('i,ij,j->', p, D, p)
        Q_hat = (Nt/(Nt-1))*pDp

        # Variance
        Term_A = (1.5-Nc)*np.power(pDp, 2)
        Term_B = (Nc-2)*np.einsum('ij,ik,i,j,k->', D, D, p, p, p)
        Term_C = 0.5*np.einsum('i,ij,j->', p, D**2, p)
        Q_var = (4/(Nc*(Nc-1)))*(Term_A + Term_B + Term_C)

        out = (Q_hat, Q_var)
    return out

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

        `float`
    """
    return np.apply_along_axis(div_base._tsallis_entropy, axis=axis, arr=p, q=q)

def func_hill(D, p, q=1):
    """ Functional Hill numbers

    Arguments:

        D: `ndarray((n,n))`. Distance matrix
        p: `ndarray(n)`. Probability distribution
        q: `float`. Order of the functional hill numbers

    Returns:

        `float`

    """
    pp = np.outer(p, p)
    if q == 1:
        Dpp = D*pp
        Q1 = np.sum(Dpp)
        out = np.exp(-np.sum((Dpp/Q1) * np.ma.log(pp))/2)
    else:
        Qq = np.sum(D*(pp**q))
        Q1 = np.sum(D*pp)
        out = (Qq/Q1)**(1/(2*(1-q)))
    return out

def leinster_cobbold(S, p, q=1):
    """ Leinster Cobbold Index

    Arguments:

        S: `ndarray((n,n))`. Similarity matrix
        p: `ndarray(n)`. Probability distribution
        q: `float`. Order of the functional hill numbers

    Returns:

        `float`

    """
    pp = np.outer(p, p)
    if q == 1:
        out = np.prod((S@p)**(-p))
    elif q==np.inf:
        out = np.min(1/(S@p))
    else:
        out = (p@((S@p)**(q-1)))**(1/(1-q))
    return out


def ricotta_szeidl(D, p):
    """ Ricotta & Szeidl's Numbers Equivalent RQE

    Arguments:

        D: `ndarray((n,n))`. Distance matrix
        p: `ndarray(n)`. Probability distribution
        q: `float`. Order of the functional hill numbers

    Returns:

        `float`

    """
    rqe = np.sum(D*np.outer(p,p))
    return 1/(1-rqe/np.max(D))
