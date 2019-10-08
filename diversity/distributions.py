# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as ss
from scipy.special import binom as C
import sklearn.metrics as skm


class FisherHypergeometricDistribution(object):
    def __init__(self, X=None, n=None, nsucc=None, ntot=None, odds_ratio=None):
        if X is not None:
            cm = skm.confusion_matrix(X[:,0], X[:,1])
            n = cm.sum(0)[1]; nsucc=cm.sum(1)[1]; ntot=cm.sum();
            odds_ratio = (X[0,0]*X[1,1])/(X[0,1]*X[1,0])

        self.n = n
        self.ntot = ntot
        self.nsucc = nsucc
        self.nfail = self.ntot - self.nsucc
        self.odds_ratio = odds_ratio
        self.xmin = np.maximum(self.n-self.nfail, 0)
        self.xmax = np.maximum(self.n, self.nsucc)


        self.domain = np.array([x for x in range(self.xmin, self.xmax + 1)])
        self.prob = np.array([C(self.nsucc, x)*C(self.nfail, self.n-x)*(self.odds_ratio**x) for x in self.domain])
        self.Z = np.sum(self.prob)
        self.prob = self.prob/self.Z
        self.cumprob = np.cumsum(self.prob)
        self.tailprob = 1 - self.cumprob

    def compute_normalization_constant(self):
        return np.sum([C(self.nsucc, y)*C(self.nfail, self.n-y)*(self.odds_ratio**y) for y in range(self.xmin, self.xmax+1)])

    def pmf(self, x):
        return self.prob[np.equal(self.domain, x)]

    def sf(self, x):
        if x + 1 < self.domain[-1]:
            tailprob = self.tailprob[np.equal(self.domain, x)]
        elif x < self.xmin:
            tailprob = 1
        else:
            tailprob = 0
        return tailprob

    def cdf(self, x):
        if x < self.xmin:
            cprob = 0
        else:
            cprob = self.cumprob[np.equal(self.domain, x)]
        return cprob

    def ppf(self, p):
        return self.domain[np.less_equal(p - self.cumprob, 0)][0]
