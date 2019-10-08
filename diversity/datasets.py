# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

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

class ClassificationData(object):
    """ Creates a classification or clustering dataset

    """
    def __init__(self,
                 nsamples=100,
                 ninformative=10,
                 nredundant=0,
                 nrepeated=0,
                 nclusters=2,
                 alpha=1,
                 class_sep=1,
                 flip_y=0.0,
                 shuffle=True,
                 feature_range=(-1, 1),
                 rng=np.random.RandomState()):
        """
        Arguments:

            nsamples: `int>0`. Number of samples in the dataset
            ninformative: `int>=0`. Number of informative features
            nredundant: `int>=0`. Number of redundant features
            nrepeated: `int>=0`. Number of repeated features
            alpha: `0<=float<=1`. Skew in the distribution of classes
            nclusters: `int>0`. Number of clusters to model
            class_sep: `0<float`. Amount of space between classes
            flip_y: `0<=float<1`. Probability of flipping labels
            shuffle: `bool`. Whether to shuffle the examples
            feature_range: `(min, max) or None`. Tuple indicating the rescaling box. If `None`, then will be left as is.
            rng: `np.random.RandomState`. Pseudorandom number generator

        """
        self.nsamples = int(nsamples )
        self.nfeatures = int(ninformative + nredundant + nrepeated)
        self.ninformative = int(ninformative)
        self.nredundant = int(nredundant)
        self.nrepeated = int(nrepeated)
        self.nclusters = int(nclusters)
        self.alpha = alpha
        self.class_sep = class_sep
        self.flip_y = flip_y
        self.shuffle = shuffle
        self.feature_range = feature_range
        self.rng = rng
        self.prob = skew_pmf(self.alpha, self.nclusters)


    def sample(self):
        n_perclass = np.round(self.nsamples*self.prob).astype(np.int)
        n_max = np.max(n_perclass)
        n_tot = n_max*self.nclusters

        X, y = make_classification(n_samples=n_tot,
                                   n_features=self.nfeatures,
                                   n_informative=self.ninformative,
                                   n_redundant=self.nredundant,
                                   n_repeated=self.nrepeated,
                                   n_classes=self.nclusters,
                                   n_clusters_per_class=1,
                                   weights=None,
                                   flip_y=self.flip_y,
                                   class_sep=self.class_sep,
                                   hypercube=True,
                                   shift=0.0,
                                   scale=1.,
                                   shuffle=False,
                                   random_state=self.rng)

        X = np.vstack([X[np.ravel(np.argwhere(np.equal(y, i)))[:n_perclass[i]]] for i in range(self.nclusters)])
        self.y = np.hstack([y[np.ravel(np.argwhere(np.equal(y, i)))[:n_perclass[i]]] for i in range(self.nclusters)])

        if self.feature_range is not None:
            self.X = MinMaxScaler(feature_range=self.feature_range).fit_transform(X)
        else:
            self.X = X

        if self.shuffle:
            idx = np.arange(X.shape[0])
            self.rng.shuffle(idx)
            self.X = self.X[idx]
            self.y = self.y[idx]
        return self.X, self.y
