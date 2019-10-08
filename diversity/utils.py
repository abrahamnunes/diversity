# -*- coding: utf-8 -*-
import numpy as np

def partition_dissimilarity_matrix(D, labels, normalize=False):
    """ Based on sample-wise dissimilarity matrix and partition ID's, finds the average dissimilarity between partitions

    Arguments:

        D: `ndarray((nsamples, nsamples))`. Dissimilarity matrix between individual observations
        labels: `ndarray(nsamples)`. Partition labels with `nlabels` unique values
        normalize: `bool`. Whether to divide all values by max.

    Returns:

        `ndarray((nlabels, nlabels))`. Symmetric between-partition dissimilarity matrix

    """
    unique_labels = np.unique(labels)
    nlabels = unique_labels.size
    Dout = np.empty((nlabels, nlabels))
    for i in range(nlabels-1):
        idx = np.equal(labels, unique_labels[i])
        for j in range(i, nlabels):
            jdx = np.equal(labels, unique_labels[j])
            dgroup = (D[idx,:])[:,jdx]
            Dout[i,j] = np.mean(dgroup)
            Dout[j,i] = np.mean(dgroup)

    if normalize:
        Dout /= np.max(Dout)

    return Dout
