from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


# mean normalize an array
def mean_normalize(feature, method=None):
    """
    Outputs mean normalized array

    method : standard deviation, range of values (Default)

    """

    if method is "std":  # standard deviation
        return (feature - np.mean(feature, axis=0)) / np.std(feature, axis=0)

    return (feature - np.mean(feature, axis=0)) / np.ptp(feature, axis=0)


# initialize a feature with default value
def initialize_feature(features, col, val=1):
    """
    Initialize a feature with a default value vector

    features: exisiting feature matrix
    val : initial value, Default is 1
    col : column to insert, index start at 0

    """
    return np.insert(features, col, val, axis=1)


# generate initial params_array
def generate_params(count=2):
    """
    generate initial parameters for features

    count: no of parameters

    """
    return np.zeros(count).reshape(-1, 1)
