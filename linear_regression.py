# usr/bin/env python3

import numpy as np


# add default feature for first parameter
def initialize_feature(features):
    return np.insert(features, 0, 1, axis=1)


# error vector
def error(features, params, train_out):
    return np.dot(features, params) - train_out


# squared error function
def squared_error(features, params, train_out):
    return np.power(error(features, params, train_out), 2)


# gradient of linear regression
def gradient(features, params, train_out, train_size=None):
    if train_size is None:
        train_size = features.shape[0]

    return error(features, params, train_out) / train_size


# cost function for linear regression
def linear_cost(features, params, train_out, train_size=None):
    if train_size is None:
        train_size = features.shape[0]

    return 0.5 * np.sum(squared_error(features, params, train_out)) / train_size


def batch_grad_descent():
    pass


def stochastic_grad_descent():
    pass


def normal_equation():
    pass


def plot_regression():
    pass
