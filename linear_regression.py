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


# check for tolerance
def under_tolerance(temp, val, tol):
    return np.sum(np.abs(temp - val)) <= tol


# batch gradienct descent for linear regression
def batch_grad_descent(features, params, train_out, train_size,
                       tolerance=None):
    if tolerance is None:
        tolerance = 0.0001

    learn_rate = 0.6
    cost_function_array = []
    params_array = np.array([])
    temp_params = np.zeros(features.shape[1])

    while True:

        for i in range(params.shape[0]):
            grd = gradient(features, params, train_out, train_size)
            feature_i = features[:, i]
            temp_params[i] = params[i] - (learn_rate * np.sum(grd * feature_i))

        params = np.copy(temp_params)

        cost_function_array.append(linear_cost(features, params, train_out,
                                               train_size))
        params_array.append(params)

        if under_tolerance(temp_params, params, tolerance):
            break

    return params, cost_function_array, params_array


def stochastic_grad_descent():
    pass


def normal_equation():
    pass


def plot_regression():
    pass
