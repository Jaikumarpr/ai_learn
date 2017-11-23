# usr/bin/env python3

import numpy as np
import data.data_helper as dh
import matplotlib.pyplot as plt


# add default feature for first parameter
def initialize_feature(features):
    return np.insert(features, 0, 1, axis=1)


# error vector
def error(features, params, train_out):
    return features.dot(params) - train_out


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
def batch_grad_descent(features, train_out, params=None, train_size=None,
                       tolerance=None):
    if tolerance is None:
        tolerance = 0.0001

    if train_size is None:
        train_size = features.shape[0]

    if params is None:
        params = np.zeros(features.shape[1])

    learn_rate = 0.06
    cost_function_array = []
    params_array = []
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


def linear_regression_plot(data, params, figurehdl, plottype='univar'):
    if plottype == 'multivar':
        pass

    minX = min(data[:, 1])
    maxX = max(data[:, 1])

    line_x = np.linspace(minX, maxX, 10)[np.newaxis]
    x = np.insert(line_x, 0, 1, axis=0).transpose()
    line_y = np.dot(x, params)
    figurehdl.scatter(data[:, 1], data[:, 2], c='r')
    figurehdl.plot(line_x[0, :], line_y, 'b')


if __name__ == '__main__':
    ip = ['temp']
    op = ['casual']
    file_path = '/Users/jaikumar/Projects/MLearn/data/bike_sharing/day.csv'
    x, y = dh.import_data(file_path, ip, op)

    # initialise the feature set
    X = initialize_feature(x)

    # run batch gradient batch_grad_descent
    theta, cost_func, theta_array = batch_grad_descent(X, y)

    print(theta)

    print(cost_func)
