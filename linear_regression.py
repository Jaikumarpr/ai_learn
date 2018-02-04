# !/usr/bin/env python3

import numpy as np
import numpy.linalg as m_inv
# import data.data_helper as dh
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# calulates sigmoid function
def sigmoid(z):
    """
    sigmoid function

    """
    return 1 / (1 + np.exp(-z))


def d_sigmoid(y):
    """
    derivative of sigmoid

    """
    return y * (1 - y)


# mean normalize an array
def mean_normalize(feature, method=None):
    """
    Outputs mean normalized array

    method : standard deviation, range of values (Default)

    """

    if method is std:  # standard deviation
        return (feature - np.mean(feature, axis=0)) / np.std(feature, axis=0)

    return (feature - np.mean(feature, axis=0)) / np.ptp(feature, axis=0)


# initialize a feature with default value
def initialize_feature(features, col, val=1):
    """
    Initialize a feature with a default value vector

    features: exisiting feature matrix
    val : initial value, Default is 1
    col : column to insert, index start at 1

    """
    return np.insert(features, col, val, axis=1)


# error vector
def delta(features, params, train_out, regression_type=None):
    """
    Calculate's the error vector

    features: feature set
    params: params vector
    train_out: output vector
    regression_type: logistic or linear(Default)

    """

    if regression_type is logistic:
        return sigmoid(features.dot(params)) - train_out

    #     matrix multiplication
    return features.dot(params) - train_out


# squared error function
def squared_delta(features, params, train_out):
    """
    Calculates the element-wise square of error vector

    features: feature set
    params: params vector
    train_out: output vector

    """
    return np.square(delta(features, params, train_out))


# cost function for linear regression
def loss(features, params, train_out, train_size=None):
    """
    Calculate the cost of plot_regression

    features: feature set
    params: params vector
    train_out: output vector
    train_size: avaialable train size, Default is feature shape

    """
    if train_size is None:
        train_size = features.shape[0]

    return 0.5 * (1 / train_size) * np.sum(squared_delta(features, params,
                                                         train_out))


# gradient of linear regression
def gradient(features, params, train_out, train_size=None, regression_type=None):
    """
    Calculates the gradient of cost function

    features: feature set
    params: params vector
    train_out: output vector
    reg_type: logistic or linear

    """
    if regression_type is logistic:
        return features.transpose().dot(delta(features, params, train_out,
                                              regression_type=logistic))

    if train_size is None:
        train_size = features.shape[0]

    return (1 / train_size) * features.transpose().dot(delta(features,
                                                            params, train_out))


# batch gradienct descent for linear regression
def batch_gradient_descent(features, train_out, params=None, train_size=None,
                           tolerance=None, learn_rate=0.06):

    if tolerance is None:
        tolerance = 0.0001

    if train_size is None:
        train_size = features.shape[0]

    if params is None:
        params = np.zeros(features.shape[1])

    loss_array = []
    params_array = []

    for i in np.arange(1500):

        # calculate the gradient vector
        grd = gradient(features, params, train_out)

        # calculate the new params
        params = params - (learn_rate * grd)

        # calculate and append to cost function for the params to a list
        loss_array.append(loss(features, params, train_out,
                               train_size))

        # append the params to a list
        params_array.append(params.tolist())

    return params, loss_array, params_array


# normal equation to calculate the params
def normal_equation(features, train_out):
    """
    Calculate the params using normal equations

     pinv(x' * x) * x' * y

    features: feature set
    train_out: output vector

    """
    x_prime = np.transpose(features)

    return m_inv(x_prime.dot(features)).dot(x_prime).dot(train_out)





# def stochastic_grad_descent():
#     pass
#
#
# def plot_regression():
#     pass


# def linear_regression_plot(data, params, figurehdl, plottype='univar'):
#     if plottype == 'multivar':
#         pass
#
#     minX = min(data[:, 1])
#     maxX = max(data[:, 1])
#
#     line_x = np.linspace(minX, maxX, 10)[np.newaxis]
#     x = np.insert(line_x, 0, 1, axis=0).transpose()
#     line_y = np.dot(x, params)
#     figurehdl.scatter(data[:, 1], data[:, 2], c='r')
#     figurehdl.plot(line_x[0, :], line_y, 'b')
#
#
# def plot_contour(features, train_out, fighdl):
#     t_0 = np.linspace(-1, 1, 200)
#     t_1 = np.linspace(-2, 2, 200)
#     Z = np.zeros((200, 200))
#
#     print(t_0.size)
#
#     for i in range(t_0.size):
#         for j in range(t_1.size):
#             Z[i, j] = linear_cost(features, [t_0[i], t_1[j]], train_out)
#
#     fighdl.contour(t_0, t_1, Z)


# def plot_surf(features, train_out, fighdl):
#     t_0 = np.linspace(-10, 10, 100)
#     t_1 = np.linspace(-10, 10, 100)
#     Z = np.zeros((100, 100))
#
#     print(t_0.size)
#
#     for i in range(t_0.size):
#         for j in range(t_1.size):
#             Z[i, j] = linear_cost(features, [t_0[i], t_1[j]], train_out)
#
#     fighdl.plot_surface(t_0, t_1, Z)

#
# if __name__ == '__main__':
#     ip = ['temp']
#     op = ['casual']
#     file_path = '/Users/jaikumar/Projects/MLearn/data/bike_sharing/day.csv'
#     x, y = dh.import_data(file_path, ip, op)
#
#     train_out = y.flatten(0) / np.max(y.flatten(0))
#     # initialise the feature set
#     X = initialize_feature(x)
#
#     # run batch gradient batch_grad_descent
#     theta, cost_func, theta_array = batch_grad_descent(X, train_out)
#
#     print(theta)
#     print(theta_array)
#     #
#     cost_func = cost_func / np.max(cost_func)
#
#     data = np.insert(X, 2, train_out, axis=1)
#
#     fig1 = plt.figure()
#     fig2 = plt.figure()
#     fig3 = plt.figure()
#     fig4 = plt.figure()
#
#     ax1 = fig1.gca()
#     ax2 = fig2.gca()
#     ax3 = fig3.add_subplot(111, projection='3d')
#     ax4 = fig4.gca()
#
#     gg = np.linspace(0, 1500, num=1500)
#
#     linear_regression_plot(data, theta, ax1, plottype='univar')
#     plot_surf(X, train_out, ax3)
#     plot_contour(X, train_out, ax4)
#
#     ax2.plot(gg, cost_func, 'b')
#
#     print(theta_array[:][0])
#     #ax4.scatter(theta_array[:, 0], theta_array[:, 1], c='r')
#
#     plt.show()
