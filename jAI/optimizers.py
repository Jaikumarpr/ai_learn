from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .preprocessing import generate_params


class bgd(object):
    """docstring for bgd."""

    def __init__(self):
        super(bgd, self).__init__()


# batch gradienct descent for linear regression
    def run(self, x, y, loss, epochs, learn_rate, theta):

        if epochs is None:
            epochs = 1000

        if learn_rate is None:
            learn_rate = 0.01

        if theta is None:
            theta = generate_params(x.shape[1])

        train_size = x.shape[0]
        cost_history = np.zeros(epochs)

        theta_array = np.array([[0], [0]])

        for i in np.arange(epochs):

            # calculate cost gradient
            d_theta = loss.gradient(x, y, theta, train_size)

            # calculate new theta
            theta = theta - (learn_rate * d_theta)

            # calculate and append to cost function for the theta to a list
            cost_history[i] = loss.cost(x, y, theta, train_size)

            # append the theta to a list
            theta_array = np.concatenate((theta_array, theta), axis=1)

        return {'theta': theta, 'cost': cost_history, 'thetalog':
                theta_array[:, 1:]}


def get_optimizer(id):

    opt = None
    if id is 'bgd':
        opt = bgd()

    return opt
