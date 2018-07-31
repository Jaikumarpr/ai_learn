from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class MSE(object):
    """docstring for MSE."""

    def __init__(self):
        super(MSE, self).__init__()

    def cost(self, x, y, theta, train_size=None):

        if train_size is None:
            train_size = x.shape[0]

        return 0.5 * (1 / train_size) * np.sum(np.square(x.dot(theta) - y))

    def gradient(self, x, y, theta, train_size=None):

        if train_size is None:
            train_size = x.shape[0]

        return (1 / train_size) * x.transpose().dot(x.dot(theta) - y)


def get_loss(loss):

    func = None

    if loss is 'MSE':
        func = MSE()

    return func
