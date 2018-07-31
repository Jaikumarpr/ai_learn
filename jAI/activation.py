
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class tanh(object):
    """Tanh activation function"""

    def activate(self, z):
        return np.tanh(z)

    def activate_prime(self, z):
        return 1 - np.tanh(z) ** 2


class RELU(object):
    """RELU activation function"""

    def activate(self, z):
        return np.maximum(z, 0, z)

    def activate_prime(self, z):
        return np.where(z >= 0, 1, 0)


class LRELU(object):
    """RELU activation function"""

    def activate(self, z):
        return np.maximum(z, 0.01 * z, z)

    def activate_prime(self, z):
        return np.where(z >= 0, 1, 0.01)


class sigmoid(object):
    """sigmoid activation function"""

    def __init__(self):
        super(sigmoid, self).__init__()

    def activate(self, z):
        return 1 / (1 + np.exp(-z))

    def activate_prime(self, z):
        return z * (1 - z)
