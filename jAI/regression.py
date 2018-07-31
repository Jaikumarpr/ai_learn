# !/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .preprocessing import generate_theta, initialize_feature
import numpy.linalg as m_inv
# import data.data_helper as dh
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def normal_equation(x, y):
    """
    Calculate the theta using normal equations

     pinv(x' * x) * x' * y

    x: feature set
    y: output vector

    """
    x_prime = np.transpose(x)

    return m_inv(x_prime.dot(x)).dot(x_prime).dot(y)
