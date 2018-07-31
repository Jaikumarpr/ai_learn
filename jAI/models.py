from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .optimizers import get_optimizer
from .loss import get_loss


class LinearRegression(object):
    """docstring for LinearRegression."""

    def __init__(self, loss='MSE', optimizer='bgd', epochs=None,
                 learning_rate=None, initial_theta=None):

        super(LinearRegression, self).__init__()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = get_optimizer(optimizer)
        self.loss = get_loss(loss)
        self.initial_theta = initial_theta

    def fit(self, x, y):

        history = self.optimizer.run(x, y, self.loss, self.epochs,
                                     self.learning_rate, self.initial_theta)

        return history
