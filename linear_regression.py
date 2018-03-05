#!/usr/bin/env python3

import numpy as np
import Modules.ml_data as datahlpr
from Modules.regression import *
from Modules.ml_plot import plot_scatter
import matplotlib.pyplot as plt 


# load data
X, Y = datahlpr.load_food_truck_data()

# plot dataset

# initialize feature
init_X = initialize_feature(X, 0)

# run linear regression_type
param, loss_h, p_array = batch_gradient_descent(init_X, Y)

print(param)

# plot iteration history
# plt.plot(loss_h)

xx = np.arange(np.min(X), np.max(X))

yy = param[0] + param[1] * xx

plt.scatter(X, Y, c='r')
plt.plot(xx, yy)
plt.show()