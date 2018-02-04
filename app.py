# usr/bin/env python3

import numpy as np
import Modules.ml_data as datahlpr
from Modules.regression import *
from Modules.ml_plot import *


# load data
X, Y = datahlpr.load_food_truck_data()

print(X)

# plot dataset
plot_scatter(X, Y, ['population', 'profit'], 'food_truck')

# initialize feature
init_X = initialize_feature(X, 0)

print(init_X)


# run linear regression_type

params, loss_array, params_array = batch_gradient_descent(init_X, Y)
print(params)
