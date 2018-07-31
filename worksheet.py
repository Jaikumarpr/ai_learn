from Modules.J_ai.J_regression import initialize_feature as init_feature
from Modules.J_ai.J_regression import batch_gradient_descent as bgd
import scipy.optimize as opt
import Modules.helpers.datahelper as dh
import numpy as np
import matplotlib.pyplot as plt

X, Y = dh.load_food_truck_data()

init_X = init_feature(X, 0)

# print(X, init_X)

param, cost_h, p_array = bgd(init_X, Y, epochs=3000)

plt.figure(1)
plt.plot(p_array[0, :], cost_h)

plt.figure(2)
plt.plot(p_array[1, :], cost_h)

xx = np.arange(np.min(X), np.max(X))

yy = param[0] + param[1] * xx

plt.figure(3)
plt.scatter(X, Y, c='r')
plt.plot(xx, yy, label="Linear Regression bgd")

plt.figure(4)
ms = np.linspace(param[0] - 20, param[0] + 20, 20)
bs = np.linspace(param[1] - 40, param[1] + 40, 40)

M, B = np.meshgrid(ms, bs)

print(M, B)
plt.show()
