from Modules.J_ai.J_regression import initialize_feature as init_feature
from Modules.J_ai.J_regression import batch_gradient_descent as bgd
from Modules.J_ai.J_regression import cost as linearcost
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
import Modules.helpers.datahelper as dh
import numpy as np
import matplotlib.pyplot as plt

X, Y = dh.load_food_truck_data()

init_X = init_feature(X, 0)

# print(X, init_X)

param, cost_h, p_array = bgd(init_X, Y, epochs=3000)

print(param)
print(min(p_array[0, :]))
print(max(p_array[0, :]))

print(min(p_array[1, :]))
print(max(p_array[1, :]))
# plt.figure(1)
# plt.plot(p_array[0, :], cost_h)
#
# plt.figure(2)
# plt.plot(p_array[1, :], cost_h)

xx = np.arange(np.min(X), np.max(X))

yy = param[0] + param[1] * xx

# plt.figure(3)
# plt.scatter(X, Y, c='r')
# plt.plot(xx, yy, label="Linear Regression bgd")

fig = plt.figure(figsize=(10, 6))
ms = np.linspace(param[0] - 50, param[0] + 50, 50)
bs = np.linspace(param[1] - 40, param[1] + 40, 40)

ax = fig.add_subplot(111, projection='3d')


M, B = np.meshgrid(ms, bs)

z = np.array([linearcost(init_X, np.array(params).reshape(-1, 1), Y)
              for params in zip(np.ravel(M), np.ravel(B))])
cost = z.reshape(M.shape)

ax.plot_surface(M, B, cost, rstride=1, cstride=1, color='b', alpha=0.2)
ax.contour(M, B, cost, 20, color='b', alpha=0.8, offset=0, stride=30)


ax.set_xlabel('Intercept')
ax.set_ylabel('Slope')
ax.set_zlabel('Cost')
ax.view_init(elev=30., azim=30)

ax.plot(param[0], param[1], cost_h[-1], markerfacecolor='r',
        markeredgecolor='r', marker='o', markersize=7)


ax.plot([p for p in p_array[0, :]], [p for p in p_array[1, :]], np.ravel(cost_h),
        markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2)
#
# ax.plot([p for p in p_array[0, :]], [p for p in p_array[1, :]], 0,
#         markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2)


plt.show()
