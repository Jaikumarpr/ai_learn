from jAI.preprocessing import initialize_feature as init_feature
from jAI.models import LinearRegression
import Modules.helpers.datahelper as dh
import numpy as np
import matplotlib.pyplot as plt

X, Y = dh.load_food_truck_data()

init_X = init_feature(X, 0)

model = LinearRegression(loss='MSE', optimizer='bgd', epochs=1000)

history = model.fit(init_X, Y)

theta = history['theta']
cost_h = history['cost']
p_array = history['thetalog']

print(theta)

plt.figure(1)
plt.plot(p_array[0, :], cost_h)

plt.figure(2)
plt.plot(p_array[1, :], cost_h)

xx = np.arange(np.min(X), np.max(X))

yy = theta[0] + theta[1] * xx

plt.figure(3)
plt.scatter(X, Y, c='r')
plt.plot(xx, yy, label="Linear Regression bgd")

fig = plt.figure(figsize=(10, 6))
ms = np.linspace(theta[0] - 50, theta[0] + 50, 50)
bs = np.linspace(theta[1] - 40, theta[1] + 40, 40)

ax = fig.add_subplot(111, projection='3d')


M, B = np.meshgrid(ms, bs)

z = np.array([model.loss.cost(init_X, Y, np.array(params).reshape(-1, 1))
              for params in zip(np.ravel(M), np.ravel(B))])
cost = z.reshape(M.shape)

ax.plot_surface(M, B, cost, rstride=1, cstride=1, color='b', alpha=0.2)
ax.contour(M, B, cost, 20, color='b', alpha=0.8, offset=0, stride=30)


ax.set_xlabel('Intercept')
ax.set_ylabel('Slope')
ax.set_zlabel('Cost')
ax.view_init(elev=30., azim=30)

ax.plot(theta[0], theta[1], cost_h[-1], markerfacecolor='r',
        markeredgecolor='r', marker='o', markersize=7)


ax.plot([p for p in p_array[0, :]], [p for p in p_array[1, :]], np.ravel(cost_h),
        markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2)

ax.plot([p for p in p_array[0, :]], [p for p in p_array[1, :]], 0,
        markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2)


plt.show()
