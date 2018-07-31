from sklearn.linear_model import LinearRegression
from Modules.J_ai.J_regression import initialize_feature as init_feature
from Modules.J_ai.J_regression import batch_gradient_descent as bgd
import scipy.optimize as opt
import Modules.data.ml_data as datahlpr
import numpy as np
import matplotlib.pyplot as plt

X, Y = datahlpr.load_food_truck_data()

init_X = init_feature(X, 0)

print(X, init_X)

param, cost_h, p_array = bgd(init_X, Y)
print(param)

xx = np.arange(np.min(X), np.max(X))

yy = param[0] + param[1] * xx

plt.scatter(X, Y, c='r')
plt.plot(xx, yy, label="Linear Regression bgd")

# =================sci learn======================

regr = LinearRegression()
regr.fit(X, Y.ravel())
print(regr.coef_,  regr.intercept_)


plt.plot(xx, regr.intercept_ + regr.coef_ * xx, label='Linear regression (Scikit-learn GLM)')

# =================sci py================================

plt.legend(loc=4)
plt.show()
