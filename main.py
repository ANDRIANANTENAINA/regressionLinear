# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:38:22 2022

@author: PC58
"""

import numpy as np
import matplotlib.pyplot as plt

surface = np.array([28, 50, 55, 60, 48, 35, 86, 65, 32, 52])
price = np.array([130, 280, 268, 320, 250, 250, 350, 300, 155, 245])

surface = surface.reshape(surface.shape[0], 1)
price = price.reshape(price.shape[0], 1)

X = np.hstack((surface, np.ones(surface.shape)))

theta = np.random.randn(2, 1)


def model(X, theta):
    return X.dot(theta)  # produit matricielle X et theta


def cost_function(X, price, theta):
    m = len(price)
    return 1 / (2 * m) * np.sum((model(X, theta) - price) ** 2)


# print(cost_function(X, price, theta))

def grad(X, price, theta):
    m = len(price)
    return 1 / m * X.T.dot(model(X, theta) - price)


def gradient_descent(X, price, theta, learning_rate, n_itertions):
    cost_history = np.zeros(n_itertions)

    for i in range(0, n_itertions):
        theta = theta - learning_rate * grad(X, price, theta)
        cost_history[i] = cost_function(X, price, theta)

    return theta, cost_history


theta_final, ch = gradient_descent(X, price, theta, learning_rate=0.0001, n_itertions=1000)
print(theta_final)

prediction = model(X, theta_final)




#plt.plot(range(1000), ch)

def coef_determination(y, pred):
    u = ((y - pred) ** 2).sum()
    v = ((y * y.mean()) ** 2).sum()

    return 1 - u / v

print(coef_determination(price, prediction))
plt.scatter(surface, price, color="red")
plt.plot(surface, prediction, c="b")
plt.show()







