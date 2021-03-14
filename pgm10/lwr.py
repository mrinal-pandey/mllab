# Implement the non-parametric Locally Weighted Regressionalgorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.mat(np.eye(m))
    for i in range(m):
        diff = point - X[i]
        weights[i, i] = np.exp((diff * diff.T) / (-2 * k * k))
    return weights

def localWeight(point, xmat, ymat, k):
    weight = kernel(point, xmat, k)
    return (X.T * weight * X).I * (X.T * weight * ymat.T)

def locallyWeightedRegression(xmat, ymat, k):
    m, n = np.shape(xmat)
    y_pred = np.zeros(m)
    for i in range(m):
        y_pred[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k)
    return y_pred

data = pd.read_csv('tips.csv')
tip = np.array(data.tip)
bill = np.array(data.total_bill)
mtip = np.mat(tip)
mbill = np.mat(bill)

m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T, mbill.T))

y_pred = locallyWeightedRegression(X, mtip, 2)
sortIndex = X[:,1].argsort(0)
xsort = X[sortIndex][:,0]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(bill, tip, color = 'blue')
ax.plot(xsort[:,1], y_pred[sortIndex], color = 'red', linewidth = 1)
plt.xlabel('Bill')
plt.ylabel('Tip')
plt.show()
