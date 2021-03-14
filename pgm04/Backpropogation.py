# Build an Artificial Neural Network by implementing the Backpropagation algorithm and test the same using appropriate data sets.

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return x * (1 - x)

X = np.array(([2, 5], [1, 4], [2, 6]), dtype = float)
Y = np.array(([92], [86], [88]), dtype = float)
X = X / np.amax(X, axis = 0)
Y = Y / 100

epoch = 10000
lr = 0.1
ni = 2
nh = 3
no = 1

wh = np.random.uniform(size = (ni, nh))
bh = np.random.uniform(size = (1, nh))
wout = np.random.uniform(size = (nh, no))
bout = np.random.uniform(size = (1, no))

for i in range(epoch):

    hinp1 = np.dot(X, wh)
    hinp = hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)

    EO = Y - output
    outgrad = derivative_sigmoid(output)
    d_output = EO * outgrad

    EH = d_output.dot(wout.T)
    hgrad = derivative_sigmoid(hlayer_act)
    d_hidden = EH * hgrad

    wh += X.T.dot(d_hidden) * lr
    bh += np.sum(d_hidden, axis = 0, keepdims = True) * lr
    wout += hlayer_act.T.dot(d_output) * lr
    bout += np.sum(d_output, axis = 0, keepdims = True) * lr

print("Input:")
print(str(X))
print()
print("Expected output:")
print(str(Y))
print()
print("Predicted output")
print(output)
print()
