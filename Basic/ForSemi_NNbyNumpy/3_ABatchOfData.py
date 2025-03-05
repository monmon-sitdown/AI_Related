#Input are not only one piece of data, but a batch of data
import numpy as np

a11 = 0.9
a21 = 0.5
a31 = 0.7
a12 = -0.8
a22 = -0.5
a32 = -0.6
a13 = 0.5
a23 = 0.8
a33 = 0.2
inputs = np.array([[a11, a21, a31],
                   [a12, a22, a32],
                   [a13, a23, a33]])

w11 = 0.8
w21 = -0.4
w31 = 0
w12 = 0.7
w22 = -0.6
w32 = 0.2
weights = np.array([[w11, w12],
                   [w21, w22],
                   [w31, w32]])

b1 = 0.1
sum = np.dot(inputs, weights) + b1

def activation_ReLu(inputs):
  return np.maximum(0, inputs)

print(sum, activation_ReLu(sum))