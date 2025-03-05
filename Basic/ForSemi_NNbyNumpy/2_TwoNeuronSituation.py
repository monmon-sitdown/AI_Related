#Two neuron : c = aW + b c' = aW' + b'
#[a1 a2 a3] .* [w1 w2 w3
#               w1' w2' w3']^T

import numpy as np

a1 = 0.9
a2 = 0.5
a3 = 0.7
inputs = np.array([a1, a2, a3])

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