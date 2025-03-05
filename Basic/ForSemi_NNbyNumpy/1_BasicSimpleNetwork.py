#The simplest network
import numpy as np

a1 = 0.9
a2 = 0.5
a3 = 0.7

w1 = 0.8
w2 = -0.4
w3 = 0

b1 = 1

sum = a1 * w1 + a2 * w2 + a3 * w3 + b1

#Activation function When x<= 0，y = 0; When x > 0， y = X ReLu
def activation_ReLu(inputs):
  return np.maximum(0, inputs)

print(activation_ReLu(sum))