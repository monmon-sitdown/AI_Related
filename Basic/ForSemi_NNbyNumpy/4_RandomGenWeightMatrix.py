#Generate a weight and a bias matrix randomly
import numpy as np

def create_weights(n_inputs, n_neurons):
  return np.random.randn(n_inputs, n_neurons)

def create_bias(n_neurons):
  return np.random.randn(n_neurons)

def activation_ReLu(inputs):
  return np.maximum(0, inputs)

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

weights = create_weights(3, 2)

#Bias matrix
b1 = np.array([0.5, 0.6])             #Caution: This is not add, but boardcast
sum = np.dot(inputs, weights) + b1

print(sum, activation_ReLu(sum))