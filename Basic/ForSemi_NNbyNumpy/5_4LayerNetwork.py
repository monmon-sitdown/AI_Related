# 2-3-4-2 Neurual Network
import numpy as np

def create_weights(n_inputs, n_neurons):
  return np.random.randn(n_inputs, n_neurons)

def create_bias(n_neurons):
  return np.random.randn(n_neurons)

def activation_ReLu(inputs):
  return np.maximum(0, inputs)

#--------------------------------------------

a11 = 0.9
a21 = 0.5
a12 = -0.8
a22 = -0.5
a13 = -0.6
a23 = -0.7
a14 = -0.3
a24 = -0.9
a15 = 0.7
a25 = 0.4

inputs = np.array([[a11, a21],
                   [a12, a22],
                   [a13, a23],
                   [a14, a24],
                   [a15, a25]])

weights1 = create_weights(2, 3)
bias1 = create_bias(3)

weights2 = create_weights(3, 4)
bias2 = create_bias(4)

weights3 = create_weights(4, 2)
bias3 = create_bias(2)

#Layer1
sum1 = np.dot(inputs, weights1) + bias1
output1 = activation_ReLu(sum1)

#Layer2
sum2 = np.dot(output1, weights2) + bias2
output2 = activation_ReLu(sum2)

#Layer3
sum3 = np.dot(output2, weights3) + bias3
output3 = activation_ReLu(sum3)
print(output3)
