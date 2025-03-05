#class Layer
import numpy as np

#-----------------------------

def create_weights(n_inputs, n_neurons):
  return np.random.randn(n_inputs, n_neurons)

def create_bias(n_neurons):
  return np.random.randn(n_neurons)

def activation_ReLu(inputs):
  return np.maximum(0, inputs)

#-----------------------------

class Layer:
  def __init__(self, n_inputs, n_neurons):
    self.weights = np.random.randn(n_inputs, n_neurons)
    self.bias = np.random.randn(n_neurons)

  def layer_forward(self, inputs):
    sum = np.dot(inputs, self.weights) + self.bias
    self.output = activation_ReLu(sum)
    return self.output

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


layer1 = Layer(2, 3)
layer2 = Layer(3, 4)
layer3 = Layer(4, 2)

output1 = layer1.layer_forward(inputs)
output2 = layer2.layer_forward(output1)
output3 = layer3.layer_forward(output2)
print(output3)

