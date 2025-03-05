import numpy as np



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

#------------------------------------------------------
#Class network

class Network:
  def __init__(self, network_shape):
    self.shape = network_shape
    self.layers = []
    for i in range(len(network_shape) - 1):
        layer = Layer(network_shape[i], network_shape[i + 1])
        self.layers.append(layer)

  def network_forward(self, inputs):
    self.output = inputs
    for layer in self.layers:
      self.output = layer.layer_forward(self.output)
    return self.output

#------------------------------------------------------

def main():
  NETWORK_SHAPE = [2, 3, 4, 2]
  
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

  network = Network(NETWORK_SHAPE)
  print(network.shape, network.network_forward(inputs))

main()