import numpy as np
#normalization function
def normalize(array):
  max_number = np.max(np.absolute(array), axis = 1, keepdims=True)
  scale_rate = np.where(max_number == 0, 0, 1/max_number)
  norm = array * scale_rate
  return norm

#Activation function softmax
#ReLu is not for the last layer, which need to show a probability that total sum equlas to 1
def activation_softmax(inputs):
  max_values = np.max(inputs, axis = 1, keepdims = True)     #Choose the max of every row, and keep the dimension as the same as the original one, which means a row vector will not become a column vector
  exp_values = np.exp(inputs - max_values)                    #Slide every value to negative  zone. The properties of the exponential function. if delta = x2 - x1 does not change, then e^(x2) / e^(x1) also not，Slide to negative  zone in case of exponential explosion
  sum_values = np.sum(exp_values, axis = 1, keepdims = True)  #Now, every value is in [0, 1], after normalization, it can be used as probability. 
  return exp_values / sum_values


def create_weights(n_inputs, n_neurons):
  return np.random.randn(n_inputs, n_neurons)

def create_bias(n_neurons):
  return np.random.randn(n_neurons)

def activation_ReLu(inputs):
  return np.maximum(0, inputs)


#------------------------------------------------------
class Layer:
  def __init__(self, n_inputs, n_neurons):
    self.weights = np.random.randn(n_inputs, n_neurons)
    self.bias = np.random.randn(n_neurons)

  def layer_forward(self, inputs):
    self.output  = np.dot(inputs, self.weights) + self.bias
    #self.output = activation_ReLu(sum)
    return self.output

#------------------------------------------------------
NETWORK_SHAPE = [2, 3, 4, 2]
class Network:
  def __init__(self, network_shape):
    self.shape = network_shape
    self.layers = []
    for i in range(len(network_shape) - 1):
        layer = Layer(network_shape[i], network_shape[i + 1])
        self.layers.append(layer)

  def network_forward(self, inputs):
    self.output = [inputs]
    for i in range(len(self.layers)):
      if i < len(self.layers) - 1:
        layer_sum = self.layers[i].layer_forward(self.output[i])
        self.output.append(normalize(activation_ReLu(layer_sum)))   #Normalization
      else:
        layer_sum = self.layers[i].layer_forward(self.output[i])
        self.output.append(activation_softmax(layer_sum))
    return self.output

#------------------------------------------------------

def main():
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