#A Simple Task：1.Generate Data and Visualization
import random
import matplotlib.pyplot as plt
import numpy as np
#------------------------------------------------------
def normalize(array):
  max_number = np.max(np.absolute(array), axis = 1, keepdims=True)
  scale_rate = np.where(max_number == 0, 0.0, 1.0/max_number)
  norm = array * scale_rate
  return norm

def activation_softmax(inputs):
  max_values = np.max(inputs, axis = 1, keepdims = True)     
  exp_values = np.exp(inputs - max_values)                    
  sum_values = np.sum(exp_values, axis = 1, keepdims = True)  
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
        self.output.append(normalize(activation_ReLu(layer_sum)))  
      else:
        layer_sum = self.layers[i].layer_forward(self.output[i])
        self.output.append(activation_softmax(layer_sum))
    return self.output

#------------------------------------------------------

NUM_DATA = 100
NETWORK_SHAPE = [2, 3, 4, 2]

def create_data(num_of_data):
  entry_list = []
  for i in range(num_of_data):
    x = random.uniform(-2, 2)
    y = random.uniform(-2, 2)
    tag = tag_entry(x, y)
    entry_list.append([x, y, tag])
  return np.array(entry_list)

def tag_entry(x, y):
  if x * x + y * y < 1:
    return 0
  else:
    return 1

def plot_data(data, title):
  color = []
  for i in data[:, 2]:
    if i == 0 :
      color.append("orange")
    else:
      color.append("blue")
  plt.scatter(data[:, 0], data[:, 1], c = color)
  plt.title(title)
  plt.show()

if __name__ == "__main__":
  data = create_data(NUM_DATA)
  plot_data(data, "data")
