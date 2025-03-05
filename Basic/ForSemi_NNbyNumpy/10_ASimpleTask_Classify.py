#A Simple Task:2. Setting the correct answer
import random
import matplotlib.pyplot as plt
import numpy as np
from tools import normalize, activation_softmax, create_weights, create_bias, activation_ReLu

#------------------------------------------------------
#Classify Function
def classify(probability):
  classification = np.rint(probability[:, 1]) #Because v[col0] + v[col1] = 1 only take 1 colomn rint四舍五入
  return classification

#------------------------------------------------------
#Data Visualization
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

#------------------------------------------------------
#Generate Data
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

#--------------------------------------------
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

if __name__ == "__main__":
  data = create_data(NUM_DATA)
  #plot_data(data, "data")
  copydata = data.copy()

  inputs = data[:, (0, 1)]    #Only the first two colomns are data
  answer = data[:, 2]

  network = Network(NETWORK_SHAPE)
  outputs = network.network_forward(inputs)
  classification = classify(outputs[-1])      #Classify for the last colomn
  copydata[:, 2] = classification
  plot_data(data, "answer")
  plot_data(copydata, "classified")
