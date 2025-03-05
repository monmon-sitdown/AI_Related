#A Simple Task:3. BackwardFunction - Adjusting matrix
import random
import matplotlib.pyplot as plt
import copy
import numpy as np
from tools import activation_ReLu

#------------------------------------------------------
def get_final_layer_preAct_demands(predicted_values, target_vector):
  target = np.zeros((len(target_vector), 2))
  target[:, 1] = target_vector
  target[:, 0] = 1 - target_vector

  for i in range(len(target_vector)):
    if np.dot(target[i], predicted_values[i]) > 0.5: 
      target[i] = np.array([0, 0])
    else:
      target[i] = (target[i] - 0.5) * 2 
  return target

#------------------------------------------------------
def precise_loss_function(predicted, real):
  real_matrix = np.zeros((len(real), 2))
  real_matrix[:, 1] = real
  real_matrix[:, 0] = 1 - real
  loss = np.sum(predicted * real_matrix, axis = 1) 
  return 1 - loss

#------------------------------------------------------
def classify(probability):
  classification = np.rint(probability[:, 1]) 
  return classification

#------------------------------------------------------

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

#------------------------------------------------------

def normalize(array):
  max_number = np.max(np.absolute(array), axis = 1, keepdims=True)
  scale_rate = np.where(max_number == 0, 0.0, 1.0 /max_number)
  norm = array * scale_rate
  return norm


def activation_softmax(inputs):
  max_values = np.max(inputs, axis = 1, keepdims = True)     
  exp_values = np.exp(inputs - max_values)                   
  sum_values = np.sum(exp_values, axis = 1, keepdims = True) 
  return exp_values / sum_values

#------------------------------------------------------

class Layer:
  def __init__(self, n_inputs, n_neurons):
    self.weights = np.random.randn(n_inputs, n_neurons)
    self.bias = np.random.randn(n_neurons)

  def layer_forward(self, inputs):
    self.output  = np.dot(inputs, self.weights) + self.bias
    return self.output

  def get_weight_adjust_matrix(self, preWeights_values, aftWeights_demands):
    return np.dot(preWeights_values.T, aftWeights_demands)

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
NUM_DATA = 5
NETWORK_SHAPE = [2, 3, 4, 2]

if __name__ == "__main__":
  data = create_data(NUM_DATA)
  copydata = data.copy()

  inputs = data[:, (0, 1)]    
  target = copy.deepcopy(data[:, 2])

  network = Network(NETWORK_SHAPE)
  outputs = network.network_forward(inputs)
  classification = classify(outputs[-1])      
  copydata[:, 2] = classification

  predicted = outputs[-1]
  loss = precise_loss_function(predicted, target)
  print(loss)

  demands = get_final_layer_preAct_demands(predicted, target)

  print(target)
  print(demands)

  adjust_matrix = network.layers[-1].get_weight_adjust_matrix(outputs[-2], demands)
  print(adjust_matrix)
