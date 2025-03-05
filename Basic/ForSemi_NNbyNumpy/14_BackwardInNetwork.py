#A Simple Task:3. BackwardFunction - backward in whole network
import random
import matplotlib.pyplot as plt
import copy
import numpy as np
from tools import activation_ReLu


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

def vector_normalize(array):
    max_number = np.max(np.absolute(array))
    scale_rate = np.where(max_number == 0, 1, 1/max_number)
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

  def layer_backward(self, preWeights_values, aftWeights_demands,):
    preWeights_demands = np.dot(aftWeights_demands, self.weights.T) 

    condition = (preWeights_values > 0)
    value_derivatives = np.where(condition, 1, 0)            

    preActs_demands = value_derivatives * preWeights_demands  
    norm_preActs_demands = normalize(preActs_demands)

    weight_adjust_matrix = self.get_weight_adjust_matrix(preWeights_values, aftWeights_demands)
    norm_weight_adjust_matrix = normalize(weight_adjust_matrix)

    return norm_preActs_demands, norm_weight_adjust_matrix

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

  def network_backword(self, layer_outputs, target_vector): #outputs are output values of all layers，target_vector is the correct answer
    preActs_demands = get_final_layer_preAct_demands(layer_outputs[-1], target_vector)  #calculate the demands function of the last layer
    for i in range(len(self.layers)):
      layer = backup_network.layers[-1 - i]                                 #from back to the front
      if i != 0:
        layer.bias += LEARNING_RATE * np.mean(preActs_demands, axis = 0)    #The correction in the final layer can easily lead to overfitting. By taking the average vertically, the bias is not adjusted when i == 0.
        layer.bias = vector_normalize(layer.bias)

      outputs = layer_outputs[-2 - i]
      preActs_demands, weight_adjust_matrix = layer.layer_backward(outputs, preActs_demands)

      layer.weights += LEARNING_RATE * weight_adjust_matrix
      layer.weights = normalize(layer.weights)
    return backup_network
#------------------------------------------------------
NUM_DATA = 100
NETWORK_SHAPE = [2, 3, 4, 2]
LEARNING_RATE = 0.015

if __name__ == "__main__":
  data = create_data(NUM_DATA)
  copydata = data.copy()

  inputs = data[:, (0, 1)]   
  target = copy.deepcopy(data[:, 2])

  network = Network(NETWORK_SHAPE)
  outputs = network.network_forward(inputs)
  classification = classify(outputs[-1])      
  copydata[:, 2] = classification
  plot_data(data, "answer")
  plot_data(copydata, "classified")

  backup_network = network.network_backword(outputs, target)
  outputs_1 = backup_network.network_forward(inputs)

  classification_1 = classify(outputs_1[-1])      #The result of backward once
  copydata[:, 2] = classification_1
  plot_data(copydata, "back once")
