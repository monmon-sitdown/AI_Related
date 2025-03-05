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

