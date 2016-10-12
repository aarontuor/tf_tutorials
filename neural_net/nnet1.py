# tensorflow nnetwork tutorial
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Retrieve data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Make a graph and set as default


# First make placeholders for inputs and targets


# Now make trainable variables


# Now compute q


# Loss function: cross entropy


# Train step: gradient descent optimizer

# Evaluate accuracy

# initialize session and variables


# training loop
