import tensorflow as tf
import numpy as np

class SimpleModel():
    """
    A class for gradient descent training arbitrary models.

    :param loss: loss_tensor defined in graph
    :param eval_tensor: For evaluating on dev set
    :param ph_dict: A dictionary of tensorflow placeholders
    :param learnrate: step_size for gradient descent
    :param debug: Whether to print debugging info
    """

    def __init__(self, loss, eval_tensor, ph_dict, learnrate=0.01, debug=False):


    def train(self, train_data, dev_data, mb=1000, num_epochs=1):
        """
        :param train_data: A DataSet object of train data.
        :param dev_data: A DataSet object of dev data.
        :param mb: The mini-batch size.
        :param num_epochs: How many epochs to train for.
        """


    def evaluate(self, data):
        """
        Evaluation function

        :param data: The data to evaluate on.
        :return: The return value of the evaluation function in numpy form
        """

    def get_feed_dict(self, batch, ph_dict):

        """
        :param batch: A dataset object.
        :param ph_dict: A dictionary where the keys match keys in batch, and the values are placeholder tensors
        :return: A feed dictionary with keys of placeholder tensors and values of numpy matrices
        """
