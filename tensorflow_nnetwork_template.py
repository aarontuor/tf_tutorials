
# coding: utf-8

# # Jupyter Notebook Install with Codefolding extension
# 
# ```bash
# $> pip install ipython
# $> pip install jupyter
# $> pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master
# $> jupyter contrib nbextension install --user 
# $> jupyter nbextension enable codefolding/main
# ```
# 

# In[ ]:

class DataSet(object):
    """
    Data structure for mini-batch gradient descent training involving non-sequential data.

    :param features: (dict) A dictionary of string label names to data matrices.
        Matrices may be of types :any:`IndexVector`, scipy sparse csr_matrix, or numpy array.
    :param labels: (dict) A dictionary of string label names to data matrices.
        Matrices may be of types :any:`IndexVector`, scipy sparse csr_matrix, or numpy array.
    :param mix: (boolean) Whether or not to shuffle per epoch.

    :examples:

        >>> import numpy as np
        >>> from antk.core.loader import DataSet
        >>> d = DataSet({'id': np.eye(5)}, labels={'ones':np.ones((5, 2))})
        >>> d #doctest: +NORMALIZE_WHITESPACE
        antk.core.DataSet object with fields:
        '_labels': {'ones': array([[ 1.,  1.],
                                   [ 1.,  1.],
                                   [ 1.,  1.],
                                   [ 1.,  1.],
                                   [ 1.,  1.]])}
        'mix_after_epoch': False
        '_num_examples': 5
        '_index_in_epoch': 0
        '_last_batch_size': 5
        '_features': {'id': array([[ 1.,  0.,  0.,  0.,  0.],
                                   [ 0.,  1.,  0.,  0.,  0.],
                                   [ 0.,  0.,  1.,  0.,  0.],
                                   [ 0.,  0.,  0.,  1.,  0.],
                                   [ 0.,  0.,  0.,  0.,  1.]])}

        >>> d.show() #doctest: +NORMALIZE_WHITESPACE
        features:
             id: (5, 5) <type 'numpy.ndarray'>
        labels:
             ones: (5, 2) <type 'numpy.ndarray'>

        >>> d.next_batch(3) #doctest: +NORMALIZE_WHITESPACE
        antk.core.DataSet object with fields:
            '_labels': {'ones': array([[ 1.,  1.],
                                       [ 1.,  1.],
                                       [ 1.,  1.]])}
            'mix_after_epoch': False
            '_num_examples': 3
            '_index_in_epoch': 0
            '_last_batch_size': 3
            '_features': {'id': array([[ 1.,  0.,  0.,  0.,  0.],
                                       [ 0.,  1.,  0.,  0.,  0.],
                                       [ 0.,  0.,  1.,  0.,  0.]])}


    """

    def __init__(self, features, labels=None, mix=False):
        self._features = features  # hashmap of feature matrices
        self._num_examples = features[features.keys()[0]].shape[0]
        if labels:
            self._labels = labels # hashmap of label matrices
        else:
            self._labels = {}
        self._index_in_epoch = 0
        self.mix_after_epoch = mix
        self._last_batch_size = self._num_examples
    
    def __repr__(self):
        attrs = vars(self)
        return 'antk.core.DataSet object with fields:\n' + '\n'.join("\t%r: %r" % item for item in attrs.items())

    # ======================================================================================
    # =============================PROPERTIES===============================================
    # ======================================================================================
    @property
    def features(self):
        """
        :attribute: (dict) A dictionary with string keys and feature matrix values.
        """
        return self._features

    @property
    def index_in_epoch(self):
        """
        :attribute: (int) The number of data points that have been trained on in a particular epoch.
        """
        return self._index_in_epoch

    @property
    def labels(self):
        """
        :attribute: (dict) A dictionary with string keys and label matrix values.
        """
        return self._labels

    @property
    def num_examples(self):
        """
        :attribute: (int) Number of rows (data points) of the matrices in this :any:`DataSet`.
        """
        return self._num_examples

    # ======================================================================================
    # =============================PUBLIC METHODS===========================================
    # ======================================================================================
    def reset_index_to_zero(self):
        """
        :method: Sets :any:`index_in_epoch` to 0.
        """
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        """
        :method:
        Return a sub DataSet of next batch-size examples.
            If no shuffling (mix=False):
                If `batch_size` is greater than the number of examples left in
                the epoch then a batch size DataSet wrapping past beginning
                (rows [index_in_epcoch:num_examples, 0::any:`num_examples`-:any:`index_in_epoch`]
                will be returned.
            If shuffling enabled (mix=True):
                If `batch_size` is greater than the number of examples left in the epoch,
                points will be shuffled and `batch_size` DataSet is returned starting from index 0.

        :param batch_size: (int) The number of rows in the matrices of the sub DataSet.
        :return: :any:`DataSet`
        """
        if batch_size != self._last_batch_size and self._index_in_epoch != 0:
            self.reset_index_to_zero()
        self._last_batch_size = batch_size
        assert batch_size <= self._num_examples
        start = self._index_in_epoch
        if self._index_in_epoch + batch_size > self._num_examples:
            if not self.mix_after_epoch:
                self._index_in_epoch = (self._index_in_epoch + batch_size) % self._num_examples
                end = self._index_in_epoch
                newbatch = DataSet(self._next_batch_(self._features, start, end),
                                   self._next_batch_(self._labels, start, end))
            else:
                self.shuffle()
                start = 0
                end = batch_size
                newbatch = DataSet(self._next_batch_(self._features, start, end),
                                   self._next_batch_(self._labels, start, end))
                self._index_in_epoch = batch_size
            return newbatch
        else:
            end = self._index_in_epoch + batch_size
            self._index_in_epoch = (batch_size + self._index_in_epoch) % self._num_examples
            if self._index_in_epoch == 0 and self.mix_after_epoch:
                self.shuffle()
            return DataSet(self._next_batch_(self._features, start, end),
                           self._next_batch_(self._labels, start, end))

    def show(self):
        """
        :method: Prints the data specs (dimensions, keys, type) in the :any:`DataSet` object
        """

        print('features:')
        for name, feature, in self.features.iteritems():
            print('\t %s: %s %s' % (name, feature.shape, type(feature)))
        print('labels:')
        for name, label in self.labels.iteritems():
            print('\t %s: %s %s' % (name, label.shape, type(label)))

    def showmore(self):
        """
        :method: Prints the data specs (dimensions, keys, type) in the :any:`DataSet` object,
        along with a sample of up to the first twenty rows for matrices in DataSet.
        """

        print('features:')
        for name, feature in self.features.iteritems():
            row = min(5, feature.shape[0])
            print('\t %s: \nFirst %s rows:\n%s\n' % (name, row, feature[0:row]))
        print('labels:')
        for name, label in self.labels.iteritems():
            row = min(5, label.shape[0])
            print('\t %s: \nFirst %s rows:\n%s\n' % (name, row, label[0:row]))

    def shuffle(self):
        """
        :method: The same random permutation is applied to the
         rows of all the matrices in :any:`features` and :any:`labels` .
        """
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._shuffle_(perm, self._features)
        self._shuffle_(perm, self._labels)

    def _shuffle_(self, order, datamap):
        '''
        :param order: A list of the indices for the row permutation
        :param datamap:
        :return: void
        Shuffles the rows an individual matrix in the :any:`DataSet` object.'
        '''
        for matrix in datamap:
            datamap[matrix] = datamap[matrix][order]

    def _next_batch_(self, datamap, start, end=None):
        '''
        :param datamap: A hash map of matrices
        :param start: starting row
        :param end: ending row
        :return: A hash map of slices of matrices from row start to row end
        '''
        if end is None:
            end = self._num_examples
        batch_data_map = {}
        if end <= start:
            start2 = 0
            end2 = end
            end = self._num_examples
            wrapdata = {}
            for matrix in datamap:
                wrapdata[matrix] = datamap[matrix][start2:end2]
                batch_data_map[matrix] = datamap[matrix][start:end]
                if sps.issparse(batch_data_map[matrix]):
                        batch_data_map[matrix] = sps.vstack([batch_data_map[matrix], wrapdata[matrix]])
                else:
                    batch_data_map[matrix] = np.concatenate([batch_data_map[matrix], wrapdata[matrix]], axis=0)
        else:
            for matrix in datamap:
                batch_data_map[matrix] = datamap[matrix][start:end]
        return batch_data_map


# # Tensorflow Introduction: Neural Network
# 
# A neural network is just a parametric function. If you can find the right parameters, a two layer neural network can 
# approximate any function!
# 
# Let $x \in \mathbb{R}^{1 \times n}, W \in \mathbb{R}^{n \times m},$ and $ U \in \mathbb{R}^{m \times p}$. A two layer neural network is the parametric function $\mathcal{q}: \mathbb{R}^{1 \times n} \rightarrow \mathbb{R}^{1 \times p}$ where 
# $\mathcal{q} = g( U f(x W + b) + c)$, $U, W, b, c$ are the parameters to be learned, and the functions $g,h$ are model choices.
# 
# ![nnet graph](nnet_graph.png)
# 
# Training a neural network involves a forward pass, which evaluates the function $q$ for a given set of parameters, and a backward pass which adjusts the parameters using gradient descent by way of the backpropagation algorithm, depending on how well $q$ approximates the training example targets. Tensorflow takes care of the math for the backward pass so we only need to worry about coding the forward pass for training.
# 
# There are several choices for $f$. Below are a few:

# In[15]:

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 

x = np.linspace(-5, 5, 100) # 100 linearly spaced numbers between -10, 10

# elementwise sigmoid
f1 = 1/(1+np.exp(-x))

# hyperbolic tangent: tanh
f2 = (np.exp(2*x) -1)/(np.exp(2*x) + 1)

# rectified linear units
f3 = np.maximum(np.zeros(x.shape), x)

plt.plot(x,f1, x, f2, x, f3)
plt.gca().set_ylim(bottom=-1.5, top=1.5)
plt.show()


# In[ ]:

# Retrieve data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Below is the computational graph of the function we want to make. Notice that $X$ is not a vector but a batch of vectors stacked together in a matrix, so we can send many vectors at a time through the neural net for training. 
# 
# ![nnet graph](batch_nnet.png)
# 

# In[ ]:

# Make a graph and set as default


# First make placeholders for inputs and targets


# Now make trainable variables


# In[ ]:

# Now compute q


# In[ ]:

# Loss function: cross entropy


# Train step: gradient descent optimizer

# Evaluate accuracy

# initialize session and variables


# In[ ]:

# training loop


# ## A reusable nnet
# This code is perfectly usable but we've written a perfectly good neural network that only works with the vectorized mnist data (or some other 784 dimensional data set)! Let's make our neural network reusable. 

# In[ ]:

# pass 1 at nnet classifier
def nnet_classifier1(x, hidden_size, output_size, activation):
    """
    First pass at a classifier
    
    :param x: Input to the network
    :param hidden_size: Size of second dimension of W the first weight matrix
    :param output_size: Size of second dimension of U the second weight matrix
    """
    return None


# ## A better reusable nnet

# In[ ]:

def nnet_classifier2(x, layers=[50,10], act=tf.nn.relu, name='nnet'):
    """
    Second pass at a classifier, eliminate repeated code. Bonus: An arbitrarilly deep neural network.
    
    :param x: Input to the network
    :param layers: Sizes of network layers
    :param act: Activation function to produce hidden layers of neural network.
    :param name: An identifier for retrieving tensors made by dnn
    """
    return None


# ## Eliminate boilerplate training code by making a reusable model class

# In[13]:

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
        
                                 


# ## Testing SimpleModel class, and nnet_classifier operation

# In[ ]:

import sys
import time
import tensorflow as tf

# Data prep ================================================================


# Make graph ============================================================

# Loss function


# Evaluate


# Make model


# Train ================================================================


