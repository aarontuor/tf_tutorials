#loading and manipulating data example
# numpy/scipy docs: http://docs.scipy.org/doc/

import numpy

ftrain = numpy.loadtxt('prog3_data/dataset1.train_features.txt')
fdev = numpy.loadtxt('prog3_data/dataset1.dev_features.txt')
ttrain = numpy.loadtxt('prog3_data/dataset1.train_targets.txt')
tdev = numpy.loadtxt('prog3_data/dataset1.dev_targets.txt')


def show(name, matrix):
    print("%s:\n \t shape: %s \n\t data struct type: %s \n\t datatype: %s\n\n%s\n" %
          (name, matrix.shape, type(matrix), matrix.dtype, str(matrix)))

def get_some_points(matrix, start, end):
    return matrix[start:end]

for name in ['ftrain', 'fdev', 'ttrain', 'tdev']:
    show(name, eval(name))

def toOnehot(X, dim):
    '''
    :param X: Vector of indices
    :param dim: Dimension of indexing
    :return: Matrix of one hots
    '''
    # empty one-hot matrix
    hotmatrix = numpy.zeros((X.shape[0], dim))
    # fill indice positions
    hotmatrix[numpy.arange(X.shape[0]), X.astype(int)] = 1
    return hotmatrix