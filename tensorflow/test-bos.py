import tensorflow as tf

from sklearn.datasets import load_boston
import numpy as np

from numpy import genfromtxt

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


# Getting data from a csv
def read_dataset(filePath,  delimiter=', '):
    return genfromtxt(filePath,  delimiter=delimiter)


# Getting data from sklean
def read_boston_data():
    boston = load_boston()
    features = np.array(boston.data)
    labels = np.array(boston.target)
    return features,  labels


features, labels = read_boston_data()


n_dim = features.shape[1]

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.ones([n_dim, 1]))
b = tf.Variable(np.random.randn())

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


