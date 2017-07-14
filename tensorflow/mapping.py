import tensorflow as tf

import numpy as np

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


array = np.array([1,2,3,4,5,6,7])

sess = tf.Session()
squares = tf.map_fn(lambda x : x*x, array)



sess.run(squares)

