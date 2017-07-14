import numpy as np

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


def random_batch(train_batch_size):

    idx = np.random.choice(50000,
                           size=train_batch_size,
                           replace=False)
    return idx
