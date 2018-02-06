import pandas as pd
from sklearn.cross_validation import train_test_split

import numpy as np



__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


data = pd.read_csv("iris.csv", index_col = 0)
X = data[data.columns[0:3]]
Y = data[data.columns[3]]

features = list(data.columns[0:3])

for feature in features:
    X[feature] = (X[feature] - X[feature].mean()) / X[feature].std()

X.as_matrix()
