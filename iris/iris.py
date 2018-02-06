import pandas as pd
from sklearn.cross_validation import train_test_split

import numpy as np
from pandas import get_dummies



__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


data = pd.read_csv("iris.csv")
X = data[data.columns[0:4]]
y = data[data.columns[4]]

features = list(data.columns[0:4])

for feature in features:
    X[feature] = (X[feature] - X[feature].mean()) / X[feature].std()

# one hot encode y
y = get_dummies(y)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)


X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
y_train = y_train.as_matrix()
y_test = y_test.as_matrix()

