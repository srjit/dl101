import pandas as pd
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from pandas import get_dummies


__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


data = pd.read_csv("iris.csv")
X = data[data.columns[0:4]]
y = data[data.columns[4]]

features = list(data.columns[0:4])
num_features = 4
num_hidden = 10
num_classes = 3

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


x = tf.placeholder(tf.float32, [None, 4])
y = tf.placeholder(tf.float32, [None, 3])


h1 = tf.Variable(tf.random_normal([4, num_hidden]))
b1 = tf.Variable(tf.random_normal([num_hidden]))

out_l = tf.Variable(tf.random_normal([num_hidden, num_classes]))
out_b = tf.Variable(tf.random_normal([num_classes]))


def network(x, h1, b1, out_l, out_b):
    layer1 = tf.add(tf.matmul(x, h1), b1)
    layer1 = tf.nn.relu(layer1)

    out_layer = tf.add(tf.matmul(layer1, out_l), out_b)
    return out_layer


learning_rate = 0.01

pred = network(x, h1, b1, out_l, out_b)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                      (logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(100):
        _, c = sess.run([optimizer, cost], feed_dict={x: X_train,
                                                      y: y_train})
        print(c)
    print("Optimiztion finished...!")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))    
