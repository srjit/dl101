import pandas as pd
import numpy as np
import tensorflow as tf

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


boston = pd.read_csv("boston.csv")

features = boston.ix[:,:12].as_matrix()

_labels = boston.ix[:,12]
labels = np.array(_labels, dtype=pd.Series)



sess = tf.InteractiveSession()


x = tf.placeholder(tf.float32, shape=[None, 12])
y_ = tf.placeholder(tf.float32, shape=[None, 1])



#Variables
W = tf.Variable(tf.zeros([12,1]))
b = tf.Variable(tf.zeros([1]))


sess.run(tf.global_variables_initializer())


#Predicted class and loss function
y = tf.matmul(x,W) + b

# loss function - cross entropy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


## Train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(5):
  begin_index = i * 100
  _features = features[begin_index : begin_index + 100]
  _labels = labels[begin_index : begin_index + 100].reshape(100,1)
  train_step.run(feed_dict={x: _features, y_: _labels})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Train Accuracy")
print(accuracy.eval(feed_dict={x: features[0:500], y_: labels[0:500].reshape(500,1)}))
