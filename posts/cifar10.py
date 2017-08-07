import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


mnist = input_data.read_data_sets('data', one_hot=True)


trainimgs = mnist.train.images
trainlabels = mnist.train.labels
testimgs = mnist.test.images
testlabels = mnist.test.labels

learning_rate = 0.001
batch_size = 128


X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])


def network(x):
    
    input_layer = tf.reshape(x, shape=[-1, 28, 28, 1])
    
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2)

    
    pool1_flat = tf.reshape(pool1, [-1, 14 * 14 * 64])
    
    dense = tf.layers.dense(inputs=pool1_flat,
                                units=4096, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense,
                                    rate=0.4,
                                    training=True)
    
    logits = tf.layers.dense(inputs=dropout, units=10)
    return logits


pred = network(X)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


epochs = 16
iterations_per_epoch = int(len(trainimgs)/batch_size)

acc = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1

    for i in range(epochs):
            print(" Executing epoch :", i+1)
            for j in range(iterations_per_epoch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
                    

            if i%3 == 0:
                    loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x,
                                                                              Y: batch_y})
                    print("Accuracy:", acc)
