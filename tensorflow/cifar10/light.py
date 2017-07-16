import cifar10
import numpy as np
import tensorflow as tf

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()


# placeholders for stuff
# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072])
labels_placeholder = tf.placeholder(tf.int64, shape=[None,])


# Define variables (these are the values we want to optimize)
weights = tf.Variable(tf.zeros([3072, 10]))
biases = tf.Variable(tf.zeros([10]))

logits = tf.matmul(images_placeholder, weights) + biases

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                          logits=logits,
                          labels=labels_placeholder))

learning_rate = 0.01
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

max_steps = 100000

batch_size = 200


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(max_steps):
        indices = np.random.choice(images_train.shape[0],
                                   batch_size)
        images_batch = images_train[indices]
        labels_batch = cls_train[indices]

        # import ipdb
        # ipdb.set_trace()

        images_batch = np.reshape(images_batch, [-1, 32 * 32 * 3])

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                images_placeholder: images_batch,
                labels_placeholder: labels_batch})
            print('Step {:5d}: training accuracy {:g}'.format(i,
                                                              train_accuracy))
        sess.run(train_step, feed_dict={images_placeholder: images_batch,
                                        labels_placeholder: labels_batch})

        


