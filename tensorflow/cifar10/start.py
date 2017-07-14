import data_utils
import cifar10
import tensorflow as tf

import os
import time
import numpy as np
from datetime import timedelta


import prettytensor as pt

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


# categorties - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

# 1. convolution
# 2. rectified linear units
# 3. max pooling
# 4. Local response normalization
# 5. AlexNet
# 6. Visualization of activities
# 7. Moving average of learned parameters
# 8. Decrementing learning rate

class_names = cifar10.load_class_names()
# X_train, y_train, X_test, y_test = data_utils.load_CIFAR10("data/CIFAR-10/cifar-10-batches-py")

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()


img_size = 32
img_size_cropped = 24
num_channels=3
num_classes = 10

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')

y_true_cls = tf.argmax(y_true, dimension=1)


## preprocessing part for a single image
def preprocessing(image, istrain):
    if istrain:

        # randomly crop the image
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        #flip the image horizontally
        image = tf.image.random_flip_left_right(image)

		# Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

		# limiting the pixel solution to 0-1 :  I don't understand what this is about
        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)

    else:
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)
    return image




## mapper function - tensorflow
def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: preprocessing(image, training), images)
    return images



distorted_images = pre_process(images=x, training=True)


def main_network(images, training):
    # Wrap the input images as a Pretty Tensor object.
    x_pretty = pt.wrap(images)

    # Pretty Tensor uses special numbers to distinguish between
    # the training and testing phases.
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    # Create the convolutional neural network using Pretty Tensor.
    # It is very similar to the previous tutorials, except
    # the use of so-called batch-normalization in the first layer.
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=num_classes, labels=y_true)

    return y_pred, loss

#helper function for creating neural net
def create_network(training):
    with tf.variable_scope('network', reuse= not training):

        images = x
        images = pre_process(images=images, training=training)

        y_pred, loss = main_network(images=images, training=training)

    return y_pred, loss


# iterations
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

#
_, loss = create_network(training=True)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

y_pred, _ = create_network(training=False)
y_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_cls, y_true_cls)

saver = tf.train.Saver()

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




def get_weights_variable(layer_name):
    with tf.variable_scope("network/" + layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable


weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')





session = tf.Session()

save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'cifar10_cnn')


session.run(tf.global_variables_initializer())


train_batch_size = 64
def random_batch():
    num_images = len(images_train)

    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch


def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

        # Save a checkpoint to disk every 1000 iterations (and last).
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            # Save all variables of the TensorFlow graph to a
            # checkpoint. Append the global_step counter
            # to the filename so we save the last several checkpoints.
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)

            print("Saved checkpoint.")

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
