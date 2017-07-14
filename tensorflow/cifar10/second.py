import tensorflow as tf
import tests

import cifar10


__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


classes = cifar10.load_class_names()

images_train, labels, train_labels_encoded = cifar10.load_training_data()
images_test, labels, test_labels_encoded = cifar10.load_test_data()

image_size = 32
number_of_classes = 10
number_of_channels = 3

image_size_cropped = 28


def pre_process_image(image, istrain):
    if istrain:

        # randomly crop the image
        image = tf.random_crop(image, size=[image_size_cropped,
                                            image_size_cropped,
                                            number_of_channels])

        # flip the image horizontally
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        image = tf.image.\
          resize_image_with_crop_or_pad(image,
                                        target_height=image_size_cropped,
                                        target_width=image_size_cropped)

    return image




def pre_process_images(images_train, istrain):
    return tf.map_fn(lambda image: pre_process_image(image, istrain),
                     images_train)


def main_network(images):
    conv1 = tf.layers.conv2d(inputs=images,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2,2],
                                    strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2,2],
                                    strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])

    dense = tf.layers.dense(inputs=pool2_flat,
                            units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4,
                                training=True)

    logits = tf.layers.dense(inputs=dropout, units=10)

    loss = None

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")    


    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }    
    
    return logits, predictions, loss, train_op





img_size = 28
num_channels = 3
num_classes=10

# placeholders for everything
x = tf.placeholder(tf.float32, shape=[100,
                                      img_size,
                                      img_size,
                                      num_channels], name='x')

y_true = tf.placeholder(tf.float32, shape=[None,
                                           num_classes], name='y')
y_true_cls = tf.argmax(y_true, dimension=1)


def create_network(training):
    images = x
    images = pre_process_images(images_train=images, istrain=training)
    logits, predictions, loss, train_op = main_network(images=images)
    return logits, predictions, loss, train_op




global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
logits, predictions, loss, train_op = create_network(training=True)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)



def optimize(num_iterations):
    for i in range(num_iterations):
        x_batch, y_true_batch = tests.random_batch()

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        if (i_global % 100 == 0) or (i == num_iterations - 1):
            print("reached iterations: ", num_iterations)


            





session = tf.Session()
session.run(tf.global_variables_initializer())

