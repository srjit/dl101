import cifar10

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

images_train = images_train.astype('float32')
labels_train = labels_train.astype('float32')

# placeholders for stuff
# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072])
labels_placeholder = tf.placeholder(tf.int32, shape=[None,10])



def create_cnn(features, labels, mode):

    # input layer reshaped
    input_layer = tf.reshape(features, [-1, 32, 32, 3])

    #add conv1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # output - 32 * 32 *  64
    #add max pooling
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2)


    pool1_flat = tf.reshape(pool1, [-1, 16 * 16 * 64])    

    # import ipdb
    # ipdb.set_trace()

    # dense layer
    dense = tf.layers.dense(inputs=pool1_flat,
                                units=16384, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense,
                                    rate=0.4,
                                    training=True)
    
    logits = tf.layers.dense(inputs=dropout, units=10)

#    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    return model_fn_lib.ModelFnOps(mode=mode,
                                   predictions=predictions,
                                   loss=loss, train_op=train_op)


cifar_classifier = learn.Estimator(
    model_fn=create_cnn, model_dir="/tmp/cifar")



cifar_classifier.fit(
    x=images_train,
    y=labels_train,
    batch_size=100,
    steps=20000)
#    monitors=[logging_hook])

metrics = {
    "accuracy":
    learn.MetricSpec(
        metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    # eval_results = cifar_classifier.evaluate(
    #     x=eval_data, y=eval_labels, metrics=metrics)
    # print(eval_results)
