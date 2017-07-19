from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile

cifar10_dataset_folder_path = 'data/CIFAR-10/cifar-10-batches-py'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('cifar-10-python.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()


        
import helper
import numpy as np

# Explore the dataset
batch_id = 2
sample_id = 4
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)        

def normalize(x)    :
    ''' normalizing in the standard deviation
    '''
    max = np.max(x)
    min = np.min(x)

    return (x - min)/(max - min)


def one_hot_encode(x):
    ''' one hot encode the labels
    '''
    nx = np.max(x) + 1
    return np.eye(nx)[x]
    

helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

     


import pickle

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))


# construction of the network
import tensorflow as tf

def neural_net_input_images(image_shape):
    return tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 3], name='x')


def neural_net_input_labels(num_classes):
    return tf.placeholder(tf.float32, [None, num_classes], name='y')

def neural_net_keep_prob_input():
    return tf.placeholder(tf.float32, name='keep_prob')



tf.reset_default_graph()
num_classes = 10


def layers(features, num_ops_fc):

    input_layer = tf.reshape(features, [-1, 32, 32, 3])

    #add conv1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # output - 32 * 32 *  64
    #add max poppoling
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2)

    conv2= tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)


    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2)
    

    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 32])

    shape = pool2_flat.get_shape().as_list()
    W = tf.Variable(tf.random_normal([shape[-1], num_ops_fc], stddev=0.1))
    b = tf.Variable(tf.zeros(num_ops_fc)) + 0.11
    fc1 = tf.nn.relu(tf.add(tf.matmul(pool2_flat, W), b))

    dropout = tf.layers.dropout(inputs=fc1,
                                rate=0.4,
                                training=True)
    

    shape = dropout.get_shape().as_list()
    W = tf.Variable(tf.random_normal([shape[-1], num_classes]))
    b = tf.Variable(tf.zeros(num_classes))
    
    return tf.add(tf.matmul(fc1, W), b)




# building th e network
tf.reset_default_graph()

x = neural_net_input_images((32, 32, 3))
y = neural_net_input_labels(10)
keep_prob = neural_net_keep_prob_input()

logits = layers(x, 384)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')




def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    '''train a single batch of x
    '''
    session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})
    

def print_stats(session, feature_batch, label_batch, cost, accuracy):

    global valid_features, valid_labels
    validation_accuracy = session.run(
        accuracy,
        feed_dict={
            x: valid_features,
            y: valid_labels,
            keep_prob: 1.0,
            }
        )
    cost = session.run(
        cost,
        feed_dict={
            x: feature_batch,
            y: label_batch,
            keep_prob: 1.0,
            }
        )
    print('Cost = {0} - Validation Accuracy = {1}'.format(cost, validation_accuracy))

    


epochs = 50
batch_size = 1024
keep_probability = 0.5


# print('Checking the Training on a Single Batch...')
# with tf.Session() as sess:
#     # Initializing the variables
#     sess.run(tf.global_variables_initializer())
    
#     # Training cycle
#     for epoch in range(epochs):
#         batch_i = 1
#         for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
#             train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
#         print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
#         print_stats(sess, batch_features, batch_labels, cost, accuracy)
    
save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)



#####################################################
