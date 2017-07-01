import numpy as np
import  tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data


__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"

## Download  and extract data
mnist = input_data.read_data_sets('data/', one_hot=True)


## MNIST Data - data exploration
trainimg   = mnist.train.images
trainlabel = mnist.train.labels

testimg    = mnist.test.images
testlabel  = mnist.test.labels


print("Plotting some images from the dataset")
nsample = 5
randidx = np.random.randint(trainimg.shape[0], size=nsample)

for i in randidx:
	curr_img   = np.reshape(trainimg[i, :], (28, 28)) # 28 by 28 matrix
	curr_label = np.argmax(trainlabel[i, :]) # Label
	plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
	plt.title("" + str(i) + "th Training Data "
               + "Label is " + str(curr_label))
	plt.show()
	print ("" + str(i) + "th Training Data "
            + "Label is " + str(curr_label))





## Learning in batches
batch_size = 100
batch_x, batch_y = mnist.train.next_batch(batch_size)


## collecting the images from random indices
random_indices = np.random.randint(trainimg.shape[0], size=batch_size)
sample = trainimg[random_indices, :]
