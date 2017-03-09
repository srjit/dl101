## History

# 1960 -> Rosenblatt -> Perceptron
# f(x) = 1 if Wx + b >0
#        0 otherwise

# Multi layer Perceptron

# All in hardware

# 1970 -> Quiet

# 1986 -> Rumelhart -> Backpropagation
# Mulitlayer networks
# Not good training

# 2006 -> Hinton
# 10 Layer Network that trains proporly
# Restricted Boltzman machine
# Sigmoid Function as activation (Not great) - deep learning

# 2010 - 2012 : Really working well
# GMM HMM
# Microsoft - Speech Recognition
# 2012 - Visual Recognition Krizhevsky, Hinton, Sutskever


# Activation functions - why isn't signmoid a good function
# why is it bad if the weights are all positive
# LeCun's paper on why tanH is better than sigmoid
# Addiitonal Read : http://stats.stackexchange.com/questions/142348/tanh-vs-sigmoid-in-neural-net








# problems with Sigmoid
# * saturated neurons kill the gradients - at 1 or 0 slope is almost zero - vector of gradients
#   will be zero
# * for sigmoid the outputs are not zeero centered
# * exp is complex to compute

# tanH - has the problem of killing gradients
# has solved the problem of centering outputs


#ReLU
# f(x) = max(0, x)
# kills the gradient if weights are negative
# problem : dead relu -
#  people usually add a bias - which is otherwise zero


# Leaky relu - max(.01x, x) 
# parametric rectifier - learn : max(αx, x) - α can be learned - the slope
# in the third quadrant for the ReLU can be choosen







# zero centre your data
# squishing data : making the covarience data


# Weight Initialization
# * don't initilize all weight to zero : all neurons learn the same thing
# * initialize with random numbers from a guassian distribution
#   ** W = .01 * np.random.randn(D, H)
#   ** works only with small networks
#   **  all activations become zero : during backprop
#       when x is small, during backpropagation : x * gradient from top, will be small
#       it gets smaller and smaller eventually and the activations will become zero


# gradient with these kinds of initialization eventually becomes zero - vanishing gradient

# -> Xavier Initialization (Glorot et.  al)
# W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in) : works well in tanH, NOT in ReLU
#

# make something unit guassian
# insert batch normalization
# batch norm layer after every convolutional layer




# what all to check while initializing the network

import numpy as np

def init_two_layer_model(input_size, hidden_size, output_size) :

    model = {}
    model['w1'] = .0001 * np.random.random(input_size, hidden_size)
    model['b1'] = np.zeros(hidden_size)

    model['w2'] = .0001 * np.random.randn(hidden_size, output_size)
    model['b2'] = np.zeros(output_size)

    return model


model = init_two_layer_model(32*32*3, 50, 10)
# loss, grad = two_layer_net(X_train, model, y_train, 0.0)
# print loss


# check with and without regularization - loss should go up
# take a sample and make sure it can overfit

# while training
# try with a small learning rate : loss is decreasing - see rate of change
# find a good learning rate

# Cross validation Startegy


# when seaching  for tthe best values of regularization and learing rate : choose the log scale

  

