
# 3 channels (depth) for the image - Is it RGB ?
# the network uses a filter with reduced dimentions that moves around
# strides - how many steps are taken while the filter is being moved around - hyperparameter

# How many filters  (N-F/ (stride)) + 1

# Image of size 7 * 7
#  N=7 F=3

#  Stride 1 = ((7-3) / 1) + 1 =  4 + 1 =5
#  Stride 2 = ((7-3) / 2) + 1 =  3
#  Stride 3 = ((7-3) / 3 ) + 1 = Non-integer  => does not fit


# It is commnon to zero pad the border of the image - add a border where everything is zero


# NUmber of paramters calculation :

#     Image :  32 * 32 * 3

#     Filter size : 10 * 5 * 5 filters, stride 1 and padding 2

#     Number of paramters in this layer ?


#     10 * (5 * 5  * 3  +  1(for the bias)) -> when there are 10 layers


# Paramters in libraries, eg. Torch

#   * nInputPlane : depth of the image
#   * nOuputPlane : How may filters - No of output planes the convolutions will produce
#   * kW : kernel width of the convolution
#   * kH : kernel height
#   * dW : step of the convolutions in width (stride)
#   * dH : step of the convolutions in height (stride)


#   Every filter is calculating a (wTX + b)  - all neurons have the same weight W -> they all share parameters

#   the neurons on different depth will be sharing the same region as well

#   Pooling -> Squishing the image

#     * max Pooling
  



# AlexNet
# ZFNet
# VGGNet
