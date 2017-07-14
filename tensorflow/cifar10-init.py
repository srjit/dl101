import sys


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

cifar_tut_directory = "/home/sree/code/models/tutorials/image/cifar10"
sys.path.insert(0, cifar_tut_directory)

import cifar10_input
data = cifar10_input.read_cifar10("/home/sree/code/dl101/tensorflow/cifar-10-batches-py")
