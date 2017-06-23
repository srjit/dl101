

## Simple Neural Network with three layers


import numpy as np

# return sigmoid or its derivative
def nonlin(x, deriv = False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1 + np.exp(-x))

    

X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

Y = np.array([[0,0,1,1]]).T

# seed for making calcs deterministic
np.random.seed(1)


#initializing weights randomly with mean to zero

# ↣ input layer : x0, x1, x2
# ↣ hidden layer (3*4)
# ↣ output layer (4*1)

# 3 input neurons and 4 neurons output
syn0 = 2 * np.random.random((3,4)) - 1

# 4 neurons and 1 output
syn1 = 2 * np.random.random((4,1)) - 1

for i in range(10000):

    # forward propogation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))


    # backpropogation - calculation of (error * df/dx)
    l2_error = Y - l2
    l2_delta = l2_error * nonlin(l2, True)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, True)

    # update the weights for next iteration
    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)

    

print("Output after training...")
print(l2)
    
