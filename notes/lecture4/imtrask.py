

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
syn0 = 2 * np.random.random((3,1)) - 1

for i in range(10000):

    # forward propogation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    l1_error = Y - l1

    # backpropogation - calculation of (error * df/dx)
    l1_delta = l1_error * nonlin(l1, True)

    # update the weights for next iteration
    syn0 += np.dot(l0.T, l1_delta)
    

print("Output after training...")
print(l1)
    
