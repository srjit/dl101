import math
import numpy as np

# Backpropogation notes


# Gradient ∇f is the vector of partial derivatives

x = -2
y = 5
z = 4

#performing the forward pass
q = x + y
f = q * z


#performing the backpropogation

# df/dz
dfdz = q

# df/dq
dfdq = z

# dq/dx
dqdx = 1

# dq/dy
dqdy = 1

dfdx = dfdq * dqdx

dfdy = dfdq * dqdy


## Every gate in the circuit computes two things during its forward pass
# 1. Local Value
# 2. Local gradient with respect to its output value

# Every gate in the circuit computes 
# 1. gradient for its output

# Gates communicate to each other if their final values has to be positive or negative





# neuron with a sigmoid activation function

# forward propogation
w0 = 2.0
x0 = -1.0

w1 = -3.0
x1 = -2.0

w2 = -3.0

a0 = w0 * x0
da0w0 = x0
da0x0 = w0

a1 = w1 * x1
da1w1 = x1
da1x1 = w1

a2 = a0 + a1
da2a0 = 1
da2a1 = 1


a3 = a2 + w2
da3a2 = 1
da3w2 = 1

a4 = a3 * -1.0
da4a3 = -a3

a5 = math.exp(a4)
da5a4 = math.exp(a4)

a6 = a5 + 1
da6a5 = 1

f = 1/a6
dfa6 = -1/math.pow(a6,2)



## Backpropogation
## tofind df/dx0 -> df/da6 * da6/da5 * da5/da4 * da4/da3 * da3/da2 * da2/da1 * da1/da0 * da0/dx0

# df/dx * (gradient from above) = -1/(1.37)^2
# applying chain rule backwards
print(dfa6)
print(dfa6 * da6a5)
print(dfa6 * da6a5 * da5a4)
print(dfa6 * da6a5 * da5a4 * da4a3)
print(dfa6 * da6a5 * da5a4 * da4a3 * da3a2)
print(dfa6 * da6a5 * da5a4 * da4a3 * da3a2 * da2a0)
print(dfa6 * da6a5 * da5a4 * da4a3 * da3a2 * da2a0 * da0x0)

# this is the effect of x0 on the final output



# Vectorized Operations
W = np.random.randn(5,10)




############################ concluding notes ############################
## more neurons are always better while creating networks, but they have to be regularized propoerly
## Depth vs Height -> no good choice
## Comon activation functions - TanH, ReLU etc
## usually only one activation function is used for all neurons