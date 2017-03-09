# mu is a hyperparamter -> Physical anology to coefficient of friction

# Update of x on gradient descent

#     X += -learning_rate * dx


# Momentum update

#    V = mu * V - learning_rate*dx
#    X += V
#  v is initialized with 0 usually
#  mu is initialized by values from .5 to .9


# Nesterov Momentum - Slight modification from the actual momentum


# * there are different types of gradient updates, choose one which will make
# convergence faster

# * adam
# * all are first order methods - uses only gradient


# decay of learning rates

# * exponential decay - usually works well
# * 1/t decay




# ** ADAM is a good choice of Gradient Update
# ** Full Batch updates can be performed with L-BFGS

# Regularization : Dropout
