import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return np.maximum(0,x)


X = np.arange(-4,5,1)
Y = f(X)

plt.plot(X,Y,'o-')
plt.ylim(-1,5); plt.grid(); plt.xlabel('$x$', fontsize=22); plt.ylabel('$f(x)$', fontsize=22)


## variation of slope is observed here - for positive numbers
## the slope is 1 - tan 45
## for negative numbers it is 0 
