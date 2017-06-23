import numpy as np

D = np.random.randn(1000, 500)
hidden_layer_sizes = [500]*10

non_linearities = ['tanh'] * len(hidden_layer_sizes)

act = {'relu' : lambda x : np.maximum(0, x), 'tanh' : lambda x: np.tanh(x)}

Hs = {}

for i in range(len(hidden_layer_sizes)):

    X = D if i == 0 else Hs[i-1]

    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]

    W = np.random.randn(fan_in, fan_out) * 0.01

    H = np.dot(X, W)
    H = act[non_linearities[i]][H]

    Hs[i] = H
    
    

