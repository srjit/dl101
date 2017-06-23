import numpy as np


data = open('alice_new.txt', 'r', encoding='utf-8').read().strip() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

hidden_size = 100
seq_length = 25
learning_rate = 1e-2

Wxh = np.random.randn(hidden_size, vocab_size) *  0.01
Whh = np.random.randn(hidden_size, vocab_size) *  0.01
Why = np.random.randn(hidden_size, vocab_size) *  0.01


bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias




def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes



### Main Program
### Initializations of stuff to zeroes
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0


while True:
    
        # batching the data
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data

    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    

    # targets are sampled 1 more into the future :
    # we are trying to predict the state which is one from the current state
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    import pdb
    
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))
        pdb.set_trace()

