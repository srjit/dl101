

- State in the box - it can modify the state which is a function of the input it receives
-_ tune the weight and change the behaviour

RNN State, h(t) = Fw(h(t-1), Xt)

t  = time

function_parameters(previous state, and input vector at some time step)

Simplest way of representation
-----------------------------

* vanilla RNN
* single hidden state h
* h(t) = tanh(Whh*h(t-1) + Wxh Xt)


** character level language models


## http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
## charcater level rnn gist :  https://gist.github.com/karpathy/d4dee566867f8291f086
