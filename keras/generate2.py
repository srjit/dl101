import numpy

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

filename = "input.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))


n_chars = len(raw_text)
n_vocab = len(chars)

# fixed length of 100 characters - 1 sequence
# 100 characters of 1 timestep input - 1 output

seq_length = 100
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)

## create 144221 vectors each of length 100
## (samples, features, timesteps)
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X/float(n_vocab)
y = np_utils.to_categorical(dataY)

model = Sequential()

## LSTM with 256 memory units
## (100 * 256)
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, nb_epoch=1, batch_size=128, callbacks=callbacks_list)




## Text Generation

filename = "weights-improvement-01-2.7491.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


int_to_char = dict((i, c) for i, c in enumerate(chars))

start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")


for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")