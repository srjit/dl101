import pandas as pd
import gensim.models.keyedvectors as word2vec
import string
import numpy as np
from functools import reduce

#input_file = "train.ft.txt"
input_file = "sample.txt"
_vocabulary = set()
lines = []

lengths = []

with open(input_file) as f:
    for line in f:
        sentiment = line.split(" ")[0]
        review = line.replace(sentiment, "").strip()

        translator=str.maketrans('','',string.punctuation)
        plane_string = review.lower().translate(translator)
        
        words = plane_string.split()
        lengths.append(len(words))
        _vocabulary = _vocabulary.union(words)
        lines.append([plane_string, sentiment])


mean_length = reduce(lambda x, y: x + y, lengths) / len(lengths)
max_length = max(lengths)
median_length = np.median(lengths)



headers = ['review','sentiment']        
data = pd.DataFrame(lines, columns=headers)


# we need the indices of words - so making it a list
wordslist = list(_vocabulary)
wordVectors = []
limit = len(wordslist)

from gensim.models import Word2Vec
model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

invalid_words = []    
def get_vector(word):
    try:
        return model[word]
    except:
        invalid_words.append(word)
        return limit

wordvectors = [get_vector(word) for word in wordslist]

#remove model from memory
import gc
del model
gc.collect()

# we have to remove invalid words from going into tf.embedding_lookup 's sentence input


sequence_len = 150
# testing for one file

#create wordvectors for every word in the vocabulary using word2vec and keep it in an array
def get_vectors_of_sentence(sentence):
    def get_index(word):
        if word in invalid_words:
            return len(wordslist)
        try:
            return wordslist.index(word)
        except:
            return len(wordslist)
    
    words = sentence.split()
    doc_vec = np.zeros(sequence_len)
    sequence =  [get_index(word) for word in words][:sequence_len]
    if(len(sequence) < sequence_len):
        sequence[len(sequence):sequence_len] = [0] * (sequence_len - len(sequence))

    return np.asarray(sequence)


# build vocabulary now
data["encoded_review"] = data["review"].apply(lambda x: get_vectors_of_sentence(x))

data["label"] = data["sentiment"].apply(lambda x: [1, 0] if x == '__label__2' else [0, 1])


batch_size = 5
input_size =  len(data)
num_classes = 2
word_vector_length = 300
lstmunits = 64

# helper functions
from random import randint
def get_train_batch():

    start_index = randint(0, input_size - batch_size)
    end_index = start_index + batch_size

    arr = np.zeros([batch_size, sequence_len])

    batch_X = (data['encoded_review'][start_index: end_index]).tolist()
    batch_Y = data['label'][start_index: end_index]

    for i in range(batch_size):
        arr[i] = batch_X[i]

    return batch_X, batch_Y
    

#testing a sample of the encoded data
sample, labels = get_train_batch()


#lstm
import tensorflow as tf
tf.reset_default_graph()

input_data = tf.placeholder(tf.int32, [batch_size, sequence_len])
labels = tf.placeholder(tf.float32, [batch_size, num_classes])


#input_data_with_embeddings = tf.placeholder(tf.zeros([batch_size, sequence_length, word_vector_length]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordvectors, input_data)

lstmcell = tf.contrib.rnn.BasicLSTMCell(lstmunits)
lstmcell = tf.contrib.rnn.DropoutWrapper(cell=lstmcell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmunits, num_classes]))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)


correctpred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctpred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):

   #next batch
    
   nextbatch, nextbatchlabels = get_train_batch();
   sess.run(optimizer, {input_data: nextbatch, labels: nextbatchlabels})


