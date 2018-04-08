import pandas as pd
import gensim.models.keyedvectors as word2vec
import string
import numpy as np
from functools import reduce
import datetime

import vectorutils

#input_file = "train.ft.txt"
#input_file = "sample.txt"

project = "bittlingmayer/amazonreviews"

input_dir = "/home/sree/.kaggle/datasets/" + project
input_file = input_dir + "/train.ft.txt"

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

# wordvectors = np.asarray([get_vector(word) for word in wordslist])

# lets save the word vectors here
word_vectors_location = input_dir + "/resources/word_vectors.npy" 
if os.path.isfile(word_vectors_location):
    wordvectors = vectorutils.load_word_vectors()
else:
    print("No word vectors found...calculating vectors from word2vec and saving it to resources...")

    import ipdb
    ipdb.set_trace()
    
    wordvectors = vectorutils.save_word_vectors(_vocabulary, project)
    

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


num_classes = 2
word_vector_length = 300
lstmunits = 64
batch_size = 24
iterations = 100000
numDimensions = 300
input_size = 100

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

    return arr, batch_Y
    

#testing a sample of the encoded data
sample, labels = get_train_batch()


import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batch_size, num_classes])
input_data = tf.placeholder(tf.int32, [batch_size, sequence_len])

data = tf.Variable(tf.zeros([batch_size, sequence_len, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordvectors,input_data)

import ipdb
ipdb.set_trace()

#lstm layer
lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstmunits)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmunits, num_classes]), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)


correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(logdir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch();

   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
   print("Epoch", i+1)

   #Write summary to Tensorboard
   # if (i % 50 == 0):
   #     summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
   #     writer.add_summary(summary, i)

   # #Save the network every 10,000 training iterations
   # if (i % 10000 == 0 and i != 0):
   #     save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
   #     print("saved to %s" % save_path)

       #

       
writer.close()

