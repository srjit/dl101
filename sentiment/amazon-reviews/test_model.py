import tensorflow as tf
from tqdm import tqdm, tqdm_pandas
import pandas as pd
import string
import numpy as np
import datetime
import os
import vectorutils

from functools import reduce


__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"




tqdm.pandas(tqdm())

project = "bittlingmayer/amazonreviews"


input_dir = "/home/sree/.kaggle/datasets/" + project
input_file = input_dir + "/test.txt"



word_list_location = input_dir + "/resources/wordlist.txt"

if os.path.isfile(word_list_location):
    print("loading file from " + word_list_location)
    vocabulary = vectorutils.load_word_list(project)
else:    
    vocabulary = vectorutils.save_word_list(input_file, project)

print("Length of vocabulary:",len(vocabulary))

lines = []
count = 0
with open(input_file) as f:
    for line in f:
        sentiment = line.split(" ")[0]
        review = line.replace(sentiment, "").strip()

        translator=str.maketrans('','',string.punctuation)
        plane_string = review.lower().translate(translator)
        
        lines.append([plane_string, sentiment])
        print(count)
        count+=1

        

headers = ['review','sentiment']        
_data = pd.DataFrame(lines, columns=headers)


word_vectors_location = input_dir + "/resources/word_vectors.npy" 
if os.path.isfile(word_vectors_location):
    print("Loading word vectors...")
    wordvectors, invalid_words = vectorutils.load_word_vectors(project)
else:
    print("No word vectors found...calculating vectors from word2vec and saving it to resources...")

    wordvectors, invalid_words = vectorutils.save_word_vectors(vocabulary, project)
    
print("Word vectors created...")
import gc
gc.collect()



sequence_len = 150
def get_vectors_of_sentence(sentence):
    def get_index(word):
        if word in invalid_words:
            return len(vocabulary)
        try:
            return vocabulary.index(word)
        except:
            return len(vocabulary)
    
    words = sentence.split()
    doc_vec = np.zeros(sequence_len)
    sequence =  [get_index(word) for word in words][:sequence_len]
    if(len(sequence) < sequence_len):
        sequence[len(sequence):sequence_len] = [0] * (sequence_len - len(sequence))
    
    return np.asarray(sequence)


# build vocabulary now
print("Vectorizing the words and encoding the sentences...")
_data["encoded_review"] = _data["review"].progress_apply(lambda x: get_vectors_of_sentence(x))

print("Setting the labels...")
_data["label"] = _data["sentiment"].apply(lambda x: [1, 0] if x == '__label__2' else [0, 1])





print("Loaded test data...")

num_classes = 2
word_vector_length = 300
lstmunits = 256
batch_size = 1000
iterations = 100000
numDimensions = 300
input_size = len(_data)




print("Creating the same neural network...")
import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batch_size, num_classes])
input_data = tf.placeholder(tf.int32, [batch_size, sequence_len])

data = tf.Variable(tf.zeros([batch_size, sequence_len, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordvectors,input_data)


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



sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))


predictions = []


def get_test_batch(start_index):

    end_index = start_index + 1000
    print("Next batch to train starting index: ", start_index)

    arr = np.zeros([batch_size, sequence_len])

    batch_X = (_data['encoded_review'][start_index: end_index]).tolist()
    batch_Y = _data['label'][start_index: end_index].tolist()

    for i in range(batch_size):
        arr[i] = batch_X[i]

    return arr, batch_Y


accuracies_of_batches = []
for  start_index in range(0, 2000, 1000):
    test_batch, actuals = get_test_batch(start_index)
    predictedSentiment = sess.run(prediction, {input_data: test_batch}).tolist()

    predictions = [1 if a > b else 0 for [a,b] in predictedSentiment]
    _actuals = [a for [a,b] in actuals]

    correctness = [1 if predictions[i] == _actuals[i] else 0 for i in range(len(_actuals))]
    accuracy = float(sum(correctness))/len(correctness)

    
    print("Accuracy of test_batch: ", accuracy)
    accuracies_of_batches.append(accuracy)



accuracy = reduce(lambda x, y: x + y, accuracies_of_batches) / len(accuracies_of_batches)
print("Final accuracy of test data:", accuracy)
    
