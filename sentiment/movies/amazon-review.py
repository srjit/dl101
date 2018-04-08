import pandas as pd
import string
import numpy as np
import datetime
import os
import vectorutils


project = "bittlingmayer/amazonreviews"


input_dir = "/home/sree/.kaggle/datasets/" + project
input_file = input_dir + "/train_head.txt"


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
data = pd.DataFrame(lines, columns=headers)


word_vectors_location = input_dir + "/resources/word_vectors.npy" 
if os.path.isfile(word_vectors_location):
    print("Loading word vectors...")
    wordvectors, invalid_words = vectorutils.load_word_vectors(project)
else:
    print("No word vectors found...calculating vectors from word2vec and saving it to resources...")

    wordvectors, invalid_words = vectorutils.save_word_vectors(vocabulary, project)
    

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
data["encoded_review"] = data["review"].apply(lambda x: get_vectors_of_sentence(x))

print("Setting the labels...")
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

