import pandas as pd
import gensim.models.keyedvectors as word2vec
import string
import numpy as np

#input_file = "train.ft.txt"
input_file = "sample.txt"
_vocabulary = set()
lines = []

lengths = []

with open(input_file) as f:
    for line in f:
        sentiment = line.split(" ")[0]
        review = line.replace(sentiment, "").strip()

        plane_string = review.lower().translate(None, string.punctuation)
        words = plane_string.split()
        lengths.append(len(words))
        _vocabulary = _vocabulary.union(words)
        lines.append([plane_string, sentiment])


mean_length = reduce(lambda x, y: x + y, lengths) / len(lengths)
max_length = max(lengths)
median_length = np.median(lengths)

headers = ['review','sentiment']        
data = pd.DataFrame(lines, columns=headers)

vocabulary = {word:index for index, word in enumerate(list(_vocabulary))}
limit = len(vocabulary)

from gensim.models import Word2Vec
model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

word_vectors = {}

sequence_len = 150
# testing for one file

def get_vectors_of_sentence(sentence):

    def get_index(word):
        try:
            return vocabulary[word]
        except:
            return limit
    
    words = sentence.split()
    doc_vec = np.zeros(sequence_len)
    return [get_index(word) for word in words][:sequence_len]

# build vocabulary now
data["doc_inv_vocab"] = data["review"].apply(lambda x: get_vectors_of_sentence(x))



        
        
        
        
        




    
    
