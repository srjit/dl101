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

wordVectors = [model[word] for word in wordslist]

# we have to remove invalid words from going into tf.embedding_lookup 's sentence input
word_vectors = {}

sequence_len = 150
# testing for one file



#create wordvectors for every word in the vocabulary using word2vec and keep it in an array



def get_vectors_of_sentence(sentence):

    def get_index(word):
        try:
            return vocabulary[word]
        except:
            return len(vocabulary)
    
    words = sentence.split()
    doc_vec = np.zeros(sequence_len)
    return [get_index(word) for word in words][:sequence_len]

# build vocabulary now
data["doc_inv_vocab"] = data["review"].apply(lambda x: get_vectors_of_sentence(x))
