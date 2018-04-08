from functools import reduce
import numpy as np
from tqdm import tqdm
import gensim.models.keyedvectors as word2vec
import pickle
import string

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


def save_word_list(input_file, projectname, filename="resources/wordlist.txt"):

    savepath = "/home/sree/.kaggle/datasets/" + projectname + "/" + filename
    
    _vocabulary = set()
    lengths = []
    count  = 0
    
    with open(input_file) as f:
        for line in f:
            sentiment = line.split(" ")[0]
            review = line.replace(sentiment, "").strip()

            translator=str.maketrans('','',string.punctuation)
            plane_string = review.lower().translate(translator)

            words = plane_string.split()
            lengths.append(len(words))
            _vocabulary = _vocabulary.union(words)
            print(count)
            count+=1

    vocabulary = list(_vocabulary)

    mean_length = reduce(lambda x, y: x + y, lengths) / len(lengths)
    max_length = max(lengths)
    median_length = np.median(lengths)

    
    with open(savepath ,"wb") as f:
        pickle.dump(vocabulary, f)

    return vocabulary
        

def load_word_list(projectname, filename="resources/wordlist.txt"):

    loadpath = "/home/sree/.kaggle/datasets/" + projectname + "/" + filename

    with open(loadpath, "rb") as fp:
        vocabulary = pickle.load(fp)

    return vocabulary
        



def save_word_vectors(_vocabulary, projectname, filename="resources/word_vectors.npy", invalid_words="resources/invalid_words.txt"):

    savepath = "/home/sree/.kaggle/datasets/" + projectname + "/" + filename
    invalid_words_path = "/home/sree/.kaggle/datasets/" + projectname + "/" + invalid_words
    
    wordslist = list(_vocabulary)
    limit = len(wordslist)

    model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    print("Word2Vec loaded...")

    invalid_words = []    
    def get_vector(word):
        try:
            return model[word]
        except:
            invalid_words.append(word)
            return limit

    wordvectors = np.zeros([len(wordslist), 300], dtype=np.float32)

    for i, word in tqdm(enumerate(wordslist)):
        wordvectors[i] = get_vector(word)

    del model
    np.save(savepath, wordvectors)

    with open(invalid_words_path ,"wb") as f:
        pickle.dump(invalid_words, f)

    
    
    return wordvectors, invalid_words


def load_word_vectors(projectname, filename="resources/word_vectors.npy", invalid_words="resources/invalid_words.txt"):

    loadpath = "/home/sree/.kaggle/datasets/" + projectname + "/" + filename
    invalid_words_path = "/home/sree/.kaggle/datasets/" + projectname + "/" + invalid_words
    
    wordvectors = np.load(loadpath)


    with open(invalid_words_path, "rb") as fp:
        invalid_words = pickle.load(fp)

        
    return wordvectors, invalid_words
    
    
