import numpy as np
from tqdm import tqdm

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


def save_word_vectors(_vocabulary, projectname, filename="word_vectors.npy"):

    savepath = "~/.kaggle/datasets/" + projectname + "/" + filename
    
    wordslist = list(_vocabulary)
    limit = len(wordslist)

    import ipdb
    ipdb.set_trace()

    from gensim.models import Word2Vec
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

    import ipdb
    ipdb.set_trace()

    np.save(wordvectors, savepath)
    return wordvectors


def load_word_vectors(projectname, filename="word_vectors.npy"):

    loadpath = "~/.kaggle/datasets/" + projectname + "/" + filename
    wordvectors = np.load(loadpath)
    return wordvectors
    
    
