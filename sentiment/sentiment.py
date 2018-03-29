import pandas as pd
import gensim.models.keyedvectors as word2vec


input_file = "train.ft.txt"
_vocabulary = set()
lines = []

with open(input_file) as f:
    for line in f:
        sentiment = line.split(" ")[0]
        review = line.replace(sentiment, "").strip()

        plane_string = lower(review).translate(None, string.punctuation)
        words = plane_string.split()
        _vocabulary.union(words)
        lines.append([plane_string, sentiment])

headers = ['review','sentiment']        
data = pd.DataFrame(lines, columns=headers)

from gensim.models import Word2Vec
#model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


# build vocabulary now



        
        
        
        
        




    
    
