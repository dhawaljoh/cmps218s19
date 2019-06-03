import gensim
import sys
import numpy as np
from numpy import linalg as LA
import time

from gensim.test.utils import common_texts, get_tmpfile, datapath
from gensim.models import Word2Vec

DATA_DIR = "../data/"

def cosine_similarity(a, b):
    """ Cosine similarity between vectors a and b. """
    cos = np.dot(a, b) / (LA.norm(a) * LA.norm(b))
    return cos

def euclidean_similarity(a, b):
    """ Euclidean similarity (1 - euclid. dist.) between vectors a and b. """
    return 1 - LA.norm(a-b)


def is_normalized(a):
    """ Check if vector a is normalized. """
    anorm = LA.norm(a)
    print (anorm)
    return anorm == 1

def load_model(filename):
    """ Load model from file and return. """
    model_file = DATA_DIR + filename
    print('Loading word vectors from ' + model_file)
    # Load Google's pre-trained Word2Vec model.
    start_time = time.time()
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
    elapsed_time = time.time() - start_time
    print('Done loading, took {:.2f} sec'.format(elapsed_time))
    return model

def main():

    #google = DATA_DIR + "GoogleNews-vectors-negative300.bin"

    ## Load Google's pre-trained Word2Vec model.
    #print('loading')
    #model = gensim.models.KeyedVectors.load_word2vec_format(google, binary=True)
    #print('done loading')

    model = load_model("GoogleNews-vectors-negative300.bin")

    # some tests
    v = model['king']
    u = model['man']
    vPrime = model['queen']
    uPrime = model['woman']

    ourModel = u + v - uPrime
    Mikolov = v - u + uPrime

    print("Words similar to the z vector generated by our method:")
    print(model.wv.similar_by_vector(ourModel, topn=10, restrict_vocab=None))
    print ("---------------------------------------------------------------")
    print("Words similar to the z vector generated by Mikolov's method:")
    print(model.wv.similar_by_vector(Mikolov, topn=10, restrict_vocab=None))
    print("----------------------------------------------------------------")
    print(model.wv.similar_by_word("king", topn=10, restrict_vocab=None))

    # TODOS
    # [V] Add functions for cosine distance and euclidean distance
    # [] Print euclidean distance between our model and woman
    # [] Print euclidean distance between Mikolov model and woman
    # [V] Check if vectors are normalized - sum squares of each vector, if sums to 1 then normalized

if __name__ == '__main__':
    main()
