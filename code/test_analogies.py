import verification
import analogies

import time
import argparse
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import pickle

DATA_DIR = '../data/'


def init():
    """ Parse CL arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--procnum", help="Number of processes", type=int)
    return parser.parse_args()

def check_analogy(model, analogy):
    """ Compute the similarity measures of the given analogy. """
    wv, wu, wvp, wup = analogy
    v, u, vp, up = model[wv], model[wu], model[wvp], model[wup]
    ourz = v + u - up
    mikz = v - u + up
    our_sim = model.wv.similar_by_vector(ourz, topn=10, restrict_vocab=None)
    mik_sim = model.wv.similar_by_vector(mikz, topn=10, restrict_vocab=None)
    return our_sim, mik_sim

def check_all_analogies_no_mp(model, analogies):
    start = time.time()
    similar_vecs = []
    for analogy in tqdm(analogies):
        similar_vecs.append(check_analogy(model, analogy))
    print('checking', len(analogies), 'analogies took:', time.time() - start)
    return similar_vecs

def check_all_analogies(model, analogies, args):
    """ Get similarity measures of all analogies usingmultiprocessing. """
    pool = mp.Pool(processes=args.procnum)
    #partitions = [analogies[i::args.procnum] for i in range(args.procnum)]
    partitions = analogies
    chunksize = int(len(partitions)/args.procnum) # should be 1
    f = partial(check_analogy, model)
    print('Computing similarities of analogies')
    similar_vecs = list(tqdm(pool.imap(f, partitions, chunksize=chunksize), total=len(partitions)))
    #similar_vecs = [check_analogy(model, analogy) for analogy in analogies]
    pool.close()
    pool.join()
    print('Done checking all analogies')
    return similar_vecs

if __name__ == '__main__':
    args = init()
    # load model
    model = verification.load_model("GoogleNews-vectors-negative300.bin")
    print('Done loading model')
    # load analogies
    # analogy type: "capital-common-countries" TODO: test other types
    #anlgtype = 'capital-common-countries'
    anlgtypes = ['capital-common-countries', 'currency', 'city-in-state', 'family', 'gram1-adjective-to-adverb', 'gram2-opposite', 'gram6-nationality-adjective', 'gram8-plural']
    for anlgtype in anlgtypes:
        anlgs = analogies.parse_google_analogies('questions-words.txt', anlgtype=anlgtype)
        anlgs = anlgs[1][0]
        print('num of analogy: ' + str(len(anlgs)))
        print('Done loading analogies')

        # test analogies: for each analogy, get the vector from the model,
        similar_vecs = check_all_analogies_no_mp(model, anlgs)
        print('Done testing analogies')
        pickle.dump(similar_vecs, open(DATA_DIR + anlgtype + '_similar_vecs.pickle', 'wb'))

