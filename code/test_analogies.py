import verification
import analogies

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
    print('blah')
    our_sim = model.wv.similar_by_vector(ourz, topn=10, restrict_vocab=None)
    print('blahblah')
    mik_sim = model.wv.similar_by_vector(Mikz, topn=10, restrict_vocab=None)
    print('blahblahblah')
    return our_sim, mik_sim

def check_all_analogies(model, analogies, args):
    """ Get similarity measures of all analogies usingmultiprocessing. """
    pool = mp.Pool(processes=args.procnum)
    partitions = [analogies[i::args.procnum] for i in range(args.procnum)]
    chunksize = int(len(partitions)/args.procnum) # should be 1
    f = partial(check_analogy, model)
    print('Computing similarities of analogies')
    similar_vecs = list(tqdm(pool.imap(f, partitions, chunksize=chunksize), total=len(partitions)))
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
    type = 'capital-common-countries'
    anlgs = analogies.parse_google_analogies('questions-words.txt', type=type)
    anlgs = anlgs[:6]
    print('num of analogy: ' + str(len(anlgs)))
    print('Done loading analogies')

    # test analogies: for each analogy, get the vector from the model,
    similar_vecs = check_all_analogies(model, anlgs, args)
    print('Done testing analogies')
    pickle.dump(similar_vecs, DATA_DIR + type + '_similar_vecs.pickle')

