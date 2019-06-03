import pickle
from tqdm import tqdm

from analogies import parse_google_analogies
import verification

anlgtypes = ['capital-common-countries', 'currency', 'city-in-state', 'family',      'gram1-adjective-to-adverb', 'gram2-opposite', 'gram6-nationality-adjective', 'gram8-    plural']
anlgtypes = ['capital-common-countries']

model = verification.load_model("GoogleNews-vectors-negative300.bin")

for anlgtype in anlgtypes:
    print(anlgtype)
    _, anlgs, _ = parse_google_analogies('questions-words.txt', anlgtype=anlgtype)
    anlgs = anlgs[0]
    results = pickle.load(open('../data/'+anlgtype+'_similar_vecs.pickle', 'rb'))
    euc_sum_ours = 0
    euc_sum_miks = 0
    cos_sum_ours = 0
    cos_sum_miks = 0
    anlg_num = len(anlgs)
    for i, anlg in enumerate(anlgs):
        print(anlg)
        target = anlg[-1]
        vec_target = model[target]
        res_ours, res_miks = results[i]
        vec_ours = model[res_ours[0][0]]
        vec_miks = model[res_miks[0][0]]
        cos_ours = verification.cosine_similarity(vec_ours, vec_target)
        cos_miks = verification.cosine_similarity(vec_miks, vec_target)
        euc_ours = verification.euclidean_similarity(vec_ours, vec_target)
        euc_miks = verification.euclidean_similarity(vec_miks, vec_target)
        cos_sum_ours += cos_ours
        cos_sum_miks += cos_miks
        euc_sum_ours += euc_ours
        euc_sum_miks += euc_miks
        #print('our cosine similarity: ' + str(cos_ours))
        #print('Mikolov cosine similarity: ' + str(cos_miks))
        #print('our euclidean similarity: ' + str(euc_ours))
        #print('Mikolov euclidean similarity: ' + str(euc_miks))
    print('our avg cosine similarity:\t' + str(cos_sum_ours/anlg_num))
    print('Mikolov avg cosine similarity:\t' + str(cos_sum_miks/anlg_num))
    print('our avg euclidean similarity:\t' + str(euc_sum_ours/anlg_num))
    print('Mikolov avg euclidean similarity:\t' + str(euc_sum_miks/anlg_num))
    print()


