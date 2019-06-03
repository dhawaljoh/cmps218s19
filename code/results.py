import pickle
from tqdm import tqdm
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib
import time

from analogies import parse_google_analogies
import verification

def project_2d(res_ours, res_miks):
    words_ours, probs_ours = zip(*res_ours)
    words_miks, probs_miks = zip(*res_miks)
    print(words_ours[0])
    vecs_ours = [model[w] for w in words_ours]
    vecs_miks = [model[w] for w in words_miks]
    proj_ours = tsne.fit_transform(vecs_ours)
    proj_miks = tsne.fit_transform(vecs_miks)
    xs_ours, ys_ours = proj_ours[:, 0], proj_ours[:, 1]
    xs_miks, ys_miks = proj_miks[:, 0], proj_miks[:, 1]
    cmap = matplotlib.cm.get_cmap('viridis')
    normalize = matplotlib.colors.Normalize(vmin=min(probs_ours), vmax=max(probs_ours))
    colors_ours = [cmap(normalize(value)) for value in probs_ours]
    colors_miks = [cmap(normalize(value)) for value in probs_miks]
    #plt.scatter(xs_ours, ys_ours, color='blue', alpha=probs_ours)
    #plt.scatter(xs_miks, ys_miks, color='red', alpha=probs_miks)
    #fig, ax = plt.subplots()
    plt.scatter(xs_ours, ys_ours, color=colors_ours)
    sc = plt.scatter(xs_miks, ys_miks, color=colors_miks)
    for w, x, y in zip(words_ours, xs_ours, ys_ours):
        plt.annotate(w, xy=(x, y), xytext=(0,0), textcoords='offset points', color='blue')
    for w, x, y in zip(words_miks, xs_miks, ys_miks):
        plt.annotate(w, xy=(x, y), xytext=(0,0), textcoords='offset points', color='red')
    #plt.colorbar(sc)
    #cax, _ = matplotlib.colorbar.make_axes(ax)
    #cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
    plt.xlim(min(xs_ours.min(), xs_miks.min())+0.00005, max(xs_ours.max(), xs_miks.max())+0.00005)
    plt.xlim(min(ys_ours.min(), ys_miks.min())+0.00005, max(ys_ours.max(), ys_miks.max())+0.00005)
    plt.savefig('temp.pdf')


if __name__ == '__main__':

    #anlgtypes = ['capital-common-countries', 'currency', 'city-in-state', 'family', 'gram1-adjective-to-adverb', 'gram2-opposite', 'gram6-nationality-adjective', 'gram8-plural']
    anlgtypes = ['capital-common-countries']

    tsne = TSNE(n_components=2, random_state=0)

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
            target = anlg[-1]
            vec_target = model[target]
            res_ours, res_miks = results[i]

            project_2d(res_ours, res_miks)

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
            break

        print('our avg cosine similarity:\t\t' + str(cos_sum_ours/anlg_num))
        print('Mikolov avg cosine similarity:\t\t' + str(cos_sum_miks/anlg_num))
        print('our avg euclidean similarity:\t\t' + str(euc_sum_ours/anlg_num))
        print('Mikolov avg euclidean similarity:\t' + str(euc_sum_miks/anlg_num))
        print()


