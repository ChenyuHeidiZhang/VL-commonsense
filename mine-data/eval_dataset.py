import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial import distance
from datasets import load_dataset
from utils import load_word_file, load_dist_file

import warnings
warnings.filterwarnings("error")

plt.rcParams.update({'font.size': 12})
plt.rc('legend', fontsize=10)

# https://huggingface.co/datasets/corypaik/coda
ds = load_dataset("corypaik/coda", ignore_verifications=True)

COLORS = [
    'black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple',
    'red', 'white', 'yellow'
]
# line_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brwon', 'pink']

color_ls = load_word_file('color')[0]
vg_dist_dict = load_dist_file('color')
#wiki_dist_dict = load_dist_file('wiki-color')

def get_color_ids(colors):
    color_ids = []
    for color in colors:
        color_ids.append(color_ls.index(color))
    return color_ids

def js_div(dist1, dist2):
    return distance.jensenshannon(dist1, dist2) ** 2, 0

def sum_vg_dist(dist_pairs):
    sum = []
    for pair in dist_pairs:
        sum.append(np.sum(pair[1]))
    return np.mean(sum)

def plot_half(dist_pairs, ax, x_axis):
    for i, pair in enumerate(dist_pairs):
        # sort two distributions together
        sorted_dists = sorted(zip(pair[0], pair[1]), reverse=True)
        tuples = zip(*sorted_dists)
        coda_dist, vg_dist = [np.array(tuple) for tuple in tuples]
        # plot the distributions
        ax.plot(x_axis, coda_dist, color=f'C{i}', linewidth=2, label=f'CoDa \'{pair[2]}\'')
        ax.plot(x_axis, vg_dist / np.sum(vg_dist), color=f'C{i}', linewidth=2, linestyle='--', label=f'VG \'{pair[2]}\'')
    ax.legend()


def plot_dists(sp_corrs, dist_pairs, group='all', num_to_plot=3):
    '''
    plot the k examples that have highest & lowest (each) corr between coda and vg distributions.
    sp_corrs: list of sp corrs
    dist_pairs: np array of triplets of (CoDa dist, VG dist, noun)
    '''
    max_idxs = np.argpartition(sp_corrs, -num_to_plot)[-num_to_plot:]
    min_idxs = np.argpartition(sp_corrs, num_to_plot)[:num_to_plot]

    #print(max_idxs, min_idxs)
    #print(np.array(sp_corrs)[max_idxs], np.array(sp_corrs)[min_idxs])
    print(sum_vg_dist(dist_pairs[max_idxs]), sum_vg_dist(dist_pairs[min_idxs]))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=[8, 5])
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('probability', fontweight='bold')
    x_axis = range(1, 12)
    plot_half(dist_pairs[max_idxs], ax1, x_axis)
    ax1.set_title('high correlation')
    plot_half(dist_pairs[min_idxs], ax2, x_axis)
    ax2.set_title('low correlation')
    ax2.set_xticks(x_axis)
    ax2.set_xlabel('colors', fontweight='bold')
    plt.savefig(f'plots/eval_dataset_plt_wiki_{group}.pdf', bbox_inches='tight')

def run(topk=11, eval_method=spearmanr):
    common_subs = []
    num_skipped = {'zero_dist':0, 'large_pval':0}
    p_vals = []
    sp_corrs = []
    sp_corrs_per_group = {0:[], 1:[], 2:[]}  # corresponding to Single, Multi, and Any
    objs = {0:[], 1:[], 2:[]}
    objs_weak_corr = {0:[], 1:[], 2:[]}
    dist_pairs = []
    dist_pairs_group = {0:[], 1:[], 2:[]}
    for split in ['train', 'validation', 'test']:  # 
        print('split:', split)
        for instance in ds[split]:
            ngram = instance['ngram']
            #if ngram in wiki_dist_dict and 
            if ngram in vg_dist_dict and ngram not in common_subs:  # do not count same subject twice (different templates but same labels)
                obj_group = instance['object_group']
                color_dist = instance['label']
                topk_color_ids = np.argsort(color_dist)[-topk:][::-1]
                color_dist = np.take(color_dist, topk_color_ids)
                color_ids = get_color_ids([COLORS[id] for id in topk_color_ids])
                vg_dist = np.take(vg_dist_dict[ngram], color_ids)
                #wiki_dist = np.take(wiki_dist_dict[ngram], color_ids)
                if np.sum(vg_dist) == 0: #or np.sum(wiki_dist) == 0:  # sp corr would be undefined if all elements of an array are 0
                    num_skipped['zero_dist'] += 1
                    continue
                val, pval = eval_method(vg_dist, color_dist)
                #val, pval = eval_method(vg_dist, wiki_dist)
                # if pval > 0.05:
                #     #print(f'skipping subject {ngram} because of large pval {pval}')
                #     num_skipped['large_pval'] += 1
                #     continue
                dist_pairs.append((color_dist, vg_dist, ngram))
                dist_pairs_group[obj_group].append((color_dist, vg_dist, ngram))
                sp_corrs.append(val)
                #p_vals.append(pval)
                sp_corrs_per_group[obj_group].append(val)
                common_subs.append(ngram)
                objs[obj_group].append(ngram)
                if val < 0.5:
                    objs_weak_corr[obj_group].append((ngram, color_dist, vg_dist))
    #print(np.median(p_vals))
    print(num_skipped)
    plot_dists(sp_corrs, np.array(dist_pairs, dtype=object))
    print('mean and std of sp corr:', np.mean(sp_corrs), np.std(sp_corrs))
    print('for Single group:', np.mean(sp_corrs_per_group[0]), np.std(sp_corrs_per_group[0]))
    plot_dists(sp_corrs_per_group[0], np.array(dist_pairs_group[0], dtype=object), 'single')
    print('for Multi group:', np.mean(sp_corrs_per_group[1]), np.std(sp_corrs_per_group[1]))
    plot_dists(sp_corrs_per_group[1], np.array(dist_pairs_group[1], dtype=object), 'multi')
    print('for Any group:', np.mean(sp_corrs_per_group[2]), np.std(sp_corrs_per_group[2]))
    plot_dists(sp_corrs_per_group[2], np.array(dist_pairs_group[2], dtype=object), 'any')

    print('num common subjects:', len(common_subs))
    print('average num of occurrences in all:', sum_vg_dist(dist_pairs))

    vg_single_dist = [dists[1] for dists in dist_pairs_group[0]]
    single_acc = np.sum(np.argmax(vg_single_dist, axis=1) == 0) / len(vg_single_dist)
    print('percentage of top1 matching for single group:', single_acc)
    
    # print('subjects per group, & subjects with weak corr:')
    for i in range(3):
        print()
        print('group', i, 'num common subjects', len(objs[i]))
        print('average num of occurrences in VG of those subjects:', sum_vg_dist(dist_pairs_group[i]))
    #     print(objs[i])
    #     for triplet in objs_weak_corr[i]:
    #         print(triplet[0])
    #         print('CoDa dist:', triplet[1])
    #         print('VG dist:', (triplet[2]/np.sum(triplet[2])).tolist())

if __name__ == '__main__':
    run(topk=11, eval_method=spearmanr)
