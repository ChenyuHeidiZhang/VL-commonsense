import json
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial import distance
from datasets import load_dataset

import warnings
warnings.filterwarnings("error")

# https://huggingface.co/datasets/corypaik/coda
ds = load_dataset("corypaik/coda", ignore_verifications=True)

COLORS = [
    'black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple',
    'red', 'white', 'yellow'
]

color_ls = []
with open('words/color-words.txt', 'r') as f:
    for line in f.readlines():
        if len(line.strip().split()) == 1:
            color_ls.append(line.strip())
def get_color_ids(colors):
    color_ids = []
    for color in colors:
        color_ids.append(color_ls.index(color))
    return color_ids

dist_file = 'distributions/color-dist.jsonl'
vg_dist_dict = json.load(open(dist_file, 'r'))

topk = 11
for split in ['train']:  #, 'validation', 'test'
    print('split:', split)
    common_subs = []
    p_vals = []
    sp_corrs = []
    sp_corrs_per_group = {0:[], 1:[], 2:[]}  # corresponding to Single, Multi, and Any
    objs = {0:[], 1:[], 2:[]}
    objs_weak_corr = {0:[], 1:[], 2:[]}
    for instance in ds[split]:
        ngram = instance['ngram']
        if ngram in vg_dist_dict and ngram not in common_subs:  # do not count same subject twice (different templates but same labels)
            color_dist = instance['label']
            topk_color_ids = np.argsort(color_dist)[-topk:][::-1]
            color_dist = np.take(color_dist, topk_color_ids)
            color_ids = get_color_ids([COLORS[id] for id in topk_color_ids])
            vg_dist = np.take(vg_dist_dict[ngram], color_ids)
            if np.sum(vg_dist) == 0:  # sp corr would be undefined if all elements of an array are 0
                continue
            #sp = kendalltau(vg_dist, color_dist)
            js_div = distance.jensenshannon(vg_dist, color_dist)**2
            val = js_div
            sp_corrs.append(val)
            #p_vals.append(sp[1])
            sp_corrs_per_group[instance['object_group']].append(val)
            objs[instance['object_group']].append(ngram)
            if val < 0.5:
                objs_weak_corr[instance['object_group']].append((ngram, color_dist, vg_dist))
            common_subs.append(ngram)
    #print(np.median(p_vals))
    print('mean and std of sp corr:', np.mean(sp_corrs), np.std(sp_corrs))
    print('for Single group:', np.mean(sp_corrs_per_group[0]), np.std(sp_corrs_per_group[0]))
    print('for Multi group:', np.mean(sp_corrs_per_group[1]), np.std(sp_corrs_per_group[1]))
    print('for Any group:', np.mean(sp_corrs_per_group[2]), np.std(sp_corrs_per_group[2]))
    print('num common subjects:', len(common_subs))
    # print('subjects per group, & subjects with weak corr:')
    # for i in range(3):
    #     print()
    #     print('group', i)
    #     print(objs[i])
    #     for triplet in objs_weak_corr[i]:
    #         print(triplet[0])
    #         print('CoDa dist:', triplet[1])
    #         print('VG dist:', (triplet[2]/np.sum(triplet[2])).tolist())

