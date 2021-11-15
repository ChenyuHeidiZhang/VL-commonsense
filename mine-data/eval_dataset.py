import json
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr
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
with open('color-words.txt', 'r') as f:
    for line in f.readlines():
        if len(line.strip().split()) == 1:
            color_ls.append(line.strip())
color_ids = []
for color in COLORS:
    color_ids.append(color_ls.index(color))

dist_file = 'distributions/color-dist.jsonl'
vg_dist_dict = json.load(open(dist_file, 'r'))

for split in ['train', 'validation', 'test']:
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
            vg_dist = np.take(vg_dist_dict[ngram], color_ids)
            if np.sum(vg_dist) == 0:  # sp corr would be undefined if all elements of an array are 0
                continue
            color_dist = instance['label']
            sp = spearmanr(vg_dist, color_dist)
            sp_corrs.append(sp[0])
            p_vals.append(sp[1])
            sp_corrs_per_group[instance['object_group']].append(sp[0])
            objs[instance['object_group']].append(ngram)
            if sp[0] < 0.5:
                objs_weak_corr[instance['object_group']].append(ngram)
            common_subs.append(ngram)
    print(np.median(p_vals))
    print('mean and var of sp corr:', np.mean(sp_corrs), np.var(sp_corrs))
    print('for Single group:', np.mean(sp_corrs_per_group[0]), np.var(sp_corrs_per_group[0]))
    print('for Multi group:', np.mean(sp_corrs_per_group[1]), np.var(sp_corrs_per_group[1]))
    print('for Any group:', np.mean(sp_corrs_per_group[2]), np.var(sp_corrs_per_group[2]))
    print('num common subjects:', len(common_subs))
    print('subjects per group:', objs)
    print('subjects with weak corr:', objs_weak_corr)

