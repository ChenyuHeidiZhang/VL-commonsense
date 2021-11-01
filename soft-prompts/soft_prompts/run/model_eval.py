from ..model import PatternModel
from ..io import load_db_general
from ..corpus import load_templates
from ..lm import construct_lm
from .experiment import read_kwargs
from .. import util

import math
import numpy as np
from scipy.stats.stats import pearsonr
import torch
import json
import os, glob
import pickle
import logging
logger = logging.getLogger(__name__)


def get_correlation(vg_dist, model_dist):
    # Compute correlation between the VG distribution and model distribution
    corr = [pearsonr(vg_dist[i], model_dist[i]) for i in range(vg_dist.shape[0])]
    avg_corr = np.mean([cor[0] for cor in corr])
    print(avg_corr)  # bert: 0.321, 0.606; oscar: 0.433, 0.411
    corr_flatten = pearsonr(vg_dist.flatten(), model_dist.flatten())
    print(corr_flatten)  # bert: 0.128, 0.224; oscar: 0.164, 0.147
    return corr, corr_flatten


def analyze_distributions(rel_type, target_dist, test_set, answer_ranks, preds):
    # get the list of object ids, and find model distribution across the objects
    objs_file = f'/home/heidi/VL-commonsense/mine-data/{rel_type}-words.txt'
    objs_f = open(objs_file, 'r').readlines()
    objs_ls = []
    single_token = True
    for line in objs_f:
        if single_token and len(line.strip().split()) > 1:
            continue
        objs_ls.append(line.strip())
    obj_ids = torch.tensor(util.tokenizer.convert_tokens_to_ids(objs_ls))

    model_dist = np.array(torch.index_select(target_dist, 1, obj_ids))
    # print(model_dist.shape)  # [69, num_objs]

    # load Visual Genome distribution across objects for the test subjects
    subjects = [rel_ins.entities[0] for rel_ins in test_set]
    # print(len(subjects))  # [69]
    dist_file = f'/home/heidi/VL-commonsense/mine-data/{rel_type}-dist.jsonl'
    vg_dist_dict = json.load(open(dist_file, 'r'))
    vg_dist = np.array([vg_dist_dict[sub] for sub in subjects])
    # vg_dist = []  # not necessary because normalization does not change pearson correlation
    # for sub in subjects:
    #     dist = np.array(vg_dist_dict[sub])
    #     dist = dist / dist.sum()
    #     vg_dist.append(dist)
    # vg_dist = np.array(vg_dist)
    # print(vg_dist.shape)    # [69, num_objs]

    print('Correlation of VG and model distributions:')
    corr, corr_flatten = get_correlation(vg_dist, model_dist)

    answer_tokens = [rel_ins.entities[1] for rel_ins in test_set]
    predicted_tokens = util.tokenizer.convert_ids_to_tokens(preds.T[0])
    # Get model distribution and VG distribution for instances where answer_rank > 3,
    # compute the correlation of two distributions,
    # print the error (sub, true_target, predicted) tuple,
    # return the error tuples and model distributions for cross-model comparisons.
    idxs = torch.nonzero(answer_ranks > 3).squeeze(1)
    print(f'predictions are off for {len(idxs)} out of {len(test_set)} instances')
    vg_dist_wrong = vg_dist[idxs]
    model_dist_wrong = model_dist[idxs]
    print('Correlation of distributions when prediction is off:')
    corr2, corr_flatten2 = get_correlation(vg_dist_wrong, model_dist_wrong)

    error_tuples = []
    for idx in idxs:
        error_tuples.append((subjects[idx], answer_tokens[idx], predicted_tokens[idx]))
    #print('\n'.join([', '.join(tup) for tup in error_tuples]))

    return error_tuples, vg_dist_wrong, model_dist_wrong


def size_correctness(rel_type, test_set, preds):
    assert rel_type in ['size_smaller', 'size_larger']
    # Retrieve the set of words in different size categories and convert them to ids.
    dir = '/home/heidi/VL-commonsense/mine-data/size_db/'
    sizes = ['tiny', 'small', 'medium', 'large', 'xlarge']
    words_dict = {sz:[] for sz in sizes}
    for filename in glob.glob(os.path.join(dir, '*.txt')):
        size = filename.split('/')[-1].split('-')[0]
        if size not in words_dict: continue
        with open(filename, 'r') as f:
            for line in f.readlines():
                words_dict[size].append(line.strip())
    word_dict_ids = {}
    for size in words_dict:
        word_dict_ids[size] = util.tokenizer.convert_tokens_to_ids(words_dict[size])

    # Get the subject words. For each sub, find its category and those word ids that are
    # larger and smaller than the word. 
    subjects = [rel_ins.entities[0] for rel_ins in test_set]
    list_sm_lg = []
    for sub in subjects:
        for i in range(1,4):
            if sub in words_dict[sizes[i]]:
                sm = []
                [sm.extend(word_dict_ids[sz]) for sz in sizes[:i]]
                lg = []
                [lg.extend(word_dict_ids[sz]) for sz in sizes[i+1:]]
                list_sm_lg.append([sm, lg])
                break
    assert len(list_sm_lg) == len(subjects)

    # For each subject, create a dict from comparable tokens to its rank in preds,
    # and sort the keys by ranks, and report the
    # percentage of smaller / larger tokens in the correct half of the ranks.
    percents = []
    id = 0 if 'small' in rel_type else 1
    for i in range(len(subjects)):
        comp_tks = list_sm_lg[i][0] + list_sm_lg[i][1]  # some of these may be [UNK]
        pred_ls = preds[i].tolist()
        rank_dict = {pred_ls.index(tk):tk for tk in comp_tks}
        ranked_tks = [v for (k,v) in sorted(rank_dict.items(), key=lambda item:item[0])]
        tks_sz_half = int(math.ceil(len(ranked_tks)/2))
        overlap_l = np.intersect1d(ranked_tks[:tks_sz_half], list_sm_lg[i][id]).shape[0]
        overlap_r = np.intersect1d(ranked_tks[tks_sz_half:], list_sm_lg[i][1-id]).shape[0]
        # print(overlap_l, overlap_r)
        percents.append((overlap_l + overlap_r) / len(ranked_tks))
    avg_perc = sum(percents) / len(percents)
    print('Percentage of size ranks that are correct:', avg_perc)


def run(lm_name, log_path=''):
    kwargs = read_kwargs()
    cfg= kwargs.pop('trainer')

    lm = construct_lm(**kwargs.pop(lm_name))
    print('lm:', lm.model_type)
    relation_db = load_db_general(**kwargs.get('db'))
    pattern_db = load_templates(**kwargs.pop('template'))

    for rel_type, pb in pattern_db.items():
        #if not rel_type in ['size_smaller', 'size_larger']: continue
        if not rel_type in ['shape']: continue

        splits = list()
        if rel_type not in relation_db['train'].banks:
            continue
        for split in ['train', 'dev', 'test']:
            splits.append(relation_db[split].banks[rel_type])
        train_set, dev_set, test_set = splits

        print(f'rel_type {rel_type} with {len(pb)} patterns.')
        model = PatternModel(
            pb, cfg.get('device'), lm, cfg.get('max_layer'), force_single_token=cfg.get('force_single_token', False),
            vocab_file=cfg.get('vocab_file', None), conditional_prompt=cfg.get('conditional_prompt', False)
        )
        if cfg.get('load_model'):
            filename = os.path.join(log_path, f'model.{rel_type}.pkl')
            best_state = pickle.load(open(filename, 'rb'))
            model.load(best_state)
        preds, answer_ranks, answer_topk, alt_ranks, target_dist = model.conditional_generate_single_slot(
            cfg.get('batch_size_no_grad'), test_set, None
        )
        print(f'Test precision: {int(answer_topk[1]) / len(test_set)}')

        if rel_type in ['size_smaller', 'size_larger']:
            size_correctness(rel_type, test_set, preds)
        else:
            return analyze_distributions(rel_type, target_dist, test_set, answer_ranks, preds)


if __name__ == '__main__':
    # error_tuples, vg_dist_wrong, model_dist_wrong = run('lm', log_path='logs/vl')
    # error_tuples2, vg_dist_wrong2, model_dist_wrong2 = run('lm2', log_path='logs/vl-oscar')
    # error_tuples3, vg_dist_wrong3, model_dist_wrong3 = run('lm3', log_path='logs/vl-dstilbert')
    run('lm', log_path='logs/vl')
    run('lm2', log_path='logs/vl-oscar')
    run('lm3', log_path='logs/vl-dstilbert')
