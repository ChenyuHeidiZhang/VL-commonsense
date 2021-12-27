from ..model import PatternModel
from ..io import load_db_general
from ..corpus import load_templates
from ..lm import construct_lm
from .experiment import read_kwargs
from .. import util

import csv
import math
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr
import torch
import json
import os, glob
import pickle
import logging
logger = logging.getLogger(__name__)

#relations = ['size_smaller', 'size_larger']
relations = ['color', 'shape', 'material'] #, 'shape', 'material', 'cooccur'
verbose = False  # verbose=True only available for non-size relations, one at a time

def load_obj_file(rel_type):
    objs_file = f'/home/heidi/VL-commonsense/mine-data/words/{rel_type}-words.txt'
    objs_ls = []
    with open(objs_file, 'r') as f:
        for line in f.readlines():
            if len(line.strip().split()) > 1:
                continue
            objs_ls.append(line.strip())
    return objs_ls

def get_correlation(vg_dist, model_dist):
    # Compute correlation between the VG distribution and model distribution
    corr_sp = [spearmanr(vg_dist[i], model_dist[i]) for i in range(vg_dist.shape[0])]
    corrs = [cor[0] for cor in corr_sp]
    print(np.mean(corrs), np.std(corrs))

    indices = np.argsort(corrs)[::-1]  # indices sorted and reversed
    return np.array(corr_sp), indices

def analyze_distributions(rel_type, target_dist, test_set, answer_ranks, preds, model_name, ptune):
    # get the list of object ids, and find model distribution across the objects
    objs_ls = load_obj_file(rel_type)
    obj_ids = torch.tensor(util.tokenizer.convert_tokens_to_ids(objs_ls))

    model_dist = np.array(torch.index_select(target_dist, 1, obj_ids))  # keeps the order of obj_ids
    # print(model_dist.shape)  # [num_test_ins, num_objs]

    # load Visual Genome distribution across objects for the test subjects
    subjects = [rel_ins.entities[0] for rel_ins in test_set]
    # print(len(subjects))  # [69]
    dist_file = f'/home/heidi/VL-commonsense/mine-data/distributions/{rel_type}-dist.jsonl'
    vg_dist_dict = json.load(open(dist_file, 'r'))
    vg_dist = np.array([vg_dist_dict[sub] for sub in subjects])

    print('Correlation of VG and model distributions:')
    corr_sp, indices = get_correlation(vg_dist, model_dist)
    print('High correlation subjects:', [subjects[i] for i in indices[:10]])
    # print('Corr and p-val:', [corr_sp[i] for i in indices[:10]])
    print('Low correlation subjects:', [subjects[i] for i in indices[-10:]])
    # print('Corr and p-val:', [corr_sp[i] for i in indices[-10:]])

    # record correlation per object
    with open('pt_corrs.csv', 'a', encoding='UTF8', newline='') as outfile:
        writer = csv.writer(outfile)
        for i, sub in enumerate(subjects):
            writer.writerow([model_name, rel_type, sub, ptune, corr_sp[i][0]])

    # answer_tokens = [rel_ins.entities[1] for rel_ins in test_set]
    # predicted_tokens = util.tokenizer.convert_ids_to_tokens(preds.T[0])
    # # Get model distribution and VG distribution for instances where answer_rank > 3,
    # # compute the correlation of two distributions,
    # # print the error (sub, true_target, predicted) tuple,
    # # return the error tuples and model distributions for cross-model comparisons.
    # idxs = torch.nonzero(answer_ranks > 3).squeeze(1)
    # print(f'predictions are off for {len(idxs)} out of {len(test_set)} instances')
    # vg_dist_wrong = vg_dist[idxs]
    # model_dist_wrong = model_dist[idxs]
    # print('Correlation of distributions when prediction is off:')
    # corr_sp_off, indices = get_correlation(vg_dist_wrong, model_dist_wrong)

    # error_tuples = []
    # for idx in idxs:
    #     error_tuples.append((subjects[idx], answer_tokens[idx], predicted_tokens[idx]))
    # #print('\n'.join([', '.join(tup) for tup in error_tuples]))

    # #return error_tuples, vg_dist_wrong, model_dist_wrong
    return corr_sp, model_dist, subjects

def calculate_acc(rel_type, test_set, target_dist):
    # get the object list of the relation
    # if max of output prob over the object list equals prob of the true target, 
    # count as correct
    objs_ls = load_obj_file(rel_type)
    obj_ids = torch.tensor(util.tokenizer.convert_tokens_to_ids(objs_ls))
    model_dist = torch.index_select(target_dist, 1, obj_ids)  # keeps the order of obj_ids

    dist_of_target = []
    for i, rel_ins in enumerate(test_set):
        obj_id = objs_ls.index(rel_ins.entities[1])
        dist_of_target.append(model_dist[i, obj_id].item())
    correct = torch.tensor(dist_of_target) == torch.max(model_dist, dim=1)[0]
    perc = correct.count_nonzero() / len(test_set)
    print('Test precision among target objects:', perc.item())


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
                list_sm_lg.append([np.unique(sm).tolist(), np.unique(lg).tolist()])
                break
    assert len(list_sm_lg) == len(subjects)

    # For each subject, create a dict from comparable tokens to its rank in preds,
    # and sort the keys by ranks, and report the
    # percentage of smaller / larger tokens in the correct half of the ranks.
    percents = []
    id = 0 if 'small' in rel_type else 1
    for i in range(len(subjects)):
        comp_tks = list_sm_lg[i][0] + list_sm_lg[i][1]  # one of each may be [UNK] (100)
        pred_ls = preds[i].tolist()
        rank_dict = {pred_ls.index(tk):tk for tk in comp_tks}
        ranked_tks = [v for (k,v) in sorted(rank_dict.items(), key=lambda item:item[0])]
        # print(len(comp_tks), len(ranked_tks))
        tks_sz_half = len(list_sm_lg[i][id])
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
        if not rel_type in relations: continue

        splits = list()
        if rel_type not in relation_db['train'].banks:
            continue
        print(f'Relation Type {rel_type}')
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
            calculate_acc(rel_type, test_set, target_dist)
            x,y,z = analyze_distributions(rel_type, target_dist, test_set, answer_ranks, preds, f"{lm.model_type}-{lm.model_size}", cfg.get('load_model'))
            if verbose:
                return x,y,z

def cross_model_corr(corr_sp1, corr_sp2, subjects):
    corr1 = [corr[0] for corr in corr_sp1]
    corr2 = [corr[0] for corr in corr_sp2]
    cor_corr = spearmanr(corr1, corr2)
    first_better = []
    second_better = []
    for i in range(len(corr1)):
        if corr1[i] - corr2[i] > 0.1:
            first_better.append(subjects[i])
        elif corr2[i] - corr1[i] > 0.1:
            second_better.append(subjects[i])
    return cor_corr, first_better, second_better

def common_subs(subs1, subs2, subs3):
    idx = np.argmin([len(subs1), len(subs2), len(subs3)])
    sub_lst = [subs1, subs2, subs3]
    subs = sub_lst[idx]
    indices = []
    for i in range(3):
        if i == idx: 
            indices.append(np.ones(len(subs), dtype=bool))
        else:
            idx_lst = []
            k = 0
            for subj in sub_lst[i]:
                if subj == subs[k]:
                    idx_lst.append(True)
                    k += 1
                else:
                    idx_lst.append(False)
            indices.append(np.array(idx_lst))
    return subs, indices

if __name__ == '__main__':
    if verbose:
        corr_sp1, model_dist1, subjects1 = run('lm', log_path='logs/vl')
        corr_sp2, model_dist2, subjects2 = run('lm2', log_path='logs/vl-oscar')
        corr_sp3, model_dist3, subjects3 = run('lm3', log_path='logs/vl-dstilbert')
        # corr_sp4, model_dist4, subjects4 = run('lm4', log_path='logs/vl-roberta')
        if not (len(subjects1) == len(subjects2) and len(subjects2) == len(subjects3)):
            subjects, indices = common_subs(subjects1, subjects2, subjects3)
            corr_sp1 = corr_sp1[indices[0]]
            corr_sp2 = corr_sp2[indices[1]]
            corr_sp3 = corr_sp3[indices[2]]
            model_dist1 = model_dist1[indices[0]]
            model_dist2 = model_dist2[indices[1]]
            model_dist3 = model_dist3[indices[2]]
            print('obtained common subjects:', len(subjects))
        else:
            subjects = subjects1
            print('num subjects:', len(subjects))

        cor_corr12, bo_bert, bo_oscar = cross_model_corr(corr_sp1, corr_sp2, subjects)
        print()
        print('Corr of bert & oscar correlations:', cor_corr12)
        print('Mean and var of corr of bert & oscar dists:')
        _, _ = get_correlation(model_dist1, model_dist2)
        print(bo_bert)
        print(bo_oscar)
        # cor_corr23, od_oscar, od_distil = cross_model_corr(corr_sp2, corr_sp3, subjects)
        # print('Corr of oscar & distil_bert correlations:', cor_corr23)
        # print('Mean and var of corr of oscar & distil_bert dists:')
        # _, _ = get_correlation(model_dist2, model_dist3)
        # print(od_oscar)
        # print(od_distil)
        # cor_corr13, bd_bert, bd_distil = cross_model_corr(corr_sp1, corr_sp3, subjects)
        # print('Corr of bert & distil_bert correlations:', cor_corr13)
        # print('Mean and var of corr of bert & distil_bert dists:')
        # _, _ = get_correlation(model_dist1, model_dist3)
        # print(bd_bert)
        # print(bd_distil)
    else:
        run('lm', log_path='logs/vl-bert')
        run('lm2', log_path='logs/vl-oscar')
        #run('lm3', log_path='logs/vl-distilbert')
        #run('lm4')
