# evaluate visual commonsense
# run in the parent directory

import os
import json
import numpy as np
from scipy.stats.stats import kendalltau, pearsonr, spearmanr
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import clip
from datasets import load_dataset
import torch
import argparse

from models import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_features(data, objs, template, model, tokenizer, model_name):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for sub, obj in tqdm(data):
            if obj not in objs:
                #all_labels.append(-1)
                print(f'test obj {obj} not in train objs')
                continue
            all_labels.append(objs.index(obj))
            text = template.replace('[X]', sub)
            if model_name=='bert':
                input_ids = torch.tensor([tokenizer.encode(text)]).to(device)
                emb = model.embeddings(input_ids=input_ids)
                features = torch.mean(model.encoder(emb)[0], dim=1)
            elif model_name=='oscar':
                input_ids = torch.tensor([tokenizer.encode(text)]).to(device)
                output = model(input_ids=input_ids)
                features = torch.mean(output[0], dim=1)
            elif model_name=='clip':
                text_input = clip.tokenize(text).to(device)
                features = model.encode_text(text_input)
            else:
                print(f'wrong model name {model_name}')
            all_features.append(features)

    return torch.cat(all_features).cpu().numpy(), np.array(all_labels)

def get_word_ids(relation, objs):
    word_ls = load_word_file(relation)
    word_ids = []
    for object in objs:
        word_ids.append(word_ls.index(object))
    return word_ids

def run(args):
    '''
    Returns the best test set accuracy and sp correlation across all templates.
    '''
    # Load the model
    model_name = args.model
    model, tokenizer = init_model(model_name, args.model_size, device)

    # Prepare the inputs
    relation = args.relation
    group = args.group
    train_data, objs = load_data(f'mine-data/db/{relation}/{group}/train.jsonl')
    test_data, _ = load_data(f'mine-data/db/{relation}/{group}/test.jsonl')
    print('total num classes:', len(objs))
    rel_name = 'color' if relation == 'coda' else relation
    templates = [f'the {rel_name} of [X]'] if args.single_prompt else load_prompts(relation)

    test_ids = []  # ids of test objs that are in train objs; those that are not are excluded when getting test features
    for i in range(len(test_data)):
        if test_data[i][1] in objs:
            test_ids.append(i)

    dist_dict = load_dist_file(relation)

    # Calculate features
    train_all_temps = []
    test_all_temps = []
    for template in templates:
        print(template)
        train_features_all, train_labels_all = get_features(train_data, objs, template, model, tokenizer, model_name)
        #print(train_features_all.shape)  # (num_examples, 512 CLIP or 768 Bert)
        test_features, test_labels = get_features(test_data, objs, template, model, tokenizer, model_name)
        train_all_temps.append((train_features_all, train_labels_all))
        test_all_temps.append((test_features, test_labels))

    step = len(train_labels_all) if args.step == -1 else args.step    
    for train_data_size in range(step, len(train_labels_all)+1, step):
        print()
        acc_all_temps = []
        corr_all_temps = []
        corr_stds = []
        sp_per_obj_all = []
        for i in range(len(templates)):  # still, same template per test set
            train_features = train_all_temps[i][0][:train_data_size]
            train_labels = train_all_temps[i][1][:train_data_size]
            test_features = test_all_temps[i][0]
            test_labels = test_all_temps[i][1]
            train_ids = np.unique(train_labels)
            print('num train classes:', len(train_ids))
            print(f'num train examples: {len(train_labels)}, num test examples: {len(test_labels)}')

            # Perform logistic regression
            classifier = LogisticRegression(random_state=0, C=0.316, max_iter=2000, verbose=0)
            classifier.fit(train_features, train_labels)

            # Evaluate using the logistic regression classifier
            predictions = classifier.predict(test_features)
            accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
            acc_all_temps.append(accuracy)
            print(f"Accuracy = {accuracy:.3f}")

            # Evaluate correlation of regression logprobs with true distributions
            logprob = classifier.predict_proba(test_features)
            # print(logprob.shape)  # num_test_examples, num_objs

            word_ids = get_word_ids(relation, [objs[id] for id in train_ids])
            sp_corrs = []
            sp_per_obj = [[] for _ in range(len(objs))]
            #idx = 0
            for idx, i in enumerate(test_ids):
                true_dist = np.take(dist_dict[test_data[i][0]], word_ids)
                if np.sum(true_dist) != 0:
                    sp = spearmanr(true_dist, logprob[idx])[0]
                    # sp = (kendalltau(true_dist, logprob[idx])[0] + sp) / 2
                    sp_corrs.append(sp)
                    sp_per_obj[test_labels[idx]].append(sp)
                #idx += 1
            sp_corr = np.mean(sp_corrs)
            corr_all_temps.append(sp_corr)
            corr_stds.append(np.std(sp_corrs))
            sp_per_obj_all.append(sp_per_obj)
            print('avg sp corr:', sp_corr)  # avg across all test examples for one template
        print('best acc across all templates:', np.max(acc_all_temps))
        max_id = np.argmax(corr_all_temps)
        print('best sp corr across all templates:', corr_all_temps[max_id])
        sp_per_obj_max = sp_per_obj_all[max_id]
        #avg_per_obj = plot_corr(sp_per_obj_max, objs, model_name, rel_name)
    return sp_per_obj_max, objs, \
        round(corr_all_temps[max_id], 3), round(corr_stds[max_id], 3), round(np.max(acc_all_temps), 1)#, avg_per_obj

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='eval parser')
    # parser.add_argument('--model', type=str, default='bert',
    #                     help='name of the model (bert, oscar, or clip)')
    # parser.add_argument('--model_size', type=str, default='base',
    #                     help='size of the model (base, large)')
    # parser.add_argument('--relation', type=str, default='shape',
    #                     help='relation to evaluate (shape, material, color, coda, cooccur)')
    # parser.add_argument('--group', type=str, default='',
    #                     help='group to evaluate (single, multi, any, or '' for all))')
    # parser.add_argument('--seed', type=int, default=1,
    #                     help='numpy random seed')
    # parser.add_argument('--step', type=int, default=-1,
    #                     help='step size of increasing training size')
    # parser.add_argument('--single_prompt', type=bool, default=False,
    #                     help='whether to use a single prompt')
    # args = parser.parse_args()
    # sp_mean, sp_std, acc = run(args)
    # print(sp_std)

    rel_types = ['color', 'shape', 'material']  # 
    models = ['bert', 'oscar', 'clip']  # 
    d = {rel: [] for rel in rel_types}
    sps_per_obj = {rel: [] for rel in rel_types}
    rel_objs = {}
    for relation in rel_types:
        #sps_per_obj_rel = []
        print(relation)
        for group in ['single', 'multi', 'any']:  # , 'single', 'multi', 'any'
            for model in models:
                args = Args(model, relation, group)
                np.random.seed(args.seed)
                sp_per_obj, objs, sp_mean, sp_std, acc = run(args)
                d[relation].append((sp_mean, sp_std, acc))
                #print(sp_per_obj)
                #sps_per_obj[relation].append(sp_per_obj)
                # rel_objs[relation] = objs
                # sps_per_obj_rel.append(sp_per_obj)
            # print('bert vs. oscar', spearmanr(sps_per_obj_rel[0], sps_per_obj_rel[1]))
            # print('bert vs. clip', spearmanr(sps_per_obj_rel[0], sps_per_obj_rel[2]))
            # print('oscar vs. clip', spearmanr(sps_per_obj_rel[1], sps_per_obj_rel[2]))

    import pandas as pd
    df = pd.DataFrame(d)
    df.to_csv('file-cls-groups.csv')
    print(df)

    #plot_corr_all_rels(sps_per_obj, models, rel_types, method='cls', rel_objs=rel_objs)

# ,color,shape,material
# 0,"(0.48, 0.216, 51.4)","(0.532, 0.134, 78.4)","(0.413, 0.156, 51.1)"
# 1,"(0.519, 0.208, 63.8)","(0.544, 0.139, 79.9)","(0.429, 0.15, 63.0)"
