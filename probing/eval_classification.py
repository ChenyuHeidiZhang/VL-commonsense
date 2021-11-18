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

from models import init_model

device = "cuda" if torch.cuda.is_available() else "cpu"
template = 'the {} of {}'

# Prepare the inputs
def get_data(file):
    data = []
    objs = set()  # for converting object words to label ids
    with open(file, 'r') as f:
        for line in f.readlines():
            js = json.loads(line)
            objs.add(js['obj'])
            data.append((js['sub'], js['obj']))
    return data, list(objs)

# ds = load_dataset("corypaik/coda", ignore_verifications=True)
# def get_coda_template(sub):
#     rand_idx = np.random.randint(10)
#     for split in ['train', 'validation', 'test']:
#         for instance in ds[split]:
#             if instance['ngram'] == sub and instance['template_group'] == 1:
#                 if rand_idx == instance['template_idx']:
#                     return instance['text']
#     return None

# Calculate features
def get_features(data, objs, relation, model, tokenizer, model_name):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for sub, obj in tqdm(data):
            if obj not in objs:
                #all_labels.append(-1)
                print(f'test obj {obj} not in train objs')
                continue
            all_labels.append(objs.index(obj))
            # text = get_coda_template(sub)
            # if text == None:
            #     text = template.format(relation, sub)
            if model_name=='bert':
                input_ids = torch.tensor([tokenizer.encode(text)]).to(device)
                emb = model.embeddings(input_ids=input_ids)
                features = torch.mean(model.encoder(emb)[0], dim=1)
            elif model_name=='oscar':
                input_ids = torch.tensor([tokenizer.encode(text)]).to(device)
                output = model(input_ids=input_ids)
                features = torch.mean(output, dim=1)
            elif model_name=='clip':
                text_input = clip.tokenize(text).to(device)
                features = model.encode_text(text_input)
            else:
                print(f'wrong model name {model_name}')
            all_features.append(features)

    return torch.cat(all_features).cpu().numpy(), np.array(all_labels)

def get_word_ids(rel_name, objs):
    word_ls = []
    with open(f'mine-data/words/{rel_name}-words.txt', 'r') as f:
        for line in f.readlines():
            if len(line.strip().split()) == 1:
                word_ls.append(line.strip())
    word_ids = []
    for object in objs:
        word_ids.append(word_ls.index(object))
    return word_ids

def run():
    parser = argparse.ArgumentParser(description='eval parser')
    parser.add_argument('--model', type=str, default='bert',
                        help='name of the model (bert, oscar, or clip)')
    parser.add_argument('--model_size', type=str, default='base',
                        help='size of the model (base, large)')
    parser.add_argument('--relation', type=str, default='shape',
                        help='relation to evaluate (shape, material, color, coda, coda_any...)')
    parser.add_argument('--group', type=str, default='',
                        help='group to evaluate (single, multi, any, or '' for all))')
    parser.add_argument('--seed', type=int, default=1,
                        help='numpy random seed')
    parser.add_argument('--step', type=int, default=-1,
                        help='step size of increasing training size')
    args = parser.parse_args()
    np.random.seed(args.seed)

    # Load the model
    model_name = args.model
    model, tokenizer = init_model(model_name, args.model_size, device)
    relation = args.relation
    group = args.group
    train_data, objs = get_data(f'mine-data/db/{relation}/{group}/train.jsonl')
    test_data, _ = get_data(f'mine-data/db/{relation}/{group}/test.jsonl')

    test_ids = []  # ids of test objs that are in train objs
    for i in range(len(test_data)):
        if test_data[i][1] in objs:
            test_ids.append(i)

    train_features_all, train_labels_all = get_features(train_data, objs, relation, model, tokenizer, model_name)
    #print(train_features.shape)  # (num_examples, 512 CLIP or 768 Bert)
    test_features, test_labels = get_features(test_data, objs, relation, model, tokenizer, model_name)
    print('total num classes:', len(objs))

    rel_name = relation.split('_')[0]
    dist_file = f'mine-data/distributions/{rel_name}-dist.jsonl'
    dist_dict = json.load(open(dist_file, 'r'))

    step = len(train_labels_all) if args.step == -1 else args.step    
    for train_data_size in range(step, len(train_labels_all)+1, step):
        print()
        p = np.random.permutation(len(train_labels_all))
        train_features = train_features_all[p][:train_data_size]
        train_labels = train_labels_all[p][:train_data_size]
        train_ids = np.unique(train_labels)
        print('num train classes:', len(train_ids))
        print(f'num train examples: {len(train_labels)}, num test examples: {len(test_labels)}')

        # Perform logistic regression
        classifier = LogisticRegression(random_state=0, C=0.316, max_iter=2000, verbose=0)
        classifier.fit(train_features, train_labels)

        # Evaluate using the logistic regression classifier
        predictions = classifier.predict(test_features)
        accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
        print(f"Accuracy = {accuracy:.3f}")

        # Evaluate correlation of regression logprobs with true distributions
        logprob = classifier.predict_proba(test_features)
        # print(logprob.shape)  # num_test_examples, num_objs

        word_ids = get_word_ids(rel_name, [objs[id] for id in train_ids])
        sp_corrs = []
        idx = 0
        for i in test_ids:
            true_dist = np.take(dist_dict[test_data[i][0]], word_ids)
            sp = spearmanr(true_dist, logprob[idx])[0]
            # sp = (kendalltau(true_dist, logprob[idx])[0] + sp) / 2
            idx += 1
            if np.sum(true_dist) != 0: sp_corrs.append(sp)
        sp_corr = np.mean(sp_corrs)
        print('avg sp corr:', sp_corr)


if __name__ == '__main__':
    run()
