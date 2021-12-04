
import utils

import json
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
Lem = WordNetLemmatizer()

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})  # prevent x labels being cut off

relations = ['color', 'shape', 'material']
templates = {
    'color': ['Y X', 'X is Y', 'X are Y'],  # (e.g. white cat, cat is white)
    'shape': ['Y X', 'X is Y', 'X are Y'],
    'material': ['Y X', 'X is made of Y', 'X are made of Y']
}


def get_nouns_set():
    # get the set of nouns in all three relations in VG
    subjs = []
    for rel in relations:
        subjs.append(set(utils.load_dist_file(rel).keys()))
    print(len(subjs[0]), len(subjs[1]), len(subjs[2]))
    noun_set = subjs[0].union(subjs[1]).union(subjs[2])
    return noun_set


def fit_template(template, idx, tokens, noun_set):
    template_tokens = word_tokenize(template)
    x_pos = template_tokens.index('X')
    y_pos = template_tokens.index('Y')
    if x_pos > y_pos and idx+(x_pos-y_pos) < len(tokens):
        after_word = Lem.lemmatize(tokens[idx+(x_pos-y_pos)])
        #if nltk.pos_tag([after_word])[0][1] == 'NN':
        if after_word in noun_set:
            fit = True
            for i in range(y_pos+1, x_pos):
                if template_tokens[i] != tokens[idx+(i-y_pos)]: fit = False
            if fit == True: return True, after_word
    elif x_pos < y_pos and idx-(y_pos-x_pos) >= 0:
        before_word = Lem.lemmatize(tokens[idx-(y_pos-x_pos)])
        #if nltk.pos_tag([before_word])[0][1] == 'NN':
        if before_word in noun_set:
            fit = True
            for i in range(x_pos+1, y_pos):
                if template_tokens[i] != tokens[idx-(y_pos-i)]: fit = False
            if fit == True: return True, before_word
    return False, None


def mine_dist(dataset, data_name, target_words):
    '''
    dists: {color: {noun1: [1,3,2,0,...]}}
    '''
    dists = {rel: {} for rel in relations}
    noun_set = get_nouns_set()
    num = 0
    for instance in tqdm(dataset):
        num += 1
        if num == 10000: break
        sents = sent_tokenize(instance['text'])
        for sent in sents:
            tokens = word_tokenize(sent)
            for idx, token in enumerate(tokens):
                for rel in relations:
                    if token not in target_words[rel]: continue
                    for temp in templates[rel]:
                        fit, word = fit_template(temp, idx, tokens, noun_set)
                        if fit:
                            if word not in dists[rel]:
                                dists[rel][word] = [0] * len(target_words[rel])
                            dists[rel][word][target_words[rel].index(token)] += 1

    for rel in relations:
        with open(f'distributions/{data_name}-{rel}-dist.jsonl', 'w') as out:
            print(rel, len(dists[rel].keys()))
            # color: 1765, shape: 1329, material: 1693
            json.dump(dists[rel], out)

    return dists


def combine_dists(dist1, dist2):
    dist_combined = {rel: {} for rel in relations}
    for rel in dist1:
        for word in dist1[rel]:
            if word in dist2[rel]:
                joint = np.array(dist1[rel][word]) + np.array(dist2[rel][word])
                dist_combined[rel][word] = joint.tolist()
            else:
                dist_combined[rel][word] = dist1[rel][word]
        for word in dist2[rel]:
            if word not in dist_combined[rel]:
                dist_combined[rel][word] = dist2[rel][word]
    
    for rel in relations:
        with open(f'distributions/text-{rel}-dist.jsonl', 'w') as out:
            json.dump(dist_combined[rel], out)


def get_db_from_dist(type):
    word_ls = np.array(utils.load_word_file(type))
    dist_dict = utils.load_dist_file(f'wiki-{type}')

    i = 0
    splits = ['train', 'test']
    out_files = [f'db/wiki-{type}/{sp}.jsonl' for sp in splits]
    with open(out_files[0], 'w') as train_f, open(out_files[1], 'w') as test_f:
        for sub in dist_dict:
            dist = np.array(dist_dict[sub])
            obj = word_ls[np.argmax(dist)]
            alt = random.choice(word_ls[np.where(dist <= np.median(dist))])
            out = train_f if i % 10 < 8 else test_f
            json.dump({"sub": sub, "obj": obj, "alt": alt}, out)
            out.write('\n')
            i += 1


def get_att_counts(word_ls, dist_dict, verbose=False):
    sum_dist = np.zeros(len(word_ls))

    for noun, dist in dist_dict.items():
        sum_dist = sum_dist + np.array(dist)

    if verbose:
        for i in range(len(word_ls)):
            print(word_ls[i], sum_dist[i])
    return sum_dist / np.sum(sum_dist)

def plot_count_comparison(relation):
    word_ls = utils.load_word_file(relation)
    dist_dict = utils.load_dist_file(f'wiki-{relation}')
    vg_dist_dict = utils.load_dist_file(relation)
    sum_dist = get_att_counts(word_ls, dist_dict)
    vg_sum_dist = get_att_counts(word_ls, vg_dist_dict)

    x_ind = range(len(word_ls))
    plt.bar(x_ind, sum_dist, alpha=0.5, label='wiki')
    plt.bar(x_ind, vg_sum_dist, alpha=0.5, label='VG')
    plt.xticks(x_ind, word_ls, rotation='vertical')
    plt.xlabel(f'{relation}s')
    plt.ylabel('count')
    plt.legend()
    plt.savefig(f'{relation}_att_counts.png')


def run():
    # Wikipedia: https://huggingface.co/datasets/wikipedia
    wikipedia_dataset = load_dataset('wikipedia', '20200501.en')['train']
    # Bookcorpus: https://huggingface.co/datasets/bookcorpus
    # books_dataset = load_dataset('bookcorpus')['train']

    target_words = {rel: utils.load_word_file(rel) for rel in relations}

    wiki_dists = mine_dist(wikipedia_dataset, 'wiki', target_words)
    # books_dists = mine_dist(books_dataset, 'books', target_words)
    # combine_dists(wiki_dists, books_dists)


if __name__ == '__main__':
    # run()
    # get_db_from_dist('material')
    plot_count_comparison('material')
