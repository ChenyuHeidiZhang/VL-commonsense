# https://psyarxiv.com/vyxpq/ experiment 4, for wikipedia data
# Analyze 1st and 2nd order co-occurrence of specific shape / material / color
# with named subjects

import json
import argparse
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from models import load_dist_file, load_word_file

# https://huggingface.co/datasets/wikipedia
wikipedia_dataset = load_dataset('wikipedia', '20200501.en')

def analyze_wiki():
    # Record: {shape: {obj1: {sub1: count, sub2: count, ...}, ...}, ...}
    #   {sub1: {shape: count, material: count, color: count}, ...}
    relations = ['shape', 'material', 'color']
    subjs = []
    for rel in relations:
        subjs.append(set(load_dist_file(rel).keys()))
    noun_set = subjs[0].union(subjs[1]).union(subjs[2])
    print('Num nouns:', len(noun_set))
    target_words = {rel: load_word_file(rel) for rel in relations}
    obj_subj_counts = {rel: {} for rel in relations}
    subj_counts = {}
    i = 0
    for instance in tqdm(wikipedia_dataset['train']):
        i += 1
        if i == 1000000: break
        sents = sent_tokenize(instance['text'])
        for sent in sents:
            tokens = word_tokenize(sent)
            for rel in relations:
                for idx, token in enumerate(tokens):
                    if token in target_words[rel]:
                        if token not in obj_subj_counts[rel]: obj_subj_counts[rel][token] = {}
                        neighbors = tokens[idx-5:idx+5]
                        overlap = list(set(neighbors) & noun_set)
                        for noun in overlap:
                            if noun not in subj_counts:
                                subj_counts[noun] = {rel:0 for rel in relations}
                            subj_counts[noun][rel] += 1
                            if noun not in obj_subj_counts[rel][token]:
                                obj_subj_counts[rel][token][noun] = 0
                            obj_subj_counts[rel][token][noun] += 1

    for rel in relations:
        with open(f'probing/wiki_count_{rel}.jsonl', 'w') as out:
            json.dump(obj_subj_counts[rel], out)
        print('RELATION', rel)
        total = 0
        for obj, sub_counts in obj_subj_counts[rel].items():
            obj_total = sum(sub_counts.values())
            print(obj, obj_total)
            total += obj_total
        print(total)
        print()

    with open('probing/wiki_rel_count.jsonl', 'w') as out:
        json.dump(subj_counts, out)


def load(rel):
    with open(f'probing/wiki_count_{rel}.jsonl', 'r') as f:
        obj_subj_counts = json.load(f)
    totals = []
    for obj, sub_counts in obj_subj_counts.items():
        obj_total = sum(sub_counts.values())
        totals.append(obj_total)
    top18 = np.sort(totals)[-18:]
    print(len(top18))
    print(np.mean(top18))


if __name__ == '__main__':
    #analyze_wiki()
    load('color')
