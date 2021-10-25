import numpy as np
import json
import random

attributes_file = 'attributes.json'

def mine_attributes(type, single_slot=True, print_every=1000, max_imgs=1000000):
    # read a list of attributes and find corresponding subjects in Visual Genome
    # output obtained pairs to file
    words_file = f'{type}-words.txt'

    print(f'loading word file of type {type}...')
    words_f = open(words_file, 'r').readlines()
    words_dict = {}
    for line in words_f:
        if single_slot and len(line.strip().split()) > 1:
            continue
        words_dict[line.strip()] = []
    word_ls = words_dict.keys()
    att_counts = {}

    print('loading attributes file...')
    attributes_f = json.load(open(attributes_file))

    print('mining attributes...')
    img_num = 0
    for img in attributes_f:
        img_num += 1
        if img_num % print_every == 0:
            print(f'processing {img_num}-th image')
        for obj in img['attributes']:
            if 'attributes' not in obj:
                continue
            for attribute in obj['attributes']:
                att = attribute
                # treat attribute "x colored" the same as "x"
                if type == 'color' and att.split()[-1] == 'colored' and att != 'light colored' and att != 'dark colored':
                    att = ' '.join(att.split()[:-1])
                # check if the attribute is in our word list
                if att in word_ls:
                    sub = obj['names'][0]
                    if att in sub: 
                        continue  # skip cases when the subject is something like "red box" 
                    if sub not in words_dict[att]:  # don't append the same attribute twice
                        words_dict[att].append(sub)
                        att_counts[(sub, att)] = 1
                    else:
                        att_counts[(sub, att)] += 1
        if img_num == max_imgs:
            break
    print(f'finished processing {img_num} images')

    # output
    i = 0
    splits = ['train', 'dev', 'test']
    out_files = [f'db/{type}/{sp}.jsonl' for sp in splits]
    with open(out_files[0], 'w') as train_f, open(out_files[1], 'w') as dev_f, open(out_files[2], 'w') as test_f:
        for key in word_ls:
            for sub in words_dict[key]:
                if att_counts[(sub, key)] > 2:  # only output when count > threshold
                    alt = random.choice(list(word_ls))
                    while (sub in words_dict[alt]):
                        alt = random.choice(list(word_ls))
                    if i % 10 < 8:
                        out = train_f
                    elif i % 10 == 8:
                        out = dev_f
                    else:
                        out = test_f
                    json.dump({"sub": sub, "obj": key, "alt": alt, "count": att_counts[(sub, key)]}, out)
                    out.write('\n')
                    i += 1


def mine_cooccurrence(print_every=1000, max_imgs=1000000):
    print('loading attributes file...')
    attributes_f = json.load(open(attributes_file))

    print('mining attributes...')
    out_file = open('co-occurrence_mined.jsonl', 'w')
    #cooccur_counts = {}
    img_num = 0
    for img in attributes_f:
        img_num += 1
        if img_num % print_every == 0:
            print(f'processing {img_num}-th image')
        names = []
        for obj in img['attributes']:
            names.append(obj['names'][0])
        json.dump({'obj_names': names}, out_file)
        out_file.write('\n')
        if img_num == max_imgs:
            break
    out_file.close()
    print(f'finished processing {img_num} images')


def mine_size(print_every=1000, max_imgs=1000000):
    print('loading attributes file...')
    attributes_f = json.load(open(attributes_file))

    print('mining attributes...')
    out_file = open('size_att_mined.jsonl', 'w')
    img_num = 0
    for img in attributes_f:
        img_num += 1
        if img_num % print_every == 0:
            print(f'processing {img_num}-th image')
        for obj in img['attributes']:
            if 'attributes' not in obj:
                continue
            for attribute in obj['attributes']:
                if attribute in ['tiny', 'small', 'medium', 'large', 'big', 'huge', 'enormous']:
                    json.dump({'size': attribute, 'obj': obj['names'][0]}, out_file)
                    out_file.write('\n')
        if img_num == max_imgs:
            break
    out_file.close()
    print(f'finished processing {img_num} images')

def extract_size():
    pass


if __name__ == "__main__":
    # mine_attributes('material')
    # mine_size()
    mine_cooccurrence()
