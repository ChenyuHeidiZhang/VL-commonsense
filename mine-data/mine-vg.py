import json
import random
import os
import numpy as np
import utils

attributes_file = 'attributes.json'

def get_db_from_dist(type, single_slot=True):
    word_ls = np.array(utils.load_word_file(type, single_slot))
    vg_dist_dict = utils.load_dist_file(type)

    i = 0
    splits = ['train', 'test']
    out_files = [f'db/{type}/{sp}.jsonl' for sp in splits]
    with open(out_files[0], 'w') as train_f, open(out_files[1], 'w') as test_f:
        for sub in vg_dist_dict:
            dist = np.array(vg_dist_dict[sub])
            obj = word_ls[np.argmax(dist)]
            alt = random.choice(word_ls[np.where(dist <= np.median(dist))])
            out = train_f if i % 10 < 8 else test_f
            json.dump({"sub": sub, "obj": obj, "alt": alt}, out)
            out.write('\n')
            i += 1

def mine_attributes(type, thres=10, single_slot=True, print_every=1000, max_imgs=1000000):
    '''
    read a list of attributes and find corresponding subjects in Visual Genome
    output obtained pairs to file
    * use `get_db_from_dist` if dist file exists
    '''

    print(f'loading word file of type {type}...')
    word_ls = utils.load_word_file(type, single_slot)
    words_dict = {}
    for word in word_ls: words_dict[word] = []

    print('loading attributes file...')
    attributes_f = json.load(open(attributes_file))

    print('mining attributes...')
    img_num = 0
    att_counts = {}
    for img in attributes_f:
        img_num += 1
        if img_num % print_every == 0:
            print(f'processing {img_num}-th image')
        for obj in img['attributes']:
            if 'attributes' not in obj:
                continue
            for attribute in obj['attributes']:
                att = utils.filter_att(attribute, type)
                # check if the attribute is in our word list
                if att in word_ls:
                    sub = utils.lemmatize(obj['names'][0])
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
    output_atts(type, word_ls, words_dict, att_counts, thres)

def output_atts(type, word_ls, words_dict, att_counts, threshold):
    i = 0
    splits = ['train', 'dev', 'test']
    out_files = [f'db/{type}/{sp}.jsonl' for sp in splits]
    with open(out_files[0], 'w') as train_f, open(out_files[1], 'w') as dev_f, open(out_files[2], 'w') as test_f:
        for key in word_ls:
            for sub in words_dict[key]:
                if att_counts[(sub, key)] > threshold:  # only output when count > threshold
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


def mine_cooccurrence(print_every=1000, max_imgs=10000):
    import itertools
    from collections import defaultdict
    from nltk.corpus import words
    ntlk_words = words.words()

    print('loading attributes file...')
    attributes_f = json.load(open(attributes_file))

    print('mining attributes...')
    noun_set = set()  # all item names in VG
    cooccur_dict = defaultdict(set)  # {sub: [objs]}
    cooccur_counts = defaultdict(int)
    img_num = 0
    for img in attributes_f:
        img_num += 1
        if img_num % print_every == 0:
            print(f'processing {img_num}-th image')
        names = set()
        for obj in img['attributes']:
            name = obj['names'][0]
            #if not any(ele in name for ele in [' ', '.', '/', '\\', '"', '(', '\'', ':']):  # only take single word items
            if name in ntlk_words:  # this is slow
                names.add(name)
        noun_set.update(names)

        for subset in itertools.combinations(names, 2):  # could really use permutation here
            cooccur_dict[subset[0]].add(subset[1])
            cooccur_counts[subset] += 1

        if img_num == max_imgs:
            break
    print(f'finished processing {img_num} images')

    # record VG distribution
    print('recording noun co-occurrrence distribution...')
    noun_list = list(noun_set)
    noun_dist = {}
    for sub in cooccur_dict:
        dist = [cooccur_counts[(sub, obj)] for obj in noun_list]
        noun_dist[sub] = dist
    dist_file = 'cooccur-dist.jsonl'
    with open(dist_file, 'w') as f:
        json.dump(noun_dist, f)

    # record list of nouns
    with open('item-words.txt', 'w') as f:
        for name in noun_list:
            f.write(name)
            f.write('\n')

    # output data
    print('outputting data...')
    i = 0
    splits = ['train', 'dev', 'test']
    out_files = [f'db/cooccur/{sp}.jsonl' for sp in splits]
    with open(out_files[0], 'w') as train_f, open(out_files[1], 'w') as dev_f, open(out_files[2], 'w') as test_f:
        for sub in cooccur_dict.keys():
            alt_list = list(noun_set - cooccur_dict[sub])
            for obj in cooccur_dict[sub]:
                if cooccur_counts[(sub, obj)] <= 5:  # only output when count > threshold
                    continue
                alt = random.choice(alt_list)
                if i % 10 < 8:
                    out = train_f
                elif i % 10 == 8:
                    out = dev_f
                else:
                    out = test_f
                json.dump({"sub": sub, "obj": obj, "alt": alt}, out)
                out.write('\n')
                i += 1


def mine_size(print_every=1000, max_imgs=1000000):
    '''
    Mine size as attributes directly from VG, but those are relative sizes.
    Only used for reference.
    '''
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

def extract_size(smaller=True, num_samples=2000):
    dir = 'size_db/'
    words_dict = {'tiny': [], 'small': [], 'medium': [], 'large': [], 'xlarge': []}
    import os, glob
    for filename in glob.glob(os.path.join(dir, '*.txt')):
        size = filename.split('/')[-1].split('-')[0]
        if size not in words_dict: continue
        with open(filename, 'r') as f:
            for line in f.readlines():
                words_dict[size].append(line.strip())
    # print(words_dict)
    sampled_pairs = {}  # {(sub, obj): alt} dict
    count = num_samples
    sizes = ['tiny', 'small', 'medium', 'large', 'xlarge']
    while count > 0:
        sub_size_id = random.randint(1,3)  # choose int from [1,2,3]
        if smaller:
            obj_size_id = random.randint(sub_size_id+1, 4)
            alt_size_id = random.randint(0, sub_size_id-1)
        else:
            obj_size_id = random.randint(0, sub_size_id-1)
            alt_size_id = random.randint(sub_size_id+1, 4)
        sub = random.choice(words_dict[sizes[sub_size_id]])
        obj = random.choice(words_dict[sizes[obj_size_id]])
        alt = random.choice(words_dict[sizes[alt_size_id]])
        if (sub, obj) not in sampled_pairs:
            sampled_pairs[(sub, obj)] = alt
            count -= 1

    i = 0
    splits = ['train', 'dev', 'test']
    name = 'smaller' if smaller else 'larger'
    out_files = [f'db/size_{name}/{sp}.jsonl' for sp in splits]
    with open(out_files[0], 'w') as train_f, open(out_files[1], 'w') as dev_f, open(out_files[2], 'w') as test_f:
        for pair, alt in sampled_pairs.items():
            if i % 10 < 8:
                out = train_f
            elif i % 10 == 8:
                out = dev_f
            else:
                out = test_f
            json.dump({"sub": pair[0], "obj": pair[1], "alt": alt}, out)
            out.write('\n')
            i += 1


if __name__ == "__main__":
    get_db_from_dist('color')
    #  mine_attributes('color', thres=utils.THRES_COLOR)
    # mine_size()
    # mine_cooccurrence()
    # extract_size(smaller=False)
