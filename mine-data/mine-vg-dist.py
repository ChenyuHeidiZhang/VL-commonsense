import numpy as np
import json
import utils

attributes_file = 'attributes.json'

def mine_attribute_dist(type, thres, single_slot=True, print_every=1000, max_imgs=1000000):
    # read a list of attributes and find corresponding subjects in Visual Genome
    # output obtained pairs to file

    print(f'loading objects file of type {type}...')
    objs_ls, obj_map = utils.load_word_file(type, single_slot)
    sub_dist = {}

    print('loading attributes file...')
    attributes_f = json.load(open(attributes_file))

    print('mining attribute distribution...')
    img_num = 0
    for img in attributes_f:
        img_num += 1
        if img_num % print_every == 0:
            print(f'processing {img_num}-th image')
        for obj in img['attributes']:
            if 'attributes' not in obj:
                continue
            subj_atts = []
            for attribute in obj['attributes']:
                att = utils.filter_att(attribute, obj_map)
                if att in objs_ls:
                    sub = utils.lemmatize(obj['names'][0])
                    if att in sub or (att == 'gray' and 'grey' in sub): 
                        continue  # skip cases when the subject is something like "red box"
                    if att in subj_atts: continue # skip if the attribute is duplicate for the current noun
                    subj_atts.append(att)
                    if sub not in sub_dist:
                        sub_dist[sub] = [0] * len(objs_ls)
                    sub_dist[sub][objs_ls.index(att)] += 1
        if img_num == max_imgs:
            break
    print(f'finished processing {img_num} images')

    # remove the distribution if the sum of occurrences is <= the threshold.
    to_remove = []
    for sub in sub_dist:
        sub_dist[sub] = np.multiply(sub_dist[sub], np.array(sub_dist[sub])>thres).tolist()
        if np.sum(sub_dist[sub]) == 0:
            to_remove.append(sub)
    for sub in to_remove: del sub_dist[sub]

    # output
    out_file = f'distributions/{type}-dist.jsonl'
    with open(out_file, 'w') as out:
        json.dump(sub_dist, out)


def mine_cooccurrence_dist(thres, print_every=1000, max_imgs=100000):
    import itertools
    from collections import defaultdict

    print('loading attributes file...')
    attributes_f = json.load(open(attributes_file))

    print('mining attributes...')
    noun_list = utils.load_word_file('cooccur')[0]  # all item names in VG
    cooccur_dict = defaultdict(set)  # {sub: [objs]}
    cooccur_counts = defaultdict(int)
    img_num = 0
    for img in attributes_f:
        img_num += 1
        if img_num % print_every == 0:
            print(f'processing {img_num}-th image')
        names = set()
        for obj in img['attributes']:
            name = utils.lemmatize(obj['names'][0])
            if name in noun_list: names.add(name)
        
        for subset in itertools.combinations(names, 2):
            cooccur_dict[subset[0]].add(subset[1])
            cooccur_counts[subset] += 1

        if img_num == max_imgs:
            break
    print(f'finished processing {img_num} images')

    # record VG distribution
    print('recording noun co-occurrrence distribution...')
    print(len(noun_list))
    noun_dist = {}
    for sub in cooccur_dict:
        dist = [cooccur_counts[(sub, obj)] for obj in noun_list]
        dist = np.multiply(dist, np.array(dist)>thres).tolist()
        if np.sum(dist) > 0:
            noun_dist[sub] = dist
    dist_file = 'distributions/cooccur-dist.jsonl'
    with open(dist_file, 'w') as f:
        json.dump(noun_dist, f)


def check_vg_atts(print_every=1000):
    print('loading attributes file...')
    attributes_f = json.load(open(attributes_file))
    att_counts = {}

    print('mining attribute distribution...')
    img_num = 0
    for img in attributes_f:
        img_num += 1
        if img_num % print_every == 0:
            print(f'processing {img_num}-th image')
        for obj in img['attributes']:
            if 'attributes' not in obj:
                continue
            for attribute in obj['attributes']:
                if attribute in att_counts:
                    att_counts[attribute] += 1
                else:
                    att_counts[attribute] = 1
    print(f'finished processing {img_num} images')
    with open('words/vg_attribute_count', 'w') as out:
        for att, count in att_counts.items():
            json.dump({att: count}, out)
            out.write('\n')


if __name__ == "__main__":
    mine_attribute_dist('material', thres=utils.THRES_MATERIAL)
    mine_cooccurrence_dist(thres=utils.THRES_COOCCUR)
    # check_vg_atts()
