import numpy as np
import json
import utils

attributes_file = 'attributes.json'

def mine_attribute_dist(type, thres=2, single_slot=True, print_every=1000, max_imgs=1000000):
    # read a list of attributes and find corresponding subjects in Visual Genome
    # output obtained pairs to file

    print(f'loading objects file of type {type}...')
    objs_ls = utils.load_word_file(type, single_slot)
    sub_dist = {}
    # obj_dist = {}
    # for obj in objs_ls: obj_dist[obj] = 0

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
            for attribute in obj['attributes']:
                att = utils.filter_att(attribute, type)
                # check if the attribute is in our word list
                if att in objs_ls:
                    sub = utils.lemmatize(obj['names'][0])
                    if att in sub: 
                        continue  # skip cases when the subject is something like "red box"
                    if sub not in sub_dist:
                        # sub_dist[sub] = np.zeros(len(objs_ls))
                        sub_dist[sub] = [0] * len(objs_ls)
                    sub_dist[sub][objs_ls.index(att)] += 1
                    # obj_dist[att] += 1
        if img_num == max_imgs:
            break
    print(f'finished processing {img_num} images')
    # print(obj_dist)

    # clean up the tail distribution
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
    # check_vg_atts()
