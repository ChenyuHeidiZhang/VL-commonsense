import numpy as np
import json

attributes_file = 'attributes.json'

def mine_attribute_dist(type, single_slot=True, print_every=1000, max_imgs=1000000):
    # read a list of attributes and find corresponding subjects in Visual Genome
    # output obtained pairs to file
    objs_file = f'{type}-words.txt'

    print(f'loading objects file of type {type}...')
    objs_f = open(objs_file, 'r').readlines()
    objs_ls = []
    for line in objs_f:
        if single_slot and len(line.strip().split()) > 1:
            continue
        objs_ls.append(line.strip())
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
            for attribute in obj['attributes']:
                att = attribute
                # treat attribute "x colored" the same as "x"
                if type == 'color' and att.split()[-1] == 'colored' and att != 'light colored' and att != 'dark colored':
                    att = ' '.join(att.split()[:-1])
                # check if the attribute is in our word list
                if att in objs_ls:
                    sub = obj['names'][0]
                    if att in sub: 
                        continue  # skip cases when the subject is something like "red box"
                    if sub not in sub_dist:
                        # sub_dist[sub] = np.zeros(len(objs_ls))
                        sub_dist[sub] = [0] * len(objs_ls)
                    sub_dist[sub][objs_ls.index(att)] += 1
        if img_num == max_imgs:
            break
    print(f'finished processing {img_num} images')

    print(objs_ls)
    # output
    out_file = f'{type}-dist.jsonl'
    with open(out_file, 'w') as out:
        json.dump(sub_dist, out)
        # for sub, counts in sub_dist.items():
        #     sub_sum = counts.sum()
        #     log_probs = (counts / sub_sum).tolist()  # using log would cause divison by 0
        #     json.dump({"sub": sub, "probs": log_probs}, out)
        #     out.write('\n')

if __name__ == "__main__":
    mine_attribute_dist('color')
