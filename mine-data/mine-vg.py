import numpy as np
import json

def mine_attributes(type, print_every=1000, max_imgs=1000000):
    # read a list of attributes and find corresponding subjects in Visual Genome
    # output obtained pairs to file
    words_file = f'{type}-words.txt'
    attributes_file = 'attributes.json'

    print(f'loading word file of type {type}...')
    words_f = open(words_file, 'r').readlines()
    print('loading attributes file...')
    attributes_f = json.load(open(attributes_file))
    words_dict = {}
    for line in words_f:
        words_dict[line.strip()] = []
    word_ls = words_dict.keys()
    att_counts = {}

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
                # check if any attribute word is in our word list
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
    out_file = 'db/' + type + '.jsonl'
    with open(out_file, 'w') as out:
        for key in word_ls:
            for sub in words_dict[key]:
                #out.write({"sub": sub, "obj": key})
                if att_counts[(sub, key)] > 2:  # only output when count > threshold
                    json.dump({"sub": sub, "obj": key, "count": att_counts[(sub, key)]}, out)
                    out.write('\n')

if __name__ == "__main__":
    mine_attributes('color')
