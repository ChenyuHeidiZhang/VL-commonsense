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

    print('mining attributes...')
    img_num = 0
    for img in attributes_f:
        img_num += 1
        if img_num % print_every == 0:
            print(f'processing {img_num}-th image')
        for obj in img['attributes']:
            if 'attributes' not in obj:
                continue
            for att in obj['attributes']:
                attribute = None
                # check if any attribute word is in our word list
                if att in word_ls:
                    attribute = att
                else:
                    # if part of the attribute (e.g. 'light' in light colored) is in our word list
                    loc = np.where([(att_word in word_ls) for att_word in att.split()])
                    if len(loc[0] > 0):
                        attribute = att.split()[loc[0][0]]
                if attribute and obj['names'][0] not in words_dict[attribute]:  # don't append the same attribute twice
                    words_dict[attribute].append(obj['names'][0])
                    # TODO: count the number of occurrences of each pair
        if img_num == max_imgs:
            break
    print(f'finished processing {img_num} images')

    # output
    out_file = words_file.split('-')[0] + '.jsonl'
    with open(out_file, 'w') as out:
        for key in word_ls:
            for sub in words_dict[key]:
                #out.write({"sub": sub, "obj": key})
                json.dump({"sub": sub, "obj": key}, out)
                out.write('\n')

if __name__ == "__main__":
    mine_attributes('material')
