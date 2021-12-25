import json
from nltk.stem.wordnet import WordNetLemmatizer
Lem = WordNetLemmatizer()

THRES_COLOR = 5
THRES_MATERIAL = 2
THRES_SHAPE = 1
THRES_COOCCUR = 8
WORD_LISTS = {
    'color': ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'silver', 'white', 'yellow'],  # 12
    'shape': ['cross', 'heart', 'octagon', 'oval', 'polygon', 'rectangle', 'rhombus', 'round', 'semicircle', 'square', 'star', 'triangle'],  # 12
    'material': ['bronze', 'ceramic', 'cloth', 'concrete', 'cotton', 'denim', 'glass', 'gold', 'iron', 'jade', 'leather', 'metal', 'paper', 'plastic', 'rubber', 'stone', 'tin', 'wood']  # 18
}

def load_word_file(type, single_slot=True):
    '''Loads the word file for color, shape, or material.'''
    if type not in WORD_LISTS.keys(): return load_words(type)

    words_file = f'words/{type}-words.txt'
    word_map = {}
    with open(words_file, 'r') as f:
        for line in f.readlines():
            splitted = line.split(':')
            specific = splitted[0].strip()
            category = splitted[1].strip() if len(splitted) == 2 else specific
            if single_slot and len(category.split()) > 1:
                continue
            word_map[specific] = category
    return WORD_LISTS[type], word_map

def load_words(type):
    words_file = f'words/{type}-words.txt'
    word_ls = []
    with open(words_file, 'r') as f:
        for line in f.readlines():
            word_ls.append(line.strip())
    return word_ls, None

def load_dist_file(type):
    dist_file = f'distributions/{type}-dist.jsonl'
    with open(dist_file, 'r') as f:
        vg_dist = json.load(f)
    return vg_dist

def filter_att(att, obj_map):
    att_split = att.split()
    # treat attribute like "x colored" the same as "x"
    if att_split[-1] == 'colored' or att_split[-1] == 'made' or att_split[-1] == 'shaped':
        att_split = att_split[:-1]
    # treat attributes like 'forest green' as green
    if len(att_split) >= 1: att = att_split[-1]

    # return the category if the attribute is in our word list
    return obj_map[att] if att in obj_map else None

def lemmatize(sub):
    sub = sub.split()
    if len(sub) > 1:  # convert plural to singulars
        sub = ' '.join(sub[:-1] + [Lem.lemmatize(sub[-1])])
    else:
        sub = Lem.lemmatize(sub[0])
    return sub
