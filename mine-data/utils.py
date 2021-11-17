from nltk.stem.wordnet import WordNetLemmatizer
Lem = WordNetLemmatizer()

THRES_COLOR = 5
THRES_SHAPE = 2
THRES_MATERIAL = 2

def load_word_file(type, single_slot=True):
    words_file = f'words/{type}-words.txt'
    word_ls = []
    with open(words_file, 'r') as f:
        for line in f.readlines():
            if single_slot and len(line.strip().split()) > 1:
                continue
            word_ls.append(line.strip())
    return word_ls

def filter_att(att, type):
    # treat attribute "x colored" the same as "x"
    if type == 'color' and att.split()[-1] == 'colored' \
        and att != 'light colored' and att != 'dark colored':
        att = ' '.join(att.split()[:-1])
    return att

def lemmatize(sub):
    sub = sub.split()
    if len(sub) > 1:  # convert plural to singulars
        sub = ' '.join(sub[:-1] + [Lem.lemmatize(sub[-1])])
    else:
        sub = Lem.lemmatize(sub[0])
    return sub
