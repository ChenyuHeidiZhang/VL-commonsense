import json
import numpy as np
import matplotlib.pyplot as plt
import clip
from transformers import BertTokenizer
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from pytorch_transformers import BertConfig
from modeling_bert import BertImgModel, BertImgForPreTraining


def init_model(model_name='bert', model_size='base', device='cpu'):
    if model_name == 'bert':
        model = BertModel.from_pretrained(f"bert-{model_size}-cased").to(device)
        tokenizer = BertTokenizer.from_pretrained(f"bert-{model_size}-cased")
    elif model_name == 'oscar':
        num = '2000000' if model_size == 'base' else '1410000'
        dir = f"soft-prompts/pretrained_{model_size}/checkpoint-{num}"
        config = BertConfig.from_pretrained(dir + "/config.json")
        model = BertImgModel.from_pretrained(dir + "/pytorch_model.bin", config=config).to(device)
        tokenizer = BertTokenizer.from_pretrained(dir)
    elif model_name == 'clip':
        model, preprocess = clip.load('ViT-B/32', device)
        tokenizer = None
    else:
        raise Exception('model name undefined')
    return model, tokenizer


def init_mlm_model(model_name='bert', model_size='base', device='cpu'):
    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-cased')
        model = BertForMaskedLM.from_pretrained(f'bert-{model_size}-cased')
    elif model_name == "mbert":
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-multilingual-cased')
        model = BertForMaskedLM.from_pretrained(f'bert-{model_size}-multilingual-cased')
    elif model_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(f'roberta-{model_size}')
        model = RobertaForMaskedLM.from_pretrained(f'roberta-{model_size}')
    elif model_name == "albert":
        if model_size == 'large': model_size = 'xxlarge'
        tokenizer = AlbertTokenizer.from_pretrained(f'albert-{model_size}-v2')
        model = AlbertForMaskedLM.from_pretrained(f'albert-{model_size}-v2')
    elif model_name == "oscar":
        num = '2000000' if model_size == 'base' else '1410000'
        dir = f"soft-prompts/pretrained_{model_size}/checkpoint-{num}"
        config = BertConfig.from_pretrained(dir + "/config.json")
        model = BertImgForPreTraining.from_pretrained(dir + "/pytorch_model.bin", config=config)
        tokenizer = BertTokenizer.from_pretrained(dir)
    elif model_name == 'distil_bert':
        config = BertConfig.from_pretrained("soft-prompts/distil_bert/config.json")
        model = BertForMaskedLM.from_pretrained("soft-prompts/distil_bert/")
        tokenizer = BertTokenizer.from_pretrained("soft-prompts/distil_bert/")
    elif model_name == 'vokenization':
        config = BertConfig.from_pretrained("soft-prompts/vlm_12L_768H_wiki/config.json")
        model = BertForMaskedLM.from_pretrained("soft-prompts/vlm_12L_768H_wiki/")
        tokenizer = BertTokenizer.from_pretrained("soft-prompts/vlm_12L_768H_wiki/")
    else:
        raise Exception('model name undefined')
    return model.to(device), tokenizer


def load_prompts(type):
    prompts_file = f'soft-prompts/prompts/vl/{type}.jsonl'
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f.readlines():
            template = json.loads(line)['template']
            prompts.append(template)
    return prompts

def load_data(file):
    data = []
    objs = set()  # for converting object words to label ids
    with open(file, 'r') as f:
        for line in f.readlines():
            js = json.loads(line)
            objs.add(js['obj'])
            data.append((js['sub'], js['obj']))
    return data, list(objs)

def load_word_file(type, single_slot=True):
    words_file = f'mine-data/words/{type}-words.txt'
    word_ls = []
    with open(words_file, 'r') as f:
        for line in f.readlines():
            if single_slot and len(line.strip().split()) > 1:
                continue
            word_ls.append(line.strip())
    return word_ls

def load_dist_file(type):
    dist_file = f'mine-data/distributions/{type}-dist.jsonl'
    with open(dist_file, 'r') as f:
        vg_dist = json.load(f)
    return vg_dist


def plot_half(dist_pairs, ax, x_axis, model):
    for i, pair in enumerate(dist_pairs):
        # sort two distributions together, order by the first
        sorted_dists = sorted(zip(pair[0], pair[1]), reverse=True)
        tuples = zip(*sorted_dists)
        vg_dist, model_dist = [np.array(tuple) for tuple in tuples]
        model_dist = np.exp(model_dist)
        # plot the distributions
        ax.plot(x_axis, vg_dist / np.sum(vg_dist), color=f'C{i}', label=f'VG \'{pair[2]}\'')
        ax.plot(x_axis, model_dist / np.sum(model_dist), color=f'C{i}', linestyle='--', label=f'{model} \'{pair[2]}\'')
    ax.legend()
    ax.set_xticks(x_axis)
    ax.set_xlabel('object ids')

def plot_dists(sp_corrs, dist_pairs, group, model, num_to_plot=3):
    '''
    plot the k examples that have highest & lowest (each) corr between coda and vg distributions.
    sp_corrs: list of sp corrs
    dist_pairs: np array of triplets of (CoDa dist, VG dist, noun)
    '''
    max_idxs = np.argsort(sp_corrs)[-num_to_plot:][::-1]
    min_idxs = np.argsort(sp_corrs)[:num_to_plot]

    #print(max_idxs, min_idxs)
    print(np.array(sp_corrs)[max_idxs], np.array(sp_corrs)[min_idxs])

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[10, 5])
    x_axis = range(1, len(dist_pairs[0][0])+1)
    plot_half(dist_pairs[max_idxs], ax1, x_axis, model)
    ax1.set_title('high correlation')
    ax1.set_ylabel('probability')
    plot_half(dist_pairs[min_idxs], ax2, x_axis, model)
    ax2.set_title('low correlation')
    if group == '': group = 'all'
    plt.savefig(f'zero_shot_plt_{group}.png')
