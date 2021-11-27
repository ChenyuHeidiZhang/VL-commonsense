# cos(large - small, subj)

import argparse
import torch
import clip
from scipy.spatial import distance
import matplotlib.pyplot as plt

from models import init_model

device = "cuda" if torch.cuda.is_available() else "cpu"

def model_output(model, tokenizer, model_name, text):
    if model_name=='bert' or model_name=='oscar':
        # 1 input word but 3 input_ids, the middle one is the token, 
        # the other two are EOS and BOS
        input_ids = torch.tensor([tokenizer.encode(text)]).to(device)
        features = model.embeddings(input_ids=input_ids)[0][1]
    elif model_name=='clip':
        text_input = clip.tokenize(text).to(device)
        features = model.encode_text(text_input)
    return features.detach().cpu().numpy()

def load_size_db(group):
    file = f'mine-data/size_db/{group}-words.txt'
    data = []
    with open(file, 'r') as f:
        for line in f.readlines():
            data.append(line.strip())
    return data

def run():
    parser = argparse.ArgumentParser(description='eval parser')
    parser.add_argument('--model', type=str, default='bert',
                        help='name of the model (bert, oscar, or clip)')
    parser.add_argument('--model_size', type=str, default='base',
                        help='size of the model (base, large)')
    args = parser.parse_args()

    model_name = args.model
    model, tokenizer = init_model(model_name, args.model_size, device)
    adj1 = 'large'
    adj2 = 'small'
    # adj1 = 'hot'
    # adj2 = 'cold'
    features1 = model_output(model, tokenizer, model_name, adj1)
    features2 = model_output(model, tokenizer, model_name, adj2)
    # calculate cosine similarity for size subjs of all 5 groups
    data = []
    groups = ['tiny', 'small', 'medium', 'large', 'huge']
    for group in groups:
        words = load_size_db(group)
        sims = []
        for subj in words:
            feature_sub = model_output(model, tokenizer, model_name, subj)
            # large sim indicates preference towards adj1
            sim = 1 - distance.cosine(features1-features2, feature_sub)
            sims.append(sim)
        data.append(sims)
    plt.boxplot(data, showfliers=False)  # , showfliers=False
    plt.xticks([1,2,3,4,5], groups)
    plt.savefig(f'probing/boxplots/size_boxplot_{model_name}.png')
    # colors = ['blue', 'purple', 'yellow', 'pink', 'red']
    # for subj in colors:
    #     feature_sub = model_output(model, tokenizer, model_name, subj)
    #     sim = 1 - distance.cosine(features1-features2, feature_sub)
    #     data.append([sim])
    # plt.boxplot(data, showfliers=False)  # , showfliers=False
    # plt.xticks([1,2,3,4,5], colors)
    # plt.savefig(f'probing/boxplots/color_boxplot_{model_name}.png')

if __name__ == '__main__':
    run()
