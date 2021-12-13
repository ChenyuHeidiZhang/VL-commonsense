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
        # For each input word, remove the EOS and BOS tokens and take the
        # avergae embedding for the middle subword tokens.
        input_ids = torch.tensor([tokenizer.encode(text)]).to(device)
        features = model.embeddings(input_ids=input_ids)[0][1:-1]
        features = torch.mean(features, dim=0)
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

def cross_model_comparison(model_names, model_size, plot='boxplot'):
    adj1 = 'large'
    adj2 = 'small'
    words = []
    groups = ['tiny', 'small', 'medium', 'large', 'huge']
    cdict = {'tiny': 'skyblue', 'small': 'cornflowerblue', 'medium': 'royalblue', 'large': 'blue', 'huge': 'navy'}
    if len(model_names) == 2 or plot=='boxplot':
        fig, ax = plt.subplots()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    data = [[] for _ in range(len(model_names))]
    for i, model_name in enumerate(model_names):
        model, tokenizer = init_model(model_name, model_size, device)
        features1 = model_output(model, tokenizer, model_name, adj1)
        features2 = model_output(model, tokenizer, model_name, adj2)
        for group in groups:
            words = load_size_db(group)
            sims = []
            for subj in words:
                feature_sub = model_output(model, tokenizer, model_name, subj)
                # large sim indicates preference towards adj1
                sim = 1 - distance.cosine(features1-features2, feature_sub)
                sims.append(sim)
            data[i].append(sims)
    if plot=='scatter':
        for i, group in enumerate(groups):
            ax.scatter(data[0][i], data[1][i], data[2][i], c=cdict[group], label=group)
            # for j, word in enumerate(load_size_db(group)):
            #     ax.annotate(word, (sims[0][i][j], sims[1][i][j]))
            ax.set_xlabel(f'{model_names[0]} cos_sim')
            ax.set_ylabel(f'{model_names[1]} cos_sim')
            if len(model_names) == 3:
                ax.set_zlabel(f'{model_names[2]} cos_sim')
        ax.legend()
    else:
        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)
        colors = ['red', 'orange', 'green']
        for i in range(len(model_names)):
            bp = plt.boxplot(data[i], showfliers=False)
            set_box_color(bp, color=colors[i])
            plt.plot([], c=colors[i], label=model_names[i])
        plt.legend()
        plt.xticks([1,2,3,4,5], labels=groups)
        plt.ylabel('cos_sim')
    plt.savefig(f'probing/plots/size_comparison_{plot}.png', bbox_inches='tight')


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
    plt.savefig(f'probing/plots/size_boxplot_{model_name}.png')


if __name__ == '__main__':
    if True:
        cross_model_comparison(['bert', 'oscar', 'clip'], 'base', plot='scatter')
    else:
        run()
