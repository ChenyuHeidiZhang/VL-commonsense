import torch
import numpy as np
import argparse
from tqdm import tqdm
from scipy.stats.stats import ModeResult, kendalltau, pearsonr, spearmanr
from scipy import stats
from models import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_token_ids(words, tokenizer, relation, model_name):
    # if model_name == 'albert':  # albert tokenizer is different when encoding single words
    #     obj_ids = tokenizer.convert_tokens_to_ids(words)
    # else:
    obj_ids = []
    for obj in words:
        token = tokenizer.encode(obj)[1]  # use the first subword token id
        if relation != 'cooccur' and token in obj_ids: continue
        obj_ids.append(token)
    return torch.tensor(obj_ids)

def get_log_probs(token_ids, model, mask_id):
    output = model(token_ids)
    hidden_states = output[0].squeeze(0)

    mask_idx = token_ids[0].tolist().index(mask_id)
    hs = hidden_states[mask_idx]
    log_probs = torch.nn.LogSoftmax(dim=0)(hs)
    return log_probs.cpu().numpy()

def get_model_dist(scores, obj_ls, obj_ids, obj_id_map):
    if obj_id_map == {}:
        return torch.index_select(torch.tensor(scores), 1, obj_ids)
    model_dist = []
    for temp_score in scores:
        dist = []
        for obj in obj_ls:
            dist.append(np.log(np.exp(temp_score[obj_id_map[obj]]).sum()))
        model_dist.append(dist)
    return torch.tensor(model_dist)


def run(args):
    '''
    Obtains the best accuracy and sp correlation across all templates for each 
    test example, and returns the averages.
    '''
    model, tokenizer = init_mlm_model(args.model, args.model_size, device)
    mask_token = tokenizer.mask_token
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    # Prepare the inputs
    relation = args.relation
    group = args.group
    test_data, _ = load_data(f'mine-data/db/{relation}/{group}/test.jsonl')
    templates = load_prompts(relation)
    print('num templates:', len(templates))
    objs_ls, objs_map = load_word_file(relation)
    obj_ids = get_token_ids(objs_ls, tokenizer, relation, args.model)
    assert len(objs_ls) == obj_ids.size(0)
    #obj_ids = torch.tensor(tokenizer.convert_tokens_to_ids(objs_ls))

    obj_id_map = {}
    if objs_map:
        for obj in objs_map:
            obj_id_map[obj] = get_token_ids(objs_map[obj], tokenizer, relation, args.model)
    #print(obj_id_map)

    print('num objs:', len(objs_ls))
    vg_dist_dict = load_dist_file(relation)

    correct = 0
    sp_corrs = []
    sp_per_obj = [[] for _ in range(len(objs_ls))]
    dist_pairs = []
    record = []
    with torch.no_grad():
        for data in tqdm(test_data):
            scores = []
            for template in templates:
                input = template.replace('[X]', data[0]).replace('[Y]', mask_token)
                token_ids = tokenizer.encode(input, return_tensors='pt').to(device)
                score = get_log_probs(token_ids, model, mask_id)
                scores.append(score)
            model_dist = get_model_dist(scores, objs_ls, obj_ids, obj_id_map)
            #model_dist = torch.index_select(torch.tensor(scores), 1, obj_ids)
            # print(model_dist.size)  # num_templates, num_objs
            #pred = (-model_dist).argsort(1)
            true_obj_idx = objs_ls.index(data[1])
            top_match = model_dist[:, true_obj_idx] == torch.max(model_dist, dim=1)[0]
            correct_idx = -1
            if torch.any(top_match):
                correct += 1
                correct_idx = np.where(top_match)[0]

            vg_dist = vg_dist_dict[data[0]]
            sp_corr = [spearmanr(vg_dist, model_dist[i])[0] for i in range(len(templates))]
            max_corr = np.max(sp_corr)
            max_idx = np.argmax(sp_corr)
            sp_corrs.append(max_corr)
            sp_per_obj[true_obj_idx].append(max_corr)

            dist_pairs.append((vg_dist, model_dist[max_idx].tolist(), data[0]))
            record.append((correct_idx, max_idx))

    #plot_dists(sp_corrs, np.array(dist_pairs, dtype=object), relation, group, args.model)
    #print('Recorded correct pred & max sp corr templates:', record)

    #avg_per_obj = plot_corr(sp_per_obj, objs_ls, args.model, relation, method='zero-shot')
    print('Prediction accuracy:', correct / len(test_data))
    print('Mean and Std of Sp Corr:', np.mean(sp_corrs), np.std(sp_corrs))

    return round(np.mean(sp_corrs),3), round(np.std(sp_corrs),3), round(correct/len(test_data)*100,1)
    #return sp_corrs, sp_per_obj #, avg_per_obj

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='zero-shot eval parser')
    # parser.add_argument('--model', type=str, default='bert',
    #                     help='name of the model (bert, roberta, albert, vokenization, oscar, or distil_bert)')
    # parser.add_argument('--model_size', type=str, default='base',
    #                     help='size of the model (base, large)')
    # parser.add_argument('--relation', type=str, default='shape',
    #                     help='relation to evaluate (shape, material, color, coda, cooccur)')
    # parser.add_argument('--group', type=str, default='',
    #                     help='group to evaluate (single, multi, any, or '' for all))')
    # args = parser.parse_args()
    # sp_mean, sp_std, acc = run(args)

    rel_types = ['coda', 'color', 'wiki-color']  # 'color', 'shape', 'material', 'cooccur', 'coda'
    models = ['bert', 'oscar', 'distil_bert', 'roberta', 'vokenization']  # 'albert'
    d = {rel: [] for rel in rel_types}
    sps_per_obj = {rel: [] for rel in rel_types}
    for relation in rel_types:
        sp_corrs = []
        print(relation)
        groups = ['']  # , 'single', 'multi', 'any'
        for group in groups:
            for model in models:
                args = Args(model, relation, group)
                sp_mean, sp_std, acc = run(args)
                d[relation].append(sp_mean)
                #d[relation].append((sp_mean, sp_std, acc))
                #sp, sp_per_obj = run(args)
                # sp_corrs.append(sp)
                #sps_per_obj[relation].append(sp_per_obj)
        #print(stats.ttest_ind(sp_corrs[0], sp_corrs[1], equal_var=False))
        #print(spearmanr(sps_per_obj[relation][0], sps_per_obj[relation][1]))
    #print(sp_per_obj)
    #plot_corr_all_rels(sps_per_obj, models, rel_types, method='zero-shot')

    import pandas as pd
    df = pd.DataFrame(d)
    #df.to_excel('file.xlsx')
    df.to_csv('heatmap_data.csv')
    print(df)
