import torch
import numpy as np
import argparse
from tqdm import tqdm
from scipy.stats.stats import kendalltau, pearsonr, spearmanr
from models import init_mlm_model, load_dist_file, load_data, load_prompts, load_word_file

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_log_probs(token_ids, model, mask_id):
    output = model(token_ids)
    hidden_states = output[0].squeeze(0)

    mask_idx = token_ids[0].tolist().index(mask_id)
    hs = hidden_states[mask_idx]
    log_probs = torch.nn.LogSoftmax(dim=0)(hs)
    return log_probs.cpu().numpy()

def run():
    parser = argparse.ArgumentParser(description='zero-shot eval parser')
    parser.add_argument('--model', type=str, default='bert',
                        help='name of the model (bert, roberta, albert, oscar, or distil_bert)')
    parser.add_argument('--model_size', type=str, default='large',
                        help='size of the model (base, large)')
    parser.add_argument('--relation', type=str, default='shape',
                        help='relation to evaluate (shape, material, color, coda, coda_any...)')
    parser.add_argument('--group', type=str, default='',
                        help='group to evaluate (single, multi, any, or '' for all))')
    parser.add_argument('--seed', type=int, default=1,
                        help='numpy random seed')
    args = parser.parse_args()

    model, tokenizer = init_mlm_model(args.model, args.model_size, device)
    mask_token = tokenizer.mask_token
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    # Prepare the inputs
    relation = args.relation
    group = args.group
    test_data, _ = load_data(f'mine-data/db/{relation}/{group}/test.jsonl')
    templates = load_prompts(relation)
    print('num templates:', len(templates))
    objs_ls = load_word_file(relation)
    obj_ids = torch.tensor(tokenizer.convert_tokens_to_ids(objs_ls))
    print('num objs:', len(objs_ls))
    vg_dist_dict = load_dist_file(relation)

    correct = 0
    sp_corrs = []
    record = []
    with torch.no_grad():
        for data in tqdm(test_data):
            scores = []
            for template in templates:
                input = template.replace('[X]', data[0]).replace('[Y]', mask_token)
                token_ids = tokenizer.encode(input, return_tensors='pt').to(device)
                score = get_log_probs(token_ids, model, mask_id)
                scores.append(score)
            model_dist = torch.index_select(torch.tensor(scores), 1, obj_ids)
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
            sp_corrs.append(np.max(sp_corr))
            sp_max_idx = np.argmax(sp_corr)

            record.append((correct_idx, sp_max_idx))

    # print('Recorded correct pred & max sp corr templates:', record)
    print('Prediction accuracy:', correct / len(test_data))
    print('Mean and Std of Sp Corr:', np.mean(sp_corrs), np.std(sp_corrs))


if __name__ == '__main__':
    run()
