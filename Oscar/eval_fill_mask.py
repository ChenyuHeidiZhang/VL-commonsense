import csv
import torch
import argparse
import difflib
import logging
import numpy as np
import pandas as pd

from transformers import BertTokenizer, BertForMaskedLM
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from tqdm import tqdm

from oscar.modeling.modeling_bert import BertImgForPreTraining
from pytorch_transformers import BertConfig, BertTokenizer

# log_softmax = torch.nn.LogSoftmax(dim=0)

# # config = BertConfig.from_pretrained("base-vg-labels/ep_67_588997/config.json")
# # model = BertImgForPreTraining.from_pretrained("base-vg-labels/ep_67_588997/pytorch_model.bin", config=config)
# # tokenizer = BertTokenizer.from_pretrained("base-vg-labels/ep_67_588997/")
# tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
# model = AlbertForMaskedLM.from_pretrained('albert-xxlarge-v2')
# mask_token = tokenizer.mask_token

# inp = f"The color of avocado is {mask_token}."
# print(inp)
# tgt1 = 'red'
# tgt2 = 'green'
# inp_ids = tokenizer.encode(inp)
# mask_id = tokenizer.convert_tokens_to_ids(mask_token)
# mask_idx = inp_ids.index(mask_id)
# # print(inp_ids, mask_id, mask_idx)

# inp_ids = torch.IntTensor(inp_ids).unsqueeze(1)
# out = model(inp_ids)
# # print(tokenizer.encode(tgt1), tokenizer.encode(tgt2))

# tgt1_id = tokenizer.encode(tgt1)[1]  # 0 for bert and oscar; 1 for albert and roberta
# tgt2_id = tokenizer.encode(tgt2)[1]
# # print(out[0].shape) #: [#tokens, 1, 30522]
# hs = out[0][mask_idx][0]
# hs = log_softmax(hs)
# prob1 = hs[tgt1_id]
# prob2 = hs[tgt2_id]
# #print(out[0][3][0][50:90], hs[50:90])
# print(prob1, prob2)

def read_data(input_file):
    """
    Load data into pandas DataFrame format.
    """
    
    df_data = pd.DataFrame(columns=['question', 'category', 'choice_correct', 'choice_wrong', 'inverse', 'compositional'])

    with open(input_file, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            df_item = {'question': row['question'],
                       'category': row['category'],
                       'choice_correct': row['correct_choice'],
                       'choice_wrong': row['wrong_choice'],
                       'inverse': row['inverse'],
                       'compositional': row['compositional']}
            df_data = df_data.append(df_item, ignore_index=True)

    return df_data


def get_lob_prob(data, lm, n=1):
    """
    Score each of the two choices of masked word. Take the log probability.
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    question = data['question']
    if uncased:
        question = question.lower()
    question = question.replace("**mask**", mask_token)

    # tokenize
    question_ids = tokenizer.encode(question)
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    mask_idx = question_ids.index(mask_id)

    question_ids = torch.IntTensor(question_ids).unsqueeze(1)
    out = model(question_ids)
    tgt1_id = tokenizer.encode(data['choice_correct'])[0]
    tgt2_id = tokenizer.encode(data['choice_wrong'])[0]

    hs = out[0][mask_idx][0]
    hs = log_softmax(hs)
    score = {}
    score["correct_score"] = hs[tgt1_id].item()
    score["wrong_score"] = hs[tgt2_id].item()
    #print(out[0][3][0][50:90], out[0][mask_idx][0][50:90])

    return score


def evaluate(args):
    """
    Evaluate a masked language model using CrowS-Pairs dataset.
    """

    print("Evaluating:")
    print("Input:", args.input_file)
    print("Model:", args.lm_model)
    print("=" * 100)

    logging.basicConfig(level=logging.INFO)

    # load data into panda DataFrame
    df_data = read_data(args.input_file)

    # supported masked language models
    if args.lm_model == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        uncased = True
    elif args.lm_model == "mbert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-multilingual-uncased')
        uncased = True
    elif args.lm_model == "oscar":
        config = BertConfig.from_pretrained("base-vg-labels/ep_67_588997/config.json")
        model = BertImgForPreTraining.from_pretrained("base-vg-labels/ep_67_588997/pytorch_model.bin", config=config)
        tokenizer = BertTokenizer.from_pretrained("base-vg-labels/ep_67_588997/")
        uncased = True
    elif args.lm_model == "oscar2":
        config = BertConfig.from_pretrained("pretrained_base/checkpoint-2000000/config.json")
        model = BertImgForPreTraining.from_pretrained("pretrained_base/checkpoint-2000000/pytorch_model.bin", config=config)
        tokenizer = BertTokenizer.from_pretrained("pretrained_base/checkpoint-2000000/")
        uncased = True

    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    mask_token = tokenizer.mask_token
    log_softmax = torch.nn.LogSoftmax(dim=0)
    # vocab = tokenizer.get_vocab()
    # with open(args.lm_model + ".vocab", "w") as f:
    #     f.write(json.dumps(vocab))

    lm = {"model": model,
          "tokenizer": tokenizer,
          "mask_token": mask_token,
          "log_softmax": log_softmax,
          "uncased": uncased
    }

    # score each choice. 
    # each row in the dataframe has the score for each choice.
    df_score = pd.DataFrame(columns=['question', 'category', "correctness",
                                     'choice_correct_score', 'choice_wrong_score',
                                     'inverse', 'compositional'])

    total_correct = 0
    N = 0
    neutral = 0
    total = len(df_data.index)
    correctness = False
    with tqdm(total=total) as pbar:
        for index, data in df_data.iterrows():
            score = get_lob_prob(data, lm)

            for stype in score.keys():
                score[stype] = round(score[stype], 3)

            N += 1
            pair_score = 0
            pbar.update(1)
            if score['correct_score'] == score['wrong_score']:
                neutral += 1
            elif score['correct_score'] > score['wrong_score']:
                correctness = True
                total_correct += 1
            else:
                correctness = False

            df_score = df_score.append({'question': data['question'],
                                        'category': data['category'],
                                        'correctness': correctness,
                                        'choice_correct_score': score['correct_score'],
                                        'choice_wrong_score': score['wrong_score'],
                                        'inverse': data['inverse'],
                                        'compositional': data['compositional']
                                      }, ignore_index=True)

    df_score.to_csv(args.output_file)
    print('=' * 100)
    print('Total examples:', N)
    print('Num. correct:', total_correct, round(total_correct / N * 100, 2))
    print("Num. neutral:", neutral, round(neutral / N * 100, 2))
    print('=' * 100)
    print()


parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input_file", type=str, help="path to input file")
parser.add_argument('-l', "--lm_model", type=str, help="pretrained LM model to use (options: bert, roberta, albert, oscar)")
parser.add_argument('-o', "--output_file", type=str, help="path to output file with sentence scores")

args = parser.parse_args()
evaluate(args)
