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
from pytorch_transformers import BertConfig

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


def get_log_prob_unigram(masked_token_ids, token_ids, mask_idx, lm):
    """
    Given a sequence of token ids, with one masked token, return the log probability of the masked token.
    """
    
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    # get model hidden states
    output = model(masked_token_ids)
    hidden_states = output[0].squeeze(0)
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    # we only need log_prob for the MASK tokens
    assert masked_token_ids[0][mask_idx] == mask_id

    hs = hidden_states[mask_idx]
    target_id = token_ids[0][mask_idx]
    log_probs = log_softmax(hs)[target_id]

    return log_probs


def get_span(seq1, seq2):
    """
    This function extract spans that are shared between two sequences.
    """

    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]
    # print(seq1, seq2)
    
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]
    return template1, template2


def mask_unigram(data, lm, n=1):
    """
    Score each version of the sentence by masking one word at a time. The score for a sentence  
    is the sum of log probability of each word in the sentence.
    n = n-gram of token that is masked, if n > 1, we mask tokens with overlapping
    n-grams.

    Later, try score each of the two choices of masked word. Take the log probability.
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    #sent1, sent2 = data["sent1"], data["sent2"]
    question = data['question']
    if uncased:
        question = question.lower()
    sent1 = question.replace("**mask**", data['choice_correct'])
    sent2 = question.replace("**mask**", data['choice_wrong'])

    # tokenize
    sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')
    sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')

    # get spans of non-changing tokens
    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])
    
    assert len(template1) == len(template2)

    N = len(template1)  # num. of tokens that can be masked
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    sent1_log_probs = 0.
    sent2_log_probs = 0.
    total_masked_tokens = 0

    # skipping CLS and SEP tokens, they'll never be masked
    for i in range(1, N-1):
        sent1_masked_token_ids = sent1_token_ids.clone().detach()
        sent2_masked_token_ids = sent2_token_ids.clone().detach()

        sent1_masked_token_ids[0][template1[i]] = mask_id
        sent2_masked_token_ids[0][template2[i]] = mask_id
        total_masked_tokens += 1

        score1 = get_log_prob_unigram(sent1_masked_token_ids, sent1_token_ids, template1[i], lm)
        score2 = get_log_prob_unigram(sent2_masked_token_ids, sent2_token_ids, template2[i], lm)

        sent1_log_probs += score1.item()
        sent2_log_probs += score2.item()

    score = {}
    # average over iterations
    score["correct_score"] = sent1_log_probs
    score["wrong_score"] = sent2_log_probs

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
    elif args.lm_model == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = RobertaForMaskedLM.from_pretrained('roberta-large')
        uncased = False
    elif args.lm_model == "albert":
        tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
        model = AlbertForMaskedLM.from_pretrained('albert-xxlarge-v2')
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
            score = mask_unigram(data, lm)

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
