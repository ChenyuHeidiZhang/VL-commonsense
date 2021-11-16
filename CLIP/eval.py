# evaluate visual commonsense

import os
import json
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import clip
from datasets import load_dataset
import torch
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM, BertOnlyMLMHead
from pytorch_transformers import BertConfig
from modeling_bert import BertImgModel

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
# bert
bert_model = BertModel.from_pretrained("bert-base-cased").to(device)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# oscar
# config = BertConfig.from_pretrained("Oscar/pretrained_base/checkpoint-2000000/config.json")
# bert_model = BertImgModel.from_pretrained("Oscar/pretrained_base/checkpoint-2000000/pytorch_model.bin", config=config).to(device)
# bert_tokenizer = BertTokenizer.from_pretrained("Oscar/pretrained_base/checkpoint-2000000/")

# Prepare the inputs
def get_data(file):
    data = []
    objs = set()  # for converting object words to label ids
    with open(file, 'r') as f:
        for line in f.readlines():
            js = json.loads(line)
            objs.add(js['obj'])
            data.append((js['sub'], js['obj']))
    return data, list(objs)

relation = 'shape'
group = ''
template = 'the {} of {}'
train_data, objs = get_data(f'mine-data/db/{relation}/{group}/train.jsonl')
test_data, _ = get_data(f'mine-data/db/{relation}/{group}/test.jsonl')

# ds = load_dataset("corypaik/coda", ignore_verifications=True)
# def get_coda_template(sub):
#     rand_idx = np.random.randint(10)
#     for split in ['train', 'validation', 'test']:
#         for instance in ds[split]:
#             if instance['ngram'] == sub and instance['template_group'] == 1:
#                 if rand_idx == instance['template_idx']:
#                     return instance['text']
#     return None

# Calculate features
def get_features(data, objs, model_name='bert'):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for sub, obj in tqdm(data):
            if obj not in objs:
                #all_labels.append(-1)
                print(f'test obj {obj} not in train objs')
                continue
            all_labels.append(objs.index(obj))
            #text = get_coda_template(sub)
            #if text == None:
            text = template.format(relation, sub)
            if model_name=='bert':
                input_ids = torch.tensor([bert_tokenizer.encode(text)]).to(device)
                emb = bert_model.embeddings(input_ids=input_ids)
                features = torch.mean(bert_model.encoder(emb)[0], dim=1)
            elif model_name=='oscar':
                input_ids = torch.tensor([bert_tokenizer.encode(text)]).to(device)
                output = bert_model(input_ids=input_ids)
                features = torch.mean(output, dim=1)
            elif model_name=='clip':
                text_input = clip.tokenize(text).to(device)
                features = model.encode_text(text_input)
            else:
                print(f'wrong model name {model_name}')
            all_features.append(features)

    return torch.cat(all_features).cpu().numpy(), np.array(all_labels)

train_features, train_labels = get_features(train_data, objs)
#print(train_features.shape)  # (num_examples, 512 CLIP or 768 Bert)
test_features, test_labels = get_features(test_data, objs)
print('num classes:', len(objs))
print(f'num train examples: {len(train_labels)}, num test examples: {len(test_labels)}')

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")

# Evaluate correlation of regression logprobs with true distributions
rel_name = relation.split('_')[0]
dist_file = f'mine-data/distributions/{rel_name}-dist.jsonl'
dist_dict = json.load(open(dist_file, 'r'))

word_ls = []
with open(f'mine-data/words/{rel_name}-words.txt', 'r') as f:
    for line in f.readlines():
        if len(line.strip().split()) == 1:
            word_ls.append(line.strip())
word_ids = []
for object in objs:
    word_ids.append(word_ls.index(object))

logprob = classifier.predict_proba(test_features)
print(logprob.shape)  # num_test_examples, num_objs
sp_corrs = []
idx = 0
for test_sub, test_obj in test_data:
    if test_obj not in objs: continue
    true_dist = np.take(dist_dict[test_sub], word_ids)
    sp_corrs.append(spearmanr(true_dist, logprob[idx]))
    idx += 1
sp_corr = np.mean(sp_corrs)
print('avg sp corr:', sp_corr)

# num objs: shape: 13, material: 21, color: 44
