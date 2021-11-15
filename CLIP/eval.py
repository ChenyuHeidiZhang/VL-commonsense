# evaluate visual commonsense

import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import clip
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

relation = 'color'
group = 'multi'
template = 'the {} of {}'
train_data, objs = get_data(f'mine-data/db/{relation}/{group}/train.jsonl')
test_data, _ = get_data(f'mine-data/db/{relation}/{group}/test.jsonl')

# Calculate features
def get_features(data, objs, model_name='bert'):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for sub, obj in tqdm(data):
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
            if obj not in objs:
                all_labels.append(-1)
                print(f'test obj {obj} not in train objs')
            else:
                all_labels.append(objs.index(obj))
    return torch.cat(all_features).cpu().numpy(), np.array(all_labels)

train_features, train_labels = get_features(train_data, objs)
#print(train_features.shape)  # (num_examples, 512 CLIP or 768 Bert)
test_features, test_labels = get_features(test_data, objs)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
classifier.fit(train_features, train_labels)
print(len(objs))

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
logprob = classifier.predict_log_proba(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")

# num objs: shape: 13, material: 21, color: 44
