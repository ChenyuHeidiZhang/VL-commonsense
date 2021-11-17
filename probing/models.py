import torch
import clip
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM, BertOnlyMLMHead
from pytorch_transformers import BertConfig
from modeling_bert import BertImgModel


def init_model(model_name='bert', device='cpu'):
    if model_name == 'bert':
        model = BertModel.from_pretrained("bert-base-cased").to(device)
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    elif model_name == 'oscar':
        config = BertConfig.from_pretrained("Oscar/pretrained_base/checkpoint-2000000/config.json")
        model = BertImgModel.from_pretrained("Oscar/pretrained_base/checkpoint-2000000/pytorch_model.bin", config=config).to(device)
        bert_tokenizer = BertTokenizer.from_pretrained("Oscar/pretrained_base/checkpoint-2000000/")
    elif model_name == 'clip':
        model, preprocess = clip.load('ViT-B/32', device)
        bert_tokenizer = None
    return model, bert_tokenizer
