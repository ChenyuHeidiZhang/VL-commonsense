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
    else:
        raise Exception('model name undefined')
    return model.to(device), tokenizer

