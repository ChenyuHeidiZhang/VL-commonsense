from transformers import (
    BertTokenizer,
    RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
)
from .modeling_bert import BertConfig, BertForMaskedLM
from pytorch_transformers import BertConfig

from torch import nn
import numpy as np

from ..util import *
from .. import util
from .language_model import LanguageModel
from .oscar.modeling.modeling_bert import BertImgForMaskedLM

class PreTrainedBert(LanguageModel):
    """
    PreTrained language model.
    The parameters of this model should be fixed. It should only be used to compute the
    likelihood of a sentence.
    """
    def __init__(
            self,
            model_type: str = 'bert',
            param_name: str = 'bert-large-cased',
            device: str = 'cuda:0',
            max_length: int = 2048,
            batch_size: int = 8,
    ) -> None:
        """
        Args:
            model_type: E.g., xlnet, bert
            param_name: E.g., xlnet-base-cased, bert-base-uncased
            device: cuda:0, cpu
            max_length: Maximum supported number of tokens. For memory concern.
            batch_size: Maximum sentences that could be processed in a batch.
        """
        model_classes = {
            'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
            'roberta': (RobertaConfig, BertForMaskedLM, RobertaTokenizer),
        }
        self.model_type = model_type
        self.model_size = param_name.split('-')[1]
        if model_type == 'oscar':
            model_size = param_name.split('-')[1]
            num = '2000000' if model_size == 'base' else '1410000'
            self.config = BertConfig.from_pretrained(f"pretrained_{model_size}/checkpoint-{num}/config.json")
            util.tokenizer = BertTokenizer.from_pretrained(f"pretrained_{model_size}/checkpoint-{num}/")
            model = BertImgForMaskedLM.from_pretrained(f"pretrained_{model_size}/checkpoint-{num}/pytorch_model.bin", config=self.config)
        elif model_type == 'distil_bert':
            self.config = BertConfig.from_pretrained("distil_bert/config.json")
            util.tokenizer = BertTokenizer.from_pretrained("distil_bert/")
            model = BertForMaskedLM.from_pretrained("distil_bert/")
        elif model_type == 'vokenization':
            self.config = BertConfig.from_pretrained("vlm_12L_768H_wiki/config.json")
            util.tokenizer = BertTokenizer.from_pretrained("vlm_12L_768H_wiki/")
            model = BertForMaskedLM.from_pretrained("vlm_12L_768H_wiki/")
        else:
            self.config, model_class, tokenizer_class = model_classes[model_type]
            util.tokenizer = tokenizer_class.from_pretrained(param_name)
            model = model_class.from_pretrained(param_name)

        self.max_length = max_length
        self.batch_size = batch_size

        super().__init__(device, model)

    @property
    def n_layer(self):
        return self.model.config.num_hidden_layers

    @property
    def dim(self):
        return self.model.config.hidden_size

    @property
    def emb(self) -> nn.Embedding:
        emb = self.model.base_model.embeddings.word_embeddings
        return emb
