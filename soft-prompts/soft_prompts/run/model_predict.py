from ..model import PatternModel
from ..io import load_db_general
from ..corpus import load_templates
from ..lm import construct_lm
from .experiment import read_kwargs
from .. import util

import os
import pickle
import logging
logger = logging.getLogger('Predictor')

kwargs = read_kwargs()
cfg= kwargs.pop('trainer')

lm = construct_lm(**kwargs.pop('lm'))
relation_db = load_db_general(**kwargs.get('db'))
pattern_db = load_templates(**kwargs.pop('template'))

for rel_type, pb in pattern_db.items():
    splits = list()
    if rel_type not in relation_db['train'].banks:
        continue
    for split in ['train', 'dev', 'test']:
        splits.append(relation_db[split].banks[rel_type])
    train_set, dev_set, test_set = splits

    # logger.info(f'Training relation {relation_bank_splits[0].relation_type} with {len(pattern_bank)} patterns.')
    logger.info(f'rel_type {rel_type} with {len(pb)} patterns.')
    model = PatternModel(
        pb, cfg.pop('device'), lm, cfg.pop('max_layer'), force_single_token=cfg.pop('force_single_token', False),
        vocab_file=cfg.pop('vocab_file', None), conditional_prompt=cfg.pop('conditional_prompt', False)
    )
    filename = os.path.join(cfg.pop('log_path'), f'model.{rel_type}.pkl')
    best_state = pickle.load(open(filename, 'rb'))
    model.load(best_state)
    after_pred, answer_ranks, answer_topk = model.conditional_generate_single_slot(
        cfg.pop('batch_size_no_grad'), test_set, None
    )
    # print(after_pred.size()) -- [5, 28996]
    # print(answer_ranks.size()) -- [5]
    # print(answer_topk.size()) -- [28996]
    print(answer_topk[1], answer_topk[10])  # answer_topk[i]: number of predictions that have answer_rank <= i
    print(answer_ranks)  # ranks of each answer in the predictions

    answer = [after_pred[i][answer_ranks[i]-1] for i in range(answer_ranks.shape[0])]
    print(answer)  # This is the same as the true answer
    pred = after_pred.T[0]
    print(pred)
    pred = util.tokenizer.convert_ids_to_tokens(pred)
    print(pred)

