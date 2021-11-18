import os

from .trex_reader import TREx, VLdb


def load_db_general(path, **kwargs):
    subs = os.listdir(path)
    ret = {sp: [] for sp in ['train', 'dev', 'test']}
    print('loading from', path)
    for sp in ret:
        files = {sub: os.path.join(path, sub, f'{sp}.jsonl') for sub in subs}
        if '/vl' in path:
            # if ret == 'dev': continue  # don't need dev
            ret[sp] = VLdb(files, **kwargs)
        else:
            ret[sp] = TREx(files, **kwargs)
    return ret
