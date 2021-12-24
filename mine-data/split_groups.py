# Split shape, material, color datasets into Single, Multi, and Any groups

import os
import json
import numpy as np
import utils

def write_data(file, data_pts, mode='w'):
    with open(file, mode) as out:
        for data in data_pts:
            json.dump(data, out)
            out.write('\n')

def split_groups(dir, spl, dist_json):
    file = os.path.join(dir, f'{spl}.jsonl')

    data_pts = [[],[],[]]
    with open(file, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            dist = np.array(dist_json[data['sub']])
            dist = np.sort(dist / np.sum(dist))[::-1]  # normalize and sort
            if dist[0] >= 0.85:
                data_pts[0].append(data)
            elif dist[:3].sum() >= 0.9:
                data_pts[1].append(data)
            else:
                data_pts[2].append(data)

    for i, group in enumerate(['single', 'multi', 'any']):
        path = os.path.join(dir, group)
        if not os.path.isdir(path):
            os.mkdir(path)
        if spl == 'train':
            write_data(os.path.join(path, 'train.jsonl'), data_pts[i], mode='w')
        else:  # dev and test splits are combined
            write_data(os.path.join(path, 'test.jsonl'), data_pts[i], mode='w')

relation = 'material'
dist_json = utils.load_dist_file(f'{relation}')
for spl in ['train', 'test']:  # 'dev'
    split_groups(f'db/{relation}', spl, dist_json)
