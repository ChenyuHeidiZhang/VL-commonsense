import numpy as np
import json
from datasets import load_dataset

ds = load_dataset("corypaik/coda", ignore_verifications=True)
COLORS = [
    'black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple',
    'red', 'white', 'yellow'
]

prompts = set()
dist = {}
data_all = {}
datasets = {}
for split in ['train', 'validation', 'test']:
    data_all[split] = []
    datasets[split] = [[], [], []]
    for instance in ds[split]:
        ngram = instance['ngram']
        prompt = instance['text'].replace(ngram, '[X]').replace('[MASK]', '[Y]')
        prompts.add(prompt)
        if ngram not in dist:
            dist[ngram] = instance['label']
            correct_color = COLORS[np.argmax(instance['label'])]
            wrong_color = COLORS[np.argmin(instance['label'])]
            js = {'sub': ngram, 'obj': correct_color, 'alt': wrong_color}
            data_all[split].append(js)
            datasets[split][instance['object_group']].append(js)

with open('distributions/coda-dist.jsonl', 'w') as f:
    json.dump(dist, f)

with open('prompts/coda.jsonl', 'w') as f:
    for temp in list(prompts):
        json.dump({'template': temp}, f)
        f.write('\n')

groups = ['single', 'multi', 'any']
for spl in ['train', 'validation', 'test']:
    split = 'dev' if spl=='validation' else spl
    with open(f'db/coda_all/{split}.jsonl', 'w') as f:
        for data in data_all[spl]:
            json.dump(data, f)
            f.write('\n')
    for i in range(3):
        with open(f'db/coda_{groups[i]}/{split}.jsonl', 'w') as f:
            for data in datasets[spl][i]:
                json.dump(data, f)
                f.write('\n')
