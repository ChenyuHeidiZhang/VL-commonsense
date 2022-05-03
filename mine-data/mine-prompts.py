import numpy as np
import json

# experimental for inspirations for prompts; not used in the end

def mine_prompts(type, print_every=1000, max_annotations=10000):
    # Read a list of sub,obj pairs and find those pairs in COCO caption annotations.
    # Output templates that connect the sub and obj.
    pairs_file = f'db/{type}.jsonl'
    captions_file = 'annotations/captions_val2017.json'

    print(f'loading sub,obj file of type {type}...')
    pairs_f = open(pairs_file, 'r').readlines()
    print('loading captions file...')
    captions_f = json.load(open(captions_file))
    pairs = []
    for line in pairs_f:
        pair = json.loads(line.strip())
        pairs.append((pair['sub'], pair['obj']))
    
    prompts = set()
    print('mining prompts...')
    ann_num = 0
    for annotation in captions_f['annotations']:
        ann_num += 1
        if ann_num % print_every == 0:
            print(f'processing {ann_num}-th annotation')
        cap = annotation['caption']
        for pair in pairs:
            cap_split = cap.split()
            if pair[0] in cap_split and pair[1] in cap_split:
                idx0 = cap_split.index(pair[0])
                idx1 = cap_split.index(pair[1])
                if idx0 < idx1:
                    # before = '' if idx0 == 0 else cap_split[idx0-1]
                    # after = '' if idx1 == len(cap_split)-1 else cap_split[idx1+1]
                    # prompt = [before, '[X]'] + cap_split[idx0+1:idx1] + ['[Y]', after]
                    prompt = ['[X]'] + cap_split[idx0+1:idx1] + ['[Y]', '.']
                else:
                    # before = '' if idx1 == 0 else cap_split[idx1-1]
                    # after = '' if idx0 == len(cap_split)-1 else cap_split[idx0+1]
                    prompt = ['[Y]'] + cap_split[idx1+1:idx0] + ['[X]', '.']
                prompts.add(' '.join(prompt))

        if ann_num == max_annotations:
            break
    print(f'finished processing {ann_num} annotations')

    # output
    out_file = 'prompts/' + type + '.jsonl'
    with open(out_file, 'w') as out:
        for template in prompts:
            json.dump({"template": template}, out)
            out.write('\n')

if __name__ == "__main__":
    mine_prompts('shape')
