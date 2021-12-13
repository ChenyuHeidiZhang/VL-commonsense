from utils import load_dist_file, load_word_file

for rel in ['shape']:
    print(rel)
    rel_dist = load_dist_file(rel)['bear']
    total = sum(rel_dist)
    words = load_word_file(rel)

    for i in range(len(rel_dist)):
        if rel_dist[i] > 2:
            print(rel_dist[i]/total, words[i])
    print()
