# Datasets: Visual Commensense Tests (ViComTe)

## Data specifics

The `words` subfolder contains relation attributes for our datasets. Representation such as "azure: blue" means that the color attribute "azure" is treated as "blue" when mined from Visual Genome.
`size_db` contains five classes of manually selected concrete nouns for building the dataset.

The `distributions` subfolder contains the distributions of subjects on attributes for the relations color, shape, material, and CoDa (Paik et al., 2021). `color-dist.jsonl`, `shape-dist.jsonl`, and `material-dist.jsonl` are mined from Visual Genome, and `wiki-*.jsonl` are mined from Wikipedia.
The attributes' order is as followed:
* color: ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'silver', 'white', 'yellow']
* shape: ['cross', 'heart', 'octagon', 'oval', 'polygon', 'rectangle', 'rhombus', 'round', 'semicircle', 'square', 'star', 'triangle']
* material: ['bronze', 'ceramic', 'cloth', 'concrete', 'cotton', 'denim', 'glass', 'gold', 'iron', 'jade', 'leather', 'metal', 'paper', 'plastic', 'rubber', 'stone', 'tin', 'wood']
* CoDa: ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple' 'red', 'white', 'yellow']
Distributions for "visual cooccurrence" can be access at [this link](`https://drive.google.com/drive/folders/1Kbd2aWLMU57Rgt8UMehUiHyyCNnqn5gY?usp=sharing`), as is too large to fit on Github.

The `db` subfolder contains (subject, object) pairs for each relation. "obj" is the top attribute associated with the "sub" and "alt" is selected from the tail distribution. This is used to calculated the top 1 accuracy. 
The data is split into 80% training and 20% testing (repeatedly select 8 into train, and then 2 into test), where the training set is used for prompt tuning and logistic regression training, and all results are reported on the test set.


## Steps taken to obtain data
1. download `attributes.json` file from Visual Genome.
2. use the `mine-vg-dist.py` script to mine relation distributions by the predefined attributes in `words`.
3. run `get_db_from_dist` in `mine-vg.py` on the selected relation to create the db.
4. run `split_groups.py` to split the db into Single, Multi, and Any groups.
5. use `mine-text.py` to mine distributions from Wikipedia, and repeat 3 and 4.
6. run `extract_size` in `mine-vg.py` to get the datasets for size.


