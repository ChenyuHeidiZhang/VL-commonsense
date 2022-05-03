# VL-commonsense

## Datasets
Data sets and code to mine data from Visual Genome are in `mine-data` directory. Also contains `eval_dataset.py` for comparing the VG dataset with CoDa and Wikipedia.

## Models
We train two models: `Distilled` and `CaptionBERT`.
The model checkpoints can be accessed at [this link](https://drive.google.com/drive/folders/1Kbd2aWLMU57Rgt8UMehUiHyyCNnqn5gY?usp=sharing)

## Probing
The `probing` directory contains code for zero shot evaluation ("best template" mode) -- `eval_zero_shot.py` -- and the logistic regression classification case -- `eval_classification.py`. `emb_for_size.py` contains code for the "adjective projection" method for size evaluation.
`plot_snippet.py` contains the plotting scripts for the heatmap and the dot-to-dot linked plot for individual objects. 

Run files in `probing` from the parent directory. E.g.
```
python probing/eval_zero_shot.py
```

## Prompt Tuning
Code for soft prompt tuning are in the `soft-prompts` directory. (source reference: https://github.com/hiaoxui/soft-prompts). Custom evaluation ("average template" case in the paper) is in `soft-prompts/soft_prompts/run/model_eval.py`. Run eval with `config-vl-eval.yaml` and prompt training with `config-vl.yaml`. See the README there for more information.


<!---
Figures: 
* Figure 2, 6, 7 (heatmap and linked plots): probing/plot_snippet.py
* Figure 3, 5 (size plots): probing/emb_for_size.py
-->
