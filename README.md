# VL-commonsense

Data sets and code to mine data from Visual Genome are in `mine-data` directory. Also contains `eval_dataset.py` for comparing the VG dataset with CoDa and Wikipedia.

Code for soft prompt tuning are in the `soft-prompts` directory. (source reference: https://github.com/hiaoxui/soft-prompts). Custom evaluation ("average template" case in the paper) is in `soft-prompts/soft_prompts/run/model_eval.py`. Run eval with `config-vl-eval.yaml` and prompt training with `config-vl.yaml`. See the README there for more information.

The `probing` directory contains code for zero shot evaluation ("best template" mode) and the logistic regression classification case. Also contains the plotting scripts for the heatmap and the dot-to-dot plot. Run files in `probing` from the parent directory.

Figures: 
* Figure 2, 6, 7 (heatmap and linked plots): probing/plot_snippet.py
* Figure 3, 5 (size plots): probing/emb_for_size.py

