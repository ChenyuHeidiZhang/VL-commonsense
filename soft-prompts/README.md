# Soft Prompts

Reference: https://github.com/hiaoxui/soft-prompts

```bibtex
@inproceedings{qin-eisner-2021-learning,
    title = "Learning How to Ask: Querying {LM}s with Mixtures of Soft Prompts",
    author = "Qin, Guanghui  and
      Eisner, Jason",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.410",
    pages = "5203--5212",
}
```

## Setup

We assume you are running soft-prompts on Linux with GPU available and anaconda3 installed.
To set up the environment, please run

```shell
conda create -n soft-prompts python=3.7
conda activate soft-prompts
conda install -y pytorch==1.7.0 cudatoolkit=11.0 -c pytorch
pip install transformers==4.0.0 pyyaml tqdm
```

## Data

The `prompts/vl` folder contains the prompts used for the vision-langauge tasks.
The `db/vl` folder contains the relations we used for experiments, including CoDa, color, shape, material, size_larger, size_smaller, and cooccur. Data is split into train and test sets; the dev set is only a placeholder and not used for our purpose.
(Some of the data used in the original paper for T-REx, Google-RE, and ConceptNet are kept for reference.)

## Experiment

To replicate our results with soft-prompt tuning, run the following commands:

```shell
cd soft-prompts
python3 -m soft_prompts.run.experiment config-vl.yaml  # to train
python3 -m soft_prompts.run.model_eval config-vl-eval.yaml  # to eval
```


