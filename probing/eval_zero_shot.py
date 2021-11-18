import torch
import argparse
from models import init_mlm_model

device = "cuda" if torch.cuda.is_available() else "cpu"

def fill_mask():
    pass

def run():
    parser = argparse.ArgumentParser(description='zero-shot eval parser')
    parser.add_argument('--model', type=str, default='bert',
                        help='name of the model (bert, roberta, albert, oscar, or distil_bert)')
    parser.add_argument('--model_size', type=str, default='base',
                        help='size of the model (base, large)')
    parser.add_argument('--relation', type=str, default='shape',
                        help='relation to evaluate (shape, material, color, coda, coda_any...)')
    parser.add_argument('--group', type=str, default='',
                        help='group to evaluate (single, multi, any, or '' for all))')
    parser.add_argument('--seed', type=int, default=1,
                        help='numpy random seed')
    args = parser.parse_args()

    model, tokenizer = init_mlm_model(args.model, args.model_size, device)

if __name__ == '__main__':
    run()
