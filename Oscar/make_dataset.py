import argparse

from torch.utils import data
from oscar.datasets.build import build_dataset

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('-d', "--data_dir", default="oscar/datasets/", type=str,
                        help="The input data dir. "
                             "Should contain the .yaml files for the task.")
    parser.add_argument('-f', "--dataset_file", default="coco_flickr30k_gqa.yaml", type=str,
                        help="The training dataset yaml file.")
    parser.add_argument("--extra_dataset_file", default=None, type=str, required=False,
                        help="The extra training dataset yaml file.")

    parser.add_argument("--chunk_start_id", default=-1, type=int,
                        help="Image Chunk Start ID")
    parser.add_argument("--chunk_end_id", default=-1, type=int,
                        help="Image Chunk End ID")
    parser.add_argument("--max_img_seq_length", default=50, type=int,
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int,
                        help="The Image Feature Dimension.")

    parser.add_argument("--use_b", type=int, default=1, help="use_b")        
    parser.add_argument("--textb_sample_mode", type=int, default=0,
                        help="0: sample from both texta&textb, "
                            "1: sample from textb, "
                            "2: sample from QA answers")
    parser.add_argument("--extra_textb_sample_mode", type=int, default=1)
    parser.add_argument("--texta_false_prob", type=float, default=0.0,
                        help="the probality that we sample wrong texta, should in [0.0, 0.5]")

    parser.add_argument("--model_name_or_path", default="pretrained_base/checkpoint-2000000/", type=str,
                        help="Path to pre-trained model or shortcut.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=35, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--on_memory", default='True',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")

    parser.add_argument("--mask_loss_for_unmatched", type=int, default=1,
                        help="masked language model loss for unmatched triplets")
    parser.add_argument(
        "--use_gtlabels",
        type=int, default=1,
        help="use groundtruth labels for text b or not"
    )

    args = parser.parse_args()
    if args.texta_false_prob < 0.5 and (args.texta_false_prob > 0 or not args.use_b):
        args.num_contrast_classes = 3
    else:
        args.num_contrast_classes = 2

    dataset = build_dataset(args)[0]
    print(dataset[0])

    # evaluate the pretrained base model on one data instance

    


if __name__ == "__main__":
    main()
