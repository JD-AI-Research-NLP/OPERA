import os
import pickle
import argparse
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
# from transformers import AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import ElectraTokenizer, ElectraModel
from transformers import AlbertTokenizer, AlbertModel

# from mspan_roberta_gcn.drop_roberta_dataset import DropReader
from tag_mspan_robert_gcn.drop_roberta_mspan_dataset import DropReader as TDropReader
from tag_mspan_robert_gcn.drop_reader import DropReader as TDropReader

import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument('--file_path', type=str, default="")
    parser.add_argument("--passage_length_limit", type=int, default=463)
    parser.add_argument("--question_length_limit", type=int, default=46)
    parser.add_argument("--processor_nums", type=int, default=1)
    parser.add_argument("--pretrain_model", type=str, default='roberta.base', help="pretrained model")
    parser.add_argument("--is_sample", default=False, action='store_true')

    args = parser.parse_args()

#     tokenizer = RobertaTokenizer.from_pretrained(os.path.join(args.input_path, args.pretrain_model))
#     tokenizer = ElectraTokenizer.from_pretrained(os.path.join(args.input_path, args.pretrain_model))
    tokenizer = AlbertTokenizer.from_pretrained(os.path.join(args.input_path, args.pretrain_model))

    print(args)
    dev_reader = TDropReader(args.pretrain_model,
                             tokenizer, args.passage_length_limit, args.question_length_limit)

    train_reader = TDropReader(args.pretrain_model,
                               tokenizer, args.passage_length_limit, args.question_length_limit,
                               skip_when_all_empty=["passage_span", "question_span", "addition_subtraction", "counting",
                                                    "multi_span"]
                               )
    if args.is_sample:
        file_name = os.path.join(args.file_path, "sample_drop_dataset_{}.json")
        data_mode = ["train", 'dev']
        for dm in data_mode:
            dpath = os.path.join(args.input_path, file_name.format(dm))

            data = dev_reader._read(dpath, args.processor_nums) if dm == "dev" else train_reader._read(dpath, args.processor_nums)            
            print("Save data to {}.".format(
                    os.path.join(os.path.join(args.output_dir, args.file_path, 'processed_data_' + args.pretrain_model),
                                 "sample_tmspan_cached_roberta_{}.pkl".format(dm))))

            with open(
                    os.path.join(os.path.join(args.output_dir, args.file_path, 'processed_data_' + args.pretrain_model),
                                 "sample_tmspan_cached_roberta_{}.pkl".format(dm)), "wb") as f:
                pickle.dump(data, f)
            f.close()

    else:
        file_name = os.path.join(args.file_path, "drop_dataset_{}.json")
        data_mode = ["train", "dev"]
        for dm in data_mode:
            import pdb
#             pdb.set_trace()
            dpath = os.path.join(args.input_path, file_name.format(dm))
            data = dev_reader._read(dpath, args.processor_nums) if dm == "dev" else train_reader._read(dpath, args.processor_nums)               
            print("Save data to {}.".format(
                    os.path.join(os.path.join(args.output_dir, args.file_path, 'processed_data_' + args.pretrain_model),
                                 "tmspan_cached_roberta_{}.pkl".format(dm))))
            with open(
                    os.path.join(os.path.join(args.output_dir, args.file_path, 'processed_data_' + args.pretrain_model),
                                 "tmspan_cached_roberta_{}.pkl".format(dm)), "wb") as f:
                pickle.dump(data, f)
            f.close()


if __name__=="__main__":
    main()