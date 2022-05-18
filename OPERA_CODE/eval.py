import argparse
import os
import json
import torch.nn as nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--data_root_dir', type=str, default='numnet_plus_data')
    parser.add_argument('--data_dir', type=str, default='drop_dataset', help='')
    parser.add_argument('--code_dir', type=str, default='.')
    parser.add_argument('--model_dir', type=str,
                        default='numnet_plus_345_LR_0.0005_BLR_1.5e-05_WD_5e-05_BWD_0.01_True')
    parser.add_argument('--max_epoch', type=str, default=10)
    parser.add_argument('--batch_size', type=str, default=16)
    parser.add_argument('--eval_batch_size', type=str, default=16)
    parser.add_argument('--TMSPAN', default=False, action='store_true')

    parser.add_argument('--add_aux_nums', default=False, action="store_true")
    parser.add_argument('--add_relation_reasoning_module', default=False, action="store_true")
    parser.add_argument('--add_sign_weight_decomp', default=False, action="store_true",
                        help="whether to decomposition the sign weight matrix")
    parser.add_argument('--add_segment', default=False, action="store_true")
    parser.add_argument('--use_gcn', default=False, action="store_true")
    parser.add_argument('--pretrain_model', type=str, default='roberta.base')

    args, _ = parser.parse_known_args()

    print(args)
    DATA_ROOT_DIR = os.path.join('..', args.data_root_dir)
    DATA_DIR = os.path.join(DATA_ROOT_DIR, args.data_dir)

    DATA_PATH = DATA_DIR  # data path
    DUMP_PATH = os.path.join(DATA_DIR, args.pretrain_model, 'prediction_drop_dataset_dev.json')  # result path
    INF_PATH = os.path.join(DATA_DIR, args.pretrain_model, 'drop_dataset_dev.json')  # origin data path
    PRE_PATH = os.path.join(DATA_DIR, args.pretrain_model, args.model_dir, 'checkpoint_best.pt')  # pretained model path

    BERT_CONFIG = "--pretrain_model {}/{}".format(DATA_DIR, args.pretrain_model)
    if args.TMSPAN:
        "Use tag_mspan model..."
        MODEL_CONFIG = "--gcn_steps 3  --tag_mspan"
    else:
        "Use mspan model..."
        MODEL_CONFIG = "--gcn_steps 3 "

    print('start to evaluation')
    TEST_CONFIG = "--eval_batch_size {} --pre_path {} --data_mode dev --dump_path {} \
                 --inf_path {}".format(args.eval_batch_size, PRE_PATH, DUMP_PATH, INF_PATH)

    eval_cmd = ' '.join(['python3', 'roberta_predict.py', TEST_CONFIG, BERT_CONFIG, MODEL_CONFIG])

    if args.use_gcn:
        eval_cmd = ' '.join([eval_cmd, '--use_gcn'])
    if args.add_aux_nums:
        eval_cmd = ' '.join([eval_cmd, '--add_aux_nums'])

    if args.add_relation_reasoning_module:
        eval_cmd = ' '.join([eval_cmd, '--add_relation_reasoning_module'])

    if args.add_sign_weight_decomp:
        eval_cmd = ' '.join([eval_cmd, '--add_sign_weight_decomp'])

    if args.add_segment:
        eval_cmd = ' '.join([eval_cmd, '--add_segment'])

    os.system(eval_cmd)
