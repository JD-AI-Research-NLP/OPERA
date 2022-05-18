import torch
from argparse import ArgumentParser


def add_data_args(parser: ArgumentParser):
    parser.add_argument("--gpu_num", default=torch.cuda.device_count(), type=int, help="training gpu num.")
    parser.add_argument("--workers", default=4, type=int, help="wroker num.")

    # parser.add_argument("--gpu_num", default=-1, type=int, help="training gpu num.")
    parser.add_argument("--data_dir", default="", type=str, required=True, help="data dir.")
    parser.add_argument("--save_dir", default="", type=str, required=True, help="save dir.")
    parser.add_argument("--log_file", default="train.log", type=str, help="train log file.")


def add_train_args(parser: ArgumentParser):
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--log_per_updates", default=20, type=int, help="log pre update size.")
    parser.add_argument("--max_epoch", default=5, type=int, help="max epoch.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="learning rate.")
    parser.add_argument("--grad_clipping", default=1.0, type=float, help="gradient clip.")
    parser.add_argument('--warmup', type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_schedule", default="warmup_linear", type=str, help="warmup schedule.")
    parser.add_argument("--optimizer", default="adam", type=str, help="train optimizer.")
    parser.add_argument('--seed', type=int, default=2018, help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--pre_path', type=str, default=None, help="Load from pre trained.")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout.")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size.")
    parser.add_argument('--eval_batch_size', type=int, default=32, help="eval batch size.")
    parser.add_argument("--eps", default=1e-8, type=float, help="ema gamma.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=0)


def add_bert_args(parser: ArgumentParser):
    parser.add_argument("--bert_learning_rate", type=float, help="bert learning rate.")
    parser.add_argument("--bert_weight_decay", type=float, help="bert weight decay.")
    parser.add_argument("--pretrain_model", type=str, help="pretrain modle path.")


def add_model_args(parser: ArgumentParser):
    parser.add_argument("--use_gcn", action="store_true", help="Using graph infomation.")
    parser.add_argument("--gcn_steps", default=1, type=int, help="max gcn steps.")
    parser.add_argument('--fp16', default=False, action="store_true")
    parser.add_argument('--add_op_layer', default=False, action="store_true")
    parser.add_argument('--pretrain_dataset', type=str, default="squad_data")
    parser.add_argument('--pretrain_stag', default=False, action="store_true")
    parser.add_argument("--use_pretrained_model", default=False, action="store_true")
    parser.add_argument("--pretrained_model_path", type=str,
                        default="squad_data/processed_data_electra.large/pretrained_model_squad/checkpoint_best.pt")
    parser.add_argument('--use_pretrained_predicted_module', default=False, action="store_true")

    parser.add_argument('--continue_train', default=False, action="store_true")
    parser.add_argument("--trained_model_path", type=str,
                        default="processed_data_electra.large/electra_trained_model_add_relation_module/checkpoint_best.pt")
    parser.add_argument('--trained_optimizer_path', type=str,
                        default="processed_data_electra.large/electra_trained_model_add_relation_module/checkpoint_best.ot")
    parser.add_argument('--continue_epoch_nums', type=int, default=5)
    parser.add_argument("--is_sample", default=False, action="store_true")


def add_inference_args(parser: ArgumentParser):
    parser.add_argument("--pre_path", type=str, help="Prepath")
    parser.add_argument("--data_mode", type=str, help="inference data mode")
    parser.add_argument("--inf_path", type=str, help="inference data path.")
    parser.add_argument("--dump_path", type=str, help="inference data path.")
    parser.add_argument('--eval_batch_size', type=int, default=32, help="eval batch size.")
