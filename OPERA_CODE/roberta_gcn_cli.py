import os
import json
import sys

sys.path.append(os.getcwd())
print(os.getcwd())
import options
import argparse
from pprint import pprint
from tools.model import DropBertModel
from mspan_roberta_gcn.roberta_batch_gen import DropBatchGen
from mspan_roberta_gcn.mspan_roberta_gcn import NumericallyAugmentedBertNet
from tag_mspan_robert_gcn.roberta_batch_gen_tmspan import DropBatchGen as TDropBatchGen
from tag_mspan_robert_gcn.tag_mspan_roberta_gcn import NumericallyAugmentedBertNet as TNumericallyAugmentedBertNet
from datetime import datetime
from tools.utils import create_logger, set_environment
from transformers import AutoModel, AutoTokenizer, AutoConfig
# from transformers import ElectraModel, ElectraTokenizer
import torch.nn as nn
import torch

from torch.utils.data import DataLoader, RandomSampler
from tag_mspan_robert_gcn.drop_dataloader import DropBatchGen, create_collate_fn


def main():


    parser = argparse.ArgumentParser("Bert training task.")
    options.add_bert_args(parser)
    options.add_model_args(parser)
    options.add_data_args(parser)
    options.add_train_args(parser)
    args = parser.parse_args()
    pprint(args)
    
    args.cuda = args.gpu_num > 0
    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f)
        
    logger = create_logger("Bert Drop Pretraining", log_file=os.path.join(args.save_dir, args.log_file))
    set_environment(args.seed, args.cuda)
    logger.info("Build Drop model.")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)
    if args.use_pretrained_model:
        print("Load from pre path {}.".format(args.pretrained_model_path))
        bert_model = AutoModel.from_config(AutoConfig.from_pretrained(args.pretrain_model))
    else:
        bert_model = AutoModel.from_pretrained(args.pretrain_model)

    if args.pretrain_stag:
        answering_abilities = ["passage_span_extraction", "question_span_extraction"]
    else:
        answering_abilities = ["passage_span_extraction", "question_span_extraction",
                               "addition_subtraction", "counting", "multiple_spans"]
        
    print('answering_abilities', answering_abilities)
    network = TNumericallyAugmentedBertNet(bert_model,
                                           hidden_size=bert_model.config.hidden_size,
                                           dropout_prob=args.dropout,
                                           use_gcn=args.use_gcn,
                                           answering_abilities=answering_abilities,
                                           gcn_steps=args.gcn_steps,
                                           add_op_layer=args.add_op_layer,
                                           bert_name=args.pretrain_model,
                                           )

    if args.use_pretrained_model:
        print("use pretrained model on the datasets")
        state_dict = torch.load(args.pretrained_model_path)
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if "bert." in key or "_op_embdding" in key or "OP_MODULES" in key:
                encoder_state_dict.update({key:value})
        network.load_state_dict(encoder_state_dict, strict=False)

    import pdb
   
    
    best_result = float("-inf")
    logger.info("Loading data...")
    collate_fn = create_collate_fn(1, args.cuda)
    train_dataset = DropBatchGen(args, data_mode="train", tokenizer=tokenizer)
    train_sampler = RandomSampler(train_dataset)
    #train_itr = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=False)
    train_itr = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=False)
    dev_dataset = DropBatchGen(args, data_mode="dev", tokenizer=tokenizer)
    dev_itr = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)
    
    num_train_steps = int(args.max_epoch * len(train_itr) / args.gradient_accumulation_steps)
    logger.info("Num update steps {}!".format(num_train_steps))
    logger.info("Build optimizer etc...")
    model = DropBertModel(args, network, num_train_step=num_train_steps)
    

    train_start = datetime.now()
    first = True
    start_epoch = 1
    end_epoch = args.max_epoch
    for epoch in range(start_epoch, end_epoch + 1):
        model.avg_reset()
#         if not first:
#             train_itr.reset()
#         first = False
        logger.info('At epoch {}'.format(epoch))
        for step, batch in enumerate(train_itr):
            model.update(batch)
            if model.step % (args.log_per_updates * args.gradient_accumulation_steps) == 0 or model.step == 1:
#                 logger.info("Updates[{0:6}] train loss[{1:.5f}] train em[{2:.5f}] f1[{3:.5f}] remaining[{4}]".format(
#                     model.updates, model.train_loss.avg, model.em_avg.avg, model.f1_avg.avg,
#                     str((datetime.now() - train_start) / (step + 1) * (num_train_steps - step - 1)).split('.')[0]))
                logger.info("Updates[{0:6}] train loss[{1:.5f}]] remaining[{2}]".format(
                    model.updates, model.train_loss.avg, str((datetime.now() - train_start) / (step + 1) * (num_train_steps - step - 1)).split('.')[0]))
                
                
                
                model.avg_reset()
            if model.step % 2000 == 0 and step != 0:
                total_num, eval_loss, eval_em, eval_f1 = model.evaluate(dev_itr)
                logger.info(
                    "Eval {} examples, result in epoch {}, eval loss {}, eval em {} eval f1 {}.".format(total_num,
                                                                                                        epoch,
                                                                                                        eval_loss,
                                                                                                        eval_em,
                                                                                                        eval_f1))
                if eval_f1 > best_result:
                    save_prefix = os.path.join(args.save_dir, "checkpoint_best")
                    model.save(save_prefix, step, epoch)
                    best_result = eval_f1
                    logger.info("Best eval F1 {} at step {} epoch {}".format(best_result, step, epoch))

        total_num, eval_loss, eval_em, eval_f1 = model.evaluate(dev_itr)
        logger.info(
            "Eval {} examples, result in epoch {}, eval loss {}, eval em {} eval f1 {}.".format(total_num, epoch,
                                                                                                eval_loss, eval_em,
                                                                                                eval_f1))

        if eval_f1 > best_result:
            save_prefix = os.path.join(args.save_dir, "checkpoint_best")
            model.save(save_prefix, step, epoch)
            best_result = eval_f1
            logger.info("Best eval F1 {} at step {} epoch {}".format(best_result, step, epoch))

        logger.info("training in {} hours!".format((datetime.now() - train_start).seconds / 3600))


if __name__ == '__main__':
    main()
