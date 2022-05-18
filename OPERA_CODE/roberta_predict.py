import json
import torch
import options
import argparse
from tqdm import tqdm
# from mspan_roberta_gcn.inference_batch_gen import DropBatchGen
# from mspan_roberta_gcn.mspan_roberta_gcn import NumericallyAugmentedBertNet
# from mspan_roberta_gcn.drop_roberta_dataset import DropReader
# from tag_mspan_robert_gcn.drop_roberta_mspan_dataset import DropReader as TDropReader
# from tag_mspan_robert_gcn.inference_batch_gen import DropBatchGen as TDropBatchGen
# from tag_mspan_robert_gcn.tag_mspan_roberta_gcn import NumericallyAugmentedBertNet as TNumericallyAugmentedBertNet

from tag_mspan_robert_gcn.tag_mspan_roberta_gcn import NumericallyAugmentedBertNet as TNumericallyAugmentedBertNet
from tag_mspan_robert_gcn.drop_dataloader import DropBatchGen, create_collate_fn
from tag_mspan_robert_gcn.drop_reader import DropReader
from tag_mspan_robert_gcn.drop_chunking_reader import DropReader as DropChunkingReader

from torch.utils.data import DataLoader, RandomSampler

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from transformers import ElectraTokenizer, ElectraModel, ElectraConfig
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig

import torch.nn as nn
import os
import pickle


def load_trained_model(args, checkpoint_path):
#     tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_model)
    bert_model = AutoModel.from_config(AutoConfig.from_pretrained(args.pretrain_model))

    
    
    
    
    print("Build bert model.")
    network = TNumericallyAugmentedBertNet(bert_model,
                                           hidden_size=bert_model.config.hidden_size,
                                           dropout_prob=0,
                                           use_gcn=args.use_gcn,
                                           gcn_steps=args.gcn_steps,
                                           bert_name=args.pretrain_model,
                                           add_op_layer = args.add_op_layer
                                           )

    if args.cuda: network.cuda()
    print("Load from pretrained path {}.".format(checkpoint_path))
    network.load_state_dict(torch.load(checkpoint_path), strict=False)
    return network


def predict_model(args, trained_model, inf_iter, output_file, output_answer_text_file):
    print("start to inference")
    with torch.no_grad():
        result = {}
        result_text = {}
        result_op_prediction = {}
        result_answer_type = {}
        import pdb
        for batch in tqdm(inf_iter):
            import pdb
#             pdb.set_trace()
            trained_model.eval()
            output_dict = trained_model(**batch)
            for i in range(len(output_dict["question_id"])):
                if output_dict["question_id"][i] not in result:
                    result[output_dict["question_id"][i]] = [output_dict['answer'][i]]
                else:
                    result[output_dict["question_id"][i]] += [output_dict['answer'][i]]

        question_ids = result.keys()

        for question_id in question_ids:
            answer_type_doc_id_map = {}
            instance_answer_of_all_docs = result[question_id]
            for doc_id, instance_answer_of_all_doc in enumerate(instance_answer_of_all_docs):
                answer_type = instance_answer_of_all_doc['answer_type']
                max_score = max(instance_answer_of_all_doc['answer_type_probs'])
                if answer_type not in answer_type_doc_id_map.keys():
                    assert not isinstance(max_score, list)
                    answer_type_doc_id_map[answer_type] = [[doc_id], max_score]
                else:
                    answer_type_doc_id_map[answer_type][0] += [doc_id]
                    assert not isinstance(max_score, list)
                    if max_score > answer_type_doc_id_map[answer_type][1]:
                        answer_type_doc_id_map[answer_type][1] = max_score

            answer_type_doc_id_map_list = sorted(answer_type_doc_id_map.items(), key=lambda item: item[1][1],
                                                 reverse=True)

            best_answer_type = answer_type_doc_id_map_list[0][0]
            best_answer_doc_id_candidate_for_instance = answer_type_doc_id_map_list[0][1][0]
            best_answer_candidates = [instance_answer_of_all_docs[id] for id in
                                      best_answer_doc_id_candidate_for_instance]
            
            if best_answer_type == "passage_span":
                selected_best_answer_text_candidate = ""
                selected_best_answer_candidate = None
                
                op_result = []
                
                best_span_score = -1e15
                for id, answer_candidate in enumerate(best_answer_candidates):
                    passage_span_best_start_log_probs = answer_candidate['passage_span_best_start_log_probs']
                    passage_span_best_end_log_probs = answer_candidate['passage_span_best_end_log_probs']
                    if passage_span_best_start_log_probs + passage_span_best_end_log_probs + max(
                            answer_candidate['answer_type_probs']) > best_span_score:
                        best_span_score = passage_span_best_start_log_probs + passage_span_best_end_log_probs + max(
                            answer_candidate['answer_type_probs'])
                        selected_best_answer_candidate = answer_candidate
                        selected_best_answer_text_candidate = answer_candidate["predicted_answer"]
                        op_result = answer_candidate["op_weight"]
                        
            if best_answer_type == "question_span":
                best_span_score = -1e15
                selected_best_answer_text_candidate = ""
                selected_best_answer_candidate = None
                op_result = []

                for id, answer_candidate in enumerate(best_answer_candidates):
                    question_span_best_start_log_probs = answer_candidate['question_span_best_start_log_probs']
                    question_span_best_end_log_probs = answer_candidate['question_span_best_end_log_probs']
                    if question_span_best_start_log_probs + question_span_best_end_log_probs + max(
                            answer_candidate['answer_type_probs']) > best_span_score:
                        best_span_score = question_span_best_start_log_probs + question_span_best_end_log_probs + max(
                            answer_candidate['answer_type_probs'])
                        selected_best_answer_candidate = answer_candidate
                        selected_best_answer_text_candidate = answer_candidate["predicted_answer"]
                        op_result = answer_candidate["op_weight"]

            if best_answer_type == "arithmetic":
                best_number_score = -1e15
                selected_best_answer_text_candidate = '0'
                selected_best_answer_candidate = None
                op_result = []
                for id, answer_candidate in enumerate(best_answer_candidates):
                    combination_log_prob = answer_candidate['best_combination_log_prob']
                    if combination_log_prob > best_number_score:
                        best_number_score = combination_log_prob
                        selected_best_answer_candidate = answer_candidate
                        selected_best_answer_text_candidate = answer_candidate["predicted_answer"]
                        op_result = answer_candidate["op_weight"]

                        
            if best_answer_type == "count":
                best_count_score = -1e15
                selected_best_answer_text_candidate = '0'
                selected_best_answer_candidate = None
                op_result = []
                for id, answer_candidate in enumerate(best_answer_candidates):
                    count_probs = max(answer_candidate['probs'])
                    if count_probs + max(answer_candidate['answer_type_probs']) > best_count_score:
                        selected_best_answer_candidate = answer_candidate
                        best_count_score = count_probs + max(answer_candidate['answer_type_probs'])
                        selected_best_answer_text_candidate = answer_candidate["predicted_answer"]

            if best_answer_type == "multiple_spans":
                best_multiple_score = -1e15
                selected_best_answer_text_candidate = ""
                selected_best_answer_candidate = None
                op_result = []
                for id, answer_candidate in enumerate(best_answer_candidates):
                    if max(answer_candidate["answer_type_probs"]) > best_multiple_score:
                        best_multiple_score = max(answer_candidate["answer_type_probs"])
                        selected_best_answer_candidate = answer_candidate
                        selected_best_answer_text_candidate = answer_candidate["predicted_answer"]
                        op_result = answer_candidate["op_weight"]

            result[question_id] = selected_best_answer_candidate
            result_text[question_id] = selected_best_answer_text_candidate
            
            op_threshold = 1/11
            op_result_hard = [1 if weight>op_threshold else 0 for weight in op_result]
            result_op_prediction[question_id] = op_result_hard
            result_answer_type[question_id]= best_answer_type
            
    data_path = os.path.dirname(output_file)

    
    print('start to write the predict file')
    with open(output_file, 'w') as f:
        print('write predict answer to file', output_file)
        json.dump(result, f, indent=4)
    f.close()

    with open(output_answer_text_file, 'w') as f:
        print("write predict answer text into file", output_answer_text_file)
        json.dump(result_text, f, indent=4)
    f.close()

    op_path = os.path.join(data_path, "op_prediction_{}.json".format(args.stag))
    with open(op_path, 'w') as f:
        json.dump(result_op_prediction, f, indent=4)
    f.close()
    
    answer_type_path = os.path.join(data_path, 'answer_type_prediction_{}.json'.format(args.stag))
    with open(answer_type_path, 'w') as f:
        json.dump(result_answer_type, f, indent=4)
    f.close()
    
    return result


def load_data(args, inf_path):
    print("Load data from {}.".format(inf_path))
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)
    
    
    
    reader = DropReader(args.pretrain_model, tokenizer, passage_length_limit=463,
                         question_length_limit=46)
    if args.is_sample:
        if not os.path.exists(
                os.path.join(args.data_root, "processed_data_{}".format(args.pretrain_model_name),
                             'sample_tmspan_cached_roberta_{}.pkl'.format(args.stag))):
            data = reader._read(inf_path, processor_num=20)
            with open(
                    os.path.join(
                        os.path.join(args.data_root,"processed_data_{}".format(args.pretrain_model_name)),
                        'sample_tmspan_cached_roberta_{}.pkl'.format(args.stag)), "wb") as f:
                pickle.dump(data, f)
        else:
            with open(os.path.join(args.data_root, "processed_data_{}".format(args.pretrain_model_name),
                                   'sample_tmspan_cached_roberta_{}.pkl'.format(args.stag)),
                      "rb") as f:
                data = pickle.load(f)
            f.close()
            print(os.path.join(args.data_root, "processed_data_{}".format(args.pretrain_model_name),
                               'sample_tmspan_cached_roberta_{}.pkl'.format(args.stag)))
    else:
        if not os.path.exists(
                os.path.join(args.data_root, "processed_data_{}".format(args.pretrain_model_name),
                             'tmspan_cached_roberta_{}.pkl'.format(args.stag))):
            data = reader._read(inf_path, processor_num=20)
            with open(
                    os.path.join(args.data_root, "processed_data_{}".format(args.pretrain_model_name),
                        'tmspan_cached_roberta_{}.pkl'.format(args.stag)), "wb") as f:
                pickle.dump(data, f)
        else:
            print("-----------------")
            with open(
                    os.path.join(args.data_root, "processed_data_{}".format(args.pretrain_model_name),
                                 'tmspan_cached_roberta_{}.pkl'.format(args.stag)),
                    "rb") as f:
                data = pickle.load(f)
            f.close()
            
    print('=======start process evaluate data=========')
    collate_fn = create_collate_fn(1, args.cuda)     
    dev_dataset = DropBatchGen(args, data_mode=args.stag, tokenizer=tokenizer, data=data)
    inf_iter = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)

    return inf_iter




def load_chunking_data(args, inf_path):
    print("Load data from {}.".format(inf_path))
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)    
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_model)
    
    reader = DropChunkingReader(args.pretrain_model, tokenizer, passage_length_limit=463,
                         question_length_limit=46, doc_chunk_stride=50)
    if args.is_sample:
        if not os.path.exists(
                os.path.join(args.data_root, "processed_data_{}".format(args.pretrain_model_name), 'chunking_data', 'sample_tmspan_cached_roberta_{}.pkl'.format(args.stag))):
            data = reader._read(inf_path, processor_num=20)
            with open(
                    os.path.join(args.data_root,"processed_data_{}".format(args.pretrain_model_name), "chunking_data", 'sample_tmspan_cached_roberta_{}.pkl'.format(args.stag)), "wb") as f:
                pickle.dump(data, f)
        else:
            with open(os.path.join(args.data_root, "processed_data_{}".format(args.pretrain_model_name), "chunking_data", 'sample_tmspan_cached_roberta_{}.pkl'.format(args.stag)), "rb") as f:
                data = pickle.load(f)
            f.close()
            print(os.path.join(args.data_root, "processed_data_{}".format(args.pretrain_model_name),"chunking_data",
                               'sample_tmspan_cached_roberta_{}.pkl'.format(args.stag)))
    else:
        if not os.path.exists(
                os.path.join(args.data_root, "processed_data_{}".format(args.pretrain_model_name),"chunking_data",
                             'tmspan_cached_roberta_{}.pkl'.format(args.stag))):
            data = reader._read(inf_path, processor_num=20)
            with open(
                    os.path.join(args.data_root, "processed_data_{}".format(args.pretrain_model_name),"chunking_data",
                        'tmspan_cached_roberta_{}.pkl'.format(args.stag)), "wb") as f:
                pickle.dump(data, f)
        else:
            print("-----------------")
            with open(
                    os.path.join(args.data_root, "processed_data_{}".format(args.pretrain_model_name),"chunking_data",
                                 'tmspan_cached_roberta_{}.pkl'.format(args.stag)),
                    "rb") as f:
                data = pickle.load(f)
            f.close()
            
    print('=======start process evaluate data=========')
    collate_fn = create_collate_fn(1, args.cuda)     
    dev_dataset = DropBatchGen(args, data_mode=args.stag, tokenizer=tokenizer,data=data)
    inf_iter = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)

    return inf_iter







if __name__ == '__main__':
    parser = argparse.ArgumentParser("Bert inference task.")
    parser.add_argument('--data_root', type=str, default="../numnet_plus_data/drop_dataset")
    parser.add_argument('--trained_model_path', type=str, default="")
    parser.add_argument("--data_dir", type=str, default="../numnet_plus_data/drop_dataset")

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)

    parser.add_argument("--warmup_schedule", default="warmup_cosine", type=str, help="warmup schedule.")
    parser.add_argument('--use_gcn', default=False, action="store_true")
    parser.add_argument('--gcn_steps', default=3, type=int)
    parser.add_argument('--pretrain_model', type=str, default="electra.large")
    parser.add_argument('--pretrain_model_name', type=str, default="electra.large")
    parser.add_argument('--add_op_layer', default=False, action="store_true")

    parser.add_argument('--inf_path', type=str, default="drop_dataset_dev.json")
    parser.add_argument('--model_dir', type=str, default="")
    parser.add_argument('--is_sample', default=False, action="store_true")
    parser.add_argument("--stag", type=str, default="dev", help="dev or test")
    parser.add_argument("--chunking", default=False, action="store_true")
    parser.add_argument("--checkpoint_path", default="", help="checkpoint_path")

    args, _ = parser.parse_known_args()
    args.cuda = torch.cuda.device_count() > 0

    
    DATA_DIR = args.data_root
    args.data_dir = os.path.join(DATA_DIR, 'processed_data_{}'.format(args.pretrain_model_name))
    if args.stag=="dev":
        if args.is_sample:
            args.inf_path = "sample_drop_dataset_{}.json".format(args.stag)
            INF_PATH = os.path.join(DATA_DIR, args.inf_path)
        else:
            args.inf_path = "drop_dataset_{}.json".format(args.stag)
            INF_PATH = os.path.join(DATA_DIR, args.inf_path)
            
    if args.stag=="test":
        args.inf_path = "drop_dataset_test_questions.json"
        INF_PATH = os.path.join(DATA_DIR, args.inf_path)
    args.pretrain_model = os.path.join(DATA_DIR, args.pretrain_model_name)

    checkpoint_path = os.path.join(DATA_DIR, args.checkpoint_path,
                                   "checkpoint_best.pt")
    
    import pdb
    #pdb.set_trace()
    if not args.chunking:
        print("checkpoint_path", checkpoint_path)
        trained_model = load_trained_model(args, checkpoint_path)
        predict_result_path_1 = os.path.join(DATA_DIR, args.checkpoint_path,
                                             "sample_{}_prediction_drop_dataset_{}.json".format(
                                                 args.is_sample, args.stag))
        predict_text_path_1 = os.path.join(DATA_DIR, args.checkpoint_path,
                                           "sample_{}_prediction_text_drop_dataset_{}.json".format(
                                               args.is_sample, args.stag))
        inf_iter_1 = load_data(args, INF_PATH)
        print('inf_iter_1', inf_iter_1)
        predict_result_model_1 = predict_model(args, trained_model, inf_iter_1, predict_result_path_1,
                                               predict_text_path_1)
        if args.stag=="dev":
            eval_cmd = "python3 drop_eval.py --gold_path {} --prediction_path {}".format(INF_PATH, predict_text_path_1)
            os.system(eval_cmd)
    else:
        print("checkpoint_path", checkpoint_path)
        trained_model = load_trained_model(args, checkpoint_path)
        predict_result_path_1 = os.path.join(DATA_DIR, args.checkpoint_path,
                                             "chunking/sample_{}_prediction_drop_dataset_{}.json".format(
                                                 args.is_sample, args.stag))
        predict_text_path_1 = os.path.join(DATA_DIR, args.checkpoint_path,
                                           "chunking/sample_{}_prediction_text_drop_dataset_{}.json".format(
                                               args.is_sample, args.stag))
        inf_iter_1 = load_chunking_data(args, INF_PATH)
        print('inf_iter_1', inf_iter_1)
        predict_result_model_1 = predict_model(args, trained_model, inf_iter_1, predict_result_path_1,
                                               predict_text_path_1)
        if args.stag=="dev":
            eval_cmd = "python3 drop_eval.py --gold_path {} --prediction_path {}".format(INF_PATH, predict_text_path_1)
            os.system(eval_cmd)
