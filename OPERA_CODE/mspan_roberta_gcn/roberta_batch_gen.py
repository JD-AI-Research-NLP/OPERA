import os
import pickle
import torch
import random
from torch.autograd import Variable


class DropBatchGen(object):
    def __init__(self, args, data_mode, tokenizer, bert_model, padding_idx=1):
        self.tokenizer = tokenizer
        self.args = args
        self.cls_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.sep_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.padding_idx = padding_idx
        self.is_train = data_mode == "train"
        self.vocab_size = len(tokenizer)
        dpath = "sample_cached_roberta_{}.pkl".format(data_mode)
        with open(os.path.join(args.data_dir, dpath), "rb") as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)

        all_data = []
        for item in data:
            question_tokens = tokenizer.convert_tokens_to_ids(item["question_tokens"])
            passage_tokens = tokenizer.convert_tokens_to_ids(item["passage_tokens"])
            all_data.append((question_tokens, passage_tokens, item))

        print("Load data size {}.".format(len(all_data)))

        self.data = DropBatchGen.make_baches(all_data, args.batch_size if self.is_train else args.eval_batch_size,
                                             self.is_train)
        self.offset = 0

        # embedding
        # self.word_embdding = bert_model.embeddings.word_embeddings

    @staticmethod
    def make_baches(data, batch_size=32, is_train=True):
        if is_train:
            random.shuffle(data)
        if is_train:
            return [
                data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[:i + batch_size - len(data)]
                for i in range(0, len(data), batch_size)]
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
            for i in range(len(self.data)):
                random.shuffle(self.data[i])
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            self.offset += 1
            q_tokens, p_tokens, metas = zip(*batch)
            bsz = len(batch)
            max_seq_len = max([len(q) + len(p) for q, p in zip(q_tokens, p_tokens)]) + 3
            max_num_len = max([1] + [len(item["number_indices"]) for item in metas])
            max_qnum_len = max([1] + [len(item["question_number_indices"]) for item in metas])
            max_aux_num_len = max([len(item["aux_number_as_tokens"]) for item in metas])

            max_pans_choice = min(8, max([1] + [len(item["answer_passage_spans"]) for item in metas]))
            max_qans_choice = min(8, max([1] + [len(item["answer_question_spans"]) for item in metas]))
            max_sign_choice = max([1] + [len(item["signs_for_add_sub_expressions"]) for item in metas])

            ## relation reasoning module
            max_sentence_num = max([1] + [len(item["all_sentence_nodes"]) for item in metas])
            max_entity_num = max([1] + [len(item["all_entities_nodes"]) for item in metas])
            max_value_num = max([1] + [len(item["all_values_nodes"]) for item in metas])

            sentence_indices_in_passage = torch.LongTensor(bsz, max_sentence_num, max_seq_len).fill_(0)
            entity_indices_in_passage = torch.LongTensor(bsz, max_entity_num, max_seq_len).fill_(0)
            value_indices_in_passage = torch.LongTensor(bsz, max_value_num).fill_(0)

            sentence_mask = torch.LongTensor(bsz, max_sentence_num).fill_(0)
            entity_mask = torch.LongTensor(bsz, max_entity_num).fill_(0)
            value_mask = torch.LongTensor(bsz, max_value_num).fill_(0)

            sentences_entities_relation = torch.LongTensor(bsz, max_sentence_num, max_entity_num).fill_(0)
            sentences_values_relation = torch.LongTensor(bsz, max_sentence_num, max_value_num).fill_(0)
            entities_values_relation = torch.LongTensor(bsz, max_entity_num, max_value_num).fill_(0)
            entities_entities_relation = torch.LongTensor(bsz, max_entity_num, max_entity_num).fill_(0)

            # question information in relation_reasoning_module
            max_question_entity_num = max([1] + [len(item["question_entity_nodes"]) for item in metas])
            max_question_value_num = max([1] + [len(item["question_value_nodes"]) for item in metas])
            question_indices = torch.LongTensor(bsz, 1, max_seq_len).fill_(0)
            entity_indices_in_question = torch.LongTensor(bsz, max_question_entity_num, max_seq_len).fill_(0)
            value_indices_in_question = torch.LongTensor(bsz, max_question_value_num).fill_(0)

            question_entities_relation = torch.LongTensor(bsz, 1, max_question_entity_num).fill_(0)
            question_values_relation = torch.LongTensor(bsz, 1, max_question_value_num).fill_(0)
            question_entities_values_relation = torch.LongTensor(bsz, max_question_entity_num,
                                                                 max_question_value_num).fill_(0)
            question_entities_entities_relation = torch.LongTensor(bsz, max_question_entity_num,
                                                                   max_question_entity_num).fill_(0)

            # same_entity_mention_relation = torch.LongTensor(bsz, max_entity_num, max_entity_num).fill_(0)
            # same_entity_mention_with_q_relation = torch.LongTensor(bsz, max_entity_num, max_question_entity_num).fill_(
            #     0)

            question_entity_mask = torch.LongTensor(bsz, max_question_entity_num).fill_(0)
            question_value_mask = torch.LongTensor(bsz, max_question_value_num).fill_(0)
            question_mask = torch.LongTensor(bsz, 1).fill_(0)

            # qa input.
            input_ids = torch.LongTensor(bsz, max_seq_len).fill_(self.padding_idx)
            input_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
            input_segments = torch.LongTensor(bsz, max_seq_len).fill_(0)
            passage_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
            question_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)

            # number infos
            number_indices = torch.LongTensor(bsz, max_num_len).fill_(-1)
            question_number_indices = torch.LongTensor(bsz, max_qnum_len).fill_(-1)
            passage_number_order = torch.LongTensor(bsz, max_num_len).fill_(-1)
            question_number_order = torch.LongTensor(bsz, max_qnum_len).fill_(-1)
            # aux_number_embeddings = torch.FloatTensor(bsz, max_aux_num_len,
            #                                          self.word_embdding.embedding_dim).fill_(-1)

            aux_number_ids = torch.LongTensor(bsz, max_aux_num_len).fill_(-1)
            aux_number_order = torch.LongTensor(bsz, max_aux_num_len).fill_(-1)

            # answer infos
            answer_as_passage_spans = torch.LongTensor(bsz, max_pans_choice, 2).fill_(-1)
            answer_as_question_spans = torch.LongTensor(bsz, max_qans_choice, 2).fill_(-1)
            answer_as_add_sub_expressions = torch.LongTensor(bsz, max_sign_choice, max_num_len).fill_(0)
            answer_as_counts = torch.LongTensor(bsz).fill_(-1)

            # answer span number
            span_num = torch.LongTensor(bsz).fill_(0)

            for i in range(bsz):
                span_num[i] = min(8, len(metas[i]["answer_texts"]))
                q_len = len(q_tokens[i])
                p_len = len(p_tokens[i])
                # input and their mask
                input_ids[i, :3 + q_len + p_len] = torch.LongTensor(
                    [self.cls_idx] + q_tokens[i] + [self.sep_idx] + p_tokens[i] + [self.sep_idx])
                input_mask[i, :3 + q_len + p_len] = 1
                question_mask[i, 1:1 + q_len] = 1
                passage_mask[i, 2 + q_len: 2 + q_len + p_len] = 1
                if self.args.add_segment:
                    input_segments[i][:2 + q_len] = 0
                    input_segments[i][2 + q_len:3 + q_len + p_len] = 1
                passage_start = q_len + 2
                question_start = 1
                # number infos
                pn_len = len(metas[i]["number_indices"]) - 1
                if pn_len > 0:
                    number_indices[i, :pn_len] = passage_start + torch.LongTensor(metas[i]["number_indices"][:pn_len])
                    passage_number_order[i, :pn_len] = torch.LongTensor(metas[i]["passage_number_order"][:pn_len])
                    number_indices[i, pn_len - 1] = 0
                qn_len = len(metas[i]["question_number_indices"]) - 1
                if qn_len > 0:
                    question_number_indices[i, :qn_len] = question_start + torch.LongTensor(
                        metas[i]["question_number_indices"][:qn_len])
                    question_number_order[i, :qn_len] = torch.LongTensor(metas[i]["question_number_order"][:qn_len])

                if metas[i]['aux_number_as_tokens']:
                    aux_number_ids[i, :] = torch.LongTensor(
                        self.tokenizer.convert_tokens_to_ids(metas[i]["aux_number_as_tokens"]))
                    aux_number_order[i] = torch.LongTensor(metas[i]["aux_number_order"])

                # answer info
                pans_len = min(len(metas[i]["answer_passage_spans"]), max_pans_choice)
                for j in range(pans_len):
                    answer_as_passage_spans[i, j, 0] = metas[i]["answer_passage_spans"][j][0] + passage_start
                    answer_as_passage_spans[i, j, 1] = metas[i]["answer_passage_spans"][j][1] + passage_start

                qans_len = min(len(metas[i]["answer_question_spans"]), max_qans_choice)
                for j in range(qans_len):
                    answer_as_question_spans[i, j, 0] = metas[i]["answer_question_spans"][j][0] + question_start
                    answer_as_question_spans[i, j, 1] = metas[i]["answer_question_spans"][j][1] + question_start

                # answer sign info
                sign_len = min(len(metas[i]["signs_for_add_sub_expressions"]), max_sign_choice)
                for j in range(sign_len):
                    answer_as_add_sub_expressions[i, j, :pn_len] = torch.LongTensor(
                        metas[i]["signs_for_add_sub_expressions"][j][:pn_len])

                # answer count info
                if len(metas[i]["counts"]) > 0:
                    answer_as_counts[i] = metas[i]["counts"][0]

                ## relation reasoning module
                for idx, (sentence_node_id, sentence_node) in enumerate(list(metas[i]["all_sentence_nodes"].items())):
                    start_index = sentence_node.get_start() + passage_start
                    end_index = sentence_node.get_end() + passage_start
                    sentence_indices_in_passage[i][idx][start_index:end_index] = 1
                    sentence_mask[i][idx] = 1

                for idx, (entity_id, entity_node) in enumerate(list(metas[i]["all_entities_nodes"].items())):
                    start_index = entity_node.get_start() + passage_start
                    end_index = entity_node.get_end() + passage_start
                    entity_indices_in_passage[i][idx][start_index:end_index] = 1
                    entity_mask[i][idx] = 1

                for idx, (value_id, value_node) in enumerate(metas[i]["all_values_nodes"].items()):
                    index = value_node.get_index() + passage_start
                    value_indices_in_passage[i][idx] = index
                    value_mask[i][idx] = 1

                ## sentence entity
                for idx, (sentence_node_id, sentence_node) in enumerate(list(metas[i]["all_sentence_nodes"].items())):
                    if sentence_node_id in metas[i]["sentences_entities"].keys():
                        entity_nodes = metas[i]["sentences_entities"][str(sentence_node_id)]
                        for id, entity_node in entity_nodes.items():
                            entity_node_id = entity_node.get_id()
                            all_entities_nodes_id = list(metas[i]["all_entities_nodes"].keys())
                            index = all_entities_nodes_id.index(entity_node_id)
                            sentences_entities_relation[i][idx][index] = 1

                ## sentence values

                for idx, (sentence_node_id, sentence_node) in enumerate(list(metas[i]["all_sentence_nodes"].items())):
                    if sentence_node_id in metas[i]["sentences_values"].keys():
                        value_nodes = metas[i]["sentences_values"][str(sentence_node_id)]
                        for id, values_node in value_nodes.items():
                            value_node_id = values_node.get_id()
                            all_values_nodes_id = list(metas[i]["all_values_nodes"].keys())
                            index = all_values_nodes_id.index(value_node_id)
                            sentences_values_relation[i][idx][index] = 1

                ## entities values

                for idx, (sentence_node_id, entity_node_pairs) in enumerate(
                        list(metas[i]["entities_entities"].items())):
                    entity_nodes1 = entity_node_pairs[0]
                    entity_nodes2 = entity_node_pairs[1]
                    for id1, entity_node1 in list(entity_nodes1.items()):
                        for id2, entity_node2 in list(entity_nodes2.items()):
                            entity_node_id1 = entity_node1.get_id()
                            entity_node_id2 = entity_node2.get_id()
                            all_entities_nodes_id = list(metas[i]["all_entities_nodes"].keys())
                            index1 = all_entities_nodes_id.index(entity_node_id1)
                            index2 = all_entities_nodes_id.index(entity_node_id2)
                            if index1 != index2:
                                entities_entities_relation[i][index1][index2] = 1

                for idx, (sentence_node_id, entity_value_node_pairs) in enumerate(
                        list(metas[i]["entities_values"].items())):
                    entity_nodes = entity_value_node_pairs[0]
                    value_nodes = entity_value_node_pairs[1]
                    for en_id, entity_node in list(entity_nodes.items()):
                        for va_id, value_node in list(value_nodes.items()):
                            entity_node_id = entity_node.get_id()
                            value_node_id = value_node.get_id()
                            all_entities_nodes_id = list(metas[i]["all_entities_nodes"].keys())
                            all_values_nodes_id = list(metas[i]["all_values_nodes"].keys())
                            entity_node_index = all_entities_nodes_id.index(entity_node_id)
                            value_node_index = all_values_nodes_id.index(value_node_id)
                            entities_values_relation[i][entity_node_index][value_node_index] = 1

                ## same entity mention
                # for entity_node_pairs in metas[i]["same_entity_mention_relation"]:
                #     entity_node1 = entity_node_pairs[0]
                #     entity_node2 = entity_node_pairs[1]
                #     entity_node1_id = entity_node1.get_id()
                #     entity_node2_id = entity_node2.get_id()
                #     all_entities_nodes_id = list(metas[i]["all_entities_nodes"].keys())
                #     entity_node1_index = all_entities_nodes_id.index(entity_node1_id)
                #     entity_node2_index = all_entities_nodes_id.index(entity_node2_id)
                #     same_entity_mention_relation[i][entity_node1_index][entity_node2_index] = 1

                ## question information in relation reasoning module
                question_indices[i][0][question_start:question_start + q_len] = 1
                question_mask[i][0] = 1
                for idx, (entity_id, entity_node) in enumerate(list(metas[i]["question_entity_nodes"].items())):
                    start_index = entity_node.get_start() + question_start
                    end_index = entity_node.get_end() + question_start

                    entity_indices_in_question[i][idx][start_index:end_index] = 1

                    question_entity_mask[i][idx] = 1

                for idx, (value_id, value_node) in enumerate(metas[i]["question_value_nodes"].items()):
                    index = value_node.get_index() + question_start
                    value_indices_in_question[i][idx] = index
                    question_value_mask[i][idx] = 1

                for idx, (question_node_id, question_node) in enumerate(
                        list(metas[i]["question_entity_nodes"].items())):
                    if question_node_id in metas[i]["question_entity_relation"].keys():
                        question_entity_nodes = metas[i]["question_entity_relation"][str(question_node_id)]
                        for id, question_entity_node in question_entity_nodes.items():
                            question_entity_node_id = question_entity_node.get_id()
                            question_entity_nodes_id = list(metas[i]["question_entity_nodes"].keys())
                            index = question_entity_nodes_id.index(question_entity_node_id)

                            question_entities_relation[i][idx][index] = 1

                ## sentence values

                for idx, (question_node_id, question_node) in enumerate(list(metas[i]["question_value_nodes"].items())):
                    if question_node_id in metas[i]["question_value_relation"].keys():
                        question_value_nodes = metas[i]["question_value_relation"][str(question_node_id)]
                        for id, question_value_node in question_value_nodes.items():
                            question_value_node_id = question_value_node.get_id()
                            question_value_nodes_id = list(metas[i]["question_value_nodes"].keys())
                            index = question_value_nodes_id.index(question_value_node_id)
                            question_values_relation[i][idx][index] = 1

                ## entities values

                for idx, (question_node_id, entity_node_pairs) in enumerate(
                        list(metas[i]["question_entity_entity_relation"].items())):
                    entity_nodes1 = entity_node_pairs[0]
                    entity_nodes2 = entity_node_pairs[1]
                    for id1, entity_node1 in list(entity_nodes1.items()):
                        for id2, entity_node2 in list(entity_nodes2.items()):
                            entity_node_id1 = entity_node1.get_id()
                            entity_node_id2 = entity_node2.get_id()
                            question_entities_nodes_id = list(metas[i]["question_entity_nodes"].keys())
                            index1 = question_entities_nodes_id.index(entity_node_id1)
                            index2 = question_entities_nodes_id.index(entity_node_id2)
                            if index1 != index2:
                                question_entities_entities_relation[i][index1][index2] = 1

                for idx, (question_node_id, entity_value_node_pairs) in enumerate(
                        list(metas[i]["question_entity_value_relation"].items())):
                    entity_nodes = entity_value_node_pairs[0]
                    value_nodes = entity_value_node_pairs[1]
                    for en_id, entity_node in list(entity_nodes.items()):
                        for va_id, value_node in list(value_nodes.items()):
                            entity_node_id = entity_node.get_id()
                            value_node_id = value_node.get_id()
                            question_entities_nodes_id = list(metas[i]["question_entity_nodes"].keys())
                            question_values_nodes_id = list(metas[i]["question_value_nodes"].keys())
                            entity_node_index = question_entities_nodes_id.index(entity_node_id)
                            value_node_index = question_values_nodes_id.index(value_node_id)
                            question_entities_values_relation[i][entity_node_index][value_node_index] = 1

                ## same entity mention
                # for entity_node_pairs in metas[i]['same_entity_mention_with_q_relation']:
                #     entity_node_p = entity_node_pairs[0]
                #     entity_node_q = entity_node_pairs[1]
                #     entity_node_p_id = entity_node_p.get_id()
                #     entity_node_q_id = entity_node_q.get_id()
                #     all_entities_nodes_id = list(metas[i]["all_entities_nodes"].keys())
                #     question_entity_nodes_id = list(metas[i]["question_entity_nodes"].keys())
                #     entity_node_p_index = all_entities_nodes_id.index(entity_node_p_id)
                #     entity_node_q_index = question_entity_nodes_id.index(entity_node_q_id)
                #     same_entity_mention_with_q_relation[i][entity_node_p_index][entity_node_q_index] = 1

            out_batch = {"input_ids": input_ids, "input_mask": input_mask, "input_segments": input_segments,
                         "passage_mask": passage_mask, "question_mask": question_mask, "number_indices": number_indices,
                         "aux_number_order": aux_number_order,
                         "aux_nums_ids": aux_number_ids,
                         "passage_number_order": passage_number_order,
                         "question_number_order": question_number_order,
                         "question_number_indices": question_number_indices,
                         "answer_as_passage_spans": answer_as_passage_spans,
                         "answer_as_question_spans": answer_as_question_spans,
                         "answer_as_add_sub_expressions": answer_as_add_sub_expressions,
                         "answer_as_counts": answer_as_counts.unsqueeze(1), "span_num": span_num.unsqueeze(1),

                         "sentence_indices_in_passage": sentence_indices_in_passage,
                         "entity_indices_in_passage": entity_indices_in_passage,
                         "value_indices_in_passage": value_indices_in_passage,
                         "sentences_entities_relation": sentences_entities_relation,
                         "sentences_values_relation": sentences_values_relation,
                         "entities_values_relation": entities_values_relation,
                         "entities_entities_relation": entities_entities_relation,
                         # "same_entity_mention_relation": same_entity_mention_relation,
                         "sentence_mask": sentence_mask,
                         "entity_mask": entity_mask,
                         "value_mask": value_mask,

                         "question_indices": question_indices,
                         "entity_indices_in_question": entity_indices_in_question,
                         "value_indices_in_question": value_indices_in_question,
                         "question_entities_relation": question_entities_relation,
                         "question_values_relation": question_values_relation,
                         "question_entities_values_relation": question_entities_values_relation,
                         "question_entities_entities_relation": question_entities_entities_relation,
                         # "same_entity_mention_with_q_relation": same_entity_mention_with_q_relation,
                         "question_entity_mask": question_entity_mask,
                         "question_value_mask": question_value_mask,

                         "metadata": metas}


            yield out_batch
