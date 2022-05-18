import os
import pickle
import torch
import random
from .token import Token

TOKEN_TYPE = ['OTHER', 'ENTITY', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL',
              'CARDINAL', 'YARD']
ENTITY_TOKEN_TYPE = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']
TOKEN_TYPE_ID = [i for i in range(len(TOKEN_TYPE))]
TOKEN_TYPE_MAP_ID = dict(zip(TOKEN_TYPE, TOKEN_TYPE_ID))
NUMBER_TOKEN_TYPE = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'YARD']
NUMBER_TOKEN_TYPE_ID = [TOKEN_TYPE_MAP_ID[token_type] for token_type in NUMBER_TOKEN_TYPE]
NUMBER_TOKEN_TYPE_MAP_ID = dict(zip(NUMBER_TOKEN_TYPE, NUMBER_TOKEN_TYPE_ID))

class DropBatchGen(object):
    def __init__(self, args, tokenizer, data, padding_idx=1, add_token_type: bool = False,
                 number_token_type_embedding=None):
        self.args = args
        self.cls_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.sep_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.padding_idx = tokenizer.pad_token_id
        self.is_train = False
        self.vocab_size = len(tokenizer)
        self.tokenizer = tokenizer
        all_data = []
        for item in data:
            question_tokens = tokenizer.convert_tokens_to_ids(item["question_tokens"])
            passage_tokens = tokenizer.convert_tokens_to_ids(item["passage_tokens"])
            question_passage_tokens = [ Token(text=item[0], idx=item[1][0], edx=item[1][1] ) for item in zip(item["question_passage_tokens"],
                    [(0,0)] + item["question_token_offsets"] + [(0,0)]+ item["passage_token_offsets"] + [(0, 0)])]
            item["question_passage_tokens"] = question_passage_tokens
            all_data.append((question_tokens, passage_tokens, item))

        print("Load data size {}.".format(len(all_data)))

        self.data = DropBatchGen.make_baches(all_data, args.batch_size if self.is_train else args.eval_batch_size,
                                                  self.is_train)
        self.offset = 0

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

            aux_number_order = None
            aux_number_ids = None

            entity_node_indices = None
            number_node_indices = None
            entity_node_mask = None
            number_node_mask = None
            entities_numbers_relation = None
            numbers_numbers_relation = None

            max_seq_len = max([len(q) + len(p) for q, p in zip(q_tokens, p_tokens)]) + 3
            max_num_len = max([1] + [len(item["number_indices"]) for item in metas])
            max_qnum_len = max([1] + [len(item["question_number_indices"]) for item in metas])
            if self.args.add_aux_nums:
                max_aux_num_len = max([len(item["aux_number_as_tokens"]) for item in metas])
                aux_number_ids = torch.LongTensor(bsz, max_aux_num_len).fill_(-1)
                aux_number_order = torch.LongTensor(bsz, max_aux_num_len).fill_(-1)

            max_pans_choice = max([1] + [len(item["answer_passage_spans"]) for item in metas])
            max_qans_choice = max([1] + [len(item["answer_question_spans"]) for item in metas])
            max_sign_choice = max([1] + [len(item["signs_for_add_sub_expressions"]) for item in metas])

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

            # answer infos
            answer_as_passage_spans = torch.LongTensor(bsz, max_pans_choice, 2).fill_(-1)
            answer_as_question_spans = torch.LongTensor(bsz, max_qans_choice, 2).fill_(-1)
            answer_as_add_sub_expressions = torch.LongTensor(bsz, max_sign_choice, max_num_len).fill_(0)
            answer_as_counts = torch.LongTensor(bsz).fill_(-1)

            # multiple span label
            max_text_answers = max([1] + [0 if len(metas[i]["multi_span"]) < 1 else
                                          len(metas[i]["multi_span"][1])
                                          for i in range(bsz)])
            max_answer_spans = max([1] + [0 if len(metas[i]["multi_span"]) < 1 else
                                          max([len(item) for item in metas[i]["multi_span"][1]])
                                          for i in range(bsz)])
            max_correct_sequences = max([1] + [0 if len(metas[i]["multi_span"]) < 1 else
                                               len(metas[i]["multi_span"][2])
                                               for i in range(bsz)])
            is_bio_mask = torch.LongTensor(bsz).fill_(0)
            bio_wordpiece_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
            answer_as_text_to_disjoint_bios = torch.LongTensor(bsz, max_text_answers, max_answer_spans,
                                                               max_seq_len).fill_(0)
            answer_as_list_of_bios = torch.LongTensor(bsz, max_correct_sequences, max_seq_len).fill_(0)
            span_bio_labels = torch.LongTensor(bsz, max_seq_len).fill_(0)

            if self.args.add_relation_reasoning_module:
                ## question sentence and passage
                # max_sentence_num_in_passage = max([1] + [len(item["all_sentence_nodes"]) for item in metas])
                max_entity_node_num_in_passage = max([1] + [len(item["all_entities_nodes"]) for item in metas])
                max_number_node_num_in_passage = max([1] + [len(item["all_number_nodes"]) for item in metas])

                # sentence_indices_in_passage = torch.LongTensor(bsz, max_sentence_num_in_passage, max_seq_len).fill_(0)
                entity_node_indices_in_passage = torch.LongTensor(bsz, max_entity_node_num_in_passage,
                                                                  max_seq_len).fill_(0)
                number_node_indices_in_passage = torch.LongTensor(bsz, max_number_node_num_in_passage,
                                                                  max_seq_len).fill_(0)

                # sentence_mask_in_passage = torch.LongTensor(bsz, max_sentence_num_in_passage).fill_(0)
                entity_node_mask_in_passage = torch.LongTensor(bsz, max_entity_node_num_in_passage).fill_(0)
                number_node_mask_in_passage = torch.LongTensor(bsz, max_number_node_num_in_passage).fill_(0)

                # question information in relation_reasoning_module
                max_entity_num_in_question = max(
                    [1] + [len(item["all_entities_nodes_in_question"]) for item in metas])
                max_number_node_num_in_question = max(
                    [1] + [len(item["all_number_nodes_in_question"]) for item in metas])

                entity_node_indices_in_question = torch.LongTensor(bsz, max_entity_num_in_question, max_seq_len).fill_(
                    0)
                number_node_indices_in_question = torch.LongTensor(bsz, max_number_node_num_in_question,
                                                                   max_seq_len).fill_(0)

                entity_mask_in_question = torch.LongTensor(bsz, max_entity_num_in_question).fill_(0)
                number_mask_in_question = torch.LongTensor(bsz, max_number_node_num_in_question).fill_(0)

                max_entity_node_num = max_entity_num_in_question + max_entity_node_num_in_passage
                max_number_node_num = max_number_node_num_in_question + max_number_node_num_in_passage
                entities_numbers_relation = torch.LongTensor(bsz, max_entity_node_num, max_number_node_num).fill_(0)
                numbers_numbers_relation = {}
                for number_type in NUMBER_TOKEN_TYPE:
                    numbers_numbers_relation_type = torch.LongTensor(bsz, max_number_node_num,
                                                                     max_number_node_num).fill_(0)
                    numbers_numbers_relation.update({number_type: numbers_numbers_relation_type})

            for i in range(bsz):
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

                if self.args.add_aux_nums:
                    aux_number_ids[i, :] = torch.LongTensor(
                        self.tokenizer.convert_tokens_to_ids(metas[i]["aux_number_as_tokens"]))
                    aux_number_order[i, :] = torch.LongTensor(metas[i]["aux_number_order"])

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
                if self.args.add_relation_reasoning_module:
                    # for idx, (sentence_node_id, sentence_node) in enumerate(
                    #         list(metas[i]["all_sentence_nodes"].items())):
                    #     start_index = sentence_node.get_start() + passage_start
                    #     end_index = sentence_node.get_end() + passage_start
                    #     sentence_indices_in_passage[i][idx][start_index:end_index] = 1
                    #     sentence_mask_in_passage[i][idx] = 1

                    for idx, (entity_id, entity_node) in enumerate(list(metas[i]["all_entities_nodes"].items())):
                        start_index = entity_node.get_start() + passage_start
                        end_index = entity_node.get_end() + passage_start
                        entity_node_indices_in_passage[i][idx][start_index:end_index] = 1
                        entity_node_mask_in_passage[i][idx] = 1

                    for idx, (number_id, number_node) in enumerate(metas[i]["all_number_nodes"].items()):
                        start_index = number_node.get_start() + passage_start
                        end_index = number_node.get_end() + passage_start
                        number_node_indices_in_passage[i][idx][start_index:end_index] = 1
                        number_node_mask_in_passage[i][idx] = 1

                    ## relation between number node and number node
                    for idx, (entity_id, entity_node) in enumerate(
                            list(metas[i]["all_entities_nodes_in_question"].items())):
                        start_index = entity_node.get_start() + question_start
                        end_index = entity_node.get_end() + question_start
                        entity_node_indices_in_question[i][idx][start_index:end_index] = 1
                        entity_mask_in_question[i][idx] = 1

                    for idx, (number_id, number_node) in enumerate(metas[i]["all_number_nodes_in_question"].items()):
                        start_index = number_node.get_start() + question_start
                        end_index = number_node.get_end() + question_start
                        number_node_indices_in_question[i][idx][start_index:end_index] = 1
                        number_mask_in_question[i][idx] = 1

                    ## question information in relation reasoning module
                    for idx, (question_node_id, entity_number_node_pairs) in enumerate(
                            list(metas[i]["entities_numbers_relation_in_question"].items())):
                        entity_nodes = entity_number_node_pairs[0]
                        number_nodes = entity_number_node_pairs[1]
                        for en_id, entity_node in list(entity_nodes.items()):
                            for num_id, number_node in list(number_nodes.items()):
                                entity_node_id = entity_node.get_id()
                                number_node_id = number_node.get_id()
                                question_entities_nodes_id = list(metas[i]["all_entities_nodes_in_question"].keys())
                                question_numbers_nodes_id = list(metas[i]["all_number_nodes_in_question"].keys())

                                if entity_node_id not in question_entities_nodes_id:
                                    print('entity_node_id', entity_node_id)
                                    print('all_entities_nodes_id', question_entities_nodes_id)
                                if number_node_id not in question_numbers_nodes_id:
                                    print('number_node_id', number_node_id)
                                    print('all_numbers_nodes_id', question_entities_nodes_id)
                                    print('number_nodes', number_nodes)

                                entity_node_index = question_entities_nodes_id.index(entity_node_id)
                                number_node_index = question_numbers_nodes_id.index(number_node_id)
                                entities_numbers_relation[i][entity_node_index][number_node_index] = 1

                    ## relstion between entity node and number node
                    for idx, (sentence_node_id, entity_value_node_pairs) in enumerate(
                            list(metas[i]["entities_numbers_relation"].items())):
                        entity_nodes = entity_value_node_pairs[0]
                        number_nodes = entity_value_node_pairs[1]
                        for en_id, entity_node in list(entity_nodes.items()):
                            for num_id, number_node in list(number_nodes.items()):
                                entity_node_id = entity_node.get_id()
                                number_node_id = number_node.get_id()
                                all_entities_nodes_id = list(metas[i]["all_entities_nodes"].keys())
                                all_numbers_nodes_id = list(metas[i]["all_number_nodes"].keys())
                                if entity_node_id not in all_entities_nodes_id:
                                    print('entity_node_id', entity_node_id)
                                    print('all_entities_nodes_id', all_entities_nodes_id)
                                if number_node_id not in all_numbers_nodes_id:
                                    print('number_node_id', number_node_id)
                                    print('all_numbers_nodes_id', all_numbers_nodes_id)
                                    print('number_nodes', number_nodes)
                                entity_node_index = all_entities_nodes_id.index(entity_node_id)
                                number_node_index = all_numbers_nodes_id.index(number_node_id)
                                entities_numbers_relation[i][entity_node_index + max_entity_num_in_question][
                                    number_node_index + max_number_node_num_in_question] = 1

                    for number_type in NUMBER_TOKEN_TYPE:
                        if number_type in metas[i]["number_type_cluster_in_question"].keys():
                            number_nodes_in_question_type = metas[i]["number_type_cluster_in_question"][number_type]
                        else:
                            number_nodes_in_question_type = {}
                        if number_type in metas[i]["number_type_cluster"].keys():
                            number_nodes_in_passage_type = metas[i]["number_type_cluster"][number_type]
                        else:
                            number_nodes_in_passage_type = {}
                        numbers_numbers_relation_type = numbers_numbers_relation[number_type]

                        for number_node1 in number_nodes_in_question_type:
                            for number_node2 in number_nodes_in_question_type:
                                number_node1_id = number_node1.get_id()
                                number_node2_id = number_node2.get_id()
                                all_numbers_nodes_id_in_question = list(metas[i]['all_number_nodes_in_question'].keys())
                                if number_node1_id not in all_numbers_nodes_id_in_question:
                                    print("i", i)
                                    print("number type", number_type)
                                    print('all_numbers_nodes_id_in_question', all_numbers_nodes_id_in_question)
                                    print('number_node1_id', number_node1_id)
                                number_node1_index = all_numbers_nodes_id_in_question.index(number_node1_id)
                                number_node2_index = all_numbers_nodes_id_in_question.index(number_node2_id)
                                if number_node1_index != number_node2_index:
                                    numbers_numbers_relation_type[i][number_node1_index][number_node2_index] = 1
                                # numbers_numbers_relation_type[i][number_node1_index][number_node2_index] = 1

                        for number_node1 in number_nodes_in_passage_type:
                            for number_node2 in number_nodes_in_passage_type:
                                number_node1_id = number_node1.get_id()
                                number_node2_id = number_node2.get_id()
                                all_numbers_nodes_id_in_passage = list(metas[i]['all_number_nodes'].keys())
                                if number_node1_id not in all_numbers_nodes_id_in_passage:
                                    print('number_node1_id', number_node1_id)
                                    print('all_numbers_nodes_id_in_passage', all_numbers_nodes_id_in_passage)

                                number_node1_index = all_numbers_nodes_id_in_passage.index(number_node1_id)
                                number_node2_index = all_numbers_nodes_id_in_passage.index(number_node2_id)
                                if number_node1_index != number_node2_index:
                                    numbers_numbers_relation_type[i][
                                        number_node1_index + max_number_node_num_in_question][
                                        number_node2_index + max_number_node_num_in_question] = 1

                                # numbers_numbers_relation_type[i][
                                #     number_node1_index + max_number_node_num_in_question][
                                #     number_node2_index + max_number_node_num_in_question] = 1

                        numbers_numbers_relation.update({number_type: numbers_numbers_relation_type})

                if self.args.add_relation_reasoning_module:
                    entity_node_indices = torch.cat([entity_node_indices_in_question, entity_node_indices_in_passage],
                                                    dim=1)
                    number_node_indices = torch.cat([number_node_indices_in_question, number_node_indices_in_passage],
                                                    dim=1)
                    entity_node_mask = torch.cat([entity_mask_in_question, entity_node_mask_in_passage], dim=1)
                    number_node_mask = torch.cat([number_mask_in_question, number_node_mask_in_passage], dim=1)

                # add multi span prediction
                cur_seq_len = q_len + p_len + 3
                bio_wordpiece_mask[i, :cur_seq_len] = torch.LongTensor(metas[i]["wordpiece_mask"][:cur_seq_len])
                if len(metas[i]["multi_span"]) > 0:
                    is_bio_mask[i] = metas[i]["multi_span"][0]
                    span_bio_labels[i, :cur_seq_len] = torch.LongTensor(metas[i]["multi_span"][-1][:cur_seq_len])
                    for j in range(len(metas[i]["multi_span"][1])):
                        for k in range(len(metas[i]["multi_span"][1][j])):
                            answer_as_text_to_disjoint_bios[i, j, k, :cur_seq_len] = torch.LongTensor(
                                metas[i]["multi_span"][1][j][k][:cur_seq_len])
                    for j in range(len(metas[i]["multi_span"][2])):
                        answer_as_list_of_bios[i, j, :cur_seq_len] = torch.LongTensor(
                            metas[i]["multi_span"][2][j][:cur_seq_len])

            out_batch = {"input_ids": input_ids, "input_mask": input_mask, "input_segments": input_segments,
                         "passage_mask": passage_mask, "question_mask": question_mask, "number_indices": number_indices,
                         "passage_number_order": passage_number_order,
                         "question_number_order": question_number_order,
                         "question_number_indices": question_number_indices,
                         "answer_as_passage_spans": answer_as_passage_spans,
                         "answer_as_question_spans": answer_as_question_spans,
                         "answer_as_add_sub_expressions": answer_as_add_sub_expressions,
                         "answer_as_counts": answer_as_counts.unsqueeze(1),
                         "answer_as_text_to_disjoint_bios": answer_as_text_to_disjoint_bios,
                         "answer_as_list_of_bios": answer_as_list_of_bios,
                         "span_bio_labels": span_bio_labels,
                         "is_bio_mask": is_bio_mask,
                         "bio_wordpiece_mask": bio_wordpiece_mask,
                         "aux_number_order": aux_number_order,
                         "aux_nums_ids": aux_number_ids,

                         "entity_node_indices": entity_node_indices,
                         "number_node_indices": number_node_indices,

                         "entity_node_mask": entity_node_mask,
                         "number_node_mask": number_node_mask,
                         "entities_numbers_relation": entities_numbers_relation,
                         "numbers_numbers_relation": numbers_numbers_relation,

                         "metadata": metas}
            if self.args.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()
                    if k == "numbers_numbers_relation" and out_batch["numbers_numbers_relation"]:
                        for type, numbers_numbers_type in list(out_batch["numbers_numbers_relation"].items()):
                            out_batch["numbers_numbers_relation"].update({type: numbers_numbers_type.cuda()})
            yield out_batch
