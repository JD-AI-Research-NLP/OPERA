import re
import json
import string
import itertools
from tqdm import tqdm
import numpy as np
from word2number.w2n import word_to_num
from typing import List, Dict, Any, Tuple
from collections import defaultdict, OrderedDict
from multiprocessing import Pool
from copy import deepcopy
import math
from mspan_roberta_gcn.util import TextProcessorSpacy
from mspan_roberta_gcn.util import Node, SentenceNode, ValueNode, EntityNode, NumberNode, Edges, SentenceAndEntity, \
    SentenceAndValue, EntityAndEntity, EntityAndValue, HeterogeneousGraph
import collections
from transformers import AutoTokenizer
from tag_mspan_robert_gcn.question_tag_parser import question_parser
import os

aux_num_list = [i for i in range(10)]


def get_number_from_word(word, improve_number_extraction=True):
    punctuation = string.punctuation.replace('-', '')
    word = word.strip(punctuation)
    word = word.replace(",", "")
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                if improve_number_extraction:
                    if re.match('^\d*1st$', word):  # ending in '1st'
                        number = int(word[:-2])
                    elif re.match('^\d*2nd$', word):  # ending in '2nd'
                        number = int(word[:-2])
                    elif re.match('^\d*3rd$', word):  # ending in '3rd'
                        number = int(word[:-2])
                    elif re.match('^\d+th$', word):  # ending in <digits>th
                        # Many occurrences are when referring to centuries (e.g "the *19th* century")
                        number = int(word[:-2])
                    elif len(word) > 1 and word[-2] == '0' and re.match('^\d+s$', word):
                        # Decades, e.g. "1960s".
                        # Other sequences of digits ending with s (there are 39 of these in the training
                        # set), do not seem to be arithmetically related, as they are usually proper
                        # names, like model numbers.
                        number = int(word[:-1])
                    elif len(word) > 4 and re.match('^\d+(\.?\d+)?/km[²2]$', word):
                        # per square kilometer, e.g "73/km²" or "3057.4/km2"
                        if '.' in word:
                            number = float(word[:-4])
                        else:
                            number = int(word[:-4])
                    elif len(word) > 6 and re.match('^\d+(\.?\d+)?/month$', word):
                        # per month, e.g "1050.95/month"
                        if '.' in word:
                            number = float(word[:-6])
                        else:
                            number = int(word[:-6])
                    else:
                        return None
                else:
                    return None
    except IndexError:
        return None
    return number


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def roberta_tokenize(text, tokenizer, is_answer=False):
    split_tokens = []
    sub_token_offsets = []

    numbers = []
    number_indices = []
    number_len = []

    word_piece_mask = []
    # char_to_word_offset = []
    word_to_char_offset = []
    prev_is_whitespace = True
    tokens = []
    for i, c in enumerate(text):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            tokens.append(c)
            word_to_char_offset.append(i)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
                word_to_char_offset.append(i)
            else:
                tokens[-1] += c
            prev_is_whitespace = False  # char_to_word_offset.append(len(tokens) - 1)

    for i, token in enumerate(tokens):
        index = word_to_char_offset[i]
        if i != 0 or is_answer:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)
        token_number = get_number_from_word(token)

        if token_number is not None:
            numbers.append(token_number)
            number_indices.append(len(split_tokens))
            number_len.append(len(sub_tokens))

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)
            sub_token_offsets.append((index, index + len(token)))

        word_piece_mask += [1]
        if len(sub_tokens) > 1:
            word_piece_mask += [0] * (len(sub_tokens) - 1)

    assert len(split_tokens) == len(sub_token_offsets)
    return split_tokens, sub_token_offsets, numbers, number_indices, number_len, word_piece_mask


def clipped_passage_num(number_indices, number_len, numbers_in_passage, plen):
    if len(number_indices) < 1 or number_indices[-1] < plen:
        return number_indices, number_len, numbers_in_passage
    lo = 0
    hi = len(number_indices) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if number_indices[mid] < plen:
            lo = mid + 1
        else:
            hi = mid
    if number_indices[lo - 1] + number_len[lo - 1] > plen:
        number_len[lo - 1] = plen - number_indices[lo - 1]
    return number_indices[:lo], number_len[:lo], numbers_in_passage[:lo]


def clipped_passage_relation_graph(all_sentence_nodes, all_entities_nodes, all_numbers_nodes,
                                   sentences_entities, sentences_numbers, plen):
    all_sentence_nodes = list(all_sentence_nodes.items())

    if len(all_sentence_nodes) < 1 or all_sentence_nodes[-1][1].end_index < plen:
        return all_sentence_nodes, all_entities_nodes, all_numbers_nodes, sentences_entities, sentences_numbers

    lo = 0
    hi = len(all_sentence_nodes)
    while lo < hi:
        mid = (lo + hi) // 2
        if all_sentence_nodes[mid][1].start_index < plen:
            lo = mid + 1
        else:
            hi = mid
    if all_sentence_nodes[lo - 1][1].end_index > plen:
        all_sentence_nodes[lo - 1][1].end_index = plen
    all_sentence_nodes = all_sentence_nodes[:lo]
    all_sentence_nodes = dict(all_sentence_nodes)

    valid_sentence_node_id = list(all_sentence_nodes.keys())
    sentence_id_include_entities = list(sentences_entities.keys())
    for id in sentence_id_include_entities:
        if id not in valid_sentence_node_id:
            del sentences_entities[id]

    if len(sentences_entities) > 0:
        last_sentence_entities = list(sentences_entities.items())[-1]  # Dict(id, node)

        last_sentence_id, last_sentence_entities_nodes = last_sentence_entities

        lo = 0
        hi = len(last_sentence_entities_nodes)
        last_sentence_entities_nodes = list(last_sentence_entities_nodes.items())
        while lo < hi:
            mid = (lo + hi) // 2
            if last_sentence_entities_nodes[mid][1].start_index < plen:
                lo = mid + 1
            else:
                hi = mid
        if last_sentence_entities_nodes[lo - 1][1].end_index > plen:
            last_sentence_entities_nodes[lo - 1][1].end_index = plen
        last_sentence_entities_nodes = last_sentence_entities_nodes[:lo]
        last_sentence_entities_nodes = dict(last_sentence_entities_nodes)
        if len(last_sentence_entities_nodes) > 0:
            sentences_entities.update({last_sentence_id: last_sentence_entities_nodes})
        else:
            del sentences_entities[last_sentence_id]
    if len(sentences_numbers) > 0:
        sentence_id_include_numbers = list(sentences_numbers.keys())
        for id in sentence_id_include_numbers:
            if id not in valid_sentence_node_id:
                del sentences_numbers[id]
        if len(sentences_numbers) > 0:
            last_sentences_numbers = list(sentences_numbers.items())[-1]
            last_sentence_id, last_sentences_numbers_nodes = last_sentences_numbers
            lo = 0
            hi = len(last_sentences_numbers_nodes)
            last_sentences_numbers_nodes = list(last_sentences_numbers_nodes.items())
            while lo < hi:
                mid = (lo + hi) // 2
                if last_sentences_numbers_nodes[mid][1].start_index < plen:
                    lo = mid + 1
                else:
                    hi = mid
            if last_sentences_numbers_nodes[lo - 1][1].end_index > plen:
                last_sentences_numbers_nodes[lo - 1][1].end_index = plen

            last_sentences_numbers_nodes = last_sentences_numbers_nodes[:lo]
            last_sentences_numbers_nodes = dict(last_sentences_numbers_nodes)

            if len(last_sentences_numbers_nodes) > 0:
                sentences_numbers.update({last_sentence_id: last_sentences_numbers_nodes})
            else:
                del sentences_numbers[last_sentence_id]

    all_entities_nodes = {}
    all_numbers_nodes = {}
    for _, entities in sentences_entities.items():
        for _, entity_node in entities.items():
            # start_index = entity_node.get_start()
            # end_index = entity_node.get_end()
            # text = entity_node.get_text()
            # entity_node = EntityNode(id=len(all_entities_nodes), start_index=start_index,
            #                          end_index=end_index, type=entity_node.get_type(),
            #                          text=text)
            all_entities_nodes.update({entity_node.get_id(): entity_node})

    for _, numbers_nodes in sentences_numbers.items():
        for _, number_node in numbers_nodes.items():
            # start_index = number_node.get_start()
            # end_index = number_node.get_end()
            # number_node = NumberNode(id=len(all_numbers_nodes), start_index=start_index, end_index=end_index,
            #                          type=number_node.get_type(), text=number_node.get_text)
            all_numbers_nodes.update({number_node.get_id(): number_node})

    for (sentence_id, entity_nodes) in sentences_entities.items():
        if sentence_id in sentences_numbers.keys():
            number_nodes = sentences_numbers[sentence_id]
            # sentences_numbers.update({sentence_id: (entity_nodes, number_nodes)})

            for e_id, entity_node in entity_nodes.items():
                if e_id not in list(all_entities_nodes.keys()):
                    print('e_id', e_id)
                    print('all_entities_nodes', list(all_entities_nodes.keys()))

            for n_id, number_node in number_nodes.items():
                if n_id not in list(all_numbers_nodes.keys()):
                    print('n_id', n_id)
                    print('number', list(all_numbers_nodes.keys()))

    return all_sentence_nodes, all_entities_nodes, all_numbers_nodes, sentences_entities, sentences_numbers


def cached_path(file_path):
    return file_path


IGNORED_TOKENS = {'a', 'an', 'the'}
MULTI_SPAN = 'multi_span'
STRIPPED_CHARACTERS = string.punctuation + ''.join([u"‘", u"’", u"´", u"`", "_"])


def whitespace_tokenize(text, ignore=False):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    """ if ignore is true, keep the original uppercase and lowercase"""
    # text = " ".join(basic_tokenizer.tokenize(text.strip())).strip()
    if ignore:
        text = text.strip()
    else:
        text = text.strip().lower()
    if not text:
        return []
    tokens = text.split()
    tokens = [token.strip(STRIPPED_CHARACTERS) for token in tokens]
    return tokens


WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}

TOKEN_TYPE = ['OTHER', 'ENTITY', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL',
              'CARDINAL', 'YARD']
ENTITY_TOKEN_TYPE = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']
TOKEN_TYPE_ID = [i for i in range(len(TOKEN_TYPE))]
TOKEN_TYPE_MAP_ID = dict(zip(TOKEN_TYPE, TOKEN_TYPE_ID))
NUMBER_TOKEN_TYPE = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'YARD']
NUMBER_TOKEN_TYPE_ID = [TOKEN_TYPE_MAP_ID[token_type] for token_type in NUMBER_TOKEN_TYPE]
NUMBER_TOKEN_TYPE_MAP_ID = dict(zip(NUMBER_TOKEN_TYPE, NUMBER_TOKEN_TYPE_ID))


class DropReader(object):
    def __init__(self, pretrain_model, tokenizer,
                 passage_length_limit: int = None, question_length_limit: int = None,
                 add_aux_nums: bool = False,
                 add_relation_reasoning_module: bool = False,
                 skip_when_all_empty: List[str] = None, instance_format: str = "drop",
                 relaxed_span_match_for_finding_labels: bool = True) -> None:

        self.pretrain_model = pretrain_model
        self.max_pieces = 512
        self._tokenizer = tokenizer
        self.add_aux_nums = add_aux_nums
        self.add_relation_reasoning_module = add_relation_reasoning_module
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_when_all_empty = skip_when_all_empty if skip_when_all_empty is not None else []
        for item in self.skip_when_all_empty:
            assert item in ["passage_span", "question_span", "addition_subtraction",
                            "counting", "multi_span"], f"Unsupported skip type: {item}"
        self.instance_format = instance_format
        self.relaxed_span_match_for_finding_labels = relaxed_span_match_for_finding_labels
        self.flexibility_threshold = 1000

        self.text_processor_spacy = TextProcessorSpacy()
        if 'roberta' in self.pretrain_model:
            self.USTRIPPED_CHARACTERS = ''.join([u"Ġ"])
            self.START_CHAR = u"Ġ"
        if 'albert' in self.pretrain_model:
            self.USTRIPPED_CHARACTERS = ''.join([u"▁"])
            self.START_CHAR = u"▁"

        if 'electra' in self.pretrain_model:
            self.USTRIPPED_CHARACTERS = ''.join([u"Ġ"])
            self.START_CHAR = u"Ġ"

    def extract_nodes(self, tokens, offset, text, entity_list, number_indices, sentence_start, all_number_nodes,
                      all_entities_nodes, token_type):

        sentences_numbers_nodes = {}
        sentences_entities_nodes = {}
        ## extract YARD number node using rules
        yard_number_items = extract_yard_num(text)
        if yard_number_items:
            for item in yard_number_items:
                item_offset = item.span()
                item_start = item_offset[0] + sentence_start
                item_end = item_offset[1] + sentence_start
                item_start, item_end = self.find_entity_index((item_start, item_end), offset)
                number_node = NumberNode(id=len(all_number_nodes), start_index=item_start,
                                         end_index=item_end, type='YARD', text=item.group())
                number_id = len(all_number_nodes)
                sentences_numbers_nodes.update({number_id: number_node})
                all_number_nodes.update({number_id: number_node})
                token_type[item_start:item_end] = ['YARD'] * (item_end - item_start)
        ## extract entity node and number node using NER
        valid_entity_list = []
        if len(entity_list) > 0:
            for id, entity in enumerate(entity_list):
                entity_tokens, _, _, _, _, _ = roberta_tokenize(entity['content'].lower(), self._tokenizer)
                entity_offset_start = entity['start_char'] + sentence_start
                if offset[-1][1] <= entity_offset_start:
                    break
                else:
                    valid_entity_list.append(entity)

        if len(valid_entity_list) > 0:
            for valid_id, entity in enumerate(valid_entity_list):
                entity_offset_start = entity['start_char'] + sentence_start
                entity_offset_end = entity['end_char'] + sentence_start
                entity_start_idx, entity_end_idx = self.find_entity_index((entity_offset_start, entity_offset_end),
                                                                          offset)
                if not (entity_start_idx != -1 and entity_end_idx != -1 and len(
                        tokens) >= entity_end_idx > entity_start_idx):
                    print('entity_start_idx', entity_start_idx)
                    print('entity_end_idx', entity_end_idx)
                    print('len of tokens', len(tokens))

                if entity['type'] in NUMBER_TOKEN_TYPE:
                    if token_type[entity_start_idx:entity_end_idx] != ['YARD'] * (
                            entity_end_idx - entity_start_idx):
                        number_id = len(all_number_nodes)
                        number_node = NumberNode(id=len(all_number_nodes), start_index=entity_start_idx,
                                                 end_index=entity_end_idx, type=entity['type'],
                                                 text=entity['content'])
                        sentences_numbers_nodes.update({number_id: number_node})
                        all_number_nodes.update({number_id: number_node})
                        token_type[entity_start_idx:entity_end_idx] = [entity['type']] * (
                                entity_end_idx - entity_start_idx)
                else:
                    if token_type[entity_start_idx:entity_end_idx] != ['YARD'] * (
                            entity_end_idx - entity_start_idx):
                        entity_id = len(all_entities_nodes)
                        entity_node = EntityNode(id=len(all_entities_nodes), start_index=entity_start_idx,
                                                 end_index=entity_end_idx, type="ENTITY", text=entity['content'])
                        sentences_entities_nodes.update({entity_id: entity_node})
                        all_entities_nodes.update({entity_id: entity_node})
                        token_type[entity_start_idx:entity_end_idx] = ['ENTITY'] * (
                                entity_end_idx - entity_start_idx)

        ## extract number node using tool word2number
        for number_idx in number_indices:
            if token_type[number_idx] not in NUMBER_TOKEN_TYPE:
                token_type[number_idx] = 'CARDINAL'
                number_id = len(all_number_nodes)
                number_node = NumberNode(id=len(all_number_nodes), start_index=number_idx,
                                         end_index=number_idx + 1, type='CARDINAL',
                                         text=tokens[number_idx])
                sentences_numbers_nodes.update({number_id: number_node})
                all_number_nodes.update({number_id: number_node})

        ## exam

        # for number_node_id1, number_node1 in all_number_nodes.items():
        #     for number_node_id2, number_node2 in all_number_nodes.items():
        #         number_node_start1 = number_node1.get_start()
        #         number_node_end1 = number_node1.get_end()
        #         number_node_start2 = number_node2.get_start()
        #         number_node_end2 = number_node2.get_end()
        #
        #         if number_node_id1 != number_node_id2:
        #             if number_node_end2 >= number_node_end1 > number_node_start2 or number_node_end1 >= number_node_end2 > number_node_start1:
        #                 print('number_node_end2', number_node_end2)
        #                 print('number_node_start2', number_node_start2)
        #                 print('number_node_end1', number_node_end1)
        #                 print('number_node_start1', number_node_start1)
        #                 print('number_node1', number_node1.text)
        #                 print('number_node2', number_node2.text)

        return all_number_nodes, all_entities_nodes, sentences_entities_nodes, sentences_numbers_nodes, token_type

    @staticmethod
    def find_entity_index(offset, offset_list):
        '''
        find the index i, statisify offsetlist[i](0)<=offset(0)<=offset(1)<=offsetlist[i](1)
        '''
        start_index = -1

        idx = 0
        while idx < len(offset_list):
            if offset_list[idx][0] <= offset[0] <= offset_list[idx][1]:
                start_index = idx
                break
            else:
                idx += 1
        end_index = start_index + 1
        # if offset[1] == offset_list[start_index][1]:
        #     return start_index, end_index
        i = start_index + 1
        for i in range(start_index + 1, len(offset_list)):
            if offset[1] <= offset_list[i][0]:
                end_index = i
                break
            if i == len(offset_list) - 1:
                end_index = len(offset_list)
        return start_index, end_index

    def _read(self, file_path: str, processor_num):
        # if `file_path` is a URL, redirect to the cache
        # print('args', self.args)
        file_path = cached_path(file_path)
        print(file_path)
        print("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        print("Reading the dataset")
        if processor_num == 1:
            instances = self.read_drop_data(dataset)
            return instances
        elif processor_num > 1:
            ##create threading
            instances = []
            dataset_size = len(dataset)
            chunk_size = math.ceil(dataset_size / processor_num)
            res = []
            p = Pool(processor_num)
            for i in range(processor_num):
                if i == processor_num - 1:
                    sub_dataset = dict(list(dataset.items())[i * chunk_size:dataset_size])
                else:
                    sub_dataset = dict(list(dataset.items())[i * chunk_size:(i + 1) * chunk_size])
                res.append(p.apply_async(self.read_drop_data, args=(sub_dataset,)))
                print(str(i) + ' processor started !')
            p.close()
            p.join()
            for i in res:
                sub_instances = i.get()
                instances.extend(sub_instances)
            return instances
        else:
            print("preocessor num must larger than 0, error!!!")
            return

    def read_drop_data(self, dataset):
        instances, skip_count = [], 0
        answer_type_nums = {'answer_passage_spans': 0, 'answer_question_spans': 0, 'counts': 0, 'multi_span': 0,
                            'signs_for_add_sub_expressions': 0}
        for passage_id, passage_info in tqdm(dataset.items()):
            passage_text = passage_info["passage"]
            passage_instance = self.passage_to_instance(passage_text, passage_id)
            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotations = []
                if "answer" in question_answer:
                    answer_annotations.append(question_answer["answer"])
                if "validated_answers" in question_answer:
                    answer_annotations += question_answer["validated_answers"]

                instance = self.text_to_instance(question_text, question_id,
                                                 answer_annotations,
                                                 **deepcopy(passage_instance)
                                                 )
                if instance is not None:
                    instances.append(instance)
                    if instance["answer_passage_spans"]:
                        answer_type_nums['answer_passage_spans'] += 1
                    if instance["answer_question_spans"]:
                        answer_type_nums['answer_question_spans'] += 1
                    if instance["counts"]:
                        answer_type_nums['counts'] += 1
                    if instance["multi_span"]:
                        answer_type_nums['multi_span'] += 1
                    if instance["signs_for_add_sub_expressions"]:
                        answer_type_nums['signs_for_add_sub_expressions'] += 1
                else:
                    skip_count += 1
        print(f"Skipped {skip_count} questions, kept {len(instances)} questions.")
        print('answer_type_nums', answer_type_nums)
        return instances

    def passage_to_instance(self, passage_text, passage_id):

        if self.add_relation_reasoning_module:
            passage_sentences = self.text_processor_spacy.sentence_split(passage_text)
            for sentence in passage_sentences:
                if len(sentence) == 0:
                    passage_sentences.remove(sentence)

            all_entities_nodes = {}
            all_numbers_nodes = {}
            all_sentence_nodes = {}
            sentence_entity_relation = {}
            sentence_number_relation = {}

            passage_sentence_text_list = []
            for sentence in passage_sentences:
                sentence_text = ' '.join(whitespace_tokenize(str(sentence)))
                passage_sentence_text_list.append(sentence_text)

            new_passage_text = ' '.join(passage_sentence_text_list)
            passage_tokens, passage_offset, numbers_in_passage, number_indices, number_len, passage_wordpiece_mask = roberta_tokenize(
                new_passage_text, self._tokenizer)
            sentence_start = 0
            number_in_sentences = []
            for sentence_id, sentence in enumerate(passage_sentences):
                sentence_text = " ".join(whitespace_tokenize(str(sentence)))
                # NER
                sentence_text_ignore_case = " ".join(whitespace_tokenize(str(sentence), ignore=True))

                entity_list = self.text_processor_spacy.ner(sentence_text_ignore_case)
                sentence_end_token_offset = sentence_start + len(sentence_text)
                sentence_end_token_index = self.find_sentence_end_token_index(passage_offset, sentence_end_token_offset)
                sentence_start_token_index = self.find_sentence_start_token_index(passage_offset, sentence_start)

                token_type = ['OTHER'] * len(passage_tokens)

                if sentence_end_token_index == -1:
                    # print('sentence_end_token_index', sentence_end_token_index)
                    # print('passage_offset', passage_offset)
                    # print('sentence_start', sentence_start)
                    # # print('sentence_offset', sentence_offset[-1][1])
                    # print('sentence_end_token_offset', sentence_end_token_offset)
                    # print('passage_tokens', passage_tokens)
                    # print('new_passage_text', new_passage_text)
                    print('passage_sentence_text_list', passage_sentence_text_list)
                    if sentence_id < len(passage_sentences) - 1:
                        sentence_start += len(sentence_text) + 1
                        continue

                if sentence_start_token_index == -1:
                    # print('sentence_start_token_index', sentence_start_token_index)
                    # print('passage_offset', passage_offset)
                    # print('sentence_start', sentence_start)
                    # print('passage_tokens', passage_tokens)
                    # print('new_passage_text', new_passage_text)
                    # print('passage_sentence_text_list', passage_sentence_text_list)
                    if sentence_id < len(passage_sentences) - 1:
                        sentence_start += len(sentence_text) + 1
                        continue

                sentence_node = SentenceNode(id=sentence_id, start_index=sentence_start_token_index,
                                             end_index=sentence_end_token_index + 1,
                                             type="sentence_node", text=sentence)
                number_in_sentence = []
                number_idx_in_sentence = []
                for idx, num_indice in enumerate(number_indices):
                    if sentence_start_token_index <= num_indice <= sentence_end_token_index:
                        number_idx_in_sentence.append(num_indice)
                        number_in_sentence.append(numbers_in_passage[idx])
                    number_in_sentences.extend(number_in_sentence)

                all_sentence_nodes.update({sentence_id: sentence_node})

                all_number_nodes, all_entities_nodes, sentences_entities_nodes, sentences_numbers_nodes, token_type = self.extract_nodes(
                    passage_tokens, passage_offset,
                    sentence_text, entity_list,
                    number_idx_in_sentence, sentence_start,
                    all_numbers_nodes,
                    all_entities_nodes, token_type)

                sentences_numbers_nodes = dict(
                    sorted(list(sentences_numbers_nodes.items()), key=lambda s: s[1].get_start()))
                if len(sentences_entities_nodes) > 0:
                    sentence_entity_relation.update({sentence_node.get_id(): sentences_entities_nodes})
                if len(sentences_numbers_nodes) > 0:
                    sentence_number_relation.update({sentence_node.get_id(): sentences_numbers_nodes})

                if sentence_id < len(passage_sentences) - 1:
                    sentence_start += len(sentence_text) + 1

            all_numbers_nodes = dict(sorted(list(all_numbers_nodes.items()), key=lambda s: s[1].get_start()))

            passage_instance = {
                "passage_id": passage_id,
                "passage_text": new_passage_text,
                "passage_tokens": passage_tokens,
                "passage_offset": passage_offset,
                "numbers_in_passage": numbers_in_passage,
                "number_indices": number_indices,
                "number_len": number_len,
                "passage_wordpiece_mask": passage_wordpiece_mask,

                "all_sentence_nodes": all_sentence_nodes,
                "all_entities_nodes": all_entities_nodes,
                "all_number_nodes": all_numbers_nodes,
                "sentence_entity_relation": sentence_entity_relation,
                "sentence_number_relation": sentence_number_relation
            }

            # exam the entity and number is have same part.
            # for entity_node_id1, entity_node1 in all_entities_nodes.items():
            #     for entity_node_id2, entity_node2 in all_entities_nodes.items():
            #         entity_node_start1 = entity_node1.get_start()
            #         entity_node_end1 = entity_node1.get_end()
            #         entity_node_start2 = entity_node2.get_start()
            #         entity_node_end2 = entity_node2.get_end()
            #
            #         if entity_node_id1 != entity_node_id2:
            #             if entity_node_end2 >= entity_node_end1 > entity_node_start2 or entity_node_end1 >= entity_node_end2 > entity_node_start1:
            #                 print('entity_node_end2', entity_node_end2)
            #                 print('entity_node_start2', entity_node_start2)
            #                 print('entity_node_end1', entity_node_end1)
            #                 print('entity_node_start1', entity_node_start1)
            #                 print('entity_node1', entity_node1.text)
            #                 print('entity_node2', entity_node2.text)
            #
            #                 print('passage_tokens', passage_tokens)
            #
            #                 print('entity_node1 token', passage_tokens[entity_node_start1:entity_node_end1])
            #                 print('entity_node2 token', passage_tokens[entity_node_start2:entity_node_end2])
            #                 print('passageText', passage_text)
            #                 print('offset', passage_offset)
            #
            #                 print('offset token', passage_offset[entity_node_start1:entity_node_end1])
            #                 print('offset token', passage_offset[entity_node_start2:entity_node_end2])
            #
            # for number_node_id1, number_node1 in all_numbers_nodes.items():
            #     for number_node_id2, number_node2 in all_numbers_nodes.items():
            #         number_node_start1 = number_node1.get_start()
            #         number_node_end1 = number_node1.get_end()
            #         number_node_start2 = number_node2.get_start()
            #         number_node_end2 = number_node2.get_end()
            #
            #         if number_node_id1 != number_node_id2:
            #             if number_node_end2 >= number_node_end1 > number_node_start2 or number_node_end1 >= number_node_end2 > number_node_start1:
            #                 print('number_node_end2', number_node_end2)
            #                 print('number_node_start2', number_node_start2)
            #                 print('number_node_end1', number_node_end1)
            #                 print('number_node_start1', number_node_start1)
            #                 print('number_node1', number_node1.text)
            #                 print('number_node2', number_node2.text)
            #
            #                 print('passage_tokens', passage_tokens)
            #
            #                 print('number_node1 token', passage_tokens[number_node_start1:number_node_end1])
            #                 print('number_node2 token', passage_tokens[number_node_start2:number_node_end2])
            #                 print('passageText', passage_text)
            #                 print('offset', passage_offset)
            #
            #                 print('offset token', passage_offset[number_node_start1:number_node_end1])
            #                 print('offset token', passage_offset[number_node_start2:number_node_end2])

            # for entity_node_id1, entity_node1 in all_entities_nodes.items():
            #     for number_node_id2, number_node2 in all_numbers_nodes.items():
            #         entity_node_start1 = entity_node1.get_start()
            #         entity_node_end1 = entity_node1.get_end()
            #         number_node_start2 = number_node2.get_start()
            #         number_node_end2 = number_node2.get_end()
            #
            #         if number_node_end2 >= entity_node_end1 > number_node_start2 or entity_node_end1 >= number_node_end2 > entity_node_start1:
            #             print('number_node_end2', number_node_end2)
            #             print('number_node_start2', number_node_start2)
            #             print('entity_node_start1', entity_node_start1)
            #             print('entity_node_end1', entity_node_end1)
            #             print('entity_node1', entity_node1.text)
            #             print('number_node2', number_node2.text)
            #
            #             print('passage_tokens', passage_tokens)
            #
            #             print('entity_node1 token', passage_tokens[entity_node_start1:entity_node_end1])
            #             print('number_node2 token', passage_tokens[number_node_start2:number_node_end2])
            #             print('passageText', passage_text)
            #             print('offset', passage_offset)
            #
            #             print('offset token', passage_offset[entity_node_start1:entity_node_end1])
            #             print('offset token', passage_offset[number_node_start2:number_node_end2])

        else:
            passage_text = " ".join(whitespace_tokenize(passage_text))
            passage_tokens, passage_offset, numbers_in_passage, number_indices, number_len, passage_wordpiece_mask = roberta_tokenize(
                passage_text, self._tokenizer)

            passage_instance = {"passage_id": passage_id,
                                "passage_text": passage_text,
                                "passage_tokens": passage_tokens,
                                "passage_offset": passage_offset,
                                "numbers_in_passage": numbers_in_passage,
                                "number_indices": number_indices,
                                "number_len": number_len,
                                "passage_wordpiece_mask": passage_wordpiece_mask,

                                "all_sentence_nodes": None,
                                "all_entities_nodes": None,
                                "all_number_nodes": None,
                                "sentence_entity_relation": None,
                                "sentence_number_relation": None
                                }

        return passage_instance

    def text_to_instance(self,
                         question_text: str,
                         question_id: str,
                         answer_annotations,
                         passage_id, passage_text, passage_tokens, passage_offset, numbers_in_passage, number_indices,
                         number_len,
                         passage_wordpiece_mask, all_sentence_nodes, all_entities_nodes, all_number_nodes,
                         sentence_entity_relation, sentence_number_relation
                         ):
        question_tag = question_parser(question_text)

        question_text_ignore_case = " ".join(whitespace_tokenize(str(question_text), ignore=True))
        question_text_entity_list = self.text_processor_spacy.ner(question_text_ignore_case)

        question_text = " ".join(whitespace_tokenize(question_text))

        question_tokens, question_offset, numbers_in_question, question_number_indices, question_number_len, question_wordpiece_mask = \
            roberta_tokenize(question_text, self._tokenizer)

        all_number_nodes_in_question = {}
        all_entities_nodes_in_question = {}
        number_type_cluster_in_question = {}
        sentence_entity_relation_in_question = {}
        sentence_number_relation_in_question = {}
        entities_numbers_relation_in_question = {}

        entities_numbers_relation = {}
        number_type_cluster = {}
        ## build the graph
        ## extract number nodes and entity nodes in 3 way: NER, YARD, word2number
        if self.add_relation_reasoning_module:
            question_node = SentenceNode(id=0, start_index=0, end_index=len(question_tokens), type="question_node",
                                         text=question_text)
            all_sentence_nodes_in_question = {}
            all_sentence_nodes_in_question.update({0: question_node})

            question_token_type = ['OTHER'] * len(question_tokens)
            all_number_nodes_in_question, all_entities_nodes_in_question, sentences_entities_nodes_in_question, sentences_numbers_nodes_in_question, question_token_type = self.extract_nodes(
                question_tokens, question_offset, question_text, question_text_entity_list, question_number_indices, 0,
                all_number_nodes_in_question, all_entities_nodes_in_question, question_token_type)

            sentences_numbers_nodes_in_question = dict(
                sorted(list(sentences_numbers_nodes_in_question.items()), key=lambda s: s[1].get_start()))
            if len(sentences_entities_nodes_in_question) > 0:
                sentence_entity_relation_in_question.update(
                    {question_node.get_id(): sentences_entities_nodes_in_question})
            if len(sentences_numbers_nodes_in_question) > 0:
                sentence_number_relation_in_question.update(
                    {question_node.get_id(): sentences_numbers_nodes_in_question})

            all_sentence_nodes_in_question, all_entities_nodes_in_question, all_number_nodes_in_question, sentence_entity_relation_in_question, sentence_number_relation_in_question = clipped_passage_relation_graph(
                all_sentence_nodes_in_question, all_entities_nodes_in_question, all_number_nodes_in_question,
                sentence_entity_relation_in_question, sentence_number_relation_in_question, self.question_length_limit)

            for (q_id, entity_nodes_in_question) in sentence_entity_relation_in_question.items():
                if q_id in sentence_number_relation_in_question.keys():
                    number_nodes_in_question = sentence_number_relation_in_question[q_id]
                    entities_numbers_relation_in_question.update(
                        {0: (entity_nodes_in_question, number_nodes_in_question)})

            for _, number_node_in_question in all_number_nodes_in_question.items():
                number_type = number_node_in_question.get_type()
                if number_type in number_type_cluster_in_question.keys():
                    number_type_cluster_in_question[number_type] = number_type_cluster_in_question[number_type] + [
                        number_node_in_question]
                else:
                    number_type_cluster_in_question.update({number_type: [number_node_in_question]})

        question_tokens = question_tokens[:self.question_length_limit]
        question_offset = question_offset[:self.question_length_limit]
        if len(question_number_indices) > 0:
            question_number_indices, question_number_len, numbers_in_question = clipped_passage_num(
                question_number_indices, question_number_len, numbers_in_question, len(question_tokens)
            )

        q_len = len(question_tokens)
        qp_tokens = [self._tokenizer.cls_token] + question_tokens + [self._tokenizer.sep_token] + passage_tokens
        qp_wordpiece_mask = [1] + question_wordpiece_mask + [1] + passage_wordpiece_mask

        if len(qp_tokens) > self.max_pieces - 1:
            qp_tokens = qp_tokens[:self.max_pieces - 1]
            passage_tokens = passage_tokens[:self.max_pieces - q_len - 3]
            passage_offset = passage_offset[:self.max_pieces - q_len - 3]
            plen = len(passage_tokens)
            number_indices, number_len, numbers_in_passage = clipped_passage_num(number_indices, number_len,
                                                                                 numbers_in_passage, plen)
            qp_wordpiece_mask = qp_wordpiece_mask[:self.max_pieces - 1]

            if self.add_relation_reasoning_module:
                all_sentence_nodes, all_entities_nodes, all_number_nodes, sentence_entity_relation, sentence_number_relation = clipped_passage_relation_graph(
                    all_sentence_nodes,
                    all_entities_nodes,
                    all_number_nodes,
                    sentence_entity_relation,
                    sentence_number_relation,
                    plen)

        for (sentence_id, entity_nodes) in sentence_entity_relation.items():
            if sentence_id in sentence_number_relation.keys():
                number_nodes = sentence_number_relation[sentence_id]
                entities_numbers_relation.update({sentence_id: (entity_nodes, number_nodes)})

                for e_id, entity_node in entity_nodes.items():
                    if e_id not in list(all_entities_nodes.keys()):
                        print('e_id', e_id)
                        print('all_entities_nodes', list(all_entities_nodes.keys()))

                for n_id, number_node in number_nodes.items():
                    if n_id not in list(all_number_nodes.keys()):
                        print('n_id', n_id)
                        print('number', list(all_number_nodes.keys()))

        for _, number_node in all_number_nodes.items():
            number_type = number_node.get_type()
            if number_type in number_type_cluster.keys():
                number_type_cluster[number_type] = number_type_cluster[number_type] + [number_node]
            else:
                number_type_cluster.update({number_type: [number_node]})

        qp_tokens += [self._tokenizer.sep_token]
        qp_wordpiece_mask += [1]

        answer_type: str = None
        answer_texts: List[str] = []
        if answer_annotations:
            # Currently we only use the first annotated answer here, but actually this doesn't affect
            # the training, because we only have one annotation for the train set.
            answer_type, answer_texts = self.extract_answer_info_from_annotation(answer_annotations[0])
            origin_answer_texts = answer_texts
            answer_texts = [" ".join(whitespace_tokenize(answer_text)) for answer_text in answer_texts]
            # if '' in answer_texts:
            #     print('origin answer_texts', origin_answer_texts)
            #     print('answer_annotations', answer_annotations)

        # Tokenize the answer text in order to find the matched span based on token
        tokenized_answer_texts = []
        specific_answer_type = "single_span"
        for answer_text in answer_texts:
            answer_tokens, _, _, _, _, _ = roberta_tokenize(answer_text, self._tokenizer, True)
            if answer_type in ["span", "spans"]:
                answer_texts = list(OrderedDict.fromkeys(answer_texts))
            if answer_type == "spans" and len(answer_texts) > 1:
                specific_answer_type = MULTI_SPAN
            tokenized_answer_text = " ".join(answer_tokens)
            if tokenized_answer_text not in tokenized_answer_texts:
                tokenized_answer_texts.append(tokenized_answer_text)

        if self.instance_format == "drop":
            def get_number_order(numbers):
                if len(numbers) < 1:
                    return None
                ordered_idx_list = np.argsort(np.array(numbers)).tolist()

                rank = 0
                number_rank = []
                for i, idx in enumerate(ordered_idx_list):
                    if i == 0 or numbers[ordered_idx_list[i]] != numbers[ordered_idx_list[i - 1]]:
                        rank += 1
                    number_rank.append(rank)

                ordered_idx_rank = zip(ordered_idx_list, number_rank)

                final_rank = sorted(ordered_idx_rank, key=lambda x: x[0])
                final_rank = [item[1] for item in final_rank]

                return final_rank

            if self.add_aux_nums:
                all_number = aux_num_list + numbers_in_passage + numbers_in_question
                all_number_order = get_number_order(all_number)
                if all_number_order is None:
                    aux_number_order = []
                    passage_number_order = []
                    question_number_order = []
                else:
                    aux_number_order = all_number_order[:len(aux_num_list)]
                    passage_number_order = all_number_order[
                                           len(aux_num_list): len(aux_num_list) + len(numbers_in_passage)]
                    question_number_order = all_number_order[len(aux_num_list) + len(numbers_in_passage):]

                number_indices = [indice + 1 for indice in number_indices]
                numbers_in_passage.append(100)
                number_indices.append(0)
                passage_number_order.append(-1)

                # hack to guarantee minimal length of padded number
                numbers_in_passage.append(0)
                number_indices.append(-1)
                passage_number_order.append(-1)

                numbers_in_question.append(0)
                question_number_indices.append(-1)
                question_number_order.append(-1)

                aux_number_order = np.array(aux_number_order)
                passage_number_order = np.array(passage_number_order)
                question_number_order = np.array(question_number_order)
                aux_numbers_as_tokens = [str(number) for number in aux_num_list]

            else:
                all_number = numbers_in_passage + numbers_in_question
                all_number_order = get_number_order(all_number)
                if all_number_order is None:
                    passage_number_order = []
                    question_number_order = []
                else:
                    passage_number_order = all_number_order[:len(numbers_in_passage)]
                    question_number_order = all_number_order[len(numbers_in_passage):]

                # hack to guarantee minimal length of padded number
                number_indices = [indice + 1 for indice in number_indices]
                numbers_in_passage.append(100)
                number_indices.append(0)
                passage_number_order.append(-1)

                # hack to guarantee minimal length of padded number
                numbers_in_passage.append(0)
                number_indices.append(-1)
                passage_number_order.append(-1)

                numbers_in_question.append(0)
                question_number_indices.append(-1)
                question_number_order.append(-1)

                passage_number_order = np.array(passage_number_order)
                question_number_order = np.array(question_number_order)
                aux_numbers_as_tokens = []
                aux_number_order = []

            numbers_as_tokens = [str(number) for number in numbers_in_passage]

            valid_passage_spans = self.find_valid_spans(self.USTRIPPED_CHARACTERS, passage_tokens,
                                                        tokenized_answer_texts) if tokenized_answer_texts else []
            if len(valid_passage_spans) > 0:
                valid_question_spans = []
            else:
                valid_question_spans = self.find_valid_spans(self.USTRIPPED_CHARACTERS, question_tokens,
                                                             tokenized_answer_texts) if tokenized_answer_texts else []

            target_numbers = []
            # `answer_texts` is a list of valid answers.
            for answer_text in answer_texts:
                number = get_number_from_word(answer_text, True)
                if number is not None:
                    target_numbers.append(number)
            valid_signs_for_add_sub_expressions: List[List[int]] = []
            valid_counts: List[int] = []
            if answer_type in ["number", "date"]:
                target_number_strs = ["%.3f" % num for num in target_numbers]
                valid_signs_for_add_sub_expressions = self.find_valid_add_sub_expressions(numbers_in_passage,
                                                                                          target_number_strs)
            if answer_type in ["number"]:
                # Currently we only support count number 0 ~ 9
                numbers_for_count = list(range(10))
                valid_counts = self.find_valid_counts(numbers_for_count, target_numbers)

            # add multi_span answer extraction
            no_answer_bios = [0] * len(qp_tokens)
            if specific_answer_type == MULTI_SPAN and (len(valid_passage_spans) > 0 or len(valid_question_spans) > 0):
                spans_dict = {}
                text_to_disjoint_bios = []
                flexibility_count = 1
                for tokenized_answer_text in tokenized_answer_texts:
                    spans = self.find_valid_spans(self.USTRIPPED_CHARACTERS, qp_tokens, [tokenized_answer_text])
                    if len(spans) == 0:
                        # possible if the passage was clipped, but not for all of the answers
                        continue
                    spans_dict[tokenized_answer_text] = spans

                    disjoint_bios = []
                    for span_ind, span in enumerate(spans):
                        bios = create_bio_labels([span], len(qp_tokens))
                        disjoint_bios.append(bios)

                    text_to_disjoint_bios.append(disjoint_bios)
                    flexibility_count *= ((2 ** len(spans)) - 1)

                answer_as_text_to_disjoint_bios = text_to_disjoint_bios

                if (flexibility_count < self.flexibility_threshold):
                    # generate all non-empty span combinations per each text
                    spans_combinations_dict = {}
                    for key, spans in spans_dict.items():
                        spans_combinations_dict[key] = all_combinations = []
                        for i in range(1, len(spans) + 1):
                            all_combinations += list(itertools.combinations(spans, i))

                    # calculate product between all the combinations per each text
                    packed_gold_spans_list = itertools.product(*list(spans_combinations_dict.values()))
                    bios_list = []
                    for packed_gold_spans in packed_gold_spans_list:
                        gold_spans = [s for sublist in packed_gold_spans for s in sublist]
                        bios = create_bio_labels(gold_spans, len(qp_tokens))
                        bios_list.append(bios)

                    answer_as_list_of_bios = bios_list
                    answer_as_text_to_disjoint_bios = [[no_answer_bios]]
                else:
                    answer_as_list_of_bios = [no_answer_bios]

                # END

                # Used for both "require-all" BIO loss and flexible loss
                bio_labels = create_bio_labels(valid_question_spans + valid_passage_spans, len(qp_tokens))
                span_bio_labels = bio_labels

                is_bio_mask = 1

                multi_span = [is_bio_mask, answer_as_text_to_disjoint_bios, answer_as_list_of_bios, span_bio_labels]
            else:
                multi_span = []

            valid_passage_spans = valid_passage_spans if specific_answer_type != MULTI_SPAN or len(
                multi_span) < 1 else []
            valid_question_spans = valid_question_spans if specific_answer_type != MULTI_SPAN or len(
                multi_span) < 1 else []

            type_to_answer_map = {"passage_span": valid_passage_spans, "question_span": valid_question_spans,
                                  "addition_subtraction": valid_signs_for_add_sub_expressions, "counting": valid_counts,
                                  "multi_span": multi_span}

            if self.skip_when_all_empty and not any(
                    type_to_answer_map[skip_type] for skip_type in self.skip_when_all_empty):
                # logger.info('my_skip_ans_type: %s' % answer_type)
                return None

            answer_info = {"answer_texts": answer_texts,  # this `answer_texts` will not be used for evaluation
                           "answer_passage_spans": valid_passage_spans,
                           "answer_question_spans": valid_question_spans,
                           "signs_for_add_sub_expressions": valid_signs_for_add_sub_expressions, "counts": valid_counts,
                           "multi_span": multi_span}

            return self.make_marginal_drop_instance(question_tokens,
                                                    passage_tokens,
                                                    qp_tokens,
                                                    numbers_as_tokens,
                                                    number_indices,
                                                    passage_number_order,
                                                    question_number_order,
                                                    question_number_indices,
                                                    qp_wordpiece_mask,
                                                    aux_numbers_as_tokens,
                                                    aux_number_order,
                                                    answer_info,

                                                    all_sentence_nodes,
                                                    all_entities_nodes,
                                                    all_number_nodes,
                                                    sentence_entity_relation,
                                                    sentence_number_relation,
                                                    entities_numbers_relation,
                                                    number_type_cluster,

                                                    all_number_nodes_in_question,
                                                    all_entities_nodes_in_question,
                                                    sentence_entity_relation_in_question,
                                                    sentence_number_relation_in_question,
                                                    entities_numbers_relation_in_question,
                                                    number_type_cluster_in_question,
                                                    question_tag,
                                                    additional_metadata={"original_passage": passage_text,
                                                                         "passage_token_offsets": passage_offset,
                                                                         "original_question": question_text,
                                                                         "question_token_offsets": question_offset,
                                                                         "original_numbers": numbers_in_passage,
                                                                         "passage_id": passage_id,
                                                                         "question_id": question_id,
                                                                         "answer_info": answer_info,
                                                                         "answer_annotations": answer_annotations})
        else:
            raise ValueError(f"Expect the instance format to be \"drop\", \"squad\" or \"bert\", "
                             f"but got {self.instance_format}")

    @staticmethod
    def extract_answer_info_from_annotation(answer_annotation: Dict[str, Any]) -> Tuple[str, List[str]]:
        answer_type = None
        # print('type of answer_annotation', type(answer_annotation))
        if isinstance(answer_annotation, list):
            print('type of answer_annotation', type(answer_annotation))
            print(answer_annotation)
        if answer_annotation["spans"]:
            answer_type = "spans"
        elif answer_annotation["number"]:
            answer_type = "number"
        elif any(answer_annotation["date"].values()):
            answer_type = "date"

        answer_content = answer_annotation[answer_type] if answer_type is not None else None

        answer_texts: List[str] = []
        if answer_type is None:  # No answer
            pass
        elif answer_type == "spans":
            # answer_content is a list of string in this case
            answer_texts = answer_content
        elif answer_type == "date":
            # answer_content is a dict with "month", "day", "year" as the keys
            date_tokens = [answer_content[key] for key in ["month", "day", "year"] if
                           key in answer_content and answer_content[key]]
            answer_texts = date_tokens
        elif answer_type == "number":
            # answer_content is a string of number
            answer_texts = [answer_content]
        return answer_type, answer_texts

    @staticmethod
    def find_valid_spans(USTRIPPED_CHARACTERS, passage_tokens: List[str], answer_texts: List[str]) -> List[
        Tuple[int, int]]:

        normalized_tokens = [token.strip(USTRIPPED_CHARACTERS) for token in passage_tokens]
        # normalized_tokens = passage_tokens
        word_positions: Dict[str, List[int]] = defaultdict(list)
        for i, token in enumerate(normalized_tokens):
            word_positions[token].append(i)
        spans = []
        for answer_text in answer_texts:
            answer_tokens = [token.strip(USTRIPPED_CHARACTERS) for token in answer_text.split()]
            # answer_tokens = answer_text.split()
            num_answer_tokens = len(answer_tokens)
            if not answer_tokens:
                # print('answer_tokens', answer_tokens)
                # print('answer_text', answer_text)
                # print('answer_texts', answer_texts)
                continue
            if answer_tokens[0] not in word_positions:
                continue
            for span_start in word_positions[answer_tokens[0]]:
                span_end = span_start  # span_end is _inclusive_
                answer_index = 1
                while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                    token = normalized_tokens[span_end + 1]
                    if answer_tokens[answer_index] == token:
                        answer_index += 1
                        span_end += 1
                    elif token in IGNORED_TOKENS:
                        span_end += 1
                    else:
                        break
                if num_answer_tokens == answer_index:
                    spans.append((span_start, span_end))
        return spans

    @staticmethod
    def find_valid_add_sub_expressions(numbers: List, targets: List, max_number_of_numbers_to_consider: int = 3) -> \
            List[List[int]]:
        valid_signs_for_add_sub_expressions = []
        # TODO: Try smaller numbers?
        for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
            possible_signs = list(itertools.product((-1, 1), repeat=number_of_numbers_to_consider))
            for number_combination in itertools.combinations(enumerate(numbers), number_of_numbers_to_consider):
                indices = [it[0] for it in number_combination]
                values = [it[1] for it in number_combination]
                for signs in possible_signs:
                    eval_value = sum(sign * value for sign, value in zip(signs, values))
                    # if eval_value in targets:
                    eval_value_str = '%.3f' % eval_value
                    if eval_value_str in targets:
                        labels_for_numbers = [0] * len(numbers)  # 0 represents ``not included''.
                        for index, sign in zip(indices, signs):
                            labels_for_numbers[index] = 1 if sign == 1 else 2  # 1 for positive, 2 for negative
                        valid_signs_for_add_sub_expressions.append(labels_for_numbers)
        return valid_signs_for_add_sub_expressions

    @staticmethod
    def find_valid_counts(count_numbers: List[int], targets: List[int]) -> List[int]:
        valid_indices = []
        for index, number in enumerate(count_numbers):
            if number in targets:
                valid_indices.append(index)
        return valid_indices

    @staticmethod
    def make_marginal_drop_instance(question_tokens: List[str],
                                    passage_tokens: List[str],
                                    question_passage_tokens: List[str],
                                    number_tokens: List[str],
                                    number_indices: List[int],
                                    passage_number_order: np.ndarray,
                                    question_number_order: np.ndarray,
                                    question_number_indices: List[int],
                                    wordpiece_mask: List[int],
                                    aux_numbers_as_tokens: List[int],
                                    aux_number_order: List[int],
                                    answer_info: Dict[str, Any] = None,
                                    all_sentence_nodes=None,
                                    all_entities_nodes=None,
                                    all_number_nodes=None,
                                    sentence_entity_relation=None,
                                    sentence_number_relation=None,
                                    entities_numbers_relation=None,
                                    number_type_cluster=None,

                                    all_number_nodes_in_question=None,
                                    all_entities_nodes_in_question=None,
                                    sentence_entity_relation_in_question=None,
                                    sentence_number_relation_in_question=None,
                                    entities_numbers_relation_in_question=None,
                                    number_type_cluster_in_question=None,
                                    question_tag=None,
                                    additional_metadata: Dict[str, Any] = None):
        metadata = {
            "question_tokens": [token for token in question_tokens],
            "passage_tokens": [token for token in passage_tokens],
            "question_passage_tokens": question_passage_tokens,
            "number_tokens": [token for token in number_tokens],
            "number_indices": number_indices,
            "question_number_indices": question_number_indices,
            "passage_number_order": passage_number_order,
            "question_number_order": question_number_order,
            "wordpiece_mask": wordpiece_mask,
            "aux_number_order": aux_number_order,
            "aux_number_as_tokens": aux_numbers_as_tokens,

            "all_sentence_nodes": all_sentence_nodes,
            "all_entities_nodes": all_entities_nodes,
            "all_number_nodes": all_number_nodes,
            "sentence_entity_relation": sentence_entity_relation,
            "sentence_number_relation": sentence_number_relation,
            "entities_numbers_relation": entities_numbers_relation,
            "number_type_cluster": number_type_cluster,
            "all_number_nodes_in_question": all_number_nodes_in_question,
            "all_entities_nodes_in_question": all_entities_nodes_in_question,
            "sentence_entity_relation_in_question": sentence_entity_relation_in_question,
            "sentence_number_relation_in_question": sentence_number_relation_in_question,
            "entities_numbers_relation_in_question": entities_numbers_relation_in_question,
            "number_type_cluster_in_question": number_type_cluster_in_question,
            "question_tag": question_tag

        }
        if answer_info:
            metadata["answer_texts"] = answer_info["answer_texts"]
            metadata["answer_passage_spans"] = answer_info["answer_passage_spans"]
            metadata["answer_question_spans"] = answer_info["answer_question_spans"]
            metadata["signs_for_add_sub_expressions"] = answer_info["signs_for_add_sub_expressions"]
            metadata["counts"] = answer_info["counts"]
            metadata["multi_span"] = answer_info["multi_span"]

        metadata.update(additional_metadata)
        return metadata

    def find_sentence_end_token_index(self, passage_offset, sentence_end_token_offset):
        # passage_end_token_offset = [token_offset[1] for token_offset in passage_offset]
        reverse_passage_end_token_offset = list(reversed(passage_offset))
        end_token_index = -1
        try:
            for i in range(len(reverse_passage_end_token_offset)):
                if sentence_end_token_offset >= reverse_passage_end_token_offset[i][1]:
                    reverse_end_token_index = i
                    end_token_index = len(reverse_passage_end_token_offset) - reverse_end_token_index - 1
                    break
        except:
            end_token_index = -1
        return end_token_index

    def find_sentence_start_token_index(self, passage_offset, sentence_start):
        # passage_start_token_offset = [token_offset[0] for token_offset in passage_offset]
        start_token_index = -1
        try:
            for i in range(len(passage_offset)):
                if sentence_start <= passage_offset[i][0]:
                    start_token_index = i
                    return start_token_index
        except:
            start_token_index = -1
        return start_token_index

    def find_entity_start_token_index(self, sentence_offset, entity_start):
        start_token_index = -1
        try:
            for i in range(len(sentence_offset)):
                if sentence_offset[i][0] <= entity_start <= sentence_offset[i][1]:
                    start_token_index = i
                    return start_token_index
        except:
            start_token_index = -1
        return start_token_index

    def find_entity_end_token_index(self, sentence_offset, entity_end):
        # passage_end_token_offset = [token_offset[1] for token_offset in passage_offset]
        reverse_sentence_end_token_offset = list(reversed(sentence_offset))
        end_token_index = -1
        try:
            for i in range(len(reverse_sentence_end_token_offset)):
                if reverse_sentence_end_token_offset[i][0] <= entity_end <= reverse_sentence_end_token_offset[i][1]:
                    reverse_end_token_index = i
                    end_token_index = len(reverse_sentence_end_token_offset) - reverse_end_token_index - 1
                    return end_token_index
        except:
            end_token_index = -1
        return end_token_index


def create_bio_labels(spans: List[Tuple[int, int]], n_labels: int):
    # initialize all labels to O
    labels = [0] * n_labels

    for span in spans:
        start = span[0]
        end = span[1]
        # create B labels
        labels[start] = 1
        # create I labels
        labels[start + 1:end + 1] = [2] * (end - start)

    return labels


def extract_yard_num(input_text):
    pattern = re.compile("\d+\.?\d*[\s+|-]yards*")
    try:
        match_items = re.finditer(pattern, input_text)
        # for item in match_items:
        #     print(item.span())
        #     print(item.group())
        return match_items
    except:
        print('no find match')
        return None
    # print(position)


#


def load_json_file(input_file):
    assert os.path.exists(input_file)
    with open(input_file, 'r') as f:
        data = json.load(f)
    f.close()
    print('finish load json file {}'.format(input_file))
    return data


if __name__ == '__main__':
    # import os

    # # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # # raw = "In 1905, 1,003 Korean immigrants, which included 802 men and 231 women and children, departed from the port of Chemulpo, Incheon aboard the ship Ilford to Salina Cruz, Oaxaca, Mexico. The journey took 45 days, after which they took a train to Coatzacoalcos, Veracruz. In the Veracruz port, another boat was taken to the port of Progreso with the final destination being the capital city of M\u00e9rida, Yucatan. They arrived in May 1905, with previously signed contracts for four years' work as indentured laborers on the Yucat\u00e1n henequen haciendas. Many of these Koreans were distributed throughout the Yucat\u00e1n in 32 henequen haciendas. The town of Motul, Yucatan, located in the heart of the henequen zone, was a destination for many of the Korean immigrants. Subsequently, in 1909, at the end of their contracts, they began a new stage in which they scattered even further  Thus, the majority of those who came were single men who made or remade their family lives with Yucatecan especially Maya women. While Korean girls were much more subject to marriages arranged by Korean parents, males had greater freedom when it came to making a family. This rapid intermarriage by Koreans, coupled with geographic dispersal, prevented the establishment of close social networks among these migrants and therefore provided the basis for Korean descendants among the Yucatan Peninsula. After that 1905 ship, no further entries of Koreans into Mexico were recorded, until many years later, leading to a new community of Koreans with completely different characteristics from those who entered in 1905. These descendants have started the Museo Conmemorativo de la Inmigraci\u00f3n Coreana a Yucat\u00e1n, a museum for the remembrance of their ancestors journey."
    # # raw2 = "The Siege of Vienna in 1529, the first quarter was the first attempt by the Ottoman Empire, led by Suleiman the Magnificent, to capture the city of Vienna, Austria. The siege signalled the pinnacle of the Ottoman Empire's power and the maximum extent of Ottoman expansion in central Europe. Thereafter, 150 years of bitter military tension and reciprocal attacks ensued, culminating in the Battle of Vienna of 1683, which marked the start of the 15-year-long Great Turkish War. The inability of the Ottomans to capture Vienna in 1529 turned the tide against almost a century of conquest throughout eastern and central Europe. The Ottoman Empire had previously annexed Central Hungary and established a vassal state in Transylvania in the wake of the Battle of Moh\u00e1cs. According to Arnold J. Toynbee, \"The failure of the first  brought to a standstill the tide of Ottoman conquest which had been flooding up the Danube Valley for a century past.\" There is speculation by some historians that Suleiman's main objective in 1529 was actually to assert Ottoman control over the whole of Hungary, the western part of which  was under Habsburg control. The decision to attack Vienna after such a long interval in Suleiman's European campaign is viewed as an opportunistic manoeuvre after his decisive victory in Hungary. Other scholars theorise that the suppression of Hungary simply marked the prologue to a later, premeditated invasion of Europe."
    # # raw3 = 'the texans would respond with fullback vonta leach getting a 1-yard touchdown run yet the raiders would answer with kicker sebastian janikowski getting a 33-yard and a 30-yard field goal'
    # # raw = "Hoping to rebound from their loss to the Patriots, the Raiders stayed at home for a Week 16 duel with the Houston Texans.,  Oakland would get the early lead in the first quarter as quarterback JaMarcus Russell completed a 20-yard touchdown pass to rookie wide receiver Chaz Schilens.,  The Texans would respond with fullback Vonta Leach getting a 1-yard touchdown run, yet the Raiders would answer with kicker Sebastian Janikowski getting a 33-yard and a 30-yard field goal.,  Houston would tie the game in the second quarter with kicker Kris Brown getting a 53-yard and a 24-yard field goal., Oakland would take the lead in the third quarter with wide receiver Johnnie Lee Higgins catching a 29-yard touchdown pass from Russell, followed up by an 80-yard punt return for a touchdown.,  The Texans tried to rally in the fourth quarter as Brown nailed a 40-yard field goal, yet the Raiders' defense would shut down any possible attempt."
    # # raw = " ".join([raw1,raw2,raw3])
    # # raw = "German has 13,444 speakers representing about 0.4% of the states population, and Vietnamese is spoken by 11,330 people, or about 0.4% of the population, many of whom live in the Asia District, Oklahoma City of Oklahoma City. Other languages include French with 8,258 speakers (0.3%), Chinese Americans with 6,413 (0.2%), Korean with 3,948 (0.1%), Arabic with 3,265 (0.1%), other Asian languages with 3,134 (0.1%), Tagalog language with 2,888 (0.1%), Japanese with 2,546 (0.1%), and African languages with 2,546 (0.1%). In addition to Cherokee, more than 25 Indigenous languages of the Americas are spoken in Oklahoma, second only to California (though, it should be noted only Cherokee exhibits language vitality at present)."
    # raw2 = "In the county, the population was spread out with 26.20% under the age of 18, 9.30% from 18 to 24, 26.50% from 25 to 44, 23.50% from 45 to 64, and 14.60% who were 65 years of age or older.  The median age was 37 years. For every 100 females there were 95.90 males.  For every 100 females age 18 and over, there were 92.50 males."
    # tokenizer = AutoTokenizer.from_pretrained('../../numnet_plus_data/drop_dataset/albert.xlarge')
    #
    # from tqdm import tqdm
    # import json
    #
    # # input_path = "../../numnet_plus_data/drop_dataset/drop_dataset_train.json"
    # # dataset = load_json_file(input_path)
    text_processor = TextProcessorSpacy()
    #
    # raw = "as of july 1 2016 kentucky had an estimated population of 4,436,974 which is an increase of 12,363 from the prior year and an increase of 97,607 or 2.2 since the year 2010"
    # passage_tokens, passage_offset, numbers_in_passage, number_indices, number_len, passage_wordpiece_mask = roberta_tokenize(
    #     raw, tokenizer)
    # print(passage_offset)
    # print(passage_tokens)
    # print(passage_offset[-1][1])
    TOKEN_TYPE = ['OTHER', 'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC',
                  'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
                  'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL',
                  'CARDINAL', 'YARD']
    TOKEN_TYPE_ID = [i for i in range(len(TOKEN_TYPE))]
    TOKEN_TYPE_MAP_ID = dict(zip(TOKEN_TYPE, TOKEN_TYPE_ID))
    NUMBER_TOKEN_TYPE = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'YARD']
    NUMBER_TOKEN_TYPE_MAP_ID = dict(
        zip(NUMBER_TOKEN_TYPE, [TOKEN_TYPE_MAP_ID[token_type] for token_type in NUMBER_TOKEN_TYPE]))
    print(TOKEN_TYPE_MAP_ID)
    print(NUMBER_TOKEN_TYPE_MAP_ID)
    raw = "In 1905, 1,003 Korean immigrants, 3.5-yard which included 802 men and 231 women 3 yards and children, departed from the port of Chemulpo, Incheon aboard the ship Ilford to Salina Cruz, Oaxaca, Mexico. The journey took 45 days, after which they took a train to Coatzacoalcos, Veracruz. In the Veracruz port, another boat was taken to the port of Progreso with the final destination being the capital city of M\u00e9rida, Yucatan. They arrived in May 1905, with previously signed contracts for four years' work as indentured laborers on the Yucat\u00e1n henequen haciendas. Many of these Koreans were distributed throughout the Yucat\u00e1n in 32 henequen haciendas. The town of Motul, Yucatan, located in the heart of the henequen zone, was a destination for many of the Korean immigrants. Subsequently, in 1909, at the end of their contracts, they began a new stage in which they scattered even further  Thus, the majority of those who came were single men who made or remade their family lives with Yucatecan especially Maya women. While Korean girls were much more subject to marriages arranged by Korean parents, males had greater freedom when it came to making a family. This rapid intermarriage by Koreans, coupled with geographic dispersal, prevented the establishment of close social networks among these migrants and therefore provided the basis for Korean descendants among the Yucatan Peninsula. After that 1905 ship, no further entries of Koreans into Mexico were recorded, until many years later, leading to a new community of Koreans with completely different characteristics from those who entered in 1905. These descendants have started the Museo Conmemorativo de la Inmigraci\u00f3n Coreana a Yucat\u00e1n, a museum for the remembrance of their ancestors journey."
    result = extract_yard_num(raw)
    # entity = text_processor.ner(raw)
    # print(entity)
    # print(result)
    if result:
        for item in result:
            print(item.group())
