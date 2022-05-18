import re
import json
import string
import itertools
from tqdm import tqdm
import numpy as np
from word2number.w2n import word_to_num
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from multiprocessing import Pool
import math
from copy import deepcopy
from mspan_roberta_gcn.util import TextProcessorSpacy
from mspan_roberta_gcn.util import Node, SentenceNode, ValueNode, EntityNode, Edges, SentenceAndEntity, \
    SentenceAndValue, EntityAndEntity, EntityAndValue, HeterogeneousGraph


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
            prev_is_whitespace = False

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

    assert len(split_tokens) == len(sub_token_offsets)
    return split_tokens, sub_token_offsets, numbers, number_indices, number_len


def clipped_passage_num(number_indices, number_len, numbers_in_passage, plen):
    if number_indices[-1] < plen:
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


def cached_path(file_path):
    return file_path


IGNORED_TOKENS = {'a', 'an', 'the'}
STRIPPED_CHARACTERS = string.punctuation + ''.join([u"‘", u"’", u"´", u"`", "_"])
USTRIPPED_CHARACTERS = ''.join([u"Ġ"])


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

aux_num_list = [i for i in range(10)]


class DropReader(object):
    def __init__(self, tokenizer,
                 passage_length_limit: int = None, question_length_limit: int = None,
                 add_aux_nums: bool = False,
                 add_relation_reasoning_module: bool = False,
                 skip_when_all_empty: List[str] = None, instance_format: str = "drop",
                 relaxed_span_match_for_finding_labels: bool = True) -> None:
        self._tokenizer = tokenizer
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.add_aux_nums = add_aux_nums
        self.add_relation_reasoning_module = add_relation_reasoning_module
        self.skip_when_all_empty = skip_when_all_empty if skip_when_all_empty is not None else []
        for item in self.skip_when_all_empty:
            assert item in ["passage_span", "question_span", "addition_subtraction",
                            "counting"], f"Unsupported skip type: {item}"
        self.instance_format = instance_format
        self.relaxed_span_match_for_finding_labels = relaxed_span_match_for_finding_labels
        self.text_processor_spacy = TextProcessorSpacy()

    @staticmethod
    def find_entity_index(offset, length, offset_list):
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

        i = start_index + 1
        for i in range(start_index + 1, len(offset_list)):
            if offset[1] < offset_list[i][0]:
                end_index = i
                break
            if i == len(offset_list) - 1:
                end_index = len(offset_list)
        return start_index, end_index

    @staticmethod
    def convert_word_to_number(word: str, try_to_include_more_numbers=False):
        """
        Currently we only support limited types of conversion.
        """
        if try_to_include_more_numbers:
            # strip all punctuations from the sides of the word, except for the negative sign
            punctruations = string.punctuation.replace('-', '')
            word = word.strip(punctruations)
            # some words may contain the comma as deliminator
            word = word.replace(",", "")
            # word2num will convert hundred, thousand ... to number, but we skip it.
            if word in ["hundred", "thousand", "million", "billion", "trillion"]:
                return None
            try:
                number = word_to_num(word)
            except ValueError:
                try:
                    number = int(word)
                except ValueError:
                    try:
                        number = float(word)
                    except ValueError:
                        number = None
            return number
        else:
            no_comma_word = word.replace(",", "")
            if no_comma_word in WORD_NUMBER_MAP:
                number = WORD_NUMBER_MAP[no_comma_word]
            else:
                try:
                    number = int(no_comma_word)
                except ValueError:
                    number = None
            return number

    def read_drop_data(self, dataset):
        instances, skip_count = [], 0
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

                instance = self.text_to_instance(question_text, question_id, answer_annotations,
                                                 **deepcopy(passage_instance))
                if instance is not None:
                    instances.append(instance)
                else:
                    skip_count += 1
        print(f"Skipped {skip_count} questions, kept {len(instances)} questions.")
        return instances

    def _read(self, file_path: str, processor_num):
        # if `file_path` is a URL, redirect to the cache
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

    def passage_to_instance(self, passage_text, passage_id):

        if self.add_relation_reasoning_module:
            passage_sentences = self.text_processor_spacy.sentencizer(passage_text)
            # passage_sentences = self.text_processor_spacy.sen2clause(passage_sentences)

            all_sentence_nodes = {}
            all_entities_nodes = {}
            all_values_nodes = {}

            sentences_entities = {}
            sentences_values = {}
            entities_entities = {}
            entities_values = {}

            passage_sentence_text_list = []

            passage_tokens = []
            passage_offset = []
            numbers_in_passage = []
            number_indices = []
            number_len = []

            sentence_start = 0
            sentence_token_len = 0

            for sentence_id, sentence in enumerate(passage_sentences):
                valid_entity_list = []
                sentence_text = " ".join(whitespace_tokenize(str(sentence)))
                passage_sentence_text_list.append(sentence_text)

                # NER
                sentence_text_ignore_case = " ".join(whitespace_tokenize(str(sentence), ignore=True))
                entity_list = self.text_processor_spacy.ner(sentence_text_ignore_case)
                # tokenize sentence
                sentence_tokens, sentence_offset, number_in_sentence, number_in_sentence_indices, number_in_sentence_len = roberta_tokenize(
                    sentence_text, self._tokenizer)
                if len(sentence_offset) <= 0:
                    continue
                if sentence_token_len + len(sentence_tokens) <= self.passage_length_limit:
                    for idx, offset in enumerate(sentence_offset):
                        start = sentence_start + offset[0]
                        end = sentence_start + offset[1]
                        new_offset = (start, end)
                        sentence_offset[idx] = new_offset
                    passage_tokens.extend(sentence_tokens)
                    passage_offset.extend(sentence_offset)

                    sentence_node = SentenceNode(id=str(sentence_id), start_index=sentence_token_len,
                                                 end_index=len(sentence_tokens) + sentence_token_len,
                                                 type="sentence_node", text=sentence)
                    all_sentence_nodes.update({str(sentence_id): sentence_node})
                    if len(entity_list) > 0:
                        for id, entity in enumerate(entity_list):
                            entity_tokens, _, _, _, _ = roberta_tokenize(entity['content'].lower(), self._tokenizer)
                            entity_offset_start = entity['start_char'] + sentence_start
                            entity_offset_end = entity['end_char'] + sentence_start
                            if passage_offset[-1][1] <= entity_offset_start:
                                break
                            else:
                                valid_entity_list.append(entity)

                    value_nodes = {}
                    if len(number_in_sentence_indices) > 0:
                        for idx, number in enumerate(number_in_sentence_indices):
                            number_in_sentence_indices[idx] += sentence_token_len
                            value_node = ValueNode(id=str(idx + len(number_indices)),
                                                   index=number_in_sentence_indices[idx],
                                                   type="value_nodes", value=number_in_sentence[idx])
                            value_nodes.update({value_node.get_id(): value_node})
                            all_values_nodes.update({value_node.get_id(): value_node})
                            ## sentence and value
                        sentences_values.update({sentence_node.get_id(): value_nodes})
                        numbers_in_passage.extend(number_in_sentence)
                        number_indices.extend(number_in_sentence_indices)
                        number_len.extend(number_in_sentence_len)

                    entity_nodes = {}
                    if len(valid_entity_list) > 0:
                        for valid_id, entity in enumerate(valid_entity_list):
                            entity_tokens, _, _, _, _ = roberta_tokenize(entity['content'], self._tokenizer)
                            entity_offset_start = entity['start_char'] + sentence_start
                            entity_offset_end = entity['end_char'] + sentence_start

                            entity_start_idx, entity_end_idx = self.find_entity_index(
                                (entity_offset_start, entity_offset_end),
                                len(entity_tokens), passage_offset)
                            assert entity_start_idx != -1 and entity_end_idx != -1 and entity_end_idx <= len(
                                passage_tokens) and entity_start_idx < entity_end_idx

                            ## filter some in number list
                            if entity_start_idx in number_in_sentence_indices:
                                continue
                            else:
                                entity_node = EntityNode(id=str(len(all_entities_nodes)),
                                                         start_index=entity_start_idx,
                                                         end_index=entity_end_idx, type="entity_node",
                                                         text=entity)
                                entity_nodes.update({entity_node.get_id(): entity_node})
                                all_entities_nodes.update({entity_node.get_id(): entity_node})
                            ## sentence between entity
                        if len(entity_nodes) > 0:
                            sentences_entities.update({sentence_node.get_id(): entity_nodes})


                elif sentence_token_len <= self.passage_length_limit < sentence_token_len + len(
                        sentence_tokens):
                    if sentence_token_len == self.passage_length_limit:
                        break
                    for idx, offset in enumerate(sentence_offset[:self.passage_length_limit - sentence_token_len]):
                        start = sentence_start + offset[0]
                        end = sentence_start + offset[1]
                        new_offset = (start, end)
                        sentence_offset[idx] = new_offset
                    passage_tokens.extend(sentence_tokens[:self.passage_length_limit - sentence_token_len])
                    passage_offset.extend(sentence_offset[:self.passage_length_limit - sentence_token_len])

                    sentence_node = SentenceNode(id=str(sentence_id), start_index=sentence_token_len,
                                                 end_index=len(sentence_tokens[
                                                               :self.passage_length_limit - sentence_token_len]) + sentence_token_len,
                                                 type="sentence_node", text=sentence)
                    all_sentence_nodes.update({str(sentence_id): sentence_node})

                    if len(number_in_sentence_indices) > 0:
                        number_in_sentence_indices, number_in_sentence_len, number_in_sentence = \
                            clipped_passage_num(list(map(lambda x: x + sentence_token_len, number_in_sentence_indices)),
                                                number_in_sentence_len, number_in_sentence, len(passage_tokens))

                        number_in_sentence_indices = list(
                            map(lambda x: x - sentence_token_len, number_in_sentence_indices))

                    value_nodes = {}
                    if len(number_in_sentence_indices) > 0:
                        for idx, number in enumerate(number_in_sentence_indices):
                            number_in_sentence_indices[idx] += sentence_token_len
                            value_node = ValueNode(id=str(idx + len(number_indices)),
                                                   index=number_in_sentence_indices[idx],
                                                   type="value_nodes", value=number_in_sentence[idx])
                            value_nodes.update({value_node.get_id(): value_node})
                            all_values_nodes.update({value_node.get_id(): value_node})
                            ## sentence and value
                        sentences_values.update({sentence_node.get_id(): value_nodes})
                        numbers_in_passage.extend(number_in_sentence)
                        number_indices.extend(number_in_sentence_indices)
                        number_len.extend(number_in_sentence_len)

                    if len(entity_list) > 0:
                        for id, entity in enumerate(entity_list):
                            entity_tokens, _, _, _, _ = roberta_tokenize(entity['content'], self._tokenizer)
                            entity_offset_start = entity['start_char'] + sentence_start
                            entity_offset_end = entity['end_char'] + sentence_start
                            if passage_offset[-1][1] <= entity_offset_start:
                                break
                            else:
                                valid_entity_list.append(entity)

                    entity_nodes = {}
                    if len(valid_entity_list) > 0:
                        for valid_id, entity in enumerate(valid_entity_list):
                            entity_tokens, _, _, _, _ = roberta_tokenize(entity['content'], self._tokenizer)
                            entity_offset_start = entity['start_char'] + sentence_start
                            entity_offset_end = entity['end_char'] + sentence_start

                            entity_start_idx, entity_end_idx = self.find_entity_index(
                                (entity_offset_start, entity_offset_end),
                                len(entity_tokens), passage_offset)
                            assert entity_start_idx != -1 and entity_end_idx != -1 and entity_end_idx <= len(
                                passage_tokens) and entity_start_idx < entity_end_idx

                            ## filter some in number list
                            if entity_start_idx in number_in_sentence_indices:
                                continue
                            else:

                                entity_node = EntityNode(id=str(len(all_entities_nodes)),
                                                         start_index=entity_start_idx,
                                                         end_index=entity_end_idx, type="entity_node",
                                                         text=entity)
                                entity_nodes.update({entity_node.get_id(): entity_node})
                                all_entities_nodes.update({entity_node.get_id(): entity_node})
                            ## sentence between entity
                        if len(entity_nodes) > 0:
                            sentences_entities.update({sentence_node.get_id(): entity_nodes})
                else:
                    break

                sentence_token_len += len(sentence_tokens)
                sentence_start = sentence_offset[-1][1] + 1

            passage_text = " ".join(passage_sentence_text_list)
            passage_text = " ".join(whitespace_tokenize(passage_text))

            ## entities values
            for (sentence_id, entity_nodes) in sentences_entities.items():
                if sentence_id in sentences_values.keys():
                    value_nodes = sentences_values[sentence_id]
                    entities_values.update({sentence_id: (entity_nodes, value_nodes)})

            ## entities_entities
            for (sentence_id, entity_nodes) in sentences_entities.items():
                entities_entities.update({sentence_id: (entity_nodes, entity_nodes)})

            ## same entity mention
            # same_entity_mention_relation = []
            # for entity_id1, entity_node1 in all_entities_nodes.items():
            #     for entity_id2, entity_node2 in all_entities_nodes.items():
            #         if entity_id1 != entity_id2 and entity_node1.get_text()["content"].lower() == \
            #                 entity_node2.get_text()["content"].lower():
            #             same_entity_mention_relation.append((entity_node1, entity_node2))

            passage_instance = {
                "passage_id": passage_id,
                "passage_text": passage_text,
                "passage_tokens": passage_tokens,
                "passage_offset": passage_offset,
                "numbers_in_passage": numbers_in_passage,
                "number_indices": number_indices,
                "all_sentence_nodes": all_sentence_nodes,
                "all_entities_nodes": all_entities_nodes,
                "all_values_nodes": all_values_nodes,
                "sentences_entities": sentences_entities,
                "sentences_values": sentences_values,
                "entities_values": entities_values,
                "entities_entities": entities_entities
                # "same_entity_mention_relation": same_entity_mention_relation
            }

        else:

            passage_text = " ".join(whitespace_tokenize(passage_text))
            passage_tokens, passage_offset, numbers_in_passage, number_indices, number_len = roberta_tokenize(
                passage_text,
                self._tokenizer)
            # p_count_number = [i for i in range(10)]
            if self.passage_length_limit is not None:
                passage_tokens = passage_tokens[: self.passage_length_limit]
                if len(number_indices) > 0:
                    number_indices, number_len, numbers_in_passage = \
                        clipped_passage_num(number_indices, number_len, numbers_in_passage, len(passage_tokens))

            passage_instance = {"passage_id": passage_id,
                                "passage_text": passage_text,
                                "passage_tokens": passage_tokens,
                                "passage_offset": passage_offset,
                                "number_indices": number_indices,
                                "number_len": number_len,
                                "numbers_in_passage": numbers_in_passage,
                                "all_sentence_nodes": None,
                                "all_entities_nodes": None,
                                "all_values_nodes": None,
                                "sentences_entities": None,
                                "sentences_values": None,
                                "entities_values": None,
                                "entities_entities": None,
                                # "same_entity_mention_relation": None
                                }

        return passage_instance

    def text_to_instance(self, question_text: str, question_id: str, answer_annotations,
                         passage_id,
                         passage_text,
                         passage_tokens=None,
                         passage_offset=None,
                         number_indices=None,
                         number_len=None,
                         numbers_in_passage=None,
                         all_sentence_nodes=None,
                         all_entities_nodes=None,
                         all_values_nodes=None,
                         sentences_entities=None,
                         sentences_values=None,
                         entities_values=None,
                         entities_entities=None,
                         # same_entity_mention_relation=None
                         ):

        question_nodes = None
        question_entity_nodes = None
        question_value_nodes = None
        question_entity_relation = None
        question_value_relation = None
        question_entity_entity_relation = None
        question_entity_value_relation = None

        if self.passage_length_limit is not None:
            passage_tokens = passage_tokens[: self.passage_length_limit]
            if len(number_indices) > 0:
                number_indices, number_len, numbers_in_passage = \
                    clipped_passage_num(number_indices, number_len, numbers_in_passage, len(passage_tokens))

        question_text_ignore_case = " ".join(whitespace_tokenize(str(question_text), ignore=True))
        question_text_entity_list = self.text_processor_spacy.ner(question_text_ignore_case)
        question_text = " ".join(whitespace_tokenize(question_text))
        question_tokens, question_offset, numbers_in_question, question_number_indices, question_number_len = roberta_tokenize(
            question_text, self._tokenizer)

        if self.question_length_limit is not None:
            question_tokens = question_tokens[: self.question_length_limit]
            question_offset = question_offset[: self.question_length_limit]
            if len(question_number_indices) > 0:
                question_number_indices, question_number_len, numbers_in_question = \
                    clipped_passage_num(question_number_indices, question_number_len, numbers_in_question,
                                        len(question_tokens))

        if self.add_relation_reasoning_module:
            question_valid_entity_list = []
            question_entity_nodes = {}
            question_value_nodes = {}
            question_entity_relation = {}
            question_value_relation = {}
            question_entity_entity_relation = {}
            question_entity_value_relation = {}
            question_nodes = {}
            question_node = SentenceNode(id="0", start_index=0, end_index=len(question_tokens), type="question_node",
                                         text=question_text)
            question_nodes.update({question_node.get_id(): question_node})

            same_entity_mention_with_q_relation = []

            if len(question_text_entity_list) > 0:
                for id, entity in enumerate(question_text_entity_list):
                    entity_tokens, _, _, _, _ = roberta_tokenize(entity['content'].lower(), self._tokenizer)
                    entity_offset_start = entity['start_char']
                    entity_offset_end = entity['end_char']
                    if question_offset[-1][1] <= entity_offset_start:
                        break
                    else:
                        question_valid_entity_list.append(entity)

            if len(question_valid_entity_list) > 0:
                for valid_id, entity in enumerate(question_valid_entity_list):
                    entity_tokens, _, _, _, _ = roberta_tokenize(entity['content'], self._tokenizer)
                    entity_offset_start = entity['start_char']
                    entity_offset_end = entity['end_char']

                    entity_start_idx, entity_end_idx = self.find_entity_index((entity_offset_start, entity_offset_end),
                                                                              len(question_tokens), question_offset)
                    if not (entity_start_idx != -1 and entity_end_idx != -1 and len(
                            question_tokens) >= entity_end_idx > entity_start_idx):
                        print('entity_start_idx', entity_start_idx)
                        print('entity_end_idx', entity_end_idx)
                        print('len of question_tokens', len(question_tokens))

                    ## filter some in number list
                    if entity_start_idx in question_number_indices:
                        continue
                    else:
                        entity_node = EntityNode(id=str(len(question_entity_nodes)),
                                                 start_index=entity_start_idx,
                                                 end_index=entity_end_idx, type="question_entity_node",
                                                 text=entity)
                        question_entity_nodes.update({entity_node.get_id(): entity_node})
                    ## sentence between entity
                if len(question_entity_nodes) > 0:
                    question_entity_relation.update({question_node.get_id(): question_entity_nodes})

                value_nodes = {}
                if len(question_number_indices) > 0:
                    for idx, number in enumerate(question_number_indices):
                        value_node = ValueNode(id=str(idx),
                                               index=question_number_indices[idx],
                                               type="question_value_nodes", value=numbers_in_question[idx])
                        value_nodes.update({value_node.get_id(): value_node})
                        question_value_nodes.update({value_node.get_id(): value_node})
                        ## sentence and value
                    question_value_relation.update({question_node.get_id(): value_nodes})

                ## entities values
                for (question_id, entity_nodes) in question_entity_relation.items():
                    if question_id in question_value_relation.keys():
                        value_nodes = question_value_relation[question_id]
                        question_entity_value_relation.update({question_id: (entity_nodes, value_nodes)})
                ## entities_entities
                for (question_id, entity_nodes) in question_entity_relation.items():
                    question_entity_entity_relation.update({question_id: (entity_nodes, entity_nodes)})

                ## same entity mention
                # for entity_id, entity_node in all_entities_nodes.items():
                #     for q_entity_id, q_entity_node in question_entity_nodes.items():
                #         if entity_node.get_text()["content"].lower() == q_entity_node.get_text()["content"].lower():
                #             same_entity_mention_with_q_relation.append((entity_node, q_entity_node))

        answer_type: str = None
        answer_texts: List[str] = []
        if answer_annotations:
            # Currently we only use the first annotated answer here, but actually this doesn't affect
            # the training, because we only have one annotation for the train set.
            answer_type, answer_texts = self.extract_answer_info_from_annotation(answer_annotations[0])
            answer_texts = [" ".join(whitespace_tokenize(answer_text)) for answer_text in answer_texts]

        # Tokenize the answer text in order to find the matched span based on token
        tokenized_answer_texts = []

        for answer_text in answer_texts:
            answer_tokens, _, _, _, _ = roberta_tokenize(answer_text, self._tokenizer, True)
            tokenized_answer_texts.append(' '.join(token for token in answer_tokens))

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
                    question_number_order = all_number_order[len(numbers_in_question):]

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

            valid_passage_spans = self.find_valid_spans(passage_tokens,
                                                        tokenized_answer_texts) if tokenized_answer_texts else []
            if len(valid_passage_spans) > 0:
                valid_question_spans = []
            else:
                valid_question_spans = self.find_valid_spans(question_tokens,
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

            type_to_answer_map = {"passage_span": valid_passage_spans, "question_span": valid_question_spans,
                                  "addition_subtraction": valid_signs_for_add_sub_expressions, "counting": valid_counts}

            if self.skip_when_all_empty and not any(
                    type_to_answer_map[skip_type] for skip_type in self.skip_when_all_empty):
                return None

            answer_info = {"answer_texts": answer_texts,  # this `answer_texts` will not be used for evaluation
                           "answer_passage_spans": valid_passage_spans, "answer_question_spans": valid_question_spans,
                           "signs_for_add_sub_expressions": valid_signs_for_add_sub_expressions, "counts": valid_counts}

            return self.make_marginal_drop_instance(question_tokens,
                                                    passage_tokens,
                                                    numbers_as_tokens,
                                                    number_indices,
                                                    passage_number_order,
                                                    question_number_order,
                                                    question_number_indices,
                                                    aux_numbers_as_tokens,
                                                    aux_number_order,
                                                    answer_info,
                                                    all_sentence_nodes,
                                                    all_entities_nodes,
                                                    all_values_nodes,
                                                    sentences_entities,
                                                    sentences_values,
                                                    entities_values,
                                                    entities_entities,
                                                    question_nodes,
                                                    question_entity_nodes,
                                                    question_value_nodes,
                                                    question_entity_relation,
                                                    question_value_relation,
                                                    question_entity_entity_relation,
                                                    question_entity_value_relation,
                                                    # same_entity_mention_relation,
                                                    # same_entity_mention_with_q_relation,
                                                    additional_metadata={"original_passage": passage_text,
                                                                         "passage_token_offsets": passage_offset,
                                                                         "original_question": question_text,
                                                                         "question_token_offsets": question_offset,
                                                                         "original_numbers": numbers_in_passage,
                                                                         "aux_numbers": aux_num_list,
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
    def find_valid_spans(passage_tokens: List[str], answer_texts: List[str]) -> List[Tuple[int, int]]:
        normalized_tokens = [token.strip(USTRIPPED_CHARACTERS) for token in passage_tokens]
        # normalized_tokens = passage_tokens
        word_positions: Dict[str, List[int]] = defaultdict(list)
        for i, token in enumerate(normalized_tokens):
            word_positions[token].append(i)
        spans = []
        for answer_text in answer_texts:
            answer_tokens = [token.strip(USTRIPPED_CHARACTERS) for token in answer_text.split()]
            num_answer_tokens = len(answer_tokens)
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
                                    number_tokens: List[str],
                                    number_indices: List[int],
                                    passage_number_order: np.ndarray,
                                    question_number_order: np.ndarray,
                                    question_number_indices: List[int],
                                    aux_numbers_as_tokens: List[int],
                                    aux_number_order: List[int],
                                    answer_info: Dict[str, Any] = None,
                                    all_sentence_nodes=None,
                                    all_entities_nodes=None,
                                    all_values_nodes=None,
                                    sentences_entities=None,
                                    sentences_values=None,
                                    entities_values=None,
                                    entities_entities=None,
                                    question_nodes=None,
                                    question_entity_nodes=None,
                                    question_value_nodes=None,
                                    question_entity_relation=None,
                                    question_value_relation=None,
                                    question_entity_entity_relation=None,
                                    question_entity_value_relation=None,
                                    # same_entity_mention_relation=None,
                                    # same_entity_mention_with_q_relation=None,
                                    additional_metadata: Dict[str, Any] = None):
        metadata = {
            "question_tokens": [token for token in question_tokens],
            "passage_tokens": [token for token in passage_tokens],
            "number_tokens": [token for token in number_tokens],
            "number_indices": number_indices,
            "question_number_indices": question_number_indices,
            "passage_number_order": passage_number_order,
            "question_number_order": question_number_order,
            "aux_number_order": aux_number_order,
            "aux_number_as_tokens": aux_numbers_as_tokens,

            "all_sentence_nodes": all_sentence_nodes,
            "all_entities_nodes": all_entities_nodes,
            "all_values_nodes": all_values_nodes,
            "sentences_entities": sentences_entities,
            "sentences_values": sentences_values,
            "entities_values": entities_values,
            "entities_entities": entities_entities,

            "question_nodes": question_nodes,
            "question_entity_nodes": question_entity_nodes,
            "question_value_nodes": question_value_nodes,
            "question_entity_relation": question_entity_relation,
            "question_value_relation": question_value_relation,
            "question_entity_entity_relation": question_entity_entity_relation,
            "question_entity_value_relation": question_entity_value_relation,
            # "same_entity_mention_relation": same_entity_mention_relation,
            # "same_entity_mention_with_q_relation": same_entity_mention_with_q_relation
        }
        if answer_info:
            metadata["answer_texts"] = answer_info["answer_texts"]
            metadata["answer_passage_spans"] = answer_info["answer_passage_spans"]
            metadata["answer_question_spans"] = answer_info["answer_question_spans"]
            metadata["signs_for_add_sub_expressions"] = answer_info["signs_for_add_sub_expressions"]
            metadata["counts"] = answer_info["counts"]

        metadata.update(additional_metadata)
        return metadata
