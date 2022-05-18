import torch
import torch.nn as nn
from typing import List, Dict, Any
from collections import OrderedDict
import torch.nn.functional as F

from tools import allennlp as util
from tools.utils import DropEmAndF1
from mspan_roberta_gcn.util import GCN, ResidualGRU, FFNLayer
from .multispan_heads import multispan_heads_mapping, remove_substring_from_prediction
from tools.allennlp import mask_mse_loss
from mspan_roberta_gcn.util import FFNLayer, GCN, ResidualGRU, SignNumFFNLayer, DigitalGCN, HeterogeneousGCN, \
    SimpleQDGAT, MultiHeadQDGAT, KV_Relation_module
from torch.autograd import Variable
import numpy as np

# from post_processor.answer_type_enhance import compute_answer_type_mask

TOKEN_TYPE = ['OTHER', 'ENTITY', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL',
              'CARDINAL', 'YARD']
ENTITY_TOKEN_TYPE = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']
TOKEN_TYPE_ID = [i for i in range(len(TOKEN_TYPE))]
TOKEN_TYPE_MAP_ID = dict(zip(TOKEN_TYPE, TOKEN_TYPE_ID))
NUMBER_TOKEN_TYPE = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'YARD']
NUMBER_TOKEN_TYPE_ID = [TOKEN_TYPE_MAP_ID[token_type] for token_type in NUMBER_TOKEN_TYPE]
NUMBER_TOKEN_TYPE_MAP_ID = dict(zip(NUMBER_TOKEN_TYPE, NUMBER_TOKEN_TYPE_ID))


def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor, topk: int):
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.

    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                          device=device)).log()
    #mport pdb
    ''' 
    span_log_mask = torch.zeros((passage_length, passage_length), device=device)
    for i in range(passage_length):
        margin = min(i+15, passage_length)
        for j in range(i, margin):
            span_log_mask[i][j] = 1
    #pdb.set_trace()
    span_log_mask = span_log_mask.log()
    '''
    
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)

    _, topk_indices = valid_span_log_probs.view(batch_size, -1).topk(topk)

    topk_start_indices = topk_indices // passage_length  ## bsz, topk, 2
    topk_end_indices = topk_indices % passage_length  ## bsz, topk, 2

    span_start_indices = best_spans // passage_length
    span_end_indices = best_spans % passage_length

    return torch.stack([span_start_indices, span_end_indices], dim=-1), torch.stack(
        [topk_start_indices, topk_end_indices], dim=-1)


def convert_number_to_str(number):
    if isinstance(number, int):
        return str(number)

    # we leave at most 3 decimal places
    num_str = '%.3f' % number

    for i in range(3):
        if num_str[-1] == '0':
            num_str = num_str[:-1]
        else:
            break

    if num_str[-1] == '.':
        num_str = num_str[:-1]

    # if number < 1, them we will omit the zero digit of the integer part
    if num_str[0] == '0' and len(num_str) > 1:
        num_str = num_str[1:]

    return num_str


class NumericallyAugmentedBertNet(nn.Module):
    """
    This class augments the QANet model with some rudimentary numerical reasoning abilities, as
    published in the original DROP paper.

    The main idea here is that instead of just predicting a passage span after doing all of the
    QANet modeling stuff, we add several different "answer abilities": predicting a span from the
    question, predicting a count, or predicting an arithmetic expression.  Near the end of the
    QANet model, we have a variable that predicts what kind of answer type we need, and each branch
    has separate modeling logic to predict that answer type.  We then marginalize over all possible
    ways of getting to the right answer through each of these answer types.
    """

    def __init__(self,
                 bert,
                 hidden_size: int,
                 dropout_prob: float = 0.1,
                 answering_abilities: List[str] = None,
                 use_gcn: bool = False,
                 gcn_steps: int = 1,
                 unique_on_multispan: bool = True,
                 multispan_head_name: str = "flexible_loss",
                 multispan_generation_top_k: int = 0,
                 multispan_prediction_beam_size: int = 1,
                 multispan_use_prediction_beam_search: bool = False,
                 multispan_use_bio_wordpiece_mask: bool = True,
                 dont_add_substrings_to_ms: bool = True,
                 trainning: bool = True,
                 add_op_layer: bool = False,
                 bert_name: str = None,
                 ) -> None:
        super(NumericallyAugmentedBertNet, self).__init__()
        self.training = trainning
        self.multispan_head_name = multispan_head_name
        self.multispan_use_prediction_beam_search = multispan_use_prediction_beam_search
        self.multispan_use_bio_wordpiece_mask = multispan_use_bio_wordpiece_mask
        self._dont_add_substrings_to_ms = dont_add_substrings_to_ms

        self.use_gcn = use_gcn
        self.bert = bert
        self.add_op_layer = add_op_layer
        self.bert_name = bert_name
        self.gcn_steps = gcn_steps
        modeling_out_dim = hidden_size
        self._drop_metrics = DropEmAndF1()
        if answering_abilities is None:
            self.answering_abilities = ["passage_span_extraction", "question_span_extraction",
                                        "addition_subtraction", "counting", "multiple_spans"]
        else:
            self.answering_abilities = answering_abilities

        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = FFNLayer(3 * hidden_size, hidden_size, len(self.answering_abilities),
                                                      dropout_prob)

        if "passage_span_extraction" in self.answering_abilities:
            self._passage_span_extraction_index = self.answering_abilities.index("passage_span_extraction")
            self._span_start_predictor = nn.Linear(4 * hidden_size, 1, bias=False)
            self._span_end_predictor = nn.Linear(4 * hidden_size, 1, bias=False)

        if "question_span_extraction" in self.answering_abilities:
            self._question_span_extraction_index = self.answering_abilities.index("question_span_extraction")

        if "addition_subtraction" in self.answering_abilities:
            self._addition_subtraction_index = self.answering_abilities.index("addition_subtraction")
            self._number_sign_predictor = FFNLayer(5 * hidden_size, hidden_size, 3, dropout_prob)

        if "counting" in self.answering_abilities:
            self._counting_index = self.answering_abilities.index("counting")
            self._count_number_predictor = FFNLayer(5 * hidden_size, hidden_size, 10, dropout_prob)

        # add multiple span prediction from https://github.com/eladsegal/project-NLP-AML
        if "multiple_spans" in self.answering_abilities:
            if self.multispan_head_name == "flexible_loss":
                self.multispan_head = multispan_heads_mapping[multispan_head_name](self.bert_name, modeling_out_dim,
                                                                                   generation_top_k=multispan_generation_top_k,
                                                                                   prediction_beam_size=multispan_prediction_beam_size)
            else:
                self.multispan_head = multispan_heads_mapping[multispan_head_name](modeling_out_dim)

            self._unique_on_multispan = unique_on_multispan

        self._dropout = torch.nn.Dropout(p=dropout_prob)

        node_dim = modeling_out_dim
        self._gcn_input_proj = nn.Linear(node_dim * 2, node_dim)
        self._gcn_enc = ResidualGRU(hidden_size, dropout_prob, 2)

        if self.use_gcn:
            self._gcn = GCN(node_dim=node_dim, iteration_steps=gcn_steps)
            self._iteration_steps = gcn_steps
            print('gcn iteration_steps=%d' % self._iteration_steps, flush=True)

        # add bert proj
        self._proj_sequence_h = nn.Linear(hidden_size, 1, bias=False)
        self._proj_number = nn.Linear(hidden_size * 2, 1, bias=False)

#         self._proj_sequence_g0 = FFNLayer(hidden_size, hidden_size, 1, dropout_prob)
        self._proj_sequence_g1 = FFNLayer(hidden_size, hidden_size, 1, dropout_prob)
        self._proj_sequence_g2 = FFNLayer(hidden_size, hidden_size, 1, dropout_prob)
        self._proj_sequence_g3 = FFNLayer(hidden_size, hidden_size, 1, dropout_prob)

        if self.add_op_layer:
            self.OP = ["addition", "substraction", "max", "min", "argmin", "argmax", "argmore", "argless", "key_value",
                       "count", "extract_span"]
            self.OP_ID = [i for i in range(len(self.OP))]
            self.OP_ID_MAP = dict(zip(self.OP, self.OP_ID))
            self._op_embedding = nn.Parameter(torch.FloatTensor(len(self.OP), hidden_size), requires_grad=True)
            nn.init.xavier_uniform(self._op_embedding)
            self.OP_MODULES = nn.ModuleDict(
                {item: nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4) for item in self.OP})

            self._answer_ability_predictor = FFNLayer(4 * hidden_size, hidden_size, len(self.answering_abilities),
                                                      dropout_prob)
            self._count_number_predictor = FFNLayer(6 * hidden_size, hidden_size, 10, dropout_prob)

            self._span_start_predictor = nn.Linear(5 * hidden_size, 1, bias=False)
            self._span_end_predictor = nn.Linear(5 * hidden_size, 1, bias=False)
            self._number_sign_predictor = FFNLayer(6 * hidden_size, hidden_size, 3, dropout_prob)
            self.op_lamda = 0.3

    def forward(self,  # type: ignore
                input_ids: torch.LongTensor,
                input_mask: torch.LongTensor,
                input_segments: torch.LongTensor,
                passage_mask: torch.LongTensor,
                question_mask: torch.LongTensor,
                number_indices: torch.LongTensor,
                passage_number_order: torch.LongTensor,
                question_number_order: torch.LongTensor,
                question_number_indices: torch.LongTensor,
                
                gnodes: torch.LongTensor,
                gnodes_len: torch.LongTensor,
                gnodes_mask: torch.LongTensor,
                gedges: torch.LongTensor,
                gedge_types: int,
                
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                answer_as_add_sub_expressions: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                answer_as_text_to_disjoint_bios: torch.LongTensor = None,
                answer_as_list_of_bios: torch.LongTensor = None,
                span_bio_labels: torch.LongTensor = None,
                bio_wordpiece_mask: torch.LongTensor = None,
                is_bio_mask: torch.LongTensor = None,
                
                question_tag: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None
                ) -> Dict[str, torch.Tensor]:

        if "multiple_spans" in self.answering_abilities:
            multispan_module = self.multispan_head.predict_module
            multispan_log_likelihood = self.multispan_head.log_likelihood
            multispan_prediction = self.multispan_head.prediction

        batch_size = input_ids.size(0)
        outputs = self.bert(input_ids, attention_mask=input_mask, token_type_ids=input_segments)
        sequence_output = outputs[0]
        sequence_output_list = [item for item in outputs[-1][-4:]]

        sequence_alg = self._gcn_input_proj(torch.cat([sequence_output_list[2], sequence_output_list[3]], dim=2))
        if (
                "passage_span_extraction" in self.answering_abilities or "question_span" in self.answering_abilities) and self.use_gcn:
            encoded_passage_for_numbers = sequence_alg
            encoded_question_for_numbers = sequence_alg
            # passage number extraction
            real_number_indices = number_indices - 1
            number_mask = (real_number_indices > -1).long()  # ??
            clamped_number_indices = util.replace_masked_values(real_number_indices, number_mask, 0)
            encoded_numbers = torch.gather(encoded_passage_for_numbers, 1,
                                           clamped_number_indices.unsqueeze(-1).expand(-1, -1,
                                                                                       encoded_passage_for_numbers.size(
                                                                                           -1)))

            # question number extraction
            question_number_mask = (question_number_indices > -1).long()
            clamped_question_number_indices = util.replace_masked_values(question_number_indices, question_number_mask,
                                                                         0)
            question_encoded_number = torch.gather(encoded_question_for_numbers, 1,
                                                   clamped_question_number_indices.unsqueeze(-1).expand(-1, -1,
                                                                                                        encoded_question_for_numbers.size(
                                                                                                            -1)))

            # graph mask
            number_order = torch.cat((passage_number_order, question_number_order), -1)
            new_graph_mask = number_order.unsqueeze(1).expand(batch_size, number_order.size(-1),
                                                              -1) > number_order.unsqueeze(-1).expand(batch_size, -1,
                                                                                                      number_order.size(
                                                                                                          -1))
            new_graph_mask = new_graph_mask.long()
            all_number_mask = torch.cat((number_mask, question_number_mask), dim=-1)
            new_graph_mask = all_number_mask.unsqueeze(1) * all_number_mask.unsqueeze(-1) * new_graph_mask
            # iteration
            d_node, q_node, d_node_weight, _ = self._gcn(d_node=encoded_numbers, q_node=question_encoded_number,
                                                         d_node_mask=number_mask, q_node_mask=question_number_mask,
                                                         graph=new_graph_mask)
            gcn_info_vec = torch.zeros((batch_size, sequence_alg.size(1) + 1, sequence_output_list[-1].size(-1)),
                                       dtype=torch.float, device=d_node.device)

            clamped_number_indices = util.replace_masked_values(real_number_indices, number_mask,
                                                                gcn_info_vec.size(1) - 1)
            gcn_info_vec.scatter_(1, clamped_number_indices.unsqueeze(-1).expand(-1, -1, d_node.size(-1)), d_node)
            gcn_info_vec = gcn_info_vec[:, :-1, :]

            sequence_output_list[2] = sequence_output_list[2] + gcn_info_vec
            sequence_output_list[0] = sequence_output_list[0] + gcn_info_vec
            sequence_output_list[1] = sequence_output_list[1] + gcn_info_vec
            sequence_output_list[3] = sequence_output_list[3] + gcn_info_vec

        if self.use_gcn:
            sequence_output_list[2] = self._gcn_enc(self._proj_ln(sequence_output_list[2]))
            sequence_output_list[0] = self._gcn_enc(self._proj_ln0(sequence_output_list[0]))
            sequence_output_list[1] = self._gcn_enc(self._proj_ln1(sequence_output_list[1]))
            sequence_output_list[3] = self._gcn_enc(self._proj_ln3(sequence_output_list[3]))
        else:
            sequence_output_list[2] = self._gcn_enc(sequence_output_list[2])
            sequence_output_list[0] = self._gcn_enc(sequence_output_list[0])
            sequence_output_list[1] = self._gcn_enc(sequence_output_list[1])
            sequence_output_list[3] = self._gcn_enc(sequence_output_list[3])

        sequence_h3_weight = self._proj_sequence_h(sequence_output_list[3]).squeeze(-1)
        passage_h3_weight = util.masked_softmax(sequence_h3_weight, passage_mask)
        passage_h3 = util.weighted_sum(sequence_output_list[3], passage_h3_weight)
        question_h3_weight = util.masked_softmax(sequence_h3_weight, question_mask)
        question_h3 = util.weighted_sum(sequence_output_list[3], question_h3_weight)

        question_g1_weight = self._proj_sequence_g1(sequence_output_list[1]).squeeze(-1)
        question_g1_weight = util.masked_softmax(question_g1_weight, question_mask)
        question_g1 = util.weighted_sum(sequence_output_list[1], question_g1_weight)

        question_g2_weight = self._proj_sequence_g2(sequence_output_list[2]).squeeze(-1)
        question_g2_weight = util.masked_softmax(question_g2_weight, question_mask)
        question_g2 = util.weighted_sum(sequence_output_list[2], question_g2_weight)

        question_g3_weight = self._proj_sequence_g3(sequence_output_list[3]).squeeze(-1)
        question_g3_weight = util.masked_softmax(question_g3_weight, question_mask)
        question_g3 = util.weighted_sum(sequence_output_list[3], question_g3_weight)

        real_number_indices = number_indices.squeeze(-1) - 1
        number_mask = (real_number_indices > -1).long()
        clamped_number_indices = util.replace_masked_values(real_number_indices, number_mask, 0)
        encoded_passage_for_numbers = torch.cat([sequence_output_list[2], sequence_output_list[3]], dim=-1)
        encoded_numbers = torch.gather(encoded_passage_for_numbers, 1,
                                       clamped_number_indices.unsqueeze(-1).expand(-1, -1,
                                                                                   encoded_passage_for_numbers.size(
                                                                                       -1)))
        number_weight = self._proj_number(encoded_numbers).squeeze(-1)
        number_mask = (number_indices > -1).long()
        number_weight = util.masked_softmax(number_weight, number_mask)
        number_vector = util.weighted_sum(encoded_numbers, number_weight)

        
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.add_op_layer:
            seq_h3 = torch.cat([question_h3, sequence_output[:, 0]], dim=1).unsqueeze(-1)
            op_weight_logit = torch.matmul(self._op_embedding.repeat(1, 2), seq_h3).squeeze(-1)
            op_weight = nn.functional.softmax(op_weight_logit, dim=-1)
            op_outputs = []
            for op_name, op_module in self.OP_MODULES.items():
                op_id = torch.tensor([self.OP_ID_MAP[op_name]])
                query = self._op_embedding[op_id].unsqueeze(1).expand(-1, batch_size, -1)
                key = sequence_output_list[3].transpose(dim0=0, dim1=1)
                value = sequence_output_list[3].transpose(dim0=0, dim1=1)
                attn_output, attn_output_weights = op_module(query, key, value)
                op_outputs.append(attn_output.transpose(dim0=0, dim1=1))
            op_outputs = torch.stack(op_outputs, dim=1).squeeze(2)
            op_outputs_h3 = util.weighted_sum(op_outputs, op_weight)
            op_h3 = util.weighted_sum(self._op_embedding.unsqueeze(0).expand(batch_size, -1, -1), op_weight)
        
        ## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if len(self.answering_abilities) > 1:
            # Shape: (batch_size, number_of_abilities)
            if self.add_op_layer:
                answer_ability_logits = self._answer_ability_predictor(
                    torch.cat([op_h3, passage_h3, question_h3, sequence_output[:, 0]], 1))
            else:
                answer_ability_logits = self._answer_ability_predictor(
                    torch.cat([passage_h3, question_h3, sequence_output[:, 0]], 1))
            answer_ability_log_probs = F.log_softmax(answer_ability_logits, -1)
            best_answer_ability = torch.argmax(answer_ability_log_probs, 1)
            top_two_answer_abilities = torch.topk(answer_ability_log_probs, k=2, dim=1)

        if "counting" in self.answering_abilities:
            # Shape: (batch_size, 10)
            if self.add_op_layer:
                count_number_logits = self._count_number_predictor(
                    torch.cat([op_outputs_h3, number_vector, passage_h3, question_h3, sequence_output[:, 0]], dim=1))
            else:
                count_number_logits = self._count_number_predictor(
                    torch.cat([number_vector, passage_h3, question_h3, sequence_output[:, 0]], dim=1))

            count_number_log_probs = torch.nn.functional.log_softmax(count_number_logits, -1)
            # Info about the best count number prediction
            # Shape: (batch_size,)
            best_count_number = torch.argmax(count_number_log_probs, -1)
            best_count_log_prob = torch.gather(count_number_log_probs, 1, best_count_number.unsqueeze(-1)).squeeze(-1)
            if len(self.answering_abilities) > 1:
                best_count_log_prob += answer_ability_log_probs[:, self._counting_index]

        if "passage_span_extraction" in self.answering_abilities or "question_span_extraction" in self.answering_abilities:
            # start 0, 2
            sequence_for_span_start = torch.cat([sequence_output_list[2],
                                                 sequence_output_list[1],
                                                 sequence_output_list[2] * question_g2.unsqueeze(1),
                                                 sequence_output_list[1] * question_g1.unsqueeze(1)],
                                                dim=2)

            # Shape: (batch_size, passage_length, modeling_dim * 2)
            sequence_for_span_end = torch.cat([sequence_output_list[3],
                                               sequence_output_list[2],
                                               sequence_output_list[3] * question_g3.unsqueeze(1),
                                               sequence_output_list[2] * question_g2.unsqueeze(1)],
                                              dim=2)

            if self.add_op_layer:
                sequence_for_span_start = torch.cat(
                    [sequence_for_span_start, op_outputs_h3.unsqueeze(1).expand(-1, sequence_alg.size(1), -1)], dim=2)
                sequence_for_span_end = torch.cat(
                    [sequence_for_span_end, op_outputs_h3.unsqueeze(1).expand(-1, sequence_alg.size(1), -1)], dim=2)

            sequence_span_start_logits = self._span_start_predictor(sequence_for_span_start).squeeze(-1)
            # Shape: (batch_size, passage_length)
            sequence_span_end_logits = self._span_end_predictor(sequence_for_span_end).squeeze(-1)
            # Shape: (batch_size, passage_length)

            if "passage_span_extraction" in self.answering_abilities:
                passage_span_start_log_probs = util.masked_log_softmax(sequence_span_start_logits, passage_mask)
                passage_span_end_log_probs = util.masked_log_softmax(sequence_span_end_logits, passage_mask)

                # Info about the best passage span prediction
                passage_span_start_logits = util.replace_masked_values(sequence_span_start_logits, passage_mask, -6e4)
                passage_span_end_logits = util.replace_masked_values(sequence_span_end_logits, passage_mask, -6e4)
                # Shage: (batch_size, topk, 2)
                best_passage_span, topk_best_passage_span = get_best_span(passage_span_start_logits,
                                                                          passage_span_end_logits, topk=10)

            if "question_span_extraction" in self.answering_abilities:
                question_span_start_log_probs = util.masked_log_softmax(sequence_span_start_logits, question_mask)
                question_span_end_log_probs = util.masked_log_softmax(sequence_span_end_logits, question_mask)

                # Info about the best question span prediction
                question_span_start_logits = util.replace_masked_values(sequence_span_start_logits, question_mask, -6e4)
                question_span_end_logits = util.replace_masked_values(sequence_span_end_logits, question_mask, -6e4)
                # Shape: (batch_size, topk, 2)
                best_question_span, topk_best_question_span = get_best_span(question_span_start_logits,
                                                                            question_span_end_logits, topk=10)

        if "addition_subtraction" in self.answering_abilities:
            alg_encoded_numbers = torch.cat(
                [encoded_numbers,
                 question_h3.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                 passage_h3.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                 sequence_output[:, 0].unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)
                 ], 2)

            if self.add_op_layer:
                alg_encoded_numbers = torch.cat(
                    [alg_encoded_numbers, op_outputs_h3.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], 2)

            # Shape: (batch_size, # of numbers in the passage, 3)
            number_sign_logits = self._number_sign_predictor(alg_encoded_numbers)
            number_sign_log_probs = torch.nn.functional.log_softmax(number_sign_logits, -1)

            # Shape: (batch_size, # of numbers in passage).
            best_signs_for_numbers = torch.argmax(number_sign_log_probs, -1)
            # For padding numbers, the best sign masked as 0 (not included).
            best_signs_for_numbers = util.replace_masked_values(best_signs_for_numbers, number_mask, 0)
            # Shape: (batch_size, # of numbers in passage)
            best_signs_log_probs = torch.gather(number_sign_log_probs, 2, best_signs_for_numbers.unsqueeze(-1)).squeeze(
                -1)
            # the probs of the masked positions should be 1 so that it will not affect the joint probability
            # TODO: this is not quite right, since if there are many numbers in the passage,
            # TODO: the joint probability would be very small.
            best_signs_log_probs = util.replace_masked_values(best_signs_log_probs, number_mask, 0)
            # Shape: (batch_size,)
            best_combination_log_prob = best_signs_log_probs.sum(-1)
            if len(self.answering_abilities) > 1:
                best_combination_log_prob += answer_ability_log_probs[:, self._addition_subtraction_index]

        # add multiple span prediction
        if bio_wordpiece_mask is None or not self.multispan_use_bio_wordpiece_mask:
            multispan_mask = input_mask
        else:
            multispan_mask = input_mask * bio_wordpiece_mask
        if "multiple_spans" in self.answering_abilities:
            if self.multispan_head_name == "flexible_loss":
                multispan_log_probs, multispan_logits = multispan_module(sequence_output, seq_mask=multispan_mask.bool())
            else:
                multispan_log_probs, multispan_logits = multispan_module(sequence_output)

        output_dict = {}

        # If answer is given, compute the loss.
        if answer_as_passage_spans is not None or answer_as_question_spans is not None or answer_as_add_sub_expressions is not None or answer_as_counts is not None:

            log_marginal_likelihood_list = []

            for answering_ability in self.answering_abilities:
                if answering_ability == "passage_span_extraction":
                    # Shape: (batch_size, # of answer spans)
                    gold_passage_span_starts = answer_as_passage_spans[:, :, 0]
                    gold_passage_span_ends = answer_as_passage_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_passage_span_mask = (gold_passage_span_starts != -1).long()
                    clamped_gold_passage_span_starts = util.replace_masked_values(gold_passage_span_starts,
                                                                                  gold_passage_span_mask, 0)
                    clamped_gold_passage_span_ends = util.replace_masked_values(gold_passage_span_ends,
                                                                                gold_passage_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_span_starts = torch.gather(passage_span_start_log_probs, 1,
                                                                          clamped_gold_passage_span_starts)
                    log_likelihood_for_passage_span_ends = torch.gather(passage_span_end_log_probs, 1,
                                                                        clamped_gold_passage_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_spans = log_likelihood_for_passage_span_starts + log_likelihood_for_passage_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_passage_spans = util.replace_masked_values(log_likelihood_for_passage_spans,
                                                                                  gold_passage_span_mask, -6e4)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_passage_span = util.logsumexp(log_likelihood_for_passage_spans)

                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span)

                elif answering_ability == "question_span_extraction":
                    # Shape: (batch_size, # of answer spans)
                    gold_question_span_starts = answer_as_question_spans[:, :, 0]
                    gold_question_span_ends = answer_as_question_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_question_span_mask = (gold_question_span_starts != -1).long()
                    clamped_gold_question_span_starts = util.replace_masked_values(gold_question_span_starts,
                                                                                   gold_question_span_mask, 0)
                    clamped_gold_question_span_ends = util.replace_masked_values(gold_question_span_ends,
                                                                                 gold_question_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_span_starts = torch.gather(question_span_start_log_probs, 1,
                                                                           clamped_gold_question_span_starts)
                    log_likelihood_for_question_span_ends = torch.gather(question_span_end_log_probs, 1,
                                                                         clamped_gold_question_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_spans = log_likelihood_for_question_span_starts + log_likelihood_for_question_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_question_spans = util.replace_masked_values(log_likelihood_for_question_spans,
                                                                                   gold_question_span_mask, -6e4)
                    # Shape: (batch_size, )
                    # pylint: disable=invalid-name
                    log_marginal_likelihood_for_question_span = util.logsumexp(log_likelihood_for_question_spans)

                    # question multi span prediction
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span)

                elif answering_ability == "addition_subtraction":
                    # The padded add-sub combinations use 0 as the signs for all numbers, and we mask them here.
                    # Shape: (batch_size, # of combinations)
                    gold_add_sub_mask = (answer_as_add_sub_expressions.sum(-1) > 0).float()
                    # Shape: (batch_size, # of numbers in the passage, # of combinations)
                    gold_add_sub_signs = answer_as_add_sub_expressions.transpose(1, 2)
                    # Shape: (batch_size, # of numbers in the passage, # of combinations)
                    log_likelihood_for_number_signs = torch.gather(number_sign_log_probs, 2, gold_add_sub_signs)
                    # the log likelihood of the masked positions should be 0
                    # so that it will not affect the joint probability
                    log_likelihood_for_number_signs = util.replace_masked_values(log_likelihood_for_number_signs,
                                                                                 number_mask.unsqueeze(-1), 0)
                    # Shape: (batch_size, # of combinations)
                    log_likelihood_for_add_subs = log_likelihood_for_number_signs.sum(1)
                    # For those padded combinations, we set their log probabilities to be very small negative value
                    log_likelihood_for_add_subs = util.replace_masked_values(log_likelihood_for_add_subs,
                                                                             gold_add_sub_mask, -6e4)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_add_sub = util.logsumexp(log_likelihood_for_add_subs)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_add_sub)

                elif answering_ability == "counting":
                    # Count answers are padded with label -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    # Shape: (batch_size, # of count answers)
                    gold_count_mask = (answer_as_counts != -1).long()
                    # Shape: (batch_size, # of count answers)
                    clamped_gold_counts = util.replace_masked_values(answer_as_counts, gold_count_mask, 0)
                    log_likelihood_for_counts = torch.gather(count_number_log_probs, 1, clamped_gold_counts)
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_counts = util.replace_masked_values(log_likelihood_for_counts, gold_count_mask,
                                                                           -6e4)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_count = util.logsumexp(log_likelihood_for_counts)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_count)

                elif answering_ability == "multiple_spans":
                    if self.multispan_head_name == "flexible_loss":
                        log_marginal_likelihood_for_multispan = \
                            multispan_log_likelihood(answer_as_text_to_disjoint_bios,
                                                     answer_as_list_of_bios,
                                                     span_bio_labels,
                                                     multispan_log_probs,
                                                     multispan_logits,
                                                     multispan_mask,
                                                     bio_wordpiece_mask,
                                                     is_bio_mask)
                    else:
                        log_marginal_likelihood_for_multispan = \
                            multispan_log_likelihood(span_bio_labels,
                                                     multispan_log_probs,
                                                     multispan_mask,
                                                     is_bio_mask,
                                                     logits=multispan_logits)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_multispan)
                else:
                    raise ValueError(f"Unsupported answering ability: {answering_ability}")

            if len(self.answering_abilities) > 1:
                # Add the ability probabilities if there are more than one abilities
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]
            output_dict["loss"] = -1 * marginal_log_likelihood.mean()

            if self.add_op_layer:
                # Count answers are padded with label -1,
                # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                # Shape: (batch_size, # of question_tag )
                gold_question_tag_mask = (question_tag != -1).long()
                # Shape: (batch_size, # of count answers)
                clamped_gold_question_tag = util.replace_masked_values(question_tag, gold_question_tag_mask, 0)
                question_tag_log_probs = torch.nn.functional.log_softmax(op_weight_logit, -1)
                log_likelihood_for_question_tag = torch.gather(question_tag_log_probs, 1, clamped_gold_question_tag)
                log_likelihood_for_question_tag = util.replace_masked_values(log_likelihood_for_question_tag,
                                                                             gold_question_tag_mask,
                                                                             -6e4)
                # Shape: (batch_size, )
                op_loss = -util.logsumexp(log_likelihood_for_question_tag).mean()
                output_dict["loss"] += self.op_lamda * op_loss

        with torch.no_grad():
            if len(self.answering_abilities) > 1:
                best_answer_ability = best_answer_ability.detach().cpu().numpy()
            if metadata is not None:
                output_dict["question_id"] = []
                output_dict["answer"] = []
                i = 0
                while i < batch_size:
                    if len(self.answering_abilities) > 1:
                        answer_index = best_answer_ability[i]
                        predicted_ability_str = self.answering_abilities[answer_index]
                    else:
                        predicted_ability_str = self.answering_abilities[0]

                    answer_json: Dict[str, Any] = {}

                    question_start = 1
                    passage_start = len(metadata[i]["question_tokens"]) + 2
                    # We did not consider multi-mention answers here
                    answer_json['answer_type_probs'] = answer_ability_log_probs[i].cpu().detach().numpy().tolist()
                    if isinstance(answer_json['answer_type_probs'], np.ndarray):
                        print('answer_json[\'answer_type_probs\'] is ndarray')

                    if predicted_ability_str == "passage_span_extraction":
                        answer_json["answer_type"] = "passage_span"
                        passage_str = metadata[i]['original_passage']
                        offsets = metadata[i]['passage_token_offsets']

                        passage_tokens = metadata[i]['passage_tokens']

                        topk_predict_span = topk_best_passage_span[i].detach().cpu().numpy().tolist()
                        top_k_start_index = [min(topk_predict_span[i][0] - passage_start, len(offsets) - 1) for i in
                                             range(len(topk_predict_span))]
                        top_k_end_index = [min(topk_predict_span[i][1] - passage_start, len(offsets) - 1) for i in
                                           range(len(topk_predict_span))]

                        predicted_span = tuple(best_passage_span[i].detach().cpu().numpy())
                        start_index = min(predicted_span[0] - passage_start, len(offsets) - 1)
                        end_index = min(predicted_span[1] - passage_start, len(offsets) - 1)

                        start_offset = offsets[start_index][0]
                        end_offset = offsets[end_index][1]
                        predicted_answer = passage_str[start_offset:end_offset]
                        
                        
                        if  predicted_answer.endswith("'s"):
                            predicted_answer = predicted_answer.replace("'s", "")
                        if " / " in predicted_answer:
                            predicted_answer = predicted_answer.replace(" / ", "/")
                        if " - " in predicted_answer:
                            predicted_answer = predicted_answer.replace(" - ", "-")
                        if " , " in predicted_answer:
                            predicted_answer = predicted_answer.replace(" , ", ", ")
                        if " ' " in predicted_answer:
                            predicted_answer = predicted_answer.replace(" ' ", "' ")
                        
                        
                        answer_json["value"] = predicted_answer
                        
                        
                        
                        
                        
                        answer_json["spans"] = [(start_offset, end_offset)]
                        answer_json['passage_span_best_start_log_probs'] = passage_span_start_log_probs[i][
                            start_index + passage_start].cpu().detach().numpy().tolist()
                        answer_json['passage_span_best_end_log_probs'] = passage_span_end_log_probs[i][
                            end_index + passage_start].cpu().detach().numpy().tolist()

                        answer_json['top_k_start_end_index'] = list(zip(top_k_start_index, top_k_end_index))
                        answer_json['top_k_start_end_offset'] = list(zip(
                            [offsets[start_index][0] for start_index in top_k_start_index],
                            [offsets[end_index][1] for end_index in top_k_end_index]))

                        answer_json["top_k_predicted_answer"] = [passage_str[i:j] for i, j in
                                                                 answer_json['top_k_start_end_offset']]

                        answer_json['passage_tokens'] = passage_tokens
                        answer_json['passage_span_top_k_start_probs'] = [
                            passage_span_start_log_probs[i][index + passage_start].cpu().detach().numpy().tolist() for
                            index in
                            top_k_start_index]
                        answer_json['passage_span_top_k_end_probs'] = [
                            passage_span_end_log_probs[i][index + passage_start].cpu().detach().numpy().tolist() for
                            index in
                            top_k_end_index]

                        if isinstance(answer_json['passage_span_top_k_start_probs'], np.ndarray):
                            print('answer_json[\'passage_span_top_k_start_probs\'] is ndarray')
                        if isinstance(answer_json['passage_span_top_k_end_probs'], np.ndarray):
                            print('answer_json[\'passage_span_top_k_end_probs\'] is ndarray')



                    elif predicted_ability_str == "question_span_extraction":
                        answer_json["answer_type"] = "question_span"
                        question_str = metadata[i]['original_question']
                        offsets = metadata[i]['question_token_offsets']
                        predicted_span = tuple(best_question_span[i].detach().cpu().numpy())
                        start_index = min(predicted_span[0] - question_start, len(offsets) - 1)
                        end_index = min(predicted_span[1] - question_start, len(offsets) - 1)

                        start_offset = offsets[start_index][0]
                        end_offset = offsets[end_index][1]
                        predicted_answer = question_str[start_offset:end_offset]
                        
                        if  predicted_answer.endswith("'s"):
                            predicted_answer = predicted_answer.replace("'s", "")
                        if " / " in predicted_answer:
                            predicted_answer = predicted_answer.replace(" / ", "/")
                        if " - " in predicted_answer:
                            predicted_answer = predicted_answer.replace(" - ", "-")
                        if " , " in predicted_answer:
                            predicted_answer = predicted_answer.replace(" , ", ", ")
                        if " ' " in predicted_answer:
                            predicted_answer = predicted_answer.replace(" ' ", "' ")
                        
                        answer_json["value"] = predicted_answer
                        answer_json["spans"] = [(start_offset, end_offset)]
                        answer_json['question_span_best_start_log_probs'] = question_span_start_log_probs[i][
                            start_index + question_start].detach().cpu().numpy().tolist()
                        answer_json['question_span_best_end_log_probs'] = question_span_end_log_probs[i][
                            end_index + question_start].detach().cpu().numpy().tolist()

                        topk_predict_span = topk_best_question_span[i].detach().cpu().numpy().tolist()
                        top_k_start_index = [min(topk_predict_span[i][0] - question_start, len(offsets) - 1) for i in
                                             range(len(topk_predict_span))]
                        top_k_end_index = [min(topk_predict_span[i][1] - question_start, len(offsets) - 1) for i in
                                           range(len(topk_predict_span))]

                        question_tokens = metadata[i]['question_tokens']
                        answer_json['question_tokens'] = question_tokens
                        answer_json['top_k_start_end_index'] = list(zip(top_k_start_index, top_k_end_index))
                        answer_json['top_k_start_end_offset'] = list(zip(
                            [offsets[start_index][0] for start_index in top_k_start_index],
                            [offsets[end_index][1] for end_index in top_k_end_index]))

                        answer_json["top_k_predicted_answer"] = [question_str[i:j] for i, j in
                                                                 answer_json['top_k_start_end_offset']]

                        answer_json['question_span_topk_start_log_probs'] = [
                            question_span_start_log_probs[i][index + question_start].cpu().detach().numpy().tolist() for
                            index in
                            top_k_start_index]
                        answer_json['question_span_topk_end_log_probs'] = [
                            question_span_end_log_probs[i][index + question_start].cpu().detach().numpy().tolist() for
                            index in
                            top_k_end_index]

                        if isinstance(answer_json['question_span_best_start_log_probs'], np.ndarray):
                            print('answer_json[\'question_span_best_start_log_probs\'] is ndarray')
                        if isinstance(answer_json['question_span_best_end_log_probs'], np.ndarray):
                            print('answer_json[\'question_span_best_end_log_probs\'] is ndarray')

                    elif predicted_ability_str == "addition_subtraction":  # plus_minus combination answer
                        answer_json["answer_type"] = "arithmetic"
                        original_numbers = metadata[i]['original_numbers']
                        sign_remap = {0: 0, 1: 1, 2: -1}
                        predicted_signs = [sign_remap[it] for it in best_signs_for_numbers[i].detach().cpu().numpy()]
                        result = sum([sign * number for sign, number in zip(predicted_signs, original_numbers)])
                        predicted_answer = convert_number_to_str(result)

                        offsets = metadata[i]['passage_token_offsets']
                        number_indices = metadata[i]['number_indices']
                        number_positions = [offsets[index - 1] for index in number_indices]
                        answer_json['numbers'] = []
                        for offset, value, sign in zip(number_positions, original_numbers, predicted_signs):
                            answer_json['numbers'].append({'span': offset, 'value': value, 'sign': sign})
                        if number_indices[-1] == -1:
                            # There is a dummy 0 number at position -1 added in some cases; we are
                            # removing that here.
                            answer_json["numbers"].pop()
                        answer_json["value"] = result
                        answer_json['best_combination_log_prob'] = best_combination_log_prob[
                            i].detach().cpu().numpy().tolist()
                        if isinstance(answer_json['best_combination_log_prob'], np.ndarray):
                            print('answer_json[\'best_combination_log_prob\'] is ndarray')

                    elif predicted_ability_str == "counting":
                        answer_json["answer_type"] = "count"
                        predicted_count = best_count_number[i].detach().cpu().numpy().tolist()
                        predicted_answer = str(predicted_count)
                        answer_json["count"] = predicted_count
                        answer_json['probs'] = count_number_log_probs[i].detach().cpu().numpy().tolist()
                        if isinstance(answer_json['probs'], np.ndarray):
                            print('answer_json[\'probs\'] is ndarray')


                    elif predicted_ability_str == "multiple_spans":
                        passage_str = metadata[i]["original_passage"]
                        question_str = metadata[i]['original_question']
                        qp_tokens = metadata[i]["question_passage_tokens"]
                        answer_json["answer_type"] = "multiple_spans"
                        answer_json['multispan_log_probs'] = multispan_log_probs[i].detach().cpu().numpy().tolist()
                        if isinstance(answer_json['multispan_log_probs'], np.ndarray):
                            print('answer_json[\'multispan_log_probs\'] is ndarray')

                        if self.multispan_head_name == "flexible_loss":
                            answer_json["value"], answer_json["spans"], invalid_spans = \
                                multispan_prediction(multispan_log_probs[i], multispan_logits[i], qp_tokens,
                                                     passage_str,
                                                     question_str,
                                                     multispan_mask[i].bool(), bio_wordpiece_mask[i].bool(),
                                                     self.multispan_use_prediction_beam_search and not self.training)
                        else:
                            answer_json["value"], answer_json["spans"], invalid_spans = \
                                multispan_prediction(multispan_log_probs[i], multispan_logits[i], qp_tokens,
                                                     passage_str,
                                                     question_str,
                                                     multispan_mask[i].bool())
                        if self._unique_on_multispan:
                            answer_json["value"] = list(OrderedDict.fromkeys(answer_json["value"]))

                            if self._dont_add_substrings_to_ms:
                                answer_json["value"] = remove_substring_from_prediction(answer_json["value"])

                        if len(answer_json["value"]) == 0:
                            best_answer_ability[i] = top_two_answer_abilities[1][i][1]
                            continue
                        predicted_answer = answer_json["value"]
                        
                        for span_id, item in enumerate(predicted_answer):
                            if item.endswith("'s"):
                                predicted_answer[span_id]= item.replace("'s","")
                            if " - " in item:
                                predicted_answer[span_id] = item.replace(" - ", "-")
                        
                    else:
                        raise ValueError(f"Unsupported answer ability: {predicted_ability_str}")
                    answer_json["predicted_answer"] = predicted_answer
                    
                    ##+++++++++++++++++++++
                    if self.add_op_layer:
                        answer_json["op_weight"] = op_weight[i].cpu().detach().numpy().tolist()
                    else:
                        answer_json["op_weight"] = []
                    ##+++++++++++++++++++++
                        
                    output_dict["question_id"].append(metadata[i]["question_id"])
                    if not self.training:
                        output_dict["answer"].append(answer_json)
                    answer_annotations = metadata[i].get('answer_annotations', [])
                    if answer_annotations:
                        self._drop_metrics(predicted_answer, answer_annotations)

                    i += 1
            return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}
