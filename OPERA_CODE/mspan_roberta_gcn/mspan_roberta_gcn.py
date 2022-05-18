import torch
import torch.nn as nn
from typing import List, Dict, Any
from tools import allennlp as util
import torch.nn.functional as F
from mspan_roberta_gcn.util import FFNLayer, GCN, ResidualGRU, SignNumFFNLayer, DigitalGCN, HeterogeneousGCN
from tools.utils import DropEmAndF1
from torch.nn.parameter import Parameter
from tools.allennlp import mask_mse_loss


def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
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
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    # best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
    _, best_spans = valid_span_log_probs.view(batch_size, -1).topk(20, dim=-1)

    # (batch_size, 20)
    span_start_indices = best_spans // passage_length
    span_end_indices = best_spans % passage_length

    # (batch_size, 20, 2)
    return torch.stack([span_start_indices, span_end_indices], dim=-1)


def best_answers_extraction(best_spans, span_num, original_str, offsets, offset_start):
    predicted_span = tuple(best_spans.detach().cpu().numpy())
    predict_answers = []
    predict_offsets = []
    for i in range(20):
        start_offset = offsets[predicted_span[i][0] - offset_start][0] if predicted_span[i][0] - offset_start < len(
            offsets) else offsets[-1][0]
        end_offset = offsets[predicted_span[i][1] - offset_start][1] if predicted_span[i][1] - offset_start < len(
            offsets) else offsets[-1][1]
        predict_answer = original_str[start_offset:end_offset]
        predict_offset = (start_offset, end_offset)
        if len(predict_answers) == 0 or all(
                [len(set(item.split()) & set(predict_answer.split())) == 0 for item in predict_answers]):
            predict_answers.append(predict_answer)
            predict_offsets.append(predict_offset)
        if len(predict_answers) >= span_num:
            break
    return predict_answers, predict_offsets


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
    def __init__(self,
                 bert,
                 hidden_size: int,
                 dropout_prob: float = 0.1,
                 answering_abilities: List[str] = None,
                 use_gcn: bool = False,
                 gcn_steps: int = 1, add_aux_nums: bool = False, aux_nums_embeddings=None,
                 add_sign_weight_decomp: bool = False, add_relation_reasoning_module: bool = False,
                 bert_name: str = None,

                 ) -> None:
        super(NumericallyAugmentedBertNet, self).__init__()
        self.add_aux_nums = add_aux_nums
        self.add_sign_weight_decomp = add_sign_weight_decomp
        self.use_gcn = use_gcn
        self.add_relation_reasoning_module = add_relation_reasoning_module
        self.bert = bert
        self.bert_name = bert_name

        modeling_out_dim = hidden_size
        self._drop_metrics = DropEmAndF1()
        if answering_abilities is None:
            self.answering_abilities = ["passage_span_extraction", "question_span_extraction",
                                        "addition_subtraction", "counting"]
        else:
            self.answering_abilities = answering_abilities

        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = FFNLayer(3 * hidden_size, hidden_size, len(self.answering_abilities),
                                                      dropout_prob)

        if "passage_span_extraction" in self.answering_abilities or "question_span_extraction" in self.answering_abilities:
            self._passage_span_extraction_index = self.answering_abilities.index("passage_span_extraction")
            self._question_span_extraction_index = self.answering_abilities.index("question_span_extraction")
            self._span_start_predictor = nn.Linear(4 * hidden_size, 1, bias=False)
            self._span_end_predictor = nn.Linear(4 * hidden_size, 1, bias=False)

        if "addition_subtraction" in self.answering_abilities:
            self._addition_subtraction_index = self.answering_abilities.index("addition_subtraction")
            if self.add_sign_weight_decomp:
                self._number_sign_predictor = SignNumFFNLayer(5 * hidden_size, hidden_size, 3, dropout_prob)
            else:
                self._number_sign_predictor = FFNLayer(5 * hidden_size, hidden_size, 3, dropout_prob)

        if "counting" in self.answering_abilities:
            self._counting_index = self.answering_abilities.index("counting")
            self._count_number_predictor = FFNLayer(5 * hidden_size, hidden_size, 10, dropout_prob)

        self._dropout = torch.nn.Dropout(p=dropout_prob)

        node_dim = modeling_out_dim
        self._gcn_input_proj = nn.Linear(node_dim * 2, node_dim)
        self._proj_ln = nn.LayerNorm(node_dim)
        self._proj_ln0 = nn.LayerNorm(node_dim)
        self._proj_ln1 = nn.LayerNorm(node_dim)
        self._proj_ln3 = nn.LayerNorm(node_dim)
        self._gcn_enc = ResidualGRU(hidden_size, dropout_prob, 2)

        if self.use_gcn:
            self._gcn = GCN(node_dim=node_dim, iteration_steps=gcn_steps)
            self._iteration_steps = gcn_steps

        # add bert proj
        self._proj_sequence_h = nn.Linear(hidden_size, 1, bias=False)
        self._proj_number = nn.Linear(hidden_size * 2, 1, bias=False)

        # add aux encoder proj
        self._proj_aux_number = nn.Linear(hidden_size, 2 * hidden_size, bias=False)

        self._proj_sequence_g0 = FFNLayer(hidden_size, hidden_size, 1, dropout_prob)
        self._proj_sequence_g1 = FFNLayer(hidden_size, hidden_size, 1, dropout_prob)
        self._proj_sequence_g2 = FFNLayer(hidden_size, hidden_size, 1, dropout_prob)

        # span num extraction
        self._proj_span_num = FFNLayer(3 * hidden_size, hidden_size, 9, dropout_prob)

        # if self.add_aux_nums:
        #     self.aux_nums_embeddings = aux_nums_embeddings
        #     self._count_number_predictor.fc2.weight = Parameter(self.aux_nums_embeddings, requires_grad=True)

        if self.add_aux_nums:
            # node_dim = modeling_out_dim
            # self._aux_num_gcn = DigitalGCN(node_dim=node_dim, iteration_steps=gcn_steps)
            self._iteration_steps = gcn_steps
            # self._aux_num_proj_number = nn.Linear(hidden_size, 1, bias=False)
            # self._count_number_predictor = FFNLayer(6 * hidden_size, hidden_size, 10, dropout_prob)
            self._aux_doc_nums_gcn = GCN(node_dim=node_dim, iteration_steps=gcn_steps)
            self._aux_doc_nums_proj_ln = nn.LayerNorm(node_dim)
            self._aux_doc_nums_proj_ln0 = nn.LayerNorm(node_dim)
            self._aux_doc_nums_proj_ln1 = nn.LayerNorm(node_dim)
            self._aux_doc_nums_proj_ln3 = nn.LayerNorm(node_dim)
            self._aux_doc_nums_gcn_enc = ResidualGRU(hidden_size, dropout_prob, 2)
            self._aux_num_proj_number = nn.Linear(hidden_size, 1, bias=False)
            self._count_number_predictor = FFNLayer(6 * hidden_size, hidden_size, 10, dropout_prob)
            self.mask_mse_loss = mask_mse_loss

        if self.add_relation_reasoning_module:
            self._proj_sequence_relation_input = nn.Linear(hidden_size, 1, bias=False)
            # node_dim = modeling_out_dim
            self._relation_gcn_input_proj = nn.Linear(node_dim * 2, node_dim)
            self._relation_gcn = HeterogeneousGCN(node_dim=node_dim, iteration_steps=gcn_steps)

    def forward(self,  # type: ignore
                input_ids: torch.LongTensor,
                input_mask: torch.LongTensor,
                input_segments: torch.LongTensor,
                passage_mask: torch.LongTensor,
                question_mask: torch.LongTensor,
                number_indices: torch.LongTensor,
                aux_number_order: torch.LongTensor,
                aux_nums_ids: torch.LongTensor,
                passage_number_order: torch.LongTensor,
                question_number_order: torch.LongTensor,
                question_number_indices: torch.LongTensor,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                answer_as_add_sub_expressions: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                span_num: torch.LongTensor = None,

                sentence_indices_in_passage: torch.LongTensor = None,
                entity_indices_in_passage: torch.LongTensor = None,
                value_indices_in_passage: torch.LongTensor = None,
                sentences_entities_relation: torch.LongTensor = None,
                sentences_values_relation: torch.LongTensor = None,
                entities_values_relation: torch.LongTensor = None,
                entities_entities_relation: torch.LongTensor = None,
                # same_entity_mention_relation: torch.LongTensor = None,
                sentence_mask: torch.LongTensor = None,
                entity_mask: torch.LongTensor = None,
                value_mask: torch.LongTensor = None,

                question_indices: torch.LongTensor = None,
                entity_indices_in_question: torch.LongTensor = None,
                value_indices_in_question: torch.LongTensor = None,
                question_entities_relation: torch.LongTensor = None,
                question_values_relation: torch.LongTensor = None,
                question_entities_values_relation: torch.LongTensor = None,
                question_entities_entities_relation: torch.LongTensor = None,
                # same_entity_mention_with_q_relation: torch.LongTensor = None,
                question_entity_mask: torch.LongTensor = None,
                question_value_mask: torch.LongTensor = None,

                metadata: List[Dict[str, Any]] = None

                ) -> Dict[str, torch.Tensor]:

        outputs = self.bert(input_ids, attention_mask=input_mask, token_type_ids=input_segments)
        sequence_output = outputs[0]
        sequence_output_list = [item for item in outputs[-1][-4:]]
        batch_size = input_ids.size(0)
        sequence_alg = self._gcn_input_proj(torch.cat([sequence_output_list[2], sequence_output_list[3]], dim=2))
        if (
                "passage_span_extraction" in self.answering_abilities or "question_span" in self.answering_abilities) and self.use_gcn:
            # M2, M3
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
                                                              -1) > number_order.unsqueeze(-1).expand(batch_size,
                                                                                                      -1,
                                                                                                      number_order.size(
                                                                                                          -1))
            new_graph_mask = new_graph_mask.long()
            all_number_mask = torch.cat((number_mask, question_number_mask), dim=-1)
            new_graph_mask = all_number_mask.unsqueeze(1) * all_number_mask.unsqueeze(-1) * new_graph_mask

            # iteration
            # encoded_numbers_aux_numbers = torch.cat([encoded_numbers, encoded_aux_number], dim=1)
            # numbers_aux_numbers_mask = torch.cat([aux_number_mask, number_mask], dim=1)

            d_node, q_node, d_node_weight, _ = self._gcn(d_node=encoded_numbers,
                                                         q_node=question_encoded_number,
                                                         d_node_mask=number_mask,
                                                         q_node_mask=question_number_mask,
                                                         graph=new_graph_mask)

            gcn_info_vec = torch.zeros((batch_size, sequence_alg.size(1) + 1, sequence_output_list[-1].size(-1)),
                                       dtype=torch.float, device=d_node.device)

            number_mask = (real_number_indices > -1).long()
            clamped_number_indices = util.replace_masked_values(real_number_indices, number_mask,
                                                                gcn_info_vec.size(1) - 1)
            gcn_info_vec.scatter_(1, clamped_number_indices.unsqueeze(-1).expand(-1, -1, d_node.size(-1)), d_node)
            gcn_info_vec = gcn_info_vec[:, :-1, :]

            sequence_output_list[2] = sequence_output_list[2] + gcn_info_vec
            sequence_output_list[0] = sequence_output_list[0] + gcn_info_vec
            sequence_output_list[1] = sequence_output_list[1] + gcn_info_vec
            sequence_output_list[3] = sequence_output_list[3] + gcn_info_vec

        if self.add_aux_nums:
            encoded_aux_num_node = self.bert.embeddings.word_embeddings(aux_nums_ids[0])
            encoded_aux_num_node = encoded_aux_num_node.unsqueeze(0).expand(batch_size, -1, -1)
            encoded_sequence = self.bert.embeddings.word_embeddings(input_ids)
            if 'albert' in self.bert_name:
                encoded_aux_num_node = self.bert.encoder.embedding_hidden_mapping_in(encoded_aux_num_node)
                encoded_sequence = self.bert.encoder.embedding_hidden_mapping_in(encoded_sequence)

            encoded_passage_for_numbers = encoded_sequence
            encoded_question_for_numbers = encoded_sequence

            real_number_indices = number_indices - 1
            number_mask = (real_number_indices > -1).long()  # ??
            clamped_number_indices = util.replace_masked_values(real_number_indices, number_mask, 0)
            encoded_numbers = torch.gather(encoded_passage_for_numbers, 1,
                                           clamped_number_indices.unsqueeze(-1).expand(-1, -1,
                                                                                       encoded_passage_for_numbers.size(
                                                                                           -1)))
            encoded_aux_doc_nums = torch.cat([encoded_aux_num_node, encoded_numbers], dim=1)
            aux_nums_mask = torch.LongTensor(aux_number_order.size(0), aux_number_order.size(1)).fill_(1).cuda()
            aux_doc_nums_mask = torch.cat([aux_nums_mask, number_mask], dim=-1)

            # question number extraction
            question_number_mask = (question_number_indices > -1).long()
            clamped_question_number_indices = util.replace_masked_values(question_number_indices,
                                                                         question_number_mask,
                                                                         0)
            question_encoded_number = torch.gather(encoded_question_for_numbers, 1,
                                                   clamped_question_number_indices.unsqueeze(-1).expand(-1, -1,
                                                                                                        encoded_question_for_numbers.size(
                                                                                                            -1)))

            aux_number_number_order = torch.cat((aux_number_order, passage_number_order, question_number_order), -1)
            new_aux_num_graph_mask = aux_number_number_order.unsqueeze(1).expand(batch_size,
                                                                                 aux_number_number_order.size(-1),
                                                                                 -1) > aux_number_number_order.unsqueeze(
                -1).expand(batch_size,
                           -1,
                           aux_number_number_order.size(
                               -1))
            new_aux_num_graph_mask = new_aux_num_graph_mask.long()

            all_number_mask = torch.cat((aux_nums_mask, number_mask, question_number_mask), dim=-1)
            new_aux_num_graph_mask = all_number_mask.unsqueeze(1) * all_number_mask.unsqueeze(
                -1) * new_aux_num_graph_mask

            # iteration
            aux_doc_node, q_node_v2, aux_doc_node_weight, _ = self._aux_doc_nums_gcn(d_node=encoded_aux_doc_nums,
                                                                                     q_node=question_encoded_number,
                                                                                     d_node_mask=aux_doc_nums_mask,
                                                                                     q_node_mask=question_number_mask,
                                                                                     graph=new_aux_num_graph_mask)

            d_node_v2 = aux_doc_node[:, aux_nums_mask.size(1):, :]
            aux_num_node = aux_doc_node[:, :aux_nums_mask.size(1), :]

            embedding_gcn_info_vec = torch.zeros(
                (batch_size, sequence_alg.size(1) + 1, sequence_output_list[-1].size(-1)),
                dtype=torch.float, device=d_node_v2.device)

            number_mask = (real_number_indices > -1).long()
            clamped_number_indices = util.replace_masked_values(real_number_indices, number_mask,
                                                                embedding_gcn_info_vec.size(1) - 1)
            embedding_gcn_info_vec.scatter_(1,
                                            clamped_number_indices.unsqueeze(-1).expand(-1, -1, d_node_v2.size(-1)),
                                            d_node_v2)
            embedding_gcn_info_vec = embedding_gcn_info_vec[:, :-1, :]

            sequence_output_list[2] = sequence_output_list[2] + embedding_gcn_info_vec
            sequence_output_list[0] = sequence_output_list[0] + embedding_gcn_info_vec
            sequence_output_list[1] = sequence_output_list[1] + embedding_gcn_info_vec
            sequence_output_list[3] = sequence_output_list[3] + embedding_gcn_info_vec

            aux_num_weight = self._aux_num_proj_number(aux_num_node).squeeze(-1)
            aux_num_weight = util.masked_softmax(aux_num_weight, aux_nums_mask)
            aux_num_vector = util.weighted_sum(aux_num_node, aux_num_weight)

        if self.add_relation_reasoning_module:
            encoded_sentence_nodes = torch.matmul(sentence_indices_in_passage.float(),
                                                  sequence_alg)  # (bsz, n_s, l) (bsz, l , h)
            sentence_token_nums = sentence_indices_in_passage.sum(-1)  # (bsz, n_s)
            sentence_token_nums = util.replace_masked_values(sentence_token_nums, sentence_mask, 1)
            encoded_sentence_nodes = torch.div(encoded_sentence_nodes,
                                               sentence_token_nums.unsqueeze(-1).expand(-1, -1,
                                                                                        encoded_sentence_nodes.size(
                                                                                            -1)))  # (bsz, n_s, H) (bsz, n_s)

            encoded_entity_nodes = torch.matmul(entity_indices_in_passage.float(),
                                                sequence_alg)  # (bsz, n_e, l) (bsz, l , h)
            entity_token_nums = entity_indices_in_passage.sum(-1)
            entity_token_nums = util.replace_masked_values(entity_token_nums, entity_mask, 1)
            encoded_entity_nodes = torch.div(encoded_entity_nodes,
                                             entity_token_nums.unsqueeze(-1).expand(-1, -1,
                                                                                    encoded_entity_nodes.size(
                                                                                        -1)))  # (bsz, n_s, H) (bsz, n_s)
            encoded_value_nodes = torch.gather(sequence_alg, dim=1,
                                               index=value_indices_in_passage.unsqueeze(-1).expand(-1, -1,
                                                                                                   sequence_alg.size(
                                                                                                       -1)))

            encoded_question_node = torch.matmul(question_indices.float(), sequence_alg)

            question_token_nums = question_indices.sum(-1)

            encoded_question_node = torch.div(encoded_question_node,
                                              question_token_nums.unsqueeze(-1).expand(-1, -1,
                                                                                       question_token_nums.size(
                                                                                           -1)))  # (bsz, n_s, H) (bsz, n_s)

            encoded_question_entity_nodes = torch.matmul(entity_indices_in_question.float(),
                                                         sequence_alg)  # (bsz, n_e, l) (bsz, l , h)
            question_entity_token_nums = entity_indices_in_question.sum(-1)
            question_entity_token_nums = util.replace_masked_values(question_entity_token_nums,
                                                                    question_entity_mask, 1)
            encoded_question_entity_nodes = torch.div(encoded_question_entity_nodes,
                                                      question_entity_token_nums.unsqueeze(-1).expand(-1, -1,
                                                                                                      encoded_question_entity_nodes.size(
                                                                                                          -1)))  # (bsz, n_s, H) (bsz, n_s)

            encoded_questiion_value_nodes = torch.gather(sequence_alg, dim=1,
                                                         index=value_indices_in_question.unsqueeze(-1).expand(-1,
                                                                                                              -1,
                                                                                                              sequence_alg.size(
                                                                                                                  -1)))

            sentence_nodes, entity_nodes, value_nodes, question_node, question_entity_nodes, question_value_nodes = self._relation_gcn(
                sentence_node=encoded_sentence_nodes,
                entity_node=encoded_entity_nodes,
                value_node=encoded_value_nodes,  #
                sentences_values_relation=sentences_values_relation,
                entities_entities_relation=entities_entities_relation,
                entities_values_relation=entities_values_relation,
                sentences_entities_relation=sentences_entities_relation,
                # same_entity_mention_relation=same_entity_mention_relation,
                sentences_node_mask=sentence_mask,
                entities_node_mask=entity_mask,
                values_node_mask=value_mask,

                question_node=encoded_question_node,
                question_entity_node=encoded_question_entity_nodes,
                question_value_node=encoded_questiion_value_nodes,
                question_entities_relation=question_entities_relation,
                question_values_relation=question_values_relation,
                question_entities_entities_relation=question_entities_entities_relation,
                question_entities_values_relation=question_entities_values_relation,
                # same_entity_mention_with_q_relation=same_entity_mention_with_q_relation,
                question_entity_mask=question_entity_mask,
                question_value_mask=question_value_mask

            )

            entity_gcn_info_vec = torch.matmul(torch.transpose(entity_indices_in_passage, dim0=1, dim1=-1).float(),
                                               entity_nodes)

            value_gcn_info_vec = torch.zeros(
                (batch_size, sequence_alg.size(1) + 1, sequence_output_list[-1].size(-1)),
                dtype=torch.float, device=value_nodes.device)

            value_indices_in_passage = util.replace_masked_values(value_indices_in_passage, value_mask,
                                                                  value_gcn_info_vec.size(1) - 1)

            value_gcn_info_vec.scatter_(1,
                                        value_indices_in_passage.unsqueeze(-1).expand(-1, -1, value_nodes.size(-1)),
                                        value_nodes)

            value_gcn_info_vec = value_gcn_info_vec[:, :-1, :]

            question_entity_gcn_info_vec = torch.matmul(
                torch.transpose(entity_indices_in_question, dim0=1, dim1=-1).float(),
                question_entity_nodes)

            question_value_gcn_info_vec = torch.zeros(
                (batch_size, sequence_alg.size(1) + 1, sequence_output_list[-1].size(-1)),
                dtype=torch.float, device=question_value_nodes.device)

            value_indices_in_question = util.replace_masked_values(value_indices_in_question, question_value_mask,
                                                                   question_value_gcn_info_vec.size(1) - 1)

            question_value_gcn_info_vec.scatter_(1,
                                                 value_indices_in_question.unsqueeze(-1).expand(-1, -1,
                                                                                                question_value_nodes.size(
                                                                                                    -1)),
                                                 question_value_nodes)

            question_value_gcn_info_vec = question_value_gcn_info_vec[:, :-1, :]

            sequence_output_list[2] = sequence_output_list[
                                          2] + entity_gcn_info_vec + value_gcn_info_vec + question_entity_gcn_info_vec + question_value_gcn_info_vec
            sequence_output_list[0] = sequence_output_list[
                                          0] + entity_gcn_info_vec + value_gcn_info_vec + question_entity_gcn_info_vec + question_value_gcn_info_vec
            sequence_output_list[1] = sequence_output_list[
                                          1] + entity_gcn_info_vec + value_gcn_info_vec + question_entity_gcn_info_vec + question_value_gcn_info_vec
            sequence_output_list[3] = sequence_output_list[
                                          3] + entity_gcn_info_vec + value_gcn_info_vec + question_entity_gcn_info_vec + question_value_gcn_info_vec

        sequence_output_list[2] = self._gcn_enc(self._proj_ln(sequence_output_list[2]))
        sequence_output_list[0] = self._gcn_enc(self._proj_ln0(sequence_output_list[0]))
        sequence_output_list[1] = self._gcn_enc(self._proj_ln1(sequence_output_list[1]))
        sequence_output_list[3] = self._gcn_enc(self._proj_ln3(sequence_output_list[3]))

        # passage hidden and question hidden
        sequence_h2_weight = self._proj_sequence_h(sequence_output_list[2]).squeeze(-1)
        passage_h2_weight = util.masked_softmax(sequence_h2_weight, passage_mask)
        passage_h2 = util.weighted_sum(sequence_output_list[2], passage_h2_weight)
        question_h2_weight = util.masked_softmax(sequence_h2_weight, question_mask)
        question_h2 = util.weighted_sum(sequence_output_list[2], question_h2_weight)

        # passage g0, g1, g2
        question_g0_weight = self._proj_sequence_g0(sequence_output_list[0]).squeeze(-1)
        question_g0_weight = util.masked_softmax(question_g0_weight, question_mask)
        question_g0 = util.weighted_sum(sequence_output_list[0], question_g0_weight)

        question_g1_weight = self._proj_sequence_g1(sequence_output_list[1]).squeeze(-1)
        question_g1_weight = util.masked_softmax(question_g1_weight, question_mask)
        question_g1 = util.weighted_sum(sequence_output_list[1], question_g1_weight)

        question_g2_weight = self._proj_sequence_g2(sequence_output_list[2]).squeeze(-1)
        question_g2_weight = util.masked_softmax(question_g2_weight, question_mask)
        question_g2 = util.weighted_sum(sequence_output_list[2], question_g2_weight)

        if len(self.answering_abilities) > 1:
            # Shape: (batch_size, number_of_abilities)
            answer_ability_logits = self._answer_ability_predictor(
                torch.cat([passage_h2, question_h2, sequence_output[:, 0]], 1))
            answer_ability_log_probs = F.log_softmax(answer_ability_logits, -1)
            best_answer_ability = torch.argmax(answer_ability_log_probs, 1)

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

        if "counting" in self.answering_abilities:
            # Shape: (batch_size, 10)
            if self.add_aux_nums:
                count_number_logits = self._count_number_predictor(
                    torch.cat([aux_num_vector, number_vector, passage_h2, question_h2, sequence_output[:, 0]], dim=1))
            else:
                count_number_logits = self._count_number_predictor(
                    torch.cat([number_vector, passage_h2, question_h2, sequence_output[:, 0]], dim=1))
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
                                                 sequence_output_list[0],
                                                 sequence_output_list[2] * question_g2.unsqueeze(1),
                                                 sequence_output_list[0] * question_g0.unsqueeze(1)],
                                                dim=2)
            sequence_span_start_logits = self._span_start_predictor(sequence_for_span_start).squeeze(-1)
            # Shape: (batch_size, passage_length, modeling_dim * 2)
            sequence_for_span_end = torch.cat([sequence_output_list[2],
                                               sequence_output_list[1],
                                               sequence_output_list[2] * question_g2.unsqueeze(1),
                                               sequence_output_list[1] * question_g1.unsqueeze(1)],
                                              dim=2)
            # Shape: (batch_size, passage_length)
            sequence_span_end_logits = self._span_end_predictor(sequence_for_span_end).squeeze(-1)
            # Shape: (batch_size, passage_length)

            # span number prediction
            span_num_logits = self._proj_span_num(torch.cat([passage_h2, question_h2, sequence_output[:, 0]], dim=1))
            span_num_log_probs = torch.nn.functional.log_softmax(span_num_logits, -1)

            best_span_number = torch.argmax(span_num_log_probs, dim=-1)

            if "passage_span_extraction" in self.answering_abilities:
                passage_span_start_log_probs = util.masked_log_softmax(sequence_span_start_logits, passage_mask)
                passage_span_end_log_probs = util.masked_log_softmax(sequence_span_end_logits, passage_mask)

                # Info about the best passage span prediction
                passage_span_start_logits = util.replace_masked_values(sequence_span_start_logits, passage_mask,
                                                                       float('-inf'))
                passage_span_end_logits = util.replace_masked_values(sequence_span_end_logits, passage_mask,
                                                                     float('-inf'))
                # Shage: (batch_size, topk, 2)
                best_passage_span = get_best_span(passage_span_start_logits, passage_span_end_logits)

            if "question_span_extraction" in self.answering_abilities:
                question_span_start_log_probs = util.masked_log_softmax(sequence_span_start_logits, question_mask)
                question_span_end_log_probs = util.masked_log_softmax(sequence_span_end_logits, question_mask)

                # Info about the best question span prediction
                question_span_start_logits = util.replace_masked_values(sequence_span_start_logits, question_mask,
                                                                        float('-inf'))
                question_span_end_logits = util.replace_masked_values(sequence_span_end_logits, question_mask,
                                                                      float('-inf'))
                # Shape: (batch_size, topk, 2)
                best_question_span = get_best_span(question_span_start_logits, question_span_end_logits)

        if "addition_subtraction" in self.answering_abilities:
            alg_encoded_numbers = torch.cat(
                [encoded_numbers,
                 question_h2.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                 passage_h2.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                 sequence_output[:, 0].unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)
                 ], 2)

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
                                                                                  gold_passage_span_mask, -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_passage_span = util.logsumexp(log_likelihood_for_passage_spans)

                    # span log probabilities
                    log_likelihood_for_passage_span_nums = torch.gather(span_num_log_probs, 1, span_num)
                    log_likelihood_for_passage_span_nums = util.replace_masked_values(
                        log_likelihood_for_passage_span_nums,
                        gold_passage_span_mask[:, :1], -1e7)
                    log_marginal_likelihood_for_passage_span_nums = util.logsumexp(log_likelihood_for_passage_span_nums)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span +
                                                        log_marginal_likelihood_for_passage_span_nums)

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
                                                                                   gold_question_span_mask, -1e7)
                    # Shape: (batch_size, )
                    # pylint: disable=invalid-name
                    log_marginal_likelihood_for_question_span = util.logsumexp(log_likelihood_for_question_spans)

                    # question multi span prediction
                    log_likelihood_for_question_span_nums = torch.gather(span_num_log_probs, 1, span_num)
                    log_marginal_likelihood_for_question_span_nums = util.logsumexp(
                        log_likelihood_for_question_span_nums)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span +
                                                        log_marginal_likelihood_for_question_span_nums)

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
                                                                             gold_add_sub_mask, -1e7)
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
                                                                           -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_count = util.logsumexp(log_likelihood_for_counts)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_count)

                else:
                    raise ValueError(f"Unsupported answering ability: {answering_ability}")
            # print(log_marginal_likelihood_list)
            if len(self.answering_abilities) > 1:
                # Add the ability probabilities if there are more than one abilities
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]
            output_dict["loss"] = - marginal_log_likelihood.mean()

            if self.add_aux_nums:
                ## number in question\
                qq_node_left = self._aux_doc_nums_gcn._qq_node_fc_left
                qq_node_right = self._aux_doc_nums_gcn._qq_node_fc_right
                dd_node_left = self._aux_doc_nums_gcn._dd_node_fc_left
                dd_node_right = self._aux_doc_nums_gcn._dd_node_fc_right
                dq_node_left = self._aux_doc_nums_gcn._qd_node_fc_left
                dq_node_right = self._aux_doc_nums_gcn._dq_node_fc_right
                qd_node_left = self._aux_doc_nums_gcn._qd_node_fc_left
                qd_node_right = self._aux_doc_nums_gcn._qd_node_fc_right

                qq_left_constraint = torch.matmul(qq_node_left(q_node_v2),
                                                  torch.transpose(q_node_v2, 1, -1).contiguous())
                qq_left_constraint_logit = torch.sigmoid(qq_left_constraint)

                qq_diagmat = torch.diagflat(torch.ones(q_node_v2.size(1), dtype=torch.long, device=q_node_v2.device))
                qq_diagmat = qq_diagmat.unsqueeze(0).expand(q_node_v2.size(0), -1, -1)
                qq_graph = question_number_mask.unsqueeze(1) * question_number_mask.unsqueeze(-1) * (1 - qq_diagmat)
                qq_left_ground_truth = qq_graph * new_aux_num_graph_mask[:, -q_node_v2.size(1):,
                                                  -q_node_v2.size(1):].cuda()

                qq_left_constraint_loss = self.mask_mse_loss(qq_left_constraint_logit, qq_left_ground_truth.float(),
                                                             qq_graph)

                qq_right_constraint = torch.matmul(qq_node_right(q_node_v2),
                                                   torch.transpose(q_node_v2, 1, -1).contiguous())
                qq_right_constraint_logit = torch.sigmoid(qq_right_constraint)
                qq_right_ground_truth = qq_graph * (
                        1 - new_aux_num_graph_mask[:, -q_node_v2.size(1):, -q_node_v2.size(1):]).cuda()
                qq_right_constraint_loss = self.mask_mse_loss(qq_right_constraint_logit, qq_right_ground_truth.float(),
                                                              qq_graph)
                output_dict["loss"] += qq_left_constraint_loss + qq_right_constraint_loss

                dd_left_constraint = torch.matmul(dd_node_left(aux_doc_node),
                                                  torch.transpose(aux_doc_node, 1, -1).contiguous())
                dd_left_constraint_logit = torch.sigmoid(dd_left_constraint)
                dd_diagmat = torch.diagflat(
                    torch.ones(aux_doc_node.size(1), dtype=torch.long, device=aux_doc_node.device))
                dd_diagmat = dd_diagmat.unsqueeze(0).expand(aux_doc_node.size(0), -1, -1)
                number_mask = (real_number_indices > -1).long()
                aux_nums_doc_nums_mask = torch.cat([aux_nums_mask, number_mask], dim=-1)
                dd_graph = aux_nums_doc_nums_mask.unsqueeze(1) * aux_nums_doc_nums_mask.unsqueeze(-1) * (1 - dd_diagmat)
                dd_left_ground_truth = dd_graph * new_aux_num_graph_mask[:, :aux_doc_node.size(1),
                                                  :aux_doc_node.size(1)].cuda()
                dd_left_ground_loss = self.mask_mse_loss(dd_left_constraint_logit, dd_left_ground_truth.float(),
                                                         dd_graph)

                dd_right_constraint = torch.matmul(dd_node_right(aux_doc_node),
                                                   torch.transpose(aux_doc_node, 1, -1).contiguous())
                dd_right_constraint_logit = torch.sigmoid(dd_right_constraint)
                dd_right_ground_truth = dd_graph * (
                        1 - new_aux_num_graph_mask[:, :aux_doc_node.size(1), :aux_doc_node.size(1)]).cuda()
                dd_right_ground_loss = self.mask_mse_loss(dd_right_constraint_logit, dd_right_ground_truth.float(),
                                                          dd_graph)
                output_dict["loss"] += dd_left_ground_loss + dd_right_ground_loss

                ## number in question and passage
                qd_left_constraint = torch.matmul(qd_node_left(q_node_v2),
                                                  torch.transpose(aux_doc_node, 1, -1).contiguous())
                qd_left_constraint_logit = torch.sigmoid(qd_left_constraint)
                qd_graph = question_number_mask.unsqueeze(-1) * aux_nums_doc_nums_mask.unsqueeze(1)
                qd_left_ground_truth = qd_graph * new_aux_num_graph_mask[:, -q_node_v2.size(1):,
                                                  :aux_doc_node.size(1)].cuda()
                qd_left_ground_loss = self.mask_mse_loss(qd_left_constraint_logit, qd_left_ground_truth.float(),
                                                         qd_graph)

                qd_right_constraint = torch.matmul(qd_node_right(q_node_v2),
                                                   torch.transpose(aux_doc_node, 1, -1).contiguous())
                qd_right_constraint_logit = torch.sigmoid(qd_right_constraint)
                qd_graph = question_number_mask.unsqueeze(-1) * aux_nums_doc_nums_mask.unsqueeze(1)
                qd_right_ground_truth = qd_graph * new_aux_num_graph_mask[:, -q_node_v2.size(1):,
                                                   :aux_doc_node.size(1)].cuda()
                qd_right_ground_loss = self.mask_mse_loss(qd_right_constraint_logit, qd_right_ground_truth.float(),
                                                          qd_graph)

                output_dict['loss'] += qd_left_ground_loss + qd_right_ground_loss

                ## number in question and passage

                dq_left_constraint = torch.matmul(dq_node_left(aux_doc_node),
                                                  torch.transpose(q_node_v2, 1, -1).contiguous())
                dq_left_constraint_logit = torch.sigmoid(dq_left_constraint)

                dq_graph = aux_nums_doc_nums_mask.unsqueeze(-1) * question_number_mask.unsqueeze(1)
                dq_left_ground_truth = dq_graph * new_aux_num_graph_mask[:, :aux_doc_node.size(1),
                                                  -q_node_v2.size(1):].cuda()
                dq_left_ground_loss = self.mask_mse_loss(dq_left_constraint_logit, dq_left_ground_truth.float(),
                                                         dq_graph)

                dq_right_constraint = torch.matmul(dq_node_right(aux_doc_node),
                                                   torch.transpose(q_node_v2, 1, -1).contiguous())
                dq_right_constraint_logit = torch.sigmoid(dq_right_constraint)
                dq_graph = aux_nums_doc_nums_mask.unsqueeze(-1) * question_number_mask.unsqueeze(1)
                dq_right_ground_truth = dq_graph * new_aux_num_graph_mask[:, :aux_doc_node.size(1),
                                                   -q_node_v2.size(1):].cuda()
                dq_right_ground_loss = self.mask_mse_loss(dq_right_constraint_logit, dq_right_ground_truth.float(),
                                                          dq_graph)
                output_dict['loss'] += dq_left_ground_loss + dq_right_ground_loss

        if metadata is not None:
            output_dict["question_id"] = []
            output_dict["answer"] = []
            for i in range(batch_size):
                if len(self.answering_abilities) > 1:
                    predicted_ability_str = self.answering_abilities[best_answer_ability[i].detach().cpu().numpy()]
                else:
                    predicted_ability_str = self.answering_abilities[0]

                answer_json: Dict[str, Any] = {}

                question_start = 1
                passage_start = len(metadata[i]["question_tokens"]) + 2
                # We did not consider multi-mention answers here
                if predicted_ability_str == "passage_span_extraction":
                    answer_json["answer_type"] = "passage_span"
                    passage_str = metadata[i]['original_passage']
                    offsets = metadata[i]['passage_token_offsets']
                    predicted_answer, predicted_spans = best_answers_extraction(best_passage_span[i],
                                                                                best_span_number[i], passage_str,
                                                                                offsets, passage_start)
                    answer_json["value"] = predicted_answer
                    answer_json["spans"] = predicted_spans
                elif predicted_ability_str == "question_span_extraction":
                    answer_json["answer_type"] = "question_span"
                    question_str = metadata[i]['original_question']
                    offsets = metadata[i]['question_token_offsets']
                    predicted_answer, predicted_spans = best_answers_extraction(best_question_span[i],
                                                                                best_span_number[i], question_str,
                                                                                offsets, question_start)
                    answer_json["value"] = predicted_answer
                    answer_json["spans"] = predicted_spans
                elif predicted_ability_str == "addition_subtraction":
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
                    answer_json['number_sign_log_probs'] = number_sign_log_probs[i, :, :].detach().cpu().numpy()

                elif predicted_ability_str == "counting":
                    answer_json["answer_type"] = "count"
                    predicted_count = best_count_number[i].detach().cpu().numpy()
                    predicted_answer = str(predicted_count)
                    answer_json["count"] = predicted_count
                else:
                    raise ValueError(f"Unsupported answer ability: {predicted_ability_str}")

                answer_json["predicted_answer"] = predicted_answer
                output_dict["question_id"].append(metadata[i]["question_id"])
                # output_dict["answer"].append(answer_json)
                answer_annotations = metadata[i].get('answer_annotations', [])
                if answer_annotations:
                    self._drop_metrics(predicted_answer, answer_annotations)

            if self.use_gcn:
                output_dict['clamped_number_indices'] = clamped_number_indices
                output_dict['node_weight'] = d_node_weight
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}
