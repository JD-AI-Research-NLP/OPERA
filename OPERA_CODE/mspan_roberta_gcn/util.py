import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import allennlp as util

import spacy
from spacy.lang.en import English
from transformers import AutoTokenizer, AutoModel
import string
import collections
import allennlp.modules.matrix_attention
from allennlp.nn import Activation

# from allennlp.modules.attention import AdditiveAttention

from allennlp.modules.matrix_attention import (
    MatrixAttention,
    BilinearMatrixAttention,
    DotProductMatrixAttention,
    LinearMatrixAttention,
)


class Node(object):
    def __init__(self, id, type):
        self.id = id
        self.type = type

    def get_id(self):
        return self.id

    def get_type(self):
        return self.type


class SentenceNode(Node):
    def __init__(self, id, start_index, end_index, type, text):
        super().__init__(id, type)
        self.id = id
        self.start_index = start_index
        self.end_index = end_index
        self.type = type
        self.text = text

    def get_start(self):
        return self.start_index

    def get_end(self):
        return self.end_index

    def get_text(self):
        return self.text


class EntityNode(Node):
    def __init__(self, id, start_index, end_index, type, text):
        super().__init__(id, type)
        self.start_index = start_index
        self.end_index = end_index
        self.type = type
        self.text = text

    def get_start(self):
        return self.start_index

    def get_end(self):
        return self.end_index

    def get_text(self):
        return self.text


class NumberNode(Node):
    def __init__(self, id, start_index, end_index, type, text):
        super().__init__(id, type)
        self.start_index = start_index
        self.end_index = end_index
        self.type = type
        self.text = text

    def get_start(self):
        return self.start_index

    def get_end(self):
        return self.end_index

    def get_text(self):
        return self.text


class ValueNode(Node):
    def __init__(self, id, index, type, value):
        super().__init__(id, type)
        self.id = id
        self.index = index
        self.type = type
        self.value = value

    def get_index(self):
        return self.index

    def get_value(self):
        return self.value


class Edges(object):
    def __init__(self, edge_id, edge_type, node1, node2):
        self.edge_id = edge_id
        self.edge_type = edge_type
        self.node1 = node1
        self.node2 = node2

    def get_id(self):
        return self.edge_id

    def get_type(self):
        return self.edge_type

    def get_node1(self):
        return self.node1

    def get_node2(self):
        return self.node2


class SentenceAndEntity(Edges):
    def __init__(self, edge_id, edge_type, node1, node2):
        super().__init__(edge_id, edge_type, node1, node2)
        self.edge_id = edge_id
        self.edge_type = edge_type
        self.node1 = node1
        self.node2 = node2


class EntityAndEntity(Edges):
    def __init__(self, edge_id, edge_type, node1, node2):
        super().__init__(edge_id, edge_type, node1, node2)
        self.edge_id = edge_id
        self.edge_type = edge_type
        self.node1 = node1
        self.node2 = node2


class SentenceAndValue(Edges):
    def __init__(self, edge_id, edge_type, node1, node2):
        super().__init__(edge_id, edge_type, node1, node2)
        self.edge_id = edge_id
        self.edge_type = edge_type
        self.node1 = node1
        self.node2 = node2


class EntityAndValue(Edges):
    def __init__(self, edge_id, edge_type, node1, node2):
        super().__init__(edge_id, edge_type, node1, node2)
        self.edge_id = edge_id
        self.edge_type = edge_type
        self.node1 = node1
        self.node2 = node2


class HeterogeneousGraph(object):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def get_type_edges(self, type):
        return self.edges[type]


class TextProcessorSpacy():
    def __init__(self):
        super(TextProcessorSpacy, self).__init__()

    def sentencizer(self, raw_content):
        nlp = English()
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer)
        doc = nlp(raw_content)
        sentences = [str(sent) for sent in doc.sents]
        return sentences

    def sen2clause(self, sentences):

        clause_list = []
        if isinstance(sentences, str):
            sen_clause = str.split(sentences, ', ')
            clause_list.extend(sen_clause)
        elif isinstance(sentences, list):
            for sen in sentences:
                sen_clauses = str.split(sen, ', ')
                if '' in sen_clauses:
                    sen_clauses.remove('')
                if ' ' in sen_clauses:
                    sen_clauses.remove(' ')
                if '.' in sen_clauses:
                    sen_clauses.remove('.')
                if '"' in sen_clauses:
                    sen_clauses.remove('"')
                clause_list.extend(sen_clauses)
        else:
            print('sen2clause input type error')
        return clause_list

    def ner(self, raw_content, exclude_entity_type_list=None):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(raw_content)
        entities_list = []
        for item in doc.ents:
            if exclude_entity_type_list is not None and item.label_ in exclude_entity_type_list: continue
            entities_dict = {'type': item.label_, 'start_char': item.start_char, 'end_char': item.end_char,
                             'content': item.text}
            # entities_dict = {'type': item.label_,  'content': item.text}
            entities_list.append(entities_dict)

        return entities_list

    def find_entity_index(self, offset, offset_list):
        '''
        find the index i, statisify offsetlist[i](0)<=offset(0)<=offset(1)<=offsetlist[i](1)
        '''
        index = -1
        for idx, item in enumerate(offset_list):
            if offset[0] >= item[0] and offset[1] <= item[1]:
                index = idx
                return index
        return index

    def extract_tag(self, doc, tag_list):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(doc)
        result = []
        if tag_list is None:
            for w in doc:
                result.append((w, w.tag_))
        else:
            for w in doc:
                if w.tag_ in tag_list:
                    result.append((w, w.tag_))

        return result

    def sentence_split(self, str_centence):
        list_ret = list()
        for s_str in str_centence.split('. '):
            if '?' in s_str:
                list_ret.extend(s_str.split('? '))
            elif '!' in s_str:
                list_ret.extend(s_str.split('! '))
            else:
                list_ret.append(s_str)
        return list_ret


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    raw = "In 1905, 1,003 Korean immigrants, which included 802 men and 231 women and children, departed from the port of Chemulpo, Incheon aboard the ship Ilford to Salina Cruz, Oaxaca, Mexico. The journey took 45 days, after which they took a train to Coatzacoalcos, Veracruz. In the Veracruz port, another boat was taken to the port of Progreso with the final destination being the capital city of M\u00e9rida, Yucatan. They arrived in May 1905, with previously signed contracts for four years' work as indentured laborers on the Yucat\u00e1n henequen haciendas. Many of these Koreans were distributed throughout the Yucat\u00e1n in 32 henequen haciendas. The town of Motul, Yucatan, located in the heart of the henequen zone, was a destination for many of the Korean immigrants. Subsequently, in 1909, at the end of their contracts, they began a new stage in which they scattered even further  Thus, the majority of those who came were single men who made or remade their family lives with Yucatecan especially Maya women. While Korean girls were much more subject to marriages arranged by Korean parents, males had greater freedom when it came to making a family. This rapid intermarriage by Koreans, coupled with geographic dispersal, prevented the establishment of close social networks among these migrants and therefore provided the basis for Korean descendants among the Yucatan Peninsula. After that 1905 ship, no further entries of Koreans into Mexico were recorded, until many years later, leading to a new community of Koreans with completely different characteristics from those who entered in 1905. These descendants have started the Museo Conmemorativo de la Inmigraci\u00f3n Coreana a Yucat\u00e1n, a museum for the remembrance of their ancestors journey."
    # raw2 = "The Siege of Vienna in 1529, the first quarter was the first attempt by the Ottoman Empire, led by Suleiman the Magnificent, to capture the city of Vienna, Austria. The siege signalled the pinnacle of the Ottoman Empire's power and the maximum extent of Ottoman expansion in central Europe. Thereafter, 150 years of bitter military tension and reciprocal attacks ensued, culminating in the Battle of Vienna of 1683, which marked the start of the 15-year-long Great Turkish War. The inability of the Ottomans to capture Vienna in 1529 turned the tide against almost a century of conquest throughout eastern and central Europe. The Ottoman Empire had previously annexed Central Hungary and established a vassal state in Transylvania in the wake of the Battle of Moh\u00e1cs. According to Arnold J. Toynbee, \"The failure of the first  brought to a standstill the tide of Ottoman conquest which had been flooding up the Danube Valley for a century past.\" There is speculation by some historians that Suleiman's main objective in 1529 was actually to assert Ottoman control over the whole of Hungary, the western part of which  was under Habsburg control. The decision to attack Vienna after such a long interval in Suleiman's European campaign is viewed as an opportunistic manoeuvre after his decisive victory in Hungary. Other scholars theorise that the suppression of Hungary simply marked the prologue to a later, premeditated invasion of Europe."
    # raw3 = 'the texans would respond with fullback vonta leach getting a 1-yard touchdown run yet the raiders would answer with kicker sebastian janikowski getting a 33-yard and a 30-yard field goal'
    # raw = "Hoping to rebound from their loss to the Patriots, the Raiders stayed at home for a Week 16 duel with the Houston Texans.,  Oakland would get the early lead in the first quarter as quarterback JaMarcus Russell completed a 20-yard touchdown pass to rookie wide receiver Chaz Schilens.,  The Texans would respond with fullback Vonta Leach getting a 1-yard touchdown run, yet the Raiders would answer with kicker Sebastian Janikowski getting a 33-yard and a 30-yard field goal.,  Houston would tie the game in the second quarter with kicker Kris Brown getting a 53-yard and a 24-yard field goal., Oakland would take the lead in the third quarter with wide receiver Johnnie Lee Higgins catching a 29-yard touchdown pass from Russell, followed up by an 80-yard punt return for a touchdown.,  The Texans tried to rally in the fourth quarter as Brown nailed a 40-yard field goal, yet the Raiders' defense would shut down any possible attempt."
    # raw = " ".join([raw1,raw2,raw3])
    # raw = "German has 13,444 speakers representing about 0.4% of the states population, and Vietnamese is spoken by 11,330 people, or about 0.4% of the population, many of whom live in the Asia District, Oklahoma City of Oklahoma City. Other languages include French with 8,258 speakers (0.3%), Chinese Americans with 6,413 (0.2%), Korean with 3,948 (0.1%), Arabic with 3,265 (0.1%), other Asian languages with 3,134 (0.1%), Tagalog language with 2,888 (0.1%), Japanese with 2,546 (0.1%), and African languages with 2,546 (0.1%). In addition to Cherokee, more than 25 Indigenous languages of the Americas are spoken in Oklahoma, second only to California (though, it should be noted only Cherokee exhibits language vitality at present)."
    # raw2 = "In the county, the population was spread out with 26.20% under the age of 18, 9.30% from 18 to 24, 26.50% from 25 to 44, 23.50% from 45 to 64, and 14.60% who were 65 years of age or older.  The median age was 37 years. For every 100 females there were 95.90 males.  For every 100 females age 18 and over, there were 92.50 males."
    # tokenizer = AutoTokenizer.from_pretrained('../../numnet_plus_data/drop_dataset/roberta.base')
    # clause_list = []
    # text_processor = TextProcessorSpacy()
    # sentences = text_processor.sentencizer(raw)
    # for sentence in sentences:
    #     entity_in_sentence = text_processor.ner(sentence)
    #     print(entity_in_sentence)
    #
    # # print(idx)

    from stanfordcorenlp import StanfordCoreNLP

    nlp = StanfordCoreNLP('./stanford-corenlp-4.1.0')
    NER = nlp.ner(raw)
    print(NER)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        self.enc_layer.flatten_parameters()
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)


class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)


class SignNumFFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(SignNumFFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)

        self.fc2 = nn.Linear(intermediate_dim, output_dim * intermediate_dim)
        self.fc3 = nn.Linear(intermediate_dim, 1)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        inter_act = self.fc2(inter_act)
        inter_act = inter_act.unsqueeze(2).reshape(input.size(0), input.size(1), 3, -1)
        inter_act = self.fc3(inter_act)
        return inter_act.squeeze(-1)


'''
class SimpleAdditiveAttention(torch.nn.Module):
    def __init__(self, encoder_dim=100, decoder_dim=50):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.W_1 = torch.nn.Linear(self.encoder_dim, self.decoder_dim, bias=False)
        self.W_2 = torch.nn.Linear(self.encoder_dim, self.decoder_dim, bias=False)

    def forward(self,
                query,  # [bsz, L,h]
                values,  # [bsz, L, h]
                attention_mask
                ):
        weights = self._get_weights(query, values)
        weights = F.leaky_relu(weights)
        zero_vec = torch.zeros_like(weights)
        weights = torch.where(attention_mask > 0, weights, zero_vec)

        return weights

    def _get_weights(self,
                     query,  # [[bsz, L,h]]
                     values  # [bsz, L, h]
                     ):
        query = query.unsqueeze(1).repeat(1, values.size(1), 1, 1)  # [bsz, L, L, h]
        values = values.unsqueeze(1).repeat(1, query.size(1), 1, 1)
        weights = torch.transpose(self.W_1(query).squeeze(-1), dim0=1, dim1=-1) + self.W_2(values).squeeze(
            -1)  # [bsz, L, L]

        return weights




class SimpleQDGAT(nn.Module):
    def __init__(self, hidden_size, iter_steps, qd, NUMBER_TOKEN_TYPE, drop_out):
        super(SimpleQDGAT, self).__init__()
        self.hidden_size = hidden_size
        # self.output_dim = output_dim
        self.iter_steps = iter_steps
        self.qd = qd
        self.NUMBER_TOKEN_TYPE = NUMBER_TOKEN_TYPE
        self.drop_out = drop_out
        self.layer_norm_eps = 1e-12

        self.W_fc = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.W_dc1 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.W_dc2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.W_dc3 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.W_dc4 = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self.W_qv = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.W_kv = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.W_vv = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self.W_qc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_kc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_vc = nn.Linear(hidden_size, hidden_size, bias=False)

        self.number_number_DATE_attention = SimpleAdditiveAttention(hidden_size, 1)
        self.number_number_TIME_attention = SimpleAdditiveAttention(hidden_size, 1)
        self.number_number_PERCENT_attention = SimpleAdditiveAttention(hidden_size, 1)
        self.number_number_MONEY_attention = SimpleAdditiveAttention(hidden_size, 1)
        self.number_number_QUANTITY_attention = SimpleAdditiveAttention(hidden_size, 1)
        self.number_number_ORDINAL_attention = SimpleAdditiveAttention(hidden_size, 1)
        self.number_number_CARDINAL_attention = SimpleAdditiveAttention(hidden_size, 1)
        self.number_number_YARD_attention = SimpleAdditiveAttention(hidden_size, 1)

        self.entity_number_attention = SimpleAdditiveAttention(hidden_size, 1)

        self.norm_layer = torch.nn.LayerNorm(hidden_size)
        self.x_q_norm_layer = torch.nn.LayerNorm(hidden_size, self.layer_norm_eps)
        self.x_k_norm_layer = torch.nn.LayerNorm(hidden_size, self.layer_norm_eps)
        self.x_v_norm_layer = torch.nn.LayerNorm(hidden_size, self.layer_norm_eps)

        self.drop_out_layer = nn.Dropout(self.drop_out)
        self.W_u = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.input_layer_norm = torch.nn.LayerNorm(2*hidden_size, self.layer_norm_eps)
        self.input_proj = torch.nn.Linear(2*hidden_size, 2*hidden_size)
        self.m_t_LN = torch.nn.LayerNorm(hidden_size, self.layer_norm_eps)

    def forward(self, node_emebdding=None, question_emebdding=None, attention_mask_entity_number_type=None,
                attention_mask_number_number_types=None, entity_node_mask=None, number_node_mask=None):

        node_encoding = node_emebdding
        number_number_types_attention = {'DATE': self.number_number_DATE_attention,
                                         'TIME': self.number_number_TIME_attention,
                                         'PERCENT': self.number_number_PERCENT_attention,
                                         'MONEY': self.number_number_MONEY_attention,
                                         'QUANTITY': self.number_number_QUANTITY_attention,
                                         'ORDINAL': self.number_number_ORDINAL_attention,
                                         'CARDINAL': self.number_number_CARDINAL_attention,
                                         'YARD': self.number_number_YARD_attention
                                         }
        w_dcs = [self.W_dc1, self.W_dc2, self.W_dc3, self.W_dc4]
        for t in range(self.iter_steps):
            m_t = w_dcs[t](F.elu(self.W_fc(question_emebdding)))
            input_encoding = torch.cat([node_encoding, node_emebdding], dim=-1)
            input_encoding = self.input_proj(input_encoding)
            input_encoding = self.input_layer_norm(input_encoding)
            if self.qd:
                x_q = self.W_qv(input_encoding) * self.W_qc(m_t)
                x_k = self.W_kv(input_encoding) * self.W_kc(m_t)
                x_v = self.W_vv(input_encoding) * self.W_vc(m_t)
                x_q = self.x_q_norm_layer(x_q)
                x_k = self.x_k_norm_layer(x_k)
                x_v = self.x_v_norm_layer(x_v)
            else:
                x_q = self.W_qv(input_encoding)
                x_k = self.W_kv(input_encoding)
                x_v = self.W_vv(input_encoding)
                # x_q = self.x_q_norm_layer(x_q)
                # x_k = self.x_k_norm_layer(x_k)
                # x_v = self.x_v_norm_layer(x_v)

            entity_number_att = self.entity_number_attention(x_q, x_k, attention_mask_entity_number_type)

            attention_mask = torch.zeros_like(attention_mask_entity_number_type)
            attention_score = torch.zeros_like(entity_number_att)
            for number_type in self.NUMBER_TOKEN_TYPE:
                attention_mask_type = attention_mask_number_number_types[number_type]
                number_number_type_attention = number_number_types_attention[number_type]
                number_number_type_att = number_number_type_attention(x_q, x_k, attention_mask_type)
                attention_score += number_number_type_att
                attention_mask += attention_mask_type

            attention_score += entity_number_att
            attention_mask += attention_mask_entity_number_type
            neginf = -1e32 * torch.ones_like(attention_score).cuda()
            attention_score = torch.where(attention_mask > 0, attention_score, neginf)

            attention_probs = util.masked_softmax(attention_score, attention_mask, dim=-1)
            zero_attention_probs = torch.zeros_like(attention_probs)
            attention_probs = torch.where(attention_mask > 0,
                                          attention_probs, zero_attention_probs)

            attention_probs = self.drop_out_layer(attention_probs)

            X_t = util.weighted_sum(x_v, attention_probs)

            node_encoding = torch.cat([node_encoding, X_t], dim=-1)
            node_encoding = self.W_u(node_encoding)
            node_encoding = self.norm_layer(node_encoding)



        return node_encoding

'''


class SimpleAdditiveAttention(torch.nn.Module):
    def __init__(self, encoder_dim=100, decoder_dim=50):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.W_1 = torch.nn.Linear(self.encoder_dim, self.decoder_dim, bias=False)
        self.W_2 = torch.nn.Linear(self.encoder_dim, self.decoder_dim, bias=False)

    def forward(self,
                query,  # [bsz, L,h]
                values,  # [bsz, L, h]
                attention_mask
                ):
        weights = self._get_weights(query, values)
        weights = F.leaky_relu(weights)
        zero_vec = torch.zeros_like(weights)
        weights = torch.where(attention_mask > 0, weights, zero_vec)

        return weights

    def _get_weights(self,
                     query,  # [[bsz, L,h]]
                     values  # [bsz, L, h]
                     ):
        query = query.unsqueeze(1).repeat(1, values.size(1), 1, 1)  # [bsz, L, L, h]
        values = values.unsqueeze(1).repeat(1, query.size(1), 1, 1)
        weights = torch.transpose(self.W_1(query).squeeze(-1), dim0=1, dim1=-1) + self.W_2(values).squeeze(
            -1)  # [bsz, L, L]

        return weights


class SimpleQDGAT(nn.Module):
    def __init__(self, hidden_size, iter_steps, qd, NUMBER_TOKEN_TYPE, drop_out):
        super(SimpleQDGAT, self).__init__()
        self.hidden_size = hidden_size
        # self.output_dim = output_dim
        self.iter_steps = iter_steps
        self.qd = qd
        self.NUMBER_TOKEN_TYPE = NUMBER_TOKEN_TYPE
        self.drop_out = drop_out
        self.layer_norm_eps = 1e-12

        self.W_fc = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.W_dc1 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.W_dc2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.W_dc3 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.W_dc4 = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self.W_qv = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.W_kv = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.W_vv = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self.W_qc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_kc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_vc = nn.Linear(hidden_size, hidden_size, bias=False)

        self.number_number_DATE_attention = LinearMatrixAttention(tensor_1_dim=hidden_size, tensor_2_dim=hidden_size,
                                                                  combination="x,y",
                                                                  activation=Activation.by_name('leaky_relu')())
        self.number_number_TIME_attention = LinearMatrixAttention(tensor_1_dim=hidden_size, tensor_2_dim=hidden_size,
                                                                  combination="x,y",
                                                                  activation=Activation.by_name('leaky_relu')())
        self.number_number_PERCENT_attention = LinearMatrixAttention(tensor_1_dim=hidden_size, tensor_2_dim=hidden_size,
                                                                     combination="x,y",
                                                                     activation=Activation.by_name('leaky_relu')())
        self.number_number_MONEY_attention = LinearMatrixAttention(tensor_1_dim=hidden_size, tensor_2_dim=hidden_size,
                                                                   combination="x,y",
                                                                   activation=Activation.by_name('leaky_relu')())
        self.number_number_QUANTITY_attention = LinearMatrixAttention(tensor_1_dim=hidden_size,
                                                                      tensor_2_dim=hidden_size, combination="x,y",
                                                                      activation=Activation.by_name('leaky_relu')())
        self.number_number_ORDINAL_attention = LinearMatrixAttention(tensor_1_dim=hidden_size, tensor_2_dim=hidden_size,
                                                                     combination="x,y",
                                                                     activation=Activation.by_name('leaky_relu')())
        self.number_number_CARDINAL_attention = LinearMatrixAttention(tensor_1_dim=hidden_size,
                                                                      tensor_2_dim=hidden_size, combination="x,y",
                                                                      activation=Activation.by_name('leaky_relu')())
        self.number_number_YARD_attention = LinearMatrixAttention(tensor_1_dim=hidden_size, tensor_2_dim=hidden_size,
                                                                  combination="x,y",
                                                                  activation=Activation.by_name('leaky_relu')())
        self.entity_number_attention = LinearMatrixAttention(tensor_1_dim=hidden_size, tensor_2_dim=hidden_size,
                                                             combination="x,y",
                                                             activation=Activation.by_name('leaky_relu')())

        self.norm_layer = torch.nn.LayerNorm(hidden_size)
        self.x_q_norm_layer = torch.nn.LayerNorm(hidden_size, self.layer_norm_eps)
        self.x_k_norm_layer = torch.nn.LayerNorm(hidden_size, self.layer_norm_eps)
        self.x_v_norm_layer = torch.nn.LayerNorm(hidden_size, self.layer_norm_eps)

        self.drop_out_layer = nn.Dropout(self.drop_out)
        self.W_u = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.input_layer_norm = torch.nn.LayerNorm(2 * hidden_size, self.layer_norm_eps)
        self.input_proj = torch.nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.m_t_LN = torch.nn.LayerNorm(hidden_size, self.layer_norm_eps)

    def forward(self, node_emebdding=None, question_emebdding=None, attention_mask_entity_number_type=None,
                attention_mask_number_number_types=None, entity_node_mask=None, number_node_mask=None):

        node_encoding = node_emebdding
        number_number_types_attention = {'DATE': self.number_number_DATE_attention,
                                         'TIME': self.number_number_TIME_attention,
                                         'PERCENT': self.number_number_PERCENT_attention,
                                         'MONEY': self.number_number_MONEY_attention,
                                         'QUANTITY': self.number_number_QUANTITY_attention,
                                         'ORDINAL': self.number_number_ORDINAL_attention,
                                         'CARDINAL': self.number_number_CARDINAL_attention,
                                         'YARD': self.number_number_YARD_attention
                                         }
        w_dcs = [self.W_dc1, self.W_dc2, self.W_dc3, self.W_dc4]
        for t in range(self.iter_steps):
            m_t = w_dcs[t](F.elu(self.W_fc(question_emebdding)))
            input_encoding = torch.cat([node_encoding, node_emebdding], dim=-1)
            if self.qd:
                x_q = self.W_qv(input_encoding) * self.W_qc(m_t)
                x_k = self.W_kv(input_encoding) * self.W_kc(m_t)
                x_v = self.W_vv(input_encoding) * self.W_vc(m_t)
                x_q = self.x_q_norm_layer(x_q)
                x_k = self.x_k_norm_layer(x_k)
                x_v = self.x_v_norm_layer(x_v)
            else:
                x_q = self.W_qv(input_encoding)
                x_k = self.W_kv(input_encoding)
                x_v = self.W_vv(input_encoding)

            zero_attention = torch.zeros(node_emebdding.size(0), node_emebdding.size(1), node_emebdding.size(1))
            entity_number_attention_score = self.entity_number_attention(x_q, x_k)
            entity_number_attention_score = torch.where(attention_mask_entity_number_type > 0,
                                                        entity_number_attention_score,
                                                        zero_attention)

            number_number_attention_mask = torch.zeros_like(attention_mask_entity_number_type)
            number_number_attention_score = torch.zeros_like(entity_number_attention_score)

            for number_type in self.NUMBER_TOKEN_TYPE:
                number_number_type_attention = number_number_types_attention[number_type]
                number_number_type_att = number_number_type_attention(x_q, x_k)
                attention_mask_type = attention_mask_number_number_types[number_type]
                number_number_type_att = torch.where(attention_mask_type > 0, number_number_type_att, zero_attention)
                number_number_attention_score += number_number_type_att
                number_number_attention_mask += attention_mask_type

            number_number_attention_probs = util.masked_softmax(number_number_attention_score,
                                                                number_number_attention_mask, dim=-1)
            entity_number_attention_probs = util.masked_softmax(entity_number_attention_score,
                                                                attention_mask_entity_number_type, dim=-1)

            X_number_number = util.weighted_sum(x_v, number_number_attention_probs)
            X_entity_number = util.weighted_sum(x_v, entity_number_attention_probs)
            X_t = X_number_number + X_entity_number

            node_encoding = torch.cat([node_encoding, X_t], dim=-1)
            node_encoding = self.W_u(node_encoding)
            node_encoding = self.norm_layer(node_encoding)

        return node_encoding


# class AdditiveAttention(torch.nn.Module):
#     def __init__(self, encoder_dim=100, decoder_dim=50, alpha=0.2):
#         super().__init__()
#
#         self.encoder_dim = encoder_dim
#         self.decoder_dim = decoder_dim
#         self.alpha = alpha
#         self.W = torch.nn.Linear(self.encoder_dim, self.decoder_dim, bias=False)
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self,
#                 query,  # [bsz, L,h]
#                 key,  # [bsz, L, h]
#                 attention_mask
#                 ):
#         input = self._prepare_attentional_mechanism_input(query, key)  # [bsz, L, L, 2h]
#         weights = self.W(input).squeeze(-1)
#         weights = self.leakyrelu(weights)
#         zero_vec = torch.zeros_like(weights)
#         weights = torch.where(attention_mask > 0, weights, zero_vec)  # [bsz, L, L]
#         return weights
#
#     def _prepare_attentional_mechanism_input(self,
#                                              query,  # [[bsz, L,h]]
#                                              key  # [bsz, L, h]
#                                              ):
#         bsz = query.size(0)
#         N = query.size(1)
#         hidden_size = query.size(-1)
#         query_repeated_in_chunk = query.repeat_interleave(N, 1)  # bsz, l*l, h
#         key_repeated_alternating = key.repeat(1, N, 1)
#         all_combinations_matrix = torch.cat([query_repeated_in_chunk, key_repeated_alternating], dim=-1)
#         return all_combinations_matrix.view(bsz, N, N, 2 * hidden_size)
#
#
# class QDGATLayer(nn.Module):
#     def __init__(self, input_features_dim, output_feature_dim, dropout, alpha, qd, NUMBER_TOKEN_TYPE):
#         super(QDGATLayer, self).__init__()
#         self.dropout = dropout
#         # self.hidden_size = hidden_size
#         self.input_features_dim = input_features_dim
#         self.output_feature_dim = output_feature_dim
#         self.alpha = alpha
#         self.qd = qd
#         self.NUMBER_TOKEN_TYPE = NUMBER_TOKEN_TYPE
#
#         # self.w_dc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
#         self.W_fc = nn.Linear(output_feature_dim, 2 * output_feature_dim, bias=False)
#
#         self.W_qv = nn.Linear(2 * input_features_dim, output_feature_dim, bias=False)
#         self.W_kv = nn.Linear(2 * input_features_dim, output_feature_dim, bias=False)
#         self.W_vv = nn.Linear(2 * input_features_dim, output_feature_dim, bias=False)
#
#         self.W_qc = nn.Linear(output_feature_dim, output_feature_dim, bias=False)
#         self.W_kc = nn.Linear(output_feature_dim, output_feature_dim, bias=False)
#         self.W_vc = nn.Linear(output_feature_dim, output_feature_dim, bias=False)
#
#         self.number_number_types_attention = nn.ModuleDict(
#             {'DATE': AdditiveAttention(2 * output_feature_dim, 1, self.alpha),
#              'TIME': AdditiveAttention(2 * output_feature_dim, 1, self.alpha),
#              'PERCENT': AdditiveAttention(2 * output_feature_dim, 1,
#                                           self.alpha),
#              'MONEY': AdditiveAttention(2 * output_feature_dim, 1, self.alpha),
#              'QUANTITY': AdditiveAttention(2 * output_feature_dim, 1,
#                                            self.alpha),
#              'ORDINAL': AdditiveAttention(2 * output_feature_dim, 1,
#                                           self.alpha),
#              'CARDINAL': AdditiveAttention(2 * output_feature_dim, 1,
#                                            self.alpha),
#              'YARD': AdditiveAttention(2 * output_feature_dim, 1, self.alpha)})
#
#         self.entity_number_attention = AdditiveAttention(2 * output_feature_dim, 1, self.alpha)
#
#         self.x_q_norm_layer = torch.nn.LayerNorm(output_feature_dim)
#         self.x_k_norm_layer = torch.nn.LayerNorm(output_feature_dim)
#         self.x_v_norm_layer = torch.nn.LayerNorm(output_feature_dim)
#         self.norm_layer = torch.nn.LayerNorm(2 * output_feature_dim)
#
#         self.drop_out_layer = nn.Dropout(self.dropout)
#
#     def forward(self, w_dc, node_emebdding=None, question_emebdding=None, attention_mask_entity_number_type=None,
#                 attention_mask_number_number_types=None, entity_node_mask=None, number_node_mask=None):
#
#         node_encoding = node_emebdding
#         m_t = w_dc(F.elu(self.W_fc(question_emebdding)))
#         input_encoding = torch.cat([node_encoding, node_emebdding], dim=-1)
#         input_encoding = F.dropout(input_encoding, self.dropout, training=self.training)
#
#         if self.qd:
#             x_q = self.W_qv(input_encoding) * self.W_qc(m_t)
#             x_k = self.W_kv(input_encoding) * self.W_kc(m_t)
#             x_v = self.W_vv(input_encoding) * self.W_vc(m_t)
#             x_q = self.x_q_norm_layer(x_q)
#             x_k = self.x_k_norm_layer(x_k)
#             x_v = self.x_v_norm_layer(x_v)
#         else:
#             x_q = self.W_qv(input_encoding)
#             x_k = self.W_kv(input_encoding)
#             x_v = self.W_vv(input_encoding)
#
#         # x_q = F.dropout(x_q, self.dropout, training=self.training)
#         # x_k = F.dropout(x_k, self.dropout, training=self.training)
#         # x_v = F.dropout(x_v, self.dropout, training=self.training)
#
#         entity_number_att = self.entity_number_attention(x_q, x_k, attention_mask_entity_number_type)
#
#         attention_mask = torch.zeros_like(attention_mask_entity_number_type)
#         attention_score = torch.zeros_like(entity_number_att)
#         for number_type in self.NUMBER_TOKEN_TYPE:
#             attention_mask_type = attention_mask_number_number_types[number_type]
#             number_number_type_attention = self.number_number_types_attention[number_type]
#             number_number_type_att = number_number_type_attention(x_q, x_k, attention_mask_type)
#             attention_score += number_number_type_att
#             attention_mask += attention_mask_type
#
#         attention_score += entity_number_att
#         attention_mask += attention_mask_entity_number_type
#         neginf = -1e32 * torch.ones_like(attention_score).cuda()
#         attention_score = torch.where(attention_mask > 0, attention_score, neginf)
#
#         attention_probs = F.softmax(attention_score, dim=-1)
#         zero_attention_probs = torch.zeros_like(attention_probs)
#         attention_probs = torch.where(attention_mask > 0,
#                                       attention_probs, zero_attention_probs)
#
#         attention_probs = F.dropout(attention_probs, self.dropout, training=self.training)
#         X_t = util.weighted_sum(x_v, attention_probs)
#
#         return X_t
#
#
# class QDGAT(nn.Module):
#     def __init__(self, hidden_size, dropout, alpha, nheads, iter_step, qd, NUMBER_TOKEN_TYPE):
#         """Dense version of QDGAT."""
#         super(QDGAT, self).__init__()
#         self.hidden_size = hidden_size
#         self.drop_out = dropout
#         self.alpha = alpha
#         self.nheads = nheads
#         self.iter_step = iter_step
#         self.qd = qd
#         self.NUMBER_TOKEN_TYPE = NUMBER_TOKEN_TYPE
#         self.attentions = nn.ModuleList(
#             [QDGATLayer(hidden_size, hidden_size, dropout, alpha, qd, NUMBER_TOKEN_TYPE) for i in
#              range(nheads)])
#
#         self.w_dc = nn.ModuleList([nn.Linear(2 * hidden_size, hidden_size, bias=False) for i in range(self.iter_step)])
#         self.W_u = nn.Linear(2 * hidden_size, hidden_size, bias=False)
#         self.norm_layer = torch.nn.LayerNorm(hidden_size)
#         self.out_attention_layer = QDGATLayer(nheads * hidden_size, hidden_size, dropout, alpha, qd, NUMBER_TOKEN_TYPE)
#
#     def forward(self, node_emebdding=None, question_emebdding=None, attention_mask_entity_number_type=None,
#                 attention_mask_number_number_types=None, entity_node_mask=None, number_node_mask=None):
#         node_encoding = node_emebdding
#         for t in range(self.iter_step):
#             # node_encoding_aux_list_t = [
#             #     att(self.w_dc[t], node_emebdding, question_emebdding, attention_mask_entity_number_type,
#             #         attention_mask_number_number_types, entity_node_mask, number_node_mask) for att in
#             #     self.attentions]
#             node_encoding_aux_t = torch.cat(
#                 [att(self.w_dc[t], node_emebdding, question_emebdding, attention_mask_entity_number_type,
#                      attention_mask_number_number_types, entity_node_mask, number_node_mask) for att in
#                  self.attentions], dim=-1)
#
#             # node_encoding_aux_t = sum(node_encoding_aux_list_t)
#             # node_encoding_aux_t = node_encoding_aux_t / len(node_encoding_aux_list_t)
#
#             node_encoding_aux_t = self.out_attention_layer(self.w_dc[t], node_encoding_aux_t, question_emebdding,
#                                                            attention_mask_entity_number_type,
#                                                            attention_mask_number_number_types, entity_node_mask,
#                                                            number_node_mask)
#
#             node_encoding = torch.cat([node_encoding, node_encoding_aux_t], dim=-1)
#             node_encoding = self.W_u(node_encoding)
#             node_encoding = self.norm_layer(node_encoding)
#         return node_encoding

class MultiHeadAdditiveAttention(torch.nn.Module):
    def __init__(self, encoder_dim=100, decoder_dim=50, alpha=0.2):
        super(MultiHeadAdditiveAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.alpha = alpha
        self.W = torch.nn.Linear(self.encoder_dim, self.decoder_dim, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self,
                query,  # [bsz, N, nhead, h] -> [bsz,nhead, N, h]
                key,  # [bsz, N, nhead, h] -> [bsz,nhead, N, h]
                attention_mask  # [bsz, N, N]
                ):
        input = self._prepare_attentional_mechanism_input(query, key)  # [bsz, nhead, L, L, 2h]
        weights = self.W(input).squeeze(-1)  # [bsz, nhead, L, L]
        # weights = self.leakyrelu(weights)
        zero_vec = torch.zeros_like(weights)
        # print(attention_mask.unsqueeze(1).expand(-1, 2, -1,-1).size())
        # print(query.size(1))
        # # print(zero_vec.size())
        weights = torch.where(attention_mask.unsqueeze(1).expand(-1, query.size(1), -1, -1) > 0, weights,
                              zero_vec)  # [bsz, nheads, L, L]
        return weights

    def _prepare_attentional_mechanism_input(self,
                                             query,  # [bsz,nhead, N, h]
                                             key  # [bsz,nhead, N, h]
                                             ):
        bsz = query.size(0)
        N = query.size(-2)
        hidden_size = query.size(-1)
        nhead = query.size(1)
        query_repeated_in_chunk = query.repeat_interleave(N, -2)  # bsz,n_head, N*N, h
        key_repeated_alternating = key.repeat(1, 1, N, 1)  # bsz,n_head, N*N, h
        all_combinations_matrix = torch.cat([query_repeated_in_chunk, key_repeated_alternating], dim=-1)
        return all_combinations_matrix.view(bsz, nhead, N, N, 2 * hidden_size)


class MultiHeadQDGAT(nn.Module):
    def __init__(self, input_feature_dim, output_feature_dim, dropout, alpha, nheads, iter_step, qd, NUMBER_TOKEN_TYPE):
        super(MultiHeadQDGAT, self).__init__()

        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim
        self.drop_out = dropout
        self.alpha = alpha
        self.nheads = nheads
        self.attention_head_size = int(output_feature_dim / self.nheads)
        self.iter_step = iter_step
        self.qd = qd
        self.NUMBER_TOKEN_TYPE = NUMBER_TOKEN_TYPE
        self.layer_norm_eps = 1e-12
        self.number_number_types_attention = nn.ModuleDict(
            {'DATE': MultiHeadAdditiveAttention(2 * self.attention_head_size, 1, self.alpha),
             'TIME': MultiHeadAdditiveAttention(2 * self.attention_head_size, 1, self.alpha),
             'PERCENT': MultiHeadAdditiveAttention(2 * self.attention_head_size, 1,
                                                   self.alpha),
             'MONEY': MultiHeadAdditiveAttention(2 * self.attention_head_size, 1, self.alpha),
             'QUANTITY': MultiHeadAdditiveAttention(2 * self.attention_head_size, 1,
                                                    self.alpha),
             'ORDINAL': MultiHeadAdditiveAttention(2 * self.attention_head_size, 1,
                                                   self.alpha),
             'CARDINAL': MultiHeadAdditiveAttention(2 * self.attention_head_size, 1,
                                                    self.alpha),
             'YARD': MultiHeadAdditiveAttention(2 * self.attention_head_size, 1, self.alpha)})

        self.entity_number_attention = MultiHeadAdditiveAttention(2 * self.attention_head_size, 1, self.alpha)

        self.w_dc = nn.ModuleList(
            [nn.Linear(2 * output_feature_dim, output_feature_dim, bias=False) for i in range(self.iter_step)])
        self.W_fc = nn.Linear(output_feature_dim, 2 * output_feature_dim, bias=False)

        self.W_qv = nn.Linear(2 * input_feature_dim, output_feature_dim, bias=False)
        self.W_kv = nn.Linear(2 * input_feature_dim, output_feature_dim, bias=False)
        self.W_vv = nn.Linear(2 * input_feature_dim, output_feature_dim, bias=False)

        self.W_qc = nn.Linear(output_feature_dim, output_feature_dim, bias=False)
        self.W_kc = nn.Linear(output_feature_dim, output_feature_dim, bias=False)
        self.W_vc = nn.Linear(output_feature_dim, output_feature_dim, bias=False)

        self.W_u = nn.Linear(2 * output_feature_dim, output_feature_dim, bias=False)

        self.x_q_norm_layer = torch.nn.LayerNorm(output_feature_dim, self.layer_norm_eps)
        self.x_k_norm_layer = torch.nn.LayerNorm(output_feature_dim, self.layer_norm_eps)
        self.x_v_norm_layer = torch.nn.LayerNorm(output_feature_dim, self.layer_norm_eps)
        self.norm_layer = torch.nn.LayerNorm(output_feature_dim, self.layer_norm_eps)
        self.input_encoding_norm_layer = torch.nn.LayerNorm(2 * output_feature_dim, self.layer_norm_eps)
        self.dense = nn.Linear(2 * output_feature_dim, 2 * output_feature_dim, bias=False)
        # self.dense_output = nn.Linear(output_feature_dim, output_feature_dim, bias=False)
        # self.output_norm_layer = torch.nn.LayerNorm(output_feature_dim, self.layer_norm_eps)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.nheads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, node_emebdding=None, question_emebdding=None, attention_mask_entity_number_type=None,
                attention_mask_number_number_types=None, entity_node_mask=None, number_node_mask=None):
        node_encoding = node_emebdding
        for t in range(self.iter_step):
            m_t = self.w_dc[t](F.elu(self.W_fc(question_emebdding)))
            input_encoding = torch.cat([node_encoding, node_emebdding], dim=-1)
            input_encoding = self.dense(input_encoding)
            input_encoding = F.dropout(input_encoding, self.drop_out, training=self.training)
            input_encoding = self.input_encoding_norm_layer(input_encoding)
            if self.qd:
                x_q = self.W_qv(input_encoding) * self.W_qc(m_t)
                x_k = self.W_kv(input_encoding) * self.W_kc(m_t)
                x_v = self.W_vv(input_encoding) * self.W_vc(m_t)
                x_q = self.x_q_norm_layer(x_q)
                x_k = self.x_k_norm_layer(x_k)
                x_v = self.x_v_norm_layer(x_v)

            else:

                x_q = self.W_qv(input_encoding)
                x_k = self.W_kv(input_encoding)
                x_v = self.W_vv(input_encoding)
                # x_q = self.x_q_norm_layer(x_q)
                # x_k = self.x_k_norm_layer(x_k)
                # x_v = self.x_v_norm_layer(x_v)

            x_q = self.transpose_for_scores(x_q)  # [bsz, N, nhead, h] -> [bsz,nhead, N, h]
            x_k = self.transpose_for_scores(x_k)  # [bsz, N, nhead, h] -> [bsz,nhead, N, h]
            x_v = self.transpose_for_scores(x_v)  # [bsz, N, nhead, h] -> [bsz,nhead, N, h]

            entity_number_att = self.entity_number_attention(x_q, x_k, attention_mask_entity_number_type)
            attention_mask = torch.zeros_like(attention_mask_entity_number_type)
            attention_score = torch.zeros_like(entity_number_att)
            attention_score += entity_number_att
            for number_type in self.NUMBER_TOKEN_TYPE:
                attention_mask_type = attention_mask_number_number_types[number_type]
                number_number_type_attention = self.number_number_types_attention[number_type]
                number_number_type_att = number_number_type_attention(x_q, x_k, attention_mask_type)
                attention_score += number_number_type_att
                attention_mask += attention_mask_type
            attention_score = F.leaky_relu(attention_score)
            attention_mask += attention_mask_entity_number_type
            assert torch.max(attention_mask) <= 1

            attention_probs = util.masked_softmax(attention_score,
                                                  attention_mask.unsqueeze(1).expand(-1, self.nheads, -1, -1), dim=-1)
            zero_attention_probs = torch.zeros_like(attention_probs)
            attention_probs = torch.where(attention_mask.unsqueeze(1).expand(-1, self.nheads, -1, -1) > 0,
                                          attention_probs, zero_attention_probs)
            attention_probs = F.dropout(attention_probs, self.drop_out, training=self.training)

            X_t = torch.matmul(attention_probs, x_v)  # bsz, nheads, N , H
            X_t = X_t.permute(0, 2, 1, 3).contiguous()  # bsz, N , nheads, H
            new_context_layer_shape = X_t.size()[:-2] + (self.output_feature_dim,)
            X_t = X_t.view(*new_context_layer_shape)  # bsz, N, nheads*H
            # X_t = F.relu(X_t)
            node_encoding = torch.cat([node_encoding, X_t], dim=-1)
            node_encoding = self.W_u(node_encoding)
            node_encoding = F.dropout(node_encoding, self.drop_out, training=self.training)
            node_encoding = self.norm_layer(node_encoding)

            # # intermediate layer
            # intermediate_state = self.dense(node_encoding)
            # intermediate_state = F.relu(intermediate_state)
            #
            #
            # ## output layer
            #
            # output_state = self.dense_output(intermediate_state)
            # output_state = F.dropout(output_state,self.drop_out, training=self.training)
            # node_encoding = self.output_norm_layer(node_encoding+output_state)

        return node_encoding


class GCN(nn.Module):

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(GCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._dd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)

        self._dd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)

    def forward(self, d_node, q_node, d_node_mask, q_node_mask, graph,
                extra_factor=None):

        d_node_len = d_node.size(1)
        q_node_len = q_node.size(1)

        diagmat = torch.diagflat(torch.ones(d_node.size(1), dtype=torch.long, device=d_node.device))
        diagmat = diagmat.unsqueeze(0).expand(d_node.size(0), -1, -1)
        dd_graph = d_node_mask.unsqueeze(1) * d_node_mask.unsqueeze(-1) * (1 - diagmat)
        dd_graph_left = dd_graph * graph[:, :d_node_len, :d_node_len]
        dd_graph_right = dd_graph * (1 - graph[:, :d_node_len, :d_node_len])

        diagmat = torch.diagflat(torch.ones(q_node.size(1), dtype=torch.long, device=q_node.device))
        diagmat = diagmat.unsqueeze(0).expand(q_node.size(0), -1, -1)
        qq_graph = q_node_mask.unsqueeze(1) * q_node_mask.unsqueeze(-1) * (1 - diagmat)
        qq_graph_left = qq_graph * graph[:, d_node_len:, d_node_len:]
        qq_graph_right = qq_graph * (1 - graph[:, d_node_len:, d_node_len:])

        dq_graph = d_node_mask.unsqueeze(-1) * q_node_mask.unsqueeze(1)
        dq_graph_left = dq_graph * graph[:, :d_node_len, d_node_len:]
        dq_graph_right = dq_graph * (1 - graph[:, :d_node_len, d_node_len:])

        qd_graph = q_node_mask.unsqueeze(-1) * d_node_mask.unsqueeze(1)
        qd_graph_left = qd_graph * graph[:, d_node_len:, :d_node_len]
        qd_graph_right = qd_graph * (1 - graph[:, d_node_len:, :d_node_len])

        d_node_neighbor_num = dd_graph_left.sum(-1) + dd_graph_right.sum(-1) + dq_graph_left.sum(
            -1) + dq_graph_right.sum(-1)
        d_node_neighbor_num_mask = (d_node_neighbor_num >= 1).long()
        d_node_neighbor_num = util.replace_masked_values(d_node_neighbor_num.float(), d_node_neighbor_num_mask, 1)

        q_node_neighbor_num = qq_graph_left.sum(-1) + qq_graph_right.sum(-1) + qd_graph_left.sum(
            -1) + qd_graph_right.sum(-1)
        q_node_neighbor_num_mask = (q_node_neighbor_num >= 1).long()
        q_node_neighbor_num = util.replace_masked_values(q_node_neighbor_num.float(), q_node_neighbor_num_mask, 1)

        all_d_weight, all_q_weight = [], []
        for step in range(self.iteration_steps):
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(d_node)).squeeze(-1)
                q_node_weight = torch.sigmoid(self._node_weight_fc(q_node)).squeeze(-1)
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((d_node, extra_factor), dim=-1))).squeeze(
                    -1)
                q_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((q_node, extra_factor), dim=-1))).squeeze(
                    -1)

            all_d_weight.append(d_node_weight)
            all_q_weight.append(q_node_weight)

            self_d_node_info = self._self_node_fc(d_node)
            self_q_node_info = self._self_node_fc(q_node)

            dd_node_info_left = self._dd_node_fc_left(d_node)
            qd_node_info_left = self._qd_node_fc_left(d_node)
            qq_node_info_left = self._qq_node_fc_left(q_node)
            dq_node_info_left = self._dq_node_fc_left(q_node)

            dd_node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                dd_graph_left,
                0)

            qd_node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                qd_graph_left,
                0)

            qq_node_weight = util.replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                qq_graph_left,
                0)

            dq_node_weight = util.replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                dq_graph_left,
                0)

            dd_node_info_left = torch.matmul(dd_node_weight, dd_node_info_left)
            qd_node_info_left = torch.matmul(qd_node_weight, qd_node_info_left)
            qq_node_info_left = torch.matmul(qq_node_weight, qq_node_info_left)
            dq_node_info_left = torch.matmul(dq_node_weight, dq_node_info_left)

            dd_node_info_right = self._dd_node_fc_right(d_node)
            qd_node_info_right = self._qd_node_fc_right(d_node)
            qq_node_info_right = self._qq_node_fc_right(q_node)
            dq_node_info_right = self._dq_node_fc_right(q_node)

            dd_node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                dd_graph_right,
                0)

            qd_node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                qd_graph_right,
                0)

            qq_node_weight = util.replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                qq_graph_right,
                0)

            dq_node_weight = util.replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                dq_graph_right,
                0)

            dd_node_info_right = torch.matmul(dd_node_weight, dd_node_info_right)
            qd_node_info_right = torch.matmul(qd_node_weight, qd_node_info_right)
            qq_node_info_right = torch.matmul(qq_node_weight, qq_node_info_right)
            dq_node_info_right = torch.matmul(dq_node_weight, dq_node_info_right)

            agg_d_node_info = (
                                      dd_node_info_left + dd_node_info_right + dq_node_info_left + dq_node_info_right) / d_node_neighbor_num.unsqueeze(
                -1)
            agg_q_node_info = (
                                      qq_node_info_left + qq_node_info_right + qd_node_info_left + qd_node_info_right) / q_node_neighbor_num.unsqueeze(
                -1)

            d_node = F.relu(self_d_node_info + agg_d_node_info)
            q_node = F.relu(self_q_node_info + agg_q_node_info)

        all_d_weight = [weight.unsqueeze(1) for weight in all_d_weight]
        all_q_weight = [weight.unsqueeze(1) for weight in all_q_weight]

        all_d_weight = torch.cat(all_d_weight, dim=1)
        all_q_weight = torch.cat(all_q_weight, dim=1)

        return d_node, q_node, all_d_weight, all_q_weight  # d_node_weight, q_node_weight


class DigitalGCN(nn.Module):
    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(DigitalGCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)

    def forward(self, node, node_mask, graph, extra_factor=None):

        # d_node_len = d_node.size(1)
        # q_node_len = q_node.size(1)
        node_len = node.size(1)
        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long, device=node.device))
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)

        node_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)
        node_graph_left = node_graph * graph[:, :node_len, :node_len]
        node_graph_right = node_graph * (1 - graph[:, :node_len, :node_len])
        node_neighbor_num = node_graph_left.sum(-1) + node_graph_right.sum(-1)

        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        all_node_weight = []

        for step in range(self.iteration_steps):
            if extra_factor is None:
                node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)
            else:
                node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(
                    -1)

            all_node_weight.append(node_weight)
            self_node_info = self._self_node_fc(node)

            node_node_info_left = self._node_fc_left(node)
            node_node_weight = util.replace_masked_values(
                node_weight.unsqueeze(1).expand(-1, node_len, -1),
                node_graph_left,
                0)

            node_node_info_left = torch.matmul(node_node_weight, node_node_info_left)

            node_node_info_right = self._node_fc_right(node)
            node_node_weight = util.replace_masked_values(
                node_weight.unsqueeze(1).expand(-1, node_len, -1),
                node_graph_right,
                0)
            node_node_info_right = torch.matmul(node_node_weight, node_node_info_right)

            agg_node_info = (node_node_info_left + node_node_info_right) / node_neighbor_num.unsqueeze(-1)

            node = F.relu(self_node_info + agg_node_info)

        all_node_weight = [weight.unsqueeze(1) for weight in all_node_weight]
        all_node_weight = torch.cat(all_node_weight, dim=1)

        return node, all_node_weight


class HeterogeneousGCN(nn.Module):
    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(HeterogeneousGCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim, 1, bias=True)

        self._self_entity_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._self_value_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._self_sentence_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._self_q_entity_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._self_q_value_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._self_q_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._sentence_entity_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._sentence_value_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._entity_value_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._entity_entity_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        # self._same_entity_entity_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        # self._same_entity_qentity_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._entity_sentence_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._value_sentence_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._value_entity_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._question_entity_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._question_value_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._question_entity_value_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._question_entity_entity_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._entity_question_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._value_question_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._question_value_entity_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

    def forward(self, sentence_node,
                entity_node,
                value_node,
                sentences_values_relation,
                entities_entities_relation,
                entities_values_relation,
                sentences_entities_relation,
                # same_entity_mention_relation,
                sentences_node_mask,
                entities_node_mask,
                values_node_mask,

                question_node,
                question_entity_node,
                question_value_node,
                question_entities_relation,
                question_values_relation,
                question_entities_entities_relation,
                question_entities_values_relation,
                # same_entity_mention_with_q_relation,
                question_entity_mask,
                question_value_mask

                ):
        sentence_node_len = sentence_node.size(1)
        entity_node_len = entity_node.size(1)
        value_node_len = value_node.size(1)

        question_node_len = question_node.size(1)
        question_entity_node_len = question_entity_node.size(1)
        question_value_node_len = question_value_node.size(1)

        for step in range(self.iteration_steps):
            entity_node_weight = torch.sigmoid(self._node_weight_fc(entity_node)).squeeze(-1)
            sentence_node_weight = torch.sigmoid(self._node_weight_fc(sentence_node)).squeeze(-1)
            value_node_weight = torch.sigmoid(self._node_weight_fc(value_node)).squeeze(-1)

            question_entity_node_weight = torch.sigmoid(self._node_weight_fc(question_entity_node)).squeeze(-1)
            question_node_weight = torch.sigmoid(self._node_weight_fc(question_node)).squeeze(-1)
            question_value_node_weight = torch.sigmoid(self._node_weight_fc(question_value_node)).squeeze(-1)

            self_entity_node_info = self._self_entity_node_fc(entity_node)
            self_value_node_info = self._self_value_node_fc(value_node)
            self_sentence_node_info = self._self_sentence_node_fc(sentence_node)

            self_question_entity_node_info = self._self_q_entity_node_fc(question_entity_node)
            self_question_value_node_info = self._self_q_value_node_fc(question_value_node)
            self_question_node_info = self._self_q_node_fc(question_node)

            sentence_entity_weight = util.replace_masked_values(
                entity_node_weight.unsqueeze(1).expand(-1, sentence_node_len, -1), sentences_entities_relation, 0)

            entity_sentence_weight = util.replace_masked_values(
                sentence_node_weight.unsqueeze(1).expand(-1, entity_node_len, -1),
                torch.transpose(sentences_entities_relation, dim0=1, dim1=-1), 0)

            sentence_value_weight = util.replace_masked_values(
                value_node_weight.unsqueeze(1).expand(-1, sentence_node_len, -1), sentences_values_relation, 0)

            value_sentence_weight = util.replace_masked_values(
                sentence_node_weight.unsqueeze(1).expand(-1, value_node_len, -1),
                torch.transpose(sentences_values_relation, dim0=1, dim1=-1), 0)

            entity_entity_weight = util.replace_masked_values(
                entity_node_weight.unsqueeze(1).expand(-1, entity_node_len, -1), entities_entities_relation, 0)

            # same_entity_entity_weight = util.replace_masked_values(
            #     entity_node_weight.unsqueeze(1).expand(-1, entity_node_len, -1), same_entity_mention_relation, 0)

            entity_value_weight = util.replace_masked_values(
                value_node_weight.unsqueeze(1).expand(-1, entity_node_len, -1), entities_values_relation, 0)

            value_entity_weight = util.replace_masked_values(
                entity_node_weight.unsqueeze(1).expand(-1, value_node_len, -1),
                torch.transpose(entities_values_relation, dim0=1, dim1=-1), 0)

            question_entity_weight = util.replace_masked_values(
                question_entity_node_weight.unsqueeze(1).expand(-1, question_node_len, -1), question_entities_relation,
                0)

            entity_question_weight = util.replace_masked_values(
                question_node_weight.unsqueeze(1).expand(-1, question_entity_node_len, -1),
                torch.transpose(question_entities_relation, dim0=1, dim1=-1), 0)

            question_value_weight = util.replace_masked_values(
                question_value_node_weight.unsqueeze(1).expand(-1, question_node_len, -1), question_values_relation, 0)

            value_question_weight = util.replace_masked_values(
                question_node_weight.unsqueeze(1).expand(-1, question_value_node_len, -1),
                torch.transpose(question_values_relation, dim0=1, dim1=-1), 0)

            question_entity_entity_weight = util.replace_masked_values(
                question_entity_node_weight.unsqueeze(1).expand(-1, question_entity_node_len, -1),
                question_entities_entities_relation, 0)

            # same_entity_qentity_weight = util.replace_masked_values(
            #     question_entity_node_weight.unsqueeze(1).expand(-1, entity_node_len, -1),
            #     same_entity_mention_with_q_relation, 0)

            # same_qentity_entity_weight = util.replace_masked_values(
            #     entity_node_weight.unsqueeze(1).expand(-1, question_entity_node_len, -1),
            #     torch.transpose(same_entity_mention_with_q_relation, dim0=1, dim1=-1), 0)

            question_entity_value_weight = util.replace_masked_values(
                question_value_node_weight.unsqueeze(1).expand(-1, question_entity_node_len, -1),
                question_entities_values_relation, 0)

            question_value_entity_weight = util.replace_masked_values(
                question_entity_node_weight.unsqueeze(1).expand(-1, question_value_node_len, -1),
                torch.transpose(question_entities_values_relation, dim0=1, dim1=-1), 0)

            sentence_entity_info = self._sentence_entity_fc(entity_node)
            entity_sentence_info = self._entity_sentence_fc(sentence_node)
            sentence_value_info = self._sentence_value_fc(value_node)
            value_sentence_info = self._value_sentence_fc(sentence_node)
            entity_entity_info = self._entity_entity_fc(entity_node)
            # same_entity_entity_info = self._same_entity_entity_fc(entity_node)
            entity_value_info = self._entity_value_fc(value_node)
            value_entity_info = self._value_entity_fc(entity_node)

            question_entity_info = self._question_entity_fc(question_entity_node)
            entity_question_info = self._entity_question_fc(question_node)
            question_value_info = self._question_value_fc(question_value_node)
            value_question_info = self._value_question_fc(question_node)
            question_entity_entity_info = self._question_entity_entity_fc(question_entity_node)
            question_entity_value_info = self._question_entity_value_fc(question_value_node)
            question_value_entity_info = self._question_value_entity_fc(question_entity_node)

            # same_entity_qentity_info = self._same_entity_qentity_fc(question_entity_node)
            # same_qentity_entity_info = self._same_entity_entity_fc(entity_node)

            sentence_entity_info = torch.matmul(sentence_entity_weight, sentence_entity_info)
            entity_sentence_info = torch.matmul(entity_sentence_weight, entity_sentence_info)
            sentence_value_info = torch.matmul(sentence_value_weight, sentence_value_info)
            value_sentence_info = torch.matmul(value_sentence_weight, value_sentence_info)
            entity_entity_info = torch.matmul(entity_entity_weight, entity_entity_info)
            # same_entity_entity_info = torch.matmul(same_entity_entity_weight, same_entity_entity_info)
            entity_value_info = torch.matmul(entity_value_weight, entity_value_info)
            value_entity_info = torch.matmul(value_entity_weight, value_entity_info)

            question_entity_info = torch.matmul(question_entity_weight, question_entity_info)
            entity_question_info = torch.matmul(entity_question_weight, entity_question_info)
            question_value_info = torch.matmul(question_value_weight, question_value_info)
            value_question_info = torch.matmul(value_question_weight, value_question_info)
            question_entity_entity_info = torch.matmul(question_entity_entity_weight, question_entity_entity_info)
            question_entity_value_info = torch.matmul(question_entity_value_weight, question_entity_value_info)
            question_value_entity_info = torch.matmul(question_value_entity_weight, question_value_entity_info)

            # same_entity_qentity_info = torch.matmul(same_entity_qentity_weight, same_entity_qentity_info)
            # same_qentity_entity_info = torch.matmul(same_qentity_entity_weight, same_qentity_entity_info)

            sentence_node_neighbor_num = sentences_entities_relation.sum(-1) + sentences_values_relation.sum(-1)
            sentence_node_neighbor_num_mask = (sentence_node_neighbor_num >= 1).long()
            sentence_node_neighbor_num = util.replace_masked_values(sentence_node_neighbor_num.float(),
                                                                    sentence_node_neighbor_num_mask, 1)
            #
            # entity_node_neighbor_num = torch.transpose(sentences_entities_relation, dim0=1, dim1=-1).sum(
            #     -1) + entities_entities_relation.sum(-1) + entities_values_relation.sum(
            #     -1) + same_entity_mention_relation.sum(-1) + same_entity_mention_with_q_relation.sum(-1)

            entity_node_neighbor_num = torch.transpose(sentences_entities_relation, dim0=1, dim1=-1).sum(
                -1) + entities_entities_relation.sum(-1) + entities_values_relation.sum(
                -1)

            entity_node_neighbor_num_mask = (entity_node_neighbor_num >= 1).long()
            entity_node_neighbor_num = util.replace_masked_values(entity_node_neighbor_num.float(),
                                                                  entity_node_neighbor_num_mask, 1)

            value_node_neighbor_num = torch.transpose(sentences_values_relation, dim0=1, dim1=-1).sum(
                -1) + torch.transpose(entities_values_relation, dim0=1, dim1=-1).sum(-1)
            value_node_neighbor_num_mask = (value_node_neighbor_num >= 1).long()
            value_node_neighbor_num = util.replace_masked_values(value_node_neighbor_num.float(),
                                                                 value_node_neighbor_num_mask, 1)

            question_node_neighbor_num = question_entities_relation.sum(-1) + question_values_relation.sum(-1)
            question_node_neighbor_num_mask = (question_node_neighbor_num >= 1).long()
            question_node_neighbor_num = util.replace_masked_values(question_node_neighbor_num.float(),
                                                                    question_node_neighbor_num_mask, 1)

            # question_entity_node_neighbor_num = torch.transpose(question_entities_relation, dim0=1, dim1=-1).sum(
            #     -1) + question_entities_entities_relation.sum(-1) + question_entities_values_relation.sum(
            #     -1) + torch.transpose(same_entity_mention_with_q_relation, dim0=1, dim1=-1).sum(-1)

            question_entity_node_neighbor_num = torch.transpose(question_entities_relation, dim0=1, dim1=-1).sum(
                -1) + question_entities_entities_relation.sum(-1) + question_entities_values_relation.sum(
                -1)

            question_entity_node_neighbor_num_mask = (question_entity_node_neighbor_num >= 1).long()
            question_entity_node_neighbor_num = util.replace_masked_values(question_entity_node_neighbor_num.float(),
                                                                           question_entity_node_neighbor_num_mask, 1)

            question_value_node_neighbor_num = torch.transpose(question_values_relation, dim0=1, dim1=-1).sum(
                -1) + torch.transpose(question_entities_values_relation, dim0=1, dim1=-1).sum(-1)
            question_value_node_neighbor_num_mask = (question_value_node_neighbor_num >= 1).long()
            question_value_node_neighbor_num = util.replace_masked_values(question_value_node_neighbor_num.float(),
                                                                          question_value_node_neighbor_num_mask, 1)

            agg_sentence_node_info = (
                                             sentence_entity_info + sentence_value_info) / sentence_node_neighbor_num.unsqueeze(
                -1)
            # agg_entity_node_info = (
            #                                entity_sentence_info + entity_value_info + entity_entity_info + same_entity_entity_info + same_entity_qentity_info) / entity_node_neighbor_num.unsqueeze(
            #     -1)

            agg_entity_node_info = (
                                           entity_sentence_info + entity_value_info + entity_entity_info) / entity_node_neighbor_num.unsqueeze(
                -1)

            agg_value_node_info = (value_sentence_info + value_entity_info) / value_node_neighbor_num.unsqueeze(-1)

            agg_question_node_info = (
                                             question_entity_info + question_value_info) / question_node_neighbor_num.unsqueeze(
                -1)
            # agg_question_entity_node_info = (
            #                                         entity_question_info + question_entity_value_info + question_entity_entity_info + same_qentity_entity_info) / question_entity_node_neighbor_num.unsqueeze(
            #     -1)

            agg_question_entity_node_info = (
                                                    entity_question_info + question_entity_value_info) / question_entity_node_neighbor_num.unsqueeze(
                -1)

            agg_question_value_node_info = (
                                                   value_question_info + question_value_entity_info) / question_value_node_neighbor_num.unsqueeze(
                -1)

            sentence_node = F.relu(self_sentence_node_info + agg_sentence_node_info)
            entity_node = F.relu(self_entity_node_info + agg_entity_node_info)
            value_node = F.relu(self_value_node_info + agg_value_node_info)

            question_node = F.relu(self_question_node_info + agg_question_node_info)
            question_entity_node = F.relu(self_question_entity_node_info + agg_question_entity_node_info)
            question_value_node = F.relu(self_question_value_node_info + agg_question_value_node_info)

            return sentence_node, entity_node, value_node, question_node, question_entity_node, question_value_node


class KV_Relation_module(nn.Module):
    def __init__(self, node_dim, iteration_steps=1):
        super(KV_Relation_module, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim, 1, bias=True)

        self._self_entity_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._self_value_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._entity_value_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._value_entity_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._entity_proj = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._value_proj = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._question_proj = torch.nn.Linear(node_dim, node_dim, bias=True)

    def forward(self,
                entity_node,
                value_node,
                question_node,
                entities_values_relation,
                entities_node_mask,
                values_node_mask,
                ):
        entity_node_len = entity_node.size(1)
        value_node_len = value_node.size(1)

        for step in range(self.iteration_steps):

#             # q_aware_entity_node = torch.cat([entity_node, question_node.expand(-1, entity_node.size(1), -1)], dim=-1)
#             # q_aware_value_node = torch.cat([value_node, question_node.expand(-1, value_node.size(1), -1)], dim=-1)

#             q_aware_entity_node = self._entity_proj(entity_node) * self._question_proj(question_node)
#             q_aware_value_node = self._value_proj(value_node) * self._question_proj(question_node)

#             entity_node_weight = torch.sigmoid(self._node_weight_fc(q_aware_entity_node)).squeeze(-1)
#             value_node_weight = torch.sigmoid(self._node_weight_fc(q_aware_value_node)).squeeze(-1)
            entity_node_weight = torch.sigmoid(self._node_weight_fc(entity_node)).squeeze(-1)
            value_node_weight = torch.sigmoid(self._node_weight_fc(value_node)).squeeze(-1)

            self_entity_node_info = self._self_entity_node_fc(entity_node)
            self_value_node_info = self._self_value_node_fc(value_node)

            entity_value_weight = util.replace_masked_values(
                value_node_weight.unsqueeze(1).expand(-1, entity_node_len, -1), entities_values_relation, 0)

            value_entity_weight = util.replace_masked_values(
                entity_node_weight.unsqueeze(1).expand(-1, value_node_len, -1),
                torch.transpose(entities_values_relation, dim0=1, dim1=-1), 0)

            entity_value_info = self._entity_value_fc(value_node)
            value_entity_info = self._value_entity_fc(entity_node)

            entity_value_info = torch.matmul(entity_value_weight, entity_value_info)
            value_entity_info = torch.matmul(value_entity_weight, value_entity_info)

            entity_node_neighbor_num = entities_values_relation.sum(-1)

            entity_node_neighbor_num_mask = (entity_node_neighbor_num >= 1).long()
            entity_node_neighbor_num = util.replace_masked_values(entity_node_neighbor_num.float(),
                                                                  entity_node_neighbor_num_mask, 1)

            value_node_neighbor_num = torch.transpose(entities_values_relation, dim0=1, dim1=-1).sum(-1)
            value_node_neighbor_num_mask = (value_node_neighbor_num >= 1).long()
            value_node_neighbor_num = util.replace_masked_values(value_node_neighbor_num.float(),
                                                                 value_node_neighbor_num_mask, 1)

            agg_entity_node_info = (entity_value_info) / entity_node_neighbor_num.unsqueeze(-1)

            agg_value_node_info = (value_entity_info) / value_node_neighbor_num.unsqueeze(-1)

            entity_node = F.relu(self_entity_node_info + agg_entity_node_info)
            value_node = F.relu(self_value_node_info + agg_value_node_info)

            return entity_node, value_node
