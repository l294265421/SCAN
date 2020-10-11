# -*- coding: utf-8 -*-


from typing import *
from overrides import overrides
import time
import copy
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.seq2vec_encoders import CnnEncoder as VectorCnnEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.fields import TextField, MetadataField, ArrayField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
import torch.nn.functional as F
from allennlp.training import metrics
from allennlp.models import BasicClassifier
from allennlp.modules import attention
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from scipy.special import expit
from allennlp.nn import util as allennlp_util
import dgl
from dgl import function as dgl_fn
from sklearn.metrics import f1_score, precision_score, recall_score

from nlp_tasks.utils import attention_visualizer
from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification import allennlp_metrics
from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification.cnn_encoder_seq2seq import CnnEncoder


class AttentionInHtt(nn.Module):
    """
    2016-Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, in_features, out_features, bias=True, softmax=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias)
        self.uw = nn.Linear(out_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        u = self.W(h)
        u = torch.tanh(u)
        similarities = self.uw(u)
        similarities = similarities.squeeze(dim=-1)
        if self.softmax:
            alpha = allennlp_util.masked_softmax(similarities, mask)
            return alpha
        else:
            return similarities


class DotProductAttentionInHtt(nn.Module):
    """
    2016-Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, in_features, out_features, bias=True, softmax=True):
        super().__init__()
        self.uw = nn.Linear(in_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        similarities = self.uw(h)
        similarities = similarities.squeeze(dim=-1)
        if self.softmax:
            alpha = allennlp_util.masked_softmax(similarities, mask)
            return alpha
        else:
            return similarities


class AverageAttention(nn.Module):
    """
    2019-emnlp-Attention is not not Explanation
    """

    def __init__(self):
        super().__init__()

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        alpha = allennlp_util.masked_softmax(mask, mask)
        return alpha
        
        
class BernoulliAttentionInHtt(nn.Module):
    """
    2016-Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias)
        self.uw = nn.Linear(out_features, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        u = self.W(h)
        u = torch.tanh(u)
        similarities = self.uw(u)
        similarities = similarities.squeeze(dim=-1)
        alpha = torch.sigmoid(similarities)
        return alpha


class AttentionInCan(nn.Module):
    """
    2019-emnlp-CAN Constrained Attention Networks for Multi-Aspect Sentiment Analysis
    """

    def __init__(self, in_features, bias=True, softmax=True):
        super().__init__()
        self.W1 = nn.Linear(in_features, in_features, bias)
        self.W2 = nn.Linear(in_features, in_features, bias)
        self.uw = nn.Linear(in_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, mask: torch.Tensor):
        u1 = self.W1(h1)
        u2 = self.W2(h2)
        u = u1 + u2
        u = torch.tanh(u)
        similarities = self.uw(u)
        similarities = similarities.squeeze(dim=-1)
        if self.softmax:
            alpha = allennlp_util.masked_softmax(similarities, mask)
            return alpha
        else:
            return similarities


class LocationMaskLayer(nn.Module):
    """
    2017-CIKM-Aspect-level Sentiment Classification with HEAT (HiErarchical ATtention) Network
    """

    def __init__(self, location_num, configuration):
        super().__init__()
        self.location_num = location_num
        self.configuration = configuration

    def forward(self, alpha: torch.Tensor):
        location_num = self.location_num
        location_matrix = torch.zeros([location_num, location_num], dtype=torch.float,
                                      device=self.configuration['device'],
                                      requires_grad=False)
        for i in range(location_num):
            for j in range(location_num):
                location_matrix[i, j] = 1 - (abs(i - j) / location_num)
        result = alpha.mm(location_matrix)
        return result


class TextInAllAspectSentimentOutModel(Model):

    def __init__(self, vocab: Vocabulary, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab)
        self.category_loss_weight = category_loss_weight
        self.sentiment_loss_weight = sentiment_loss_weight

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def reduce(self, nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def pad_dgl_graph(self, graphs, max_node_num):
        graphs_padded = []
        for graph in graphs:
            graph_padded = copy.deepcopy(graph)
            node_num = graph.number_of_nodes()
            graph_padded.add_nodes(max_node_num - node_num)
            graphs_padded.append(graph_padded)
        return graphs_padded

    def no_grad_for_acd_parameter(self):
        self.set_grad_for_acd_parameter(requires_grad=False)

    def set_grad_for_acd_parameter(self, requires_grad=True):
        pass

    def set_grad_for_acsc_parameter(self, requires_grad=True):
        pass

    def _get_model_visualization_picture_filepath(self, configuration: dict, words: list):
        savefig_dir = configuration['savefig_dir']
        if not savefig_dir:
            return None
        filename = '%s-%s.svg' % ('-'.join(words[:3]), str(time.time()))
        filename = re.sub('/', '', filename)
        return os.path.join(savefig_dir, filename)


from dgl import DGLGraph


class GatEdge:
    """
    story information of a edge in gat for visualization
    """

    def __init__(self, src_ids: List[int], dst_id: int, alphas: List[float]):
        self.dst_id = dst_id
        self.src_ids = src_ids
        self.alphas = alphas

    def add(self, other_edge: 'GatEdge'):
        """

        :param other_edge:  同样的边
        :return:
        """
        if self.src_ids != other_edge.src_ids or self.dst_id != other_edge.dst_id:
            print('add error')
            return
        for i in range(len(self.alphas)):
            self.alphas[i] += other_edge.alphas[i]

    def divide(self, number: int):
        self.alphas = [alpha / number for alpha in self.alphas]

    def __str__(self):
        return '%s-%s-%s' % (str(self.dst_id), str(self.src_ids), str(self.alphas))


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, opt: dict={}):
        super().__init__()
        self.opt = opt
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges: dgl.EdgeBatch):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges: dgl.EdgeBatch):
        # message UDF for equation (3) & (4)
        result = {'z': edges.src['z'], 'node_ids': edges.src['node_ids'], 'e': edges.data['e']}
        return result

    def reduce_func(self, nodes: dgl.NodeBatch):
        # for visualization
        src_node_ids = nodes.mailbox['node_ids']
        dst_node_ids = nodes.data['node_ids']

        # reduce UDF for equation (3) & (4)
        # equation (3)

        e = nodes.mailbox['e']
        alpha = F.softmax(e, dim=1)
        # equation (4)
        z = nodes.mailbox['z']
        h = torch.sum(alpha * z, dim=1)

        self.for_visualization.append({'src_node_ids': src_node_ids, 'dst_node_ids': dst_node_ids,
                                       'alpha': alpha})

        return {'h': h}

    def forward(self, h, g: List[DGLGraph]):
        self.for_visualization = []

        batched_graph = dgl.batch(g)
        feature = h.view([-1, h.size()[-1]])

        # equation (1)
        z = self.fc(feature)
        batched_graph.ndata['z'] = z

        # 加入node id，用于attention可视化
        node_ids = batched_graph.nodes()
        batched_graph.ndata['node_ids'] = node_ids

        # equation (2)
        batched_graph.apply_edges(self.edge_attention)
        # equation (3) & (4)
        batched_graph.update_all(self.message_func, self.reduce_func)

        ug = dgl.unbatch(batched_graph)
        # output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        # for visualization in local machine
        output = [torch.unsqueeze(g.ndata.pop('h'), 0) for g in ug]
        output = torch.cat(output, 0)

        # 对for_visualization按照sample进行拆分
        sample_num = len(g)
        node_num_per_sample = h.shape[1]
        edges_of_samples = [[] for _ in range(sample_num)]
        if 'gat_visualization' in self.opt and self.opt['gat_visualization']:
            for edges in self.for_visualization:
                edge_num = edges['dst_node_ids'].shape[0]
                for i in range(edge_num):
                    src_ids = edges['src_node_ids'][i].cpu().numpy().tolist()
                    src_ids_real = [e % node_num_per_sample for e in src_ids]
                    dst_id = edges['dst_node_ids'][i].cpu().numpy().tolist()
                    dst_id_real = dst_id % node_num_per_sample
                    sample_index = dst_id // node_num_per_sample
                    alphas = edges['alpha'][i].squeeze(dim=-1).cpu().numpy().tolist()
                    edge = GatEdge(src_ids_real, dst_id_real, alphas)
                    edges_of_samples[sample_index].append(edge)

        return output, edges_of_samples


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', opt: dict={}):
        super().__init__()
        self.opt = opt
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, opt))
        self.merge = merge

    def forward(self, h, g: List[DGLGraph]):
        head_outs = [attn_head(h, g) for attn_head in self.heads]
        head_outs_feature = [e[0] for e in head_outs]
        head_outs_attention = [e[1] for e in head_outs]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs_feature, dim=2), head_outs_attention
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs_feature)), head_outs_attention


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, opt: dict={}):
        super().__init__()
        self.opt = opt
        self.layer1 = MultiHeadGATLayer(in_dim, int(out_dim / num_heads), num_heads, opt=opt)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        # self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, h, g: List[DGLGraph]):
        h, attention = self.layer1(h, g)
        # h = F.elu(h)
        # h = self.layer2(h, g)
        # 兼容之前保存的模型
        # return h, attention
        if 'gat_visualization' in self.opt and self.opt['gat_visualization']:
            return h, attention
        else:
            return h


class GATForAspectCategory(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=-1, num_heads=4, opt: dict={}):
        super().__init__()
        self.opt = opt
        self.layer1 = MultiHeadGATLayer(in_dim, int(out_dim / num_heads), num_heads, opt=opt)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        # self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, h, g: List[DGLGraph]):
        h, attention = self.layer1(h, g)
        # h = F.elu(h)
        # h = self.layer2(h, g)
        # return h, attention
        if 'gat_visualization' in self.opt and self.opt['gat_visualization']:
            return h, attention
        else:
            return h


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h': h}


class DglGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, opt, bias=True, activation=F.relu):
        super().__init__()
        self.opt = opt
        self.in_features = in_features
        self.out_features = out_features
        self.apply_mod = NodeApplyModule(in_features, out_features, activation, bias=bias)

    def _sum(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def _mean(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        mean = torch.mean(m, 1)
        return {'h': mean}

    def forward(self, text, graphs):
        hidden = text
        batched_graph = dgl.batch(graphs)
        feature = hidden.view([-1, hidden.size()[-1]])
        if feature.size()[0] != batched_graph.number_of_nodes():
            print('error')
        batched_graph.ndata['h'] = feature

        gcn_msg = dgl_fn.copy_src(src='h', out='m')
        gcn_reduce = self._mean

        batched_graph.update_all(gcn_msg, gcn_reduce)
        batched_graph.apply_nodes(func=self.apply_mod)

        ug = dgl.unbatch(batched_graph)
        output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        output = torch.cat(output, 0)
        return output


class DglGraphConvolutionForAspectCategory(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, opt, bias=True, activation=F.relu):
        super().__init__()
        self.opt = opt
        self.in_features = in_features
        self.out_features = out_features
        self.apply_mod = NodeApplyModule(in_features, out_features, activation, bias=bias)

    def _sum(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def _mean(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        mean = torch.mean(m, 1)
        return {'h': mean}

    def forward(self, text, graphs):
        hidden = text
        batched_graph = dgl.batch(graphs)
        feature = hidden.view([-1, hidden.size()[-1]])
        if feature.size()[0] != batched_graph.number_of_nodes():
            print('error')
        batched_graph.ndata['h'] = feature

        gcn_msg = dgl_fn.copy_src(src='h', out='m')
        gcn_reduce = self._mean

        batched_graph.update_all(gcn_msg, gcn_reduce)
        batched_graph.apply_nodes(func=self.apply_mod)

        ug = dgl.unbatch(batched_graph)
        output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        output = torch.cat(output, 0)
        return output


class DglGraphAverage(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, opt, bias=True, activation=F.relu):
        super().__init__()
        self.opt = opt
        self.in_features = in_features
        self.out_features = out_features

    def _sum(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def _mean(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        mean = torch.mean(m, 1)
        return {'h': mean}

    def forward(self, text, graphs):
        hidden = text
        batched_graph = dgl.batch(graphs)
        feature = hidden.view([-1, hidden.size()[-1]])
        if feature.size()[0] != batched_graph.number_of_nodes():
            print('error')
        batched_graph.ndata['h'] = feature

        gcn_msg = dgl_fn.copy_src(src='h', out='m')
        gcn_reduce = self._mean

        batched_graph.update_all(gcn_msg, gcn_reduce)

        ug = dgl.unbatch(batched_graph)
        output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        output = torch.cat(output, 0)
        return output


class DglGraphAverageForAspectCategory(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, opt, bias=True, activation=F.relu):
        super().__init__()
        self.opt = opt
        self.in_features = in_features
        self.out_features = out_features

    def _sum(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def _mean(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        mean = torch.mean(m, 1)
        return {'h': mean}

    def forward(self, text, graphs):
        hidden = text
        batched_graph = dgl.batch(graphs)
        feature = hidden.view([-1, hidden.size()[-1]])
        if feature.size()[0] != batched_graph.number_of_nodes():
            print('error')
        batched_graph.ndata['h'] = feature

        gcn_msg = dgl_fn.copy_src(src='h', out='m')
        gcn_reduce = self._mean

        batched_graph.update_all(gcn_msg, gcn_reduce)

        ug = dgl.unbatch(batched_graph)
        output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        output = torch.cat(output, 0)
        return output


class DglGraphAttentionForAspectCategory(nn.Module):
    """
    计算在所有aspect的attention下孩子节点的权重，所有权重求和然后归一化，即得到最终的
    孩子节点权重
    """
    def __init__(self, in_features, out_features, aspect_attentions, opt, bias=True, activation=F.relu):
        super().__init__()
        self.opt = opt
        self.aspect_attentions: List[AttentionInHtt] = aspect_attentions
        self.in_features = in_features
        self.out_features = out_features

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def _message_func(self, edges: dgl.EdgeBatch):
        result = {'m': edges.src['h'], 'm_sentiment': edges.src['h_sentiment']}
        return result

    def _sum(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def _mean(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        mean = torch.mean(m, 1)
        return {'h': mean}

    def _attention(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        alphas = []
        for i in range(len(self.aspect_attentions)):
            aspect_attention = self.aspect_attentions[i]
            alpha = aspect_attention(m, None)
            alphas.append(alpha.unsqueeze(1))
        alpha_cat = torch.cat(alphas, dim=1)
        alpha_final = torch.mean(alpha_cat, dim=1)
        h = self.element_wise_mul(m, alpha_final, return_not_sum_result=False)
        m_sentiment = nodes.mailbox['m_sentiment']
        h_sentiment = self.element_wise_mul(m_sentiment, alpha_final, return_not_sum_result=False)
        return {'h': h, 'h_sentiment': h_sentiment}

    def forward(self, aspect_representation, graphs, sentiment_representation=None):
        if sentiment_representation is None:
            sentiment_representation_flag = False
            sentiment_representation = aspect_representation
        else:
            sentiment_representation_flag = True
        hidden = aspect_representation
        batched_graph = dgl.batch(graphs)
        feature = hidden.view([-1, hidden.size()[-1]])
        feature_sentiment = sentiment_representation.view([-1, sentiment_representation.size()[-1]])
        if feature.size()[0] != batched_graph.number_of_nodes():
            print('error')
        batched_graph.ndata['h'] = feature
        batched_graph.ndata['h_sentiment'] = feature_sentiment

        # gcn_msg = dgl_fn.copy_src(src='h', out='m')
        gcn_msg = self._message_func
        gcn_reduce = self._attention

        batched_graph.update_all(gcn_msg, gcn_reduce)

        ug = dgl.unbatch(batched_graph)
        output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        output = torch.cat(output, 0)

        output_sentiment = [torch.unsqueeze(g.ndata.pop('h_sentiment'), 0).to(self.opt['device']) for g in ug]
        output_sentiment = torch.cat(output_sentiment, 0)
        if sentiment_representation_flag:
            return output, output_sentiment
        else:
            return output


class DglGraphAttentionForAspectCategoryWithDottedLines(nn.Module):
    """
    计算在所有aspect的attention下孩子节点的权重，所有权重求和然后归一化，即得到最终的
    孩子节点权重
    """
    def __init__(self, in_features, out_features, aspect_attentions, opt, bias=True, activation=F.relu):
        super().__init__()
        self.opt = opt
        self.aspect_attentions: List[AttentionInHtt] = aspect_attentions
        self.in_features = in_features
        self.out_features = out_features

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def _message_func(self, edges: dgl.EdgeBatch):
        result = {'m': edges.src['h'], 'dotted_line_masks': edges.data['dotted_line_masks'],
                  'm_sentiment': edges.src['h_sentiment']}
        return result

    def _sum(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        accum = torch.sum(m, 1)
        return {'h': accum}

    def _mean(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        mean = torch.mean(m, 1)
        return {'h': mean}

    def _attention(self, nodes: dgl.NodeBatch):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        m = nodes.mailbox['m']
        dotted_line_masks = nodes.mailbox['dotted_line_masks'].to(self.opt['device'])
        # for i in range(dotted_line_masks.size()[0]):
        #     print(dotted_line_masks[i])
        alphas = []
        for i in range(len(self.aspect_attentions)):
            aspect_attention = self.aspect_attentions[i]
            alpha_temp = aspect_attention(m, None)
            alpha = alpha_temp * dotted_line_masks.float()
            alphas.append(alpha.unsqueeze(1))
        alpha_cat = torch.cat(alphas, dim=1)
        alpha_final = torch.mean(alpha_cat, dim=1)
        h = self.element_wise_mul(m, alpha_final, return_not_sum_result=False)
        m_sentiment = nodes.mailbox['m_sentiment']
        h_sentiment = self.element_wise_mul(m_sentiment, alpha_final, return_not_sum_result=False)
        return {'h': h, 'h_sentiment': h_sentiment}

    def forward(self, aspect_representation, graphs, sentiment_representation=None):
        if sentiment_representation is None:
            sentiment_representation_flag = False
            # 方便其它逻辑顺利执行
            sentiment_representation = aspect_representation
        else:
            sentiment_representation_flag = True
        hidden = aspect_representation
        batched_graph = dgl.batch(graphs)
        feature = hidden.view([-1, hidden.size()[-1]])
        feature_sentiment = sentiment_representation.view([-1, sentiment_representation.size()[-1]])
        if feature.size()[0] != batched_graph.number_of_nodes():
            print('error')
        batched_graph.ndata['h'] = feature
        batched_graph.ndata['h_sentiment'] = feature_sentiment

        # gcn_msg = dgl_fn.copy_src(src='h', out='m')
        gcn_msg = self._message_func
        gcn_reduce = self._attention

        batched_graph.update_all(gcn_msg, gcn_reduce)

        ug = dgl.unbatch(batched_graph)
        output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        output = torch.cat(output, 0)

        output_sentiment = [torch.unsqueeze(g.ndata.pop('h_sentiment'), 0).to(self.opt['device']) for g in ug]
        output_sentiment = torch.cat(output_sentiment, 0)
        if sentiment_representation_flag:
            return output, output_sentiment
        else:
            return output


class SentenceConsituencyAwareModelV8(TextInAllAspectSentimentOutModel):
    """
    """
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 aspect_embedder: TextFieldEmbedder, categories: list, polarities: list, vocab: Vocabulary,
                 configuration: dict, category_loss_weight=1, sentiment_loss_weight=1):
        super().__init__(vocab, category_loss_weight=category_loss_weight, sentiment_loss_weight=sentiment_loss_weight)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.aspect_embedder = aspect_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        aspect_encoder_input_size = word_embedding_dim
        if self.configuration['aspect_position']:
            aspect_encoder_input_size += position_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(aspect_encoder_input_size, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(aspect_encoder_input_size, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(aspect_encoder_input_size, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)
        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        lstm_input_size = word_embedding_dim
        if self.configuration['position']:
            lstm_input_size += position_embedder.get_output_dim()
        num_layers = self.configuration['lstm_layer_num_in_lstm']
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['pipeline_with_acd']:
            self.category_fcs_pipeline_with_acd = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
            self.category_fcs_pipeline_with_acd = nn.ModuleList(self.category_fcs)

        sentiment_fc_input_size = word_embedding_dim
        if not self.configuration['share_sentiment_classifier']:
            self.sentiment_fcs = [nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                                nn.ReLU(),
                                                nn.Linear(sentiment_fc_input_size, self.polarity_num))
                                  for _ in range(self.category_num)]
            self.sentiment_fcs = nn.ModuleList(self.sentiment_fcs)
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                              nn.ReLU(),
                                              nn.Linear(sentiment_fc_input_size, self.polarity_num))

        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)
        # 只负责生成针对aspect category的节点表示
        aspect_graph = self.configuration['aspect_graph']
        if aspect_graph == 'attention':
            self.gnn_for_aspect = DglGraphAttentionForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                     self.embedding_layer_aspect_attentions,
                                                                     configuration)
        elif aspect_graph == 'attention_with_dotted_lines':
            self.gnn_for_aspect = DglGraphAttentionForAspectCategoryWithDottedLines(word_embedding_dim,
                                                                                    word_embedding_dim,
                                                                                    self.embedding_layer_aspect_attentions,
                                                                                    configuration)
        elif aspect_graph == 'average':
            self.gnn_for_aspect = DglGraphAverageForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)
        elif aspect_graph == 'gcn':
            self.gnn_for_aspect = DglGraphConvolutionForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)
        elif aspect_graph == 'gat':
            self.gnn_for_aspect = GATForAspectCategory(word_embedding_dim, word_embedding_dim, opt=self.configuration)
        else:
            raise NotImplementedError(aspect_graph)

        sentiment_graph = self.configuration['sentiment_graph']
        if sentiment_graph == 'average':
            self.gnn_for_sentiment = DglGraphAverage(word_embedding_dim, word_embedding_dim, self.configuration)
        elif sentiment_graph == 'gcn':
            self.gnn_for_sentiment = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        elif sentiment_graph == 'gat':
            self.gnn_for_sentiment = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)
        else:
            raise NotImplementedError(sentiment_graph)

        if self.configuration['acd_encoder_mode_for_sentiment_attention_warmup'] == 'mixed':
            if self.configuration['gnn_for_sentiment_attention_warmup'] == 'average':
                  self.gnn_for_sentiment_attention_warmup = DglGraphAverage(word_embedding_dim, word_embedding_dim, self.configuration)
            else:
                self.gnn_for_sentiment_attention_warmup = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'gcn':
            acd_layers.append(self.gc_aspect_category)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        acd_layers.append(self.gnn_for_aspect)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor,
                position: torch.Tensor, polarity_mask: torch.Tensor, sample: list, aspects: torch.Tensor=None):
        mask = get_text_field_mask(tokens)

        mask_for_graph = get_text_field_mask(tokens_for_graph)

        max_len = mask.size()[1]
        max_len_for_graph = mask_for_graph.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)

        graphs_with_dotted_line = [e['graph_with_dotted_line'] for e in sample]
        graphs_with_dotted_line_padded = self.pad_dgl_graph(graphs_with_dotted_line, max_len_for_graph)
        if not self.configuration['aspect_graph_with_dotted_line']:
            graphs_with_dotted_line_padded = graphs_padded

        position_embeddings = self.position_embedder(position)

        word_embeddings = self.word_embedder(tokens)

        word_embeddings_fc_input = word_embeddings
        if self.configuration['aspect_position']:
            word_embeddings_fc_input = torch.cat([word_embeddings, position_embeddings], dim=-1)
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings_fc_input)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings_fc_input)

        aspects_separate = [{'aspect': aspects['aspect'][:, i].unsqueeze(1)} for i in range(self.category_num)]
        aspect_embeddings_singles = [self.aspect_embedder(aspects_separate[i]).squeeze(1) for i in
                                     range(self.category_num)]
        aspects_seprate_repeat = [{'aspect': aspects_separate[i]['aspect'].expand_as(tokens['tokens'])} for i in
                                  range(self.category_num)]
        aspect_embeddings_separate = [self.aspect_embedder(aspects_seprate_repeat[i]) for i in range(self.category_num)]

        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)

        lstm_input = word_embeddings
        if self.configuration['position']:
            lstm_input = torch.cat([lstm_input, position_embeddings], dim=-1)
            lstm_input = self.dropout_after_embedding_layer(lstm_input)
        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout_after_lstm_layer(lstm_result)
        # 在lstm_result上做图卷积
        lstm_result_padded_for_graph = padder(lstm_result)

        if self.configuration['attention_warmup']:
            graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph, graphs_with_dotted_line_padded)
            if self.configuration['acd_encoder_mode_for_sentiment_attention_warmup'] == 'same':
                lstm_result_for_graph = graph_output_for_aspect
            else:
                lstm_result_for_graph = self.gnn_for_sentiment_attention_warmup(word_embeddings_fc_for_graph,
                                                                                graphs_with_dotted_line_padded)
        else:
            acd_sc_encoder_mode = self.configuration['acd_sc_encoder_mode']
            if acd_sc_encoder_mode == 'same':
                acd_encoder_mode = self.configuration['acd_encoder_mode']
                if acd_encoder_mode == 'simple':
                    graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                  graphs_with_dotted_line_padded)
                    # lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                    #                                             graphs_with_dotted_line_padded)
                    lstm_result_for_graph = graph_output_for_aspect
                elif acd_encoder_mode == 'complex':
                    graph_output_for_aspect = self.gnn_for_aspect(lstm_result_padded_for_graph,
                                                                  graphs_with_dotted_line_padded)
                    # lstm_result_for_graph = self.gnn_for_aspect(lstm_result_padded_for_graph,
                    #                                             graphs_with_dotted_line_padded)
                    lstm_result_for_graph = graph_output_for_aspect
                else:
                    graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                  graphs_with_dotted_line_padded)
                    lstm_result_for_graph = self.gnn_for_sentiment(word_embeddings_fc_for_graph,
                                                                   graphs_with_dotted_line_padded)
                    if self.configuration['gat_visualization']:
                        graph_output_for_aspect, gat_alpha_for_aspect = graph_output_for_aspect
                        lstm_result_for_graph, gat_alpha_for_sentiment = lstm_result_for_graph

                        # average gat alpha
                        head_num = len(gat_alpha_for_aspect)
                        gat_alpha_for_aspect_average = copy.deepcopy(gat_alpha_for_aspect[0])
                        for gat_alpha_of_one_head in gat_alpha_for_aspect[1:]:
                            for k in range(len(gat_alpha_of_one_head[0])):
                                gat_alpha_for_aspect_average[0][k].add(gat_alpha_of_one_head[0][k])
                        for edge in gat_alpha_for_aspect_average[0]:
                            edge.divide(head_num)
                        gat_alpha_for_sentiment_average = copy.deepcopy(gat_alpha_for_sentiment[0])
                        for gat_alpha_of_one_head in gat_alpha_for_sentiment[1:]:
                            for k in range(len(gat_alpha_of_one_head[0])):
                                gat_alpha_for_sentiment_average[0][k].add(gat_alpha_of_one_head[0][k])
                        for edge in gat_alpha_for_sentiment_average[0]:
                            edge.divide(head_num)


                        print()
            else:
                # mixed
                sentiment_encoder_with_own_gnn = self.configuration['sentiment_encoder_with_own_gnn']
                if sentiment_encoder_with_own_gnn:
                    graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                  graphs_with_dotted_line_padded)
                    lstm_result_for_graph = self.gnn_for_sentiment(lstm_result_padded_for_graph, graphs_padded)
                else:
                    # attention
                    graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                                         lstm_result_padded_for_graph,
                                                                                         graphs_with_dotted_line_padded)

        if not self.configuration['constituency_tree']:
            graph_output_for_aspect, lstm_result_for_graph = word_embeddings_fc, lstm_result
            mask_for_graph = mask

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(graph_output_for_aspect, mask_for_graph)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(graph_output_for_aspect, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        # lstm_result_with_position = torch.cat([lstm_result, position_embeddings], dim=-1)
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        for i in range(self.category_num):
            # sentiment
            # word_representation_for_sentiment = torch.cat([graph_output2, lstm_result], dim=-1)
            word_representation_for_sentiment = lstm_result_for_graph
            sentiment_alpha = embedding_layer_category_alphas[i]
            sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                     return_not_sum_result=False)
            if not self.configuration['share_sentiment_classifier']:
                sentiment_output = self.sentiment_fcs[i](sentiment_output_temp)
            else:
                sentiment_output = self.sentiment_fc(sentiment_output_temp)
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

            final_sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_outputs.append(final_sentiment_output)

        # --warmup=True --pipeline=True --attention_warmup_init=True --attention_warmup==False --pipeline_with_acd==True
        if self.configuration['acd_warmup'] and self.configuration['pipeline'] \
            and self.configuration['attention_warmup_init'] and not self.configuration['attention_warmup'] \
            and self.configuration['pipeline_with_acd']:
            final_category_outputs = []
            final_category_outputs_auxillary = []
            for i in range(self.category_num):
                word_representation_for_sentiment = lstm_result_for_graph
                sentiment_alpha = embedding_layer_category_alphas[i]
                category_output = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                        return_not_sum_result=False)
                for j in range(self.category_num):
                    final_category_output = self.category_fcs_pipeline_with_acd[j](category_output)
                    if i == j:
                        final_category_outputs.append(final_category_output)
                    else:
                        final_category_outputs_auxillary.append(final_category_output)

        output = {}
        output['alpha'] = embedding_layer_category_alphas
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])

            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                           label_temp)
                interactive_loss += interactive_loss_temp
            total_interactive_loss = interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1)

            loss = self.category_loss_weight * (total_category_loss + total_interactive_loss) + \
                   self.sentiment_loss_weight * total_sentiment_loss

            # Sparse Regularization Orthogonal Regularization
            if self.configuration['sparse_reg'] or self.configuration['orthogonal_reg']:
                reg_loss = 0
                for j in range(len(sample)):
                    polarity_mask_of_one_sample = polarity_mask[j]
                    category_alpha_of_one_sample = [embedding_layer_category_alphas[k][j] for k in range(self.category_num)]
                    category_alpha_of_mentioned = []
                    category_alpha_of_not_mentioned = []
                    for k in range(self.category_num):
                        if polarity_mask_of_one_sample[k] == 1:
                            category_alpha_of_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                        else:
                            category_alpha_of_not_mentioned.append(category_alpha_of_one_sample[k].unsqueeze(0))
                    if len(category_alpha_of_not_mentioned) != 0:
                        category_alpha_of_not_mentioned = torch.cat(category_alpha_of_not_mentioned, dim=0)
                        category_alpha_of_not_mentioned = torch.mean(category_alpha_of_not_mentioned, dim=0, keepdim=True)
                        category_alpha_of_mentioned.append(category_alpha_of_not_mentioned)

                    category_eye = torch.eye(len(category_alpha_of_mentioned))
                    category_alpha_of_mentioned = torch.cat(category_alpha_of_mentioned, dim=0)
                    category_alpha_similarity = torch.mm(category_alpha_of_mentioned, category_alpha_of_mentioned.t())

                    if self.configuration['sparse_reg'] and self.configuration['orthogonal_reg']:
                        pass
                    elif self.configuration['sparse_reg']:
                        for m in range(len(category_alpha_of_mentioned)):
                            for n in range(len(category_alpha_of_mentioned)):
                                if m != n:
                                    category_eye[m][n] = category_alpha_similarity[m][n]
                    else:
                        # orthogonal_reg
                        for m in range(len(category_alpha_of_mentioned)):
                            category_eye[m][m] = category_alpha_similarity[m][m]
                    # category_eye = nn_util.move_to_device(category_eye, self.configuration['device'])
                    category_eye = category_eye.to(self.configuration['device'])
                    category_alpha_similarity = category_alpha_similarity.to(self.configuration['device'])
                    category_reg_loss = category_alpha_similarity - category_eye
                    category_reg_loss = torch.norm(category_reg_loss)
                    reg_loss += category_reg_loss
                loss += (reg_loss * self.configuration['attention_lamda'] / len(sample))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                if self.configuration['constituency_tree']:
                    words = sample[i]['words_for_graph']
                    print('inner_nodes:-------------------------------------------')
                    for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                        print('%d-%s-%s' % (len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))
                else:
                    words = sample[i]['words']

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                # if sum(label_true) <= 1:
                #     continue
                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]

                titles_sentiment_temp = []
                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                    if sentiment_true_index == -100:
                        polarity_temp = '-100'
                    else:
                        polarity_temp = self.polarites[sentiment_true_index]
                    titles_sentiment_temp.append('true: %s - pred: %s - %s' % (str(polarity_temp),
                                                                      str(pred_sentiment[j][
                                                                              i].detach().cpu().numpy()),
                                                                      str(self.polarites)))
                titles = ['true: %s - pred: %s - %s' % (str(label[i][j].detach().cpu().numpy()),
                                                          str(pred_category[j][i].detach().cpu().numpy()),
                                                          titles_sentiment_temp[j])
                          for j in range(self.category_num)]
                print(words)
                print(visual_attentions_category)
                print(attention_labels)
                print(titles)
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence_backup(words, visual_attentions_category,
                                                                       attention_labels, titles,
                                                                       savefig_filepath=savefig_filepath)

                # sentiment embedding layer
                # visual_attentions = [embedding_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
                # visual_attentions = [lstm_layer_sentiment_alphas[j][i][: len(words)].detach().cpu().numpy()
                #                      for j in range(self.category_num)]
                # titles = ['true: %s - pred: %s - %s' % (str(label[i + self.category_num][j].detach().cpu().numpy()),
                #                                         str(pred_sentiment[j][i].detach().cpu().numpy()),
                #                                         str(self.polarites))
                #           for j in range(self.category_num)]
                # attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions, attention_labels,
                #                                                        titles)

                # sentiment lstm layer
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class ConstituencyBert(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        aspect_encoder_input_size = word_embedding_dim
        if self.configuration['aspect_position']:
            aspect_encoder_input_size += position_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(aspect_encoder_input_size, word_embedding_dim, bias=True)
        elif self.configuration['lstm_or_fc_after_embedding_layer'] == 'bilstm':
            self.embedding_layer_lstm = torch.nn.LSTM(aspect_encoder_input_size, int(word_embedding_dim / 2), batch_first=True,
                                                      bidirectional=True, num_layers=1)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(aspect_encoder_input_size, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)

        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            num_layers = self.configuration['lstm_layer_num_in_bert']
            bilstm_hidden_size_in_bert = self.configuration['bilstm_hidden_size_in_bert']
            if bilstm_hidden_size_in_bert == 0:
                bilstm_hidden_size_in_bert = int(word_embedding_dim / 2)
            self.lstm = torch.nn.LSTM(768, bilstm_hidden_size_in_bert, batch_first=True,
                                      bidirectional=True, num_layers=num_layers,
                                      dropout=self.configuration['dropout_in_bert'])
            hidden_size = bilstm_hidden_size_in_bert * 2
        else:
            hidden_size = 768
        if self.configuration['only_bert']:
            self.sentiment_fc = nn.Sequential(
                # nn.Linear(768, 768),
                # nn.ReLU(),
                nn.Linear(768, self.polarity_num))
        else:
            self.sentiment_fc = nn.Sequential(
                # nn.Linear(hidden_size, hidden_size),
                # nn.ReLU(),
                nn.Linear(hidden_size, self.polarity_num))
        self.sentiment_fc_for_attention_warmup = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(word_embedding_dim, self.polarity_num))
        self.bert_word_embedder = bert_word_embedder

        self.dropout_after_embedding_layer = nn.Dropout(self.configuration['dropout_in_bert'])

        # 只负责生成针对aspect category的节点表示
        aspect_graph = self.configuration['aspect_graph']
        if aspect_graph == 'attention':
            self.gnn_for_aspect = DglGraphAttentionForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                     self.embedding_layer_aspect_attentions,
                                                                     configuration)
        elif aspect_graph == 'attention_with_dotted_lines':
            self.gnn_for_aspect = DglGraphAttentionForAspectCategoryWithDottedLines(word_embedding_dim,
                                                                                    word_embedding_dim,
                                                                                    self.embedding_layer_aspect_attentions,
                                                                                    configuration)
        elif aspect_graph == 'average':
            self.gnn_for_aspect = DglGraphAverageForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                   self.configuration)
        elif aspect_graph == 'gcn':
            self.gnn_for_aspect = DglGraphConvolutionForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                       self.configuration)
        elif aspect_graph == 'gat':
            self.gnn_for_aspect = GATForAspectCategory(word_embedding_dim, word_embedding_dim, opt=self.configuration)
        else:
            raise NotImplementedError(aspect_graph)

        sentiment_graph = self.configuration['sentiment_graph']
        if sentiment_graph == 'average':
            self.gnn_for_sentiment = DglGraphAverage(hidden_size, hidden_size, self.configuration)
        elif sentiment_graph == 'gcn':
            self.gnn_for_sentiment = DglGraphConvolution(hidden_size, hidden_size, configuration)
        elif sentiment_graph == 'gat':
            self.gnn_for_sentiment = GAT(hidden_size, hidden_size, hidden_size, 4,
                                         self.configuration)
        else:
            raise NotImplementedError(sentiment_graph)

        if self.configuration['acd_encoder_mode_for_sentiment_attention_warmup'] == 'mixed':
            if self.configuration['gnn_for_sentiment_attention_warmup'] == 'average':
                self.gnn_for_sentiment_attention_warmup = DglGraphAverage(word_embedding_dim, word_embedding_dim,
                                                                          self.configuration)
            else:
                self.gnn_for_sentiment_attention_warmup = GAT(word_embedding_dim, word_embedding_dim,
                                                              word_embedding_dim, 4, self.configuration)

    def set_bert_word_embedder(self, bert_word_embedder: TextFieldEmbedder=None):
        self.bert_word_embedder = bert_word_embedder

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        acd_layers.append(self.gnn_for_aspect)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def set_grad_for_acsc_parameter(self, requires_grad=True):
        acsc_layers = []
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            acsc_layers.append(self.lstm)
        acsc_layers.append(self.sentiment_fc)
        acsc_layers.append(self.gnn_for_sentiment)
        for layer in acsc_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad
        bert_model = self.bert_word_embedder._token_embedders['bert'].bert_model
        for param in bert_model.parameters():
            param.requires_grad = requires_grad

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor,
                position: torch.Tensor, polarity_mask: torch.Tensor, sample: list, bert: torch.Tensor) -> torch.Tensor:
        bert_mask = bert['mask']
        # bert_word_embeddings = self.bert_word_embedder(bert)
        token_type_ids = bert['bert-type-ids']
        offsets = bert['bert-offsets']
        bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets)

        mask = get_text_field_mask(tokens)

        mask_for_graph = get_text_field_mask(tokens_for_graph)

        max_len_for_graph = mask_for_graph.size()[1]
        max_len = mask.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)

        graphs_with_dotted_line = [e['graph_with_dotted_line'] for e in sample]
        graphs_with_dotted_line_padded = self.pad_dgl_graph(graphs_with_dotted_line, max_len_for_graph)
        if not self.configuration['aspect_graph_with_dotted_line']:
            graphs_with_dotted_line_padded = graphs_padded

        word_embeddings = self.word_embedder(tokens)
        word_embeddings_size = word_embeddings.size()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)
        graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph, graphs_with_dotted_line_padded)
        graph_output_for_aspect = self.gnn_for_aspect(word_embeddings_fc_for_graph, graphs_with_dotted_line_padded)
        if self.configuration['acd_encoder_mode_for_sentiment_attention_warmup'] == 'same':
            lstm_result_for_graph = graph_output_for_aspect
        else:
            lstm_result_for_graph = self.gnn_for_sentiment_attention_warmup(word_embeddings_fc_for_graph,
                                                                            graphs_with_dotted_line_padded)

        bert_clses_of_all_aspect = []
        embedding_layer_sentiment_outputs = []
        for i in range(self.category_num):
            bert_clses_of_aspect = bert_word_embeddings[:, i, 0, :]
            bert_clses_of_all_aspect.append(bert_clses_of_aspect)

            if not self.configuration['only_bert']:
                bert_word_embeddings_of_aspect = bert_word_embeddings[:, i, :, :]
                aspect_word_embeddings_from_bert = []
                for j in range(len(sample)):
                    aspect_word_embeddings_from_bert_of_one_sample = []
                    all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
                    for k in range(word_embeddings_size[1]):
                        if k in all_word_indices_in_bert:
                            word_indices_in_bert = all_word_indices_in_bert[k]
                            word_bert_embeddings = []
                            for word_index_in_bert in word_indices_in_bert:
                                word_bert_embedding = bert_word_embeddings_of_aspect[j][word_index_in_bert]
                                word_bert_embeddings.append(word_bert_embedding)
                            if len(word_bert_embeddings) == 0:
                                print()
                            if len(word_bert_embeddings) > 1:
                                word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in
                                                                  word_bert_embeddings]
                                word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                                word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                                word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                            else:
                                word_bert_embeddings_ave = word_bert_embeddings[0]
                            aspect_word_embeddings_from_bert_of_one_sample.append(
                                torch.unsqueeze(word_bert_embeddings_ave, 0))
                        else:
                            zero = torch.zeros_like(aspect_word_embeddings_from_bert_of_one_sample[-1])
                            aspect_word_embeddings_from_bert_of_one_sample.append(zero)
                    aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                        aspect_word_embeddings_from_bert_of_one_sample, dim=0)
                    aspect_word_embeddings_from_bert.append(
                        torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
                aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)
                if self.configuration['lstm_layer_num_in_bert'] != 0:
                    aspect_word_embeddings_from_bert_cat, _ = self.lstm(aspect_word_embeddings_from_bert_cat)

                # 在lstm_result上做图卷积
                aspect_word_embeddings_from_bert_cat_padded_for_graph = padder(aspect_word_embeddings_from_bert_cat)

                sentiment_encoder_with_own_gnn = self.configuration['sentiment_encoder_with_own_gnn']
                if sentiment_encoder_with_own_gnn:
                    aspect_word_embeddings_from_bert_cat_result_for_graph = self.gnn_for_sentiment(
                        aspect_word_embeddings_from_bert_cat_padded_for_graph, graphs_with_dotted_line_padded)
                else:
                    # attention
                    _, aspect_word_embeddings_from_bert_cat_result_for_graph = self.gnn_for_aspect(
                        word_embeddings_fc_for_graph,
                        graphs_with_dotted_line_padded,
                       aspect_word_embeddings_from_bert_cat_padded_for_graph)

                embedding_layer_sentiment_outputs.append(aspect_word_embeddings_from_bert_cat_result_for_graph)

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(graph_output_for_aspect, mask_for_graph)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(graph_output_for_aspect, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []
        sentiment_output_clses_soft = []
        for i in range(self.category_num):
            sentiment_output_temp = bert_clses_of_all_aspect[i]
            if self.configuration['dropout_after_cls']:
                sentiment_output_temp = self.dropout_after_embedding_layer(sentiment_output_temp)
            sentiment_output_cls = self.sentiment_fc(sentiment_output_temp)
            sentiment_output_clses_soft.append(torch.softmax(sentiment_output_cls, dim=-1))
            if self.configuration['only_bert']:
                sentiment_output = sentiment_output_cls
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                # sentiment
                aspect_word_embeddings_from_bert = embedding_layer_sentiment_outputs[i]
                word_representation_for_sentiment = self.dropout_after_embedding_layer(aspect_word_embeddings_from_bert)

                sentiment_alpha = embedding_layer_category_alphas[i]
                if self.configuration['attention_warmup']:
                    word_representation_for_sentiment = lstm_result_for_graph
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                              return_not_sum_result=False)
                if self.configuration['attention_warmup']:
                    sentiment_output_not_mil = self.sentiment_fc_for_attention_warmup(sentiment_output_temp)
                else:
                    sentiment_output_not_mil = self.sentiment_fc(sentiment_output_temp)
                if self.configuration['concat_cls_vector'] and not self.configuration['attention_warmup']:
                    sentiment_output = sentiment_output_not_mil + sentiment_output_cls
                else:
                    sentiment_output = sentiment_output_not_mil
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

            sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_output = sentiment_output
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                if not self.configuration['only_sc']:
                    total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                           label_temp)
                interactive_loss += interactive_loss_temp
            loss += (interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words_for_graph']
                print('inner_nodes:-------------------------------------------')
                for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                    print('%d-%s-%s' % (
                    len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue

                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]

                titles_sentiment_temp = []
                for j in range(self.category_num):
                    c_label = label[i][j].detach().cpu().numpy().tolist()
                    sentiment_true_index = int(label[i][j + self.category_num].detach().cpu().numpy().tolist())
                    if sentiment_true_index == -100:
                        polarity_temp = '-100'
                    else:
                        polarity_temp = self.polarites[sentiment_true_index]
                    titles_sentiment_temp.append('true: %s - pred: %s - %s' % (str(polarity_temp),
                                                                               str(pred_sentiment[j][
                                                                                       i].detach().cpu().numpy()),
                                                                               str(self.polarites)))
                titles = ['true: %s - pred: %s - %s' % (str(label[i][j].detach().cpu().numpy()),
                                                        str(pred_category[j][i].detach().cpu().numpy()),
                                                        titles_sentiment_temp[j])
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles, savefig_filepath)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class ConstituencyBertSingle(TextInAllAspectSentimentOutModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 categories: list, polarities: list, vocab: Vocabulary, configuration: dict,
                 bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.categories = categories
        self.polarites = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)

        word_embedding_dim = word_embedder.get_output_dim()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            self.embedding_layer_fc = nn.Linear(word_embedding_dim, word_embedding_dim, bias=True)
        else:
            self.embedding_layer_lstm = torch.nn.LSTM(word_embedding_dim, word_embedding_dim, batch_first=True,
                                                      bidirectional=False, num_layers=1)

        self.embedding_layer_aspect_attentions = [AttentionInHtt(word_embedding_dim,
                                                                 word_embedding_dim)
                                                  for _ in range(self.category_num)]
        self.embedding_layer_aspect_attentions = nn.ModuleList(self.embedding_layer_aspect_attentions)

        self.category_fcs = [nn.Linear(word_embedding_dim, 1) for _ in range(self.category_num)]
        self.category_fcs = nn.ModuleList(self.category_fcs)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            num_layers = self.configuration['lstm_layer_num_in_bert']
            self.lstm = torch.nn.LSTM(768, int(word_embedding_dim / 2), batch_first=True,
                                      bidirectional=True, num_layers=num_layers,
                                      dropout=self.configuration['dropout_in_bert'])
            hidden_size = word_embedding_dim
        else:
            hidden_size = 768
        self.sentiment_fc = nn.Sequential(nn.Linear(hidden_size, self.polarity_num))

        self.sentiment_fc_for_attention_warmup = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(word_embedding_dim, self.polarity_num))

        self.bert_word_embedder = bert_word_embedder

        self.dropout_after_embedding_layer = nn.Dropout(self.configuration['dropout_in_bert'])

        # 只负责生成针对aspect category的节点表示
        self.gnn_for_aspect = DglGraphAttentionForAspectCategory(word_embedding_dim, word_embedding_dim,
                                                                 self.embedding_layer_aspect_attentions, configuration)

        # self.gnn_for_sentiment = DglGraphConvolution(word_embedding_dim, word_embedding_dim, configuration)
        self.gnn_for_sentiment = DglGraphAverage(word_embedding_dim, word_embedding_dim, self.configuration)
        # self.gnn_for_sentiment = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)
        # self.gnn_for_sentiment = DglGraphAttentionForAspectCategory(word_embedding_dim, word_embedding_dim, self.configuration)

    def set_grad_for_acd_parameter(self, requires_grad=True):
        acd_layers = []
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            acd_layers.append(self.embedding_layer_fc)
        else:
            acd_layers.append(self.embedding_layer_lstm)
        acd_layers.append(self.embedding_layer_aspect_attentions)
        acd_layers.append(self.category_fcs)
        acd_layers.append(self.gnn_for_aspect)
        for layer in acd_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad

    def set_grad_for_acsc_parameter(self, requires_grad=True):
        acsc_layers = []
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            acsc_layers.append(self.lstm)
        acsc_layers.append(self.sentiment_fc)
        for layer in acsc_layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad
        bert_model = self.bert_word_embedder._token_embedders['bert'].bert_model
        for param in bert_model.parameters():
            param.requires_grad = requires_grad

    def set_bert_word_embedder(self, bert_word_embedder: TextFieldEmbedder=None):
        self.bert_word_embedder = bert_word_embedder

    def forward(self, tokens: Dict[str, torch.Tensor], tokens_for_graph: Dict[str, torch.Tensor], label: torch.Tensor,
                position: torch.Tensor, polarity_mask: torch.Tensor, sample: list, bert: torch.Tensor) -> torch.Tensor:
        mask_for_graph = get_text_field_mask(tokens_for_graph)

        bert_mask = bert['mask']
        bert_word_embeddings = self.bert_word_embedder(bert)

        mask = get_text_field_mask(tokens)

        max_len_for_graph = mask_for_graph.size()[1]
        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len_for_graph)
        max_len = mask.size()[1]
        padder = nn.ConstantPad2d((0, 0, 0, max_len_for_graph - max_len), 0)

        word_embeddings = self.word_embedder(tokens)
        word_embeddings_size = word_embeddings.size()
        if self.configuration['lstm_or_fc_after_embedding_layer'] == 'fc':
            word_embeddings_fc = self.embedding_layer_fc(word_embeddings)
        else:
            word_embeddings_fc, (_, _) = self.embedding_layer_lstm(word_embeddings)

        embedding_layer_category_outputs = []
        embedding_layer_category_alphas = []
        embedding_layer_sentiment_outputs = []
        embedding_layer_sentiment_alphas = []

        # 在word_embeddings_fc上做图卷积，attention是基于图卷积的输出
        word_embeddings_fc_for_graph = padder(word_embeddings_fc)
        graph_output_for_aspect, lstm_result_for_graph = self.gnn_for_aspect(word_embeddings_fc_for_graph,
                                                                             word_embeddings_fc_for_graph,
                                                                             graphs_padded)

        bert_clses_of_all_aspect = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(graph_output_for_aspect, mask_for_graph)
            embedding_layer_category_alphas.append(alpha)

            category_output = self.element_wise_mul(graph_output_for_aspect, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

        bert_clses_of_aspect = bert_word_embeddings[:, 0, 0, :]
        bert_clses_of_all_aspect.append(bert_clses_of_aspect)

        if not self.configuration['only_bert']:
            bert_word_embeddings_of_aspect = bert_word_embeddings[:, 0, :, :]
            aspect_word_embeddings_from_bert = []
            for j in range(len(sample)):
                aspect_word_embeddings_from_bert_of_one_sample = []
                all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
                for k in range(word_embeddings_size[1]):
                    if k in all_word_indices_in_bert:
                        word_indices_in_bert = all_word_indices_in_bert[k]
                        word_bert_embeddings = []
                        for word_index_in_bert in word_indices_in_bert:
                            word_bert_embedding = bert_word_embeddings_of_aspect[j][word_index_in_bert]
                            word_bert_embeddings.append(word_bert_embedding)
                        if len(word_bert_embeddings) == 0:
                            print()
                        if len(word_bert_embeddings) > 1:
                            word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                            word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                            word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                            word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                        else:
                            word_bert_embeddings_ave = word_bert_embeddings[0]
                        aspect_word_embeddings_from_bert_of_one_sample.append(
                            torch.unsqueeze(word_bert_embeddings_ave, 0))
                    else:
                        zero = torch.zeros_like(aspect_word_embeddings_from_bert_of_one_sample[-1])
                        aspect_word_embeddings_from_bert_of_one_sample.append(zero)
                aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                    aspect_word_embeddings_from_bert_of_one_sample, dim=0)
                aspect_word_embeddings_from_bert.append(
                    torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
            aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)
            if self.configuration['lstm_layer_num_in_bert'] != 0:
                aspect_word_embeddings_from_bert_cat, _ = self.lstm(aspect_word_embeddings_from_bert_cat)

            # 在lstm_result上做图卷积
            aspect_word_embeddings_from_bert_cat_padded_for_graph = padder(aspect_word_embeddings_from_bert_cat)
            aspect_word_embeddings_from_bert_cat_result_for_graph = self.gnn_for_sentiment(
                aspect_word_embeddings_from_bert_cat_padded_for_graph, graphs_padded)
            embedding_layer_sentiment_outputs.append(aspect_word_embeddings_from_bert_cat_result_for_graph)

        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        lstm_layer_words_sentiment_soft = []

        for i in range(self.category_num):
            sentiment_output_temp = bert_clses_of_all_aspect[0]
            sentiment_output_cls = self.sentiment_fc(sentiment_output_temp)
            if self.configuration['only_bert']:
                sentiment_output = sentiment_output_cls
                lstm_layer_sentiment_outputs.append(sentiment_output)
            else:
                # sentiment
                aspect_word_embeddings_from_bert = embedding_layer_sentiment_outputs[0]
                word_representation_for_sentiment = self.dropout_after_embedding_layer(aspect_word_embeddings_from_bert)

                sentiment_alpha = embedding_layer_category_alphas[i]
                if self.configuration['attention_warmup']:
                    word_representation_for_sentiment = lstm_result_for_graph
                sentiment_output_temp = self.element_wise_mul(word_representation_for_sentiment, sentiment_alpha,
                                                              return_not_sum_result=False)
                if self.configuration['attention_warmup']:
                    sentiment_output_not_mil = self.sentiment_fc_for_attention_warmup(sentiment_output_temp)
                else:
                    sentiment_output_not_mil = self.sentiment_fc(sentiment_output_temp)
                if self.configuration['concat_cls_vector']:
                    sentiment_output = sentiment_output_not_mil + sentiment_output_cls
                else:
                    sentiment_output = sentiment_output_not_mil
                lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_category_outputs_auxillary = []
        final_lstm_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            category_output = embedding_layer_category_outputs[i]
            for j in range(self.category_num):
                final_category_output = self.category_fcs[j](category_output)
                if i == j:
                    final_category_outputs.append(final_category_output)
                else:
                    final_category_outputs_auxillary.append(final_category_output)

            sentiment_output = lstm_layer_sentiment_outputs[i]
            final_sentiment_output = sentiment_output
            final_sentiment_outputs.append(final_sentiment_output)

        output = {}
        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i])
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i])
            loss = 0
            total_category_loss = 0
            total_sentiment_loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                total_category_loss += category_temp_loss
                if not self.configuration['only_acd']:
                    total_sentiment_loss += sentiment_temp_loss

            loss = self.category_loss_weight * total_category_loss + self.sentiment_loss_weight * total_sentiment_loss

            interactive_loss = 0
            for i in range(len(final_category_outputs_auxillary)):
                label_temp = torch.zeros_like(category_labels[0])
                interactive_loss_temp = self.category_loss(final_category_outputs_auxillary[i].squeeze(dim=-1),
                                                           label_temp)
                interactive_loss += interactive_loss_temp
            loss += (interactive_loss * self.configuration['interactive_loss_lamda'] / (self.category_num - 1))

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)

            output['loss'] = loss

        # visualize attention
        pred_category = [torch.sigmoid(e) for e in final_category_outputs]
        pred_sentiment = [torch.nn.functional.softmax(e, dim=-1) for e in final_sentiment_outputs]
        output['pred_category'] = pred_category
        output['pred_sentiment'] = pred_sentiment
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words_for_graph']
                print('inner_nodes:-------------------------------------------')
                for inner_node_index, inner_node in enumerate(sample[i]['inner_nodes']):
                    print('%d-%s-%s' % (
                        len(sample[i]['words']) + inner_node_index, inner_node.labels[0], inner_node.text))

                attention_labels = [e.split('/')[0] for e in self.categories]

                # 只可视化mams
                label_true = label[i].detach().cpu().numpy()[: self.category_num]
                if sum(label_true) <= 1:
                    continue

                # category
                visual_attentions_category = [embedding_layer_category_alphas[j][i][: len(words)].detach().cpu().numpy()
                                              for j in range(self.category_num)]
                titles = ['true: %s - pred: %s' % (str(label[i][j].detach().cpu().numpy()),
                                                   str(pred_category[j][i].detach().cpu().numpy()))
                          for j in range(self.category_num)]
                savefig_filepath = super()._get_model_visualization_picture_filepath(self.configuration, words)
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_attentions_category,
                                                                       attention_labels, titles, savefig_filepath)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset),
            'category_f1': self._f1.get_metric(reset)['fscore']
        }
        return metrics


class Estimator:

    def estimate(self, ds: Iterable[Instance]) -> dict:
        raise NotImplementedError('estimate')


class TextInAllAspectSentimentOutEstimator(Estimator):
    def __init__(self, model: Model, iterator: DataIterator, categories: list, polarities: list,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.categories = categories
        self.polarities = polarities
        self._sentiment_accuracy = metrics.CategoricalAccuracy()
        self._sentiment_accuracy_temp = metrics.CategoricalAccuracy()
        self._aspect_f1 = allennlp_metrics.BinaryF1(0.5)
        self._aspect_f1_temp = allennlp_metrics.BinaryF1(0.5)
        self.cuda_device = cuda_device
        self.configuration = configuration
        self.other_metrics = {}
        self.debug = False

    def _get_other_metrics(self, reset=True):
        result = self.other_metrics
        if reset:
            self.other_metrics = {}
        return result

    def _print_tensor(self, tensors: List):
        print('------------------------------------------------------')
        list_list = [e.detach().cpu().numpy().tolist() if not isinstance(e, np.ndarray) else e.tolist() for e in tensors]
        for k in range(len(list_list[0])):
            format_str = '-'.join(['%s'] * len(list_list))
            values = tuple(e[k] for e in list_list)
            print(format_str % values)

    def _acd_aspect_and_metrics(self, category_labels, aspect_pred):
        acd_aspect_and_metrics = {}
        for i, aspect in enumerate(self.categories):
            aspect_label_i = category_labels[i].detach().cpu().numpy().astype(int)
            aspect_pred_i = (aspect_pred[i].squeeze(dim=-1).detach().cpu().numpy() > 0.5).astype(int)
            if self.debug:
                self._print_tensor([aspect_pred_i, aspect_label_i])
            aspect_f1 = f1_score(aspect_label_i, aspect_pred_i, average='binary')
            aspect_precision = precision_score(aspect_label_i, aspect_pred_i, average='binary')
            aspect_recall = recall_score(aspect_label_i, aspect_pred_i, average='binary')
            acd_aspect_and_metrics[aspect] = {
                'f1': aspect_f1,
                'precision': aspect_precision,
                'recall': aspect_recall
            }
        return acd_aspect_and_metrics

    def _acsc_aspect_and_metrics(self, polarity_labels, sentiment_pred, polarity_masks):
        acsc_aspect_and_metrics = {}
        for i, aspect in enumerate(self.categories):
            aspect_sentiment_label_i = polarity_labels[i]
            aspect_sentiment_pred_i = sentiment_pred[i]
            aspect_sentiment_mask_i = polarity_masks[i]
            self._sentiment_accuracy_temp(aspect_sentiment_pred_i, aspect_sentiment_label_i,
                                          aspect_sentiment_mask_i)
            aspect_acc_temp = self._sentiment_accuracy_temp.get_metric(reset=True),
            acsc_aspect_and_metrics[aspect] = {
                'acc': aspect_acc_temp[0],
            }
        return acsc_aspect_and_metrics

    def _polarity_metrics(self, sentiment_logit, sentiment_label, sentiment_mask):
        sentiment_label_pred_list = sentiment_logit.argmax(dim=-1).detach().cpu().numpy().tolist()
        sentiment_label_list = sentiment_label.detach().cpu().numpy().tolist()
        sentiment_mask_list = sentiment_mask.detach().cpu().numpy().tolist()
        sentiment_label_pred_final = []
        sentiment_label_final = []
        for i in range(len(sentiment_mask_list)):
            sentiment_label_list_i = sentiment_label_list[i]
            sentiment_label_pred_list_i = sentiment_label_pred_list[i]
            sentiment_mask_list_i = sentiment_mask_list[i]
            if sentiment_mask_list_i == 0:
                continue
            sentiment_label_pred_final.append(sentiment_label_pred_list_i)
            sentiment_label_final.append(sentiment_label_list_i)
        sentiment_f1s = f1_score(np.array(sentiment_label_final), np.array(sentiment_label_pred_final), average=None,
                                 labels=list(range(len(self.polarities))))
        sentiment_precisions = precision_score(np.array(sentiment_label_final), np.array(sentiment_label_pred_final),
                                               average=None, labels=list(range(len(self.polarities))))
        sentiment_recalls = recall_score(np.array(sentiment_label_final), np.array(sentiment_label_pred_final),
                                         average=None, labels=list(range(len(self.polarities))))
        polarity_metrics = {}
        for i, polarity in enumerate(self.polarities):
            polarity_metrics[polarity] = {
                'f1': sentiment_f1s[i],
                'precision': sentiment_precisions[i],
                'recall': sentiment_recalls[i]
            }
        return polarity_metrics

    def _merge_micro_f1(self, merge_label_real, merge_logit_real):
        tp = 0
        pred_total = 0
        true_total = 0
        for i in range(merge_logit_real.shape[0]):
            pred = merge_logit_real[i]
            true = merge_label_real[i]
            if pred != 0:
                pred_total += 1
            if true != 0:
                true_total += 1
            if pred == true != 0:
                tp += 1
        if pred_total == 0:
            pred_total = 0.0000000000000001
        if true_total == 0:
            true_total = 0.0000000000000001
        p = tp / pred_total
        r = tp / true_total
        if p == 0 and r == 0:
            f1 = 0
        else:
            f1 = 2 * (p * r) / (p + r)
        return f1

    def _inner_estimate(self, label, polarity_mask, aspect_pred, sentiment_pred, merge_pred):
        category_labels = []
        polarity_labels = []
        merge_labeles = []
        polarity_masks = []
        category_num = len(self.categories)
        for i in range(category_num):
            category_labels.append(label[:, i])
            polarity_labels.append(label[:, i + category_num])
            polarity_masks.append(polarity_mask[:, i])
            merge_labeles.append(label[:, i + category_num * 2])
        if self.debug:
            self._print_tensor([label] + category_labels + polarity_labels + merge_labeles)
            self._print_tensor([polarity_mask] + polarity_masks)
        acd_aspect_and_metrics = self._acd_aspect_and_metrics(category_labels, aspect_pred)
        self.other_metrics['acd_metrics'] = acd_aspect_and_metrics
        # category f1
        category_prob = torch.cat(aspect_pred).squeeze()
        category_label = torch.cat(category_labels)
        self._aspect_f1(category_prob, category_label)

        if not self.configuration['only_acd']:
            acsc_aspect_and_metrics = self._acsc_aspect_and_metrics(polarity_labels, sentiment_pred,
                                                                    polarity_masks)
            self.other_metrics['acsc_metrics'] = acsc_aspect_and_metrics

            # sentiment accuracy
            sentiment_logit = torch.cat(sentiment_pred)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._sentiment_accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            polarity_metrics = self._polarity_metrics(sentiment_logit, sentiment_label,
                                                      sentiment_mask)
            self.other_metrics['polarity_metrics'] = polarity_metrics

            # merge
            merge_logit = torch.cat(merge_pred)
            merge_pred_aspect_indicator = (merge_logit.argmax(dim=-1) != 0)
            merge_pred_aspect_indicator = nn_util.move_to_device(merge_pred_aspect_indicator, self.cuda_device)

            merge_label = torch.cat(merge_labeles)
            merge_label_aspect_indicator = (merge_label != 0)
            merge_label_aspect_indicator = nn_util.move_to_device(merge_label_aspect_indicator, self.cuda_device)

            merge_aspect_indicator = merge_pred_aspect_indicator | merge_label_aspect_indicator
            if self.debug:
                self._print_tensor([merge_logit, merge_pred_aspect_indicator, merge_label, merge_label_aspect_indicator, merge_aspect_indicator])

            merge_logit_real = merge_logit[merge_aspect_indicator].argmax(dim=-1).detach().cpu().numpy()
            merge_label_real = merge_label[merge_aspect_indicator].detach().cpu().numpy()
            if self.debug:
                self._print_tensor([merge_logit_real, merge_label_real])
            # merge_micro_f1 = f1_score(merge_label_real, merge_logit_real, average='micro')
            merge_micro_f1 = self._merge_micro_f1(merge_label_real, merge_logit_real)
            self.other_metrics['merge_micro_f1'] = merge_micro_f1

    def estimate(self, ds: Iterable[Instance]) -> dict:
        self.model.eval()
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(pred_generator, total=self.iterator.get_num_batches(ds))
        with torch.no_grad():
            labels = []
            polarity_masks = []
            pred_categorys = []
            pred_sentiments = []
            pred_merges = []
            for batch in pred_generator_tqdm:
                label = batch['label']
                labels.append(label)

                polarity_mask = batch['polarity_mask']
                polarity_masks.append(polarity_mask)

                batch = nn_util.move_to_device(batch, self.cuda_device)
                out_dict = self.model(**batch)
                pred_category = out_dict['pred_category']
                pred_categorys.append(pred_category)
                if not self.configuration['only_acd']:
                    pred_sentiment = out_dict['pred_sentiment']
                    pred_sentiments.append(pred_sentiment)

                    if 'merge_pred' in out_dict:
                        pred_merge = out_dict['merge_pred']
                    else:
                        pred_merge = []
                        for i in range(len(self.categories)):
                            pred_category_i = pred_category[i].detach().clone().squeeze(-1)
                            pred_sentiment_i = torch.softmax(pred_sentiment[i], dim=-1)
                            aspect_threshold = 0.5 if 'aspect_threshold' not in self.configuration else self.configuration['aspect_threshold']
                            pred_category_i_indicator = pred_category_i > aspect_threshold
                            pred_category_i_indicator_not = pred_category_i <= aspect_threshold
                            if self.debug:
                                print(i)
                                self._print_tensor([pred_category_i, pred_sentiment_i, pred_category_i_indicator, pred_category_i_indicator_not])
                            pred_category_i[pred_category_i_indicator] = 0
                            pred_category_i[pred_category_i_indicator_not] = 1.1
                            pred_category_i = pred_category_i.unsqueeze(-1)
                            if self.debug:
                                self._print_tensor([pred_category[i], pred_category_i, torch.cat([pred_category_i, pred_sentiment_i], dim=-1)])
                            pred_merge.append(torch.cat([pred_category_i, pred_sentiment_i], dim=-1))
                    pred_merges.append(pred_merge)
            label_final = torch.cat(labels, dim=0)
            polarity_mask_final = torch.cat(polarity_masks, dim=0)
            pred_category_final = []
            pred_sentiment_final = []
            pred_merge_final = []
            for i in range(len(self.categories)):
                pred_category_i = [e[i] for e in pred_categorys]
                pred_category_i_cat = torch.cat(pred_category_i, dim=0)
                pred_category_final.append(pred_category_i_cat)
                if not self.configuration['only_acd']:
                    pred_sentiment_i = [e[i] for e in pred_sentiments]
                    pred_sentiment_i_cat = torch.cat(pred_sentiment_i, dim=0)
                    pred_sentiment_final.append(pred_sentiment_i_cat)

                    pred_merge_i = [e[i] for e in pred_merges]
                    pred_merge_i_cat = torch.cat(pred_merge_i, dim=0)
                    pred_merge_final.append(pred_merge_i_cat)

            # self._estimate(label_final, polarity_mask_final, pred_category_final, pred_sentiment_final)
            self._inner_estimate(label_final, polarity_mask_final, pred_category_final, pred_sentiment_final,
                                 pred_merge_final)
        return {'sentiment_acc': self._sentiment_accuracy.get_metric(reset=True),
                'category_f1': self._aspect_f1.get_metric(reset=True),
                'other_metrics': self._get_other_metrics()}


class TextInAllAspectSentimentOutEstimatorAll(Estimator):
    def __init__(self, model: Model, iterator: DataIterator, categories: list, polarities: list,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.categories = categories
        self.polarities = polarities
        self._accuracy = metrics.CategoricalAccuracy()
        self._f1 = allennlp_metrics.BinaryF1(0.5)
        self.cuda_device = cuda_device
        self.configuration = configuration

    def _estimate(self, batch) -> np.ndarray:
        label = batch['label']
        polarity_mask = batch['polarity_mask']
        category_labels = []
        polarity_labels = []
        polarity_masks = []
        for i in range(len(self.categories)):
            category_labels.append(label[:, i])
            polarity_labels.append(label[:, i + len(self.categories)])
            polarity_masks.append(polarity_mask[:, i])

        out_dict = self.model(**batch)
        pred_category = out_dict['pred_category']

        if not self.configuration['only_acd']:
            pred_sentiment = out_dict['pred_sentiment']

            sentiment_logit = torch.cat(pred_sentiment)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

        # category f1
        category_prob = torch.cat(pred_category).squeeze()
        category_label = torch.cat(category_labels)
        self._f1(category_prob, category_label)

    def estimate(self, ds: Iterable[Instance]) -> dict:
        self.model.eval()
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                self._estimate(batch)
        return {'sentiment_acc': self._accuracy.get_metric(reset=True),
                'category_f1': self._f1.get_metric(reset=True)}
