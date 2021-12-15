import sys
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from relnet.agent.fnapprox.gnn_regressor import GNNRegressor
from relnet.utils.config_utils import get_device_placement

sys.path.append('/usr/lib/pytorch_structure2vec/s2v_lib')
from pytorch_util import weights_init

from relnet.common.modules.custom_mod import JaggedMaxModule


def greedy_actions(q_values, v_p, banned_list):
    actions = []
    offset = 0
    banned_acts = []
    prefix_sum = v_p.data.cpu().numpy()
    for i in range(len(prefix_sum)):
        n_nodes = prefix_sum[i] - offset

        if banned_list is not None and banned_list[i] is not None:
            for j in banned_list[i]:
                banned_acts.append(offset + j)
        offset = prefix_sum[i]

    q_values = q_values.data.clone()
    q_values.resize_(len(q_values))

    banned = torch.LongTensor(banned_acts)
    device_placement = get_device_placement()
    if device_placement == 'GPU':
        banned = banned.cuda()

    if len(banned_acts):
        min_tensor = torch.tensor(float(np.finfo(np.float32).min))
        if device_placement == 'GPU':
            min_tensor = min_tensor.cuda()
        q_values.index_fill_(0, banned, min_tensor)

    # if len(banned_acts):
    #     q_values[banned_acts, :] = torch.float(min_val)
    jmax = JaggedMaxModule()
    values, actions = jmax(Variable(q_values), v_p)
    return actions.data, values.data


class QNet(GNNRegressor, nn.Module):
    def __init__(self, hyperparams, s2v_module):
        super().__init__(hyperparams, s2v_module)

        embed_dim = hyperparams['latent_dim']

        self.linear_1 = nn.Linear(embed_dim * 2, hyperparams['hidden'])
        self.linear_out = nn.Linear(hyperparams['hidden'], 1)
        weights_init(self)

        self.num_node_feats = 2
        self.num_edge_feats = 0

        if s2v_module is None:
            self.s2v = self.model(latent_dim=embed_dim,
                                  output_dim=0,
                                  num_node_feats=self.num_node_feats,
                                  num_edge_feats=self.num_edge_feats,
                                  max_lv=hyperparams['max_lv'])
        else:
            self.s2v = s2v_module

    def add_offset(self, actions, v_p):
        prefix_sum = v_p.data.cpu().numpy()

        shifted = []
        for i in range(len(prefix_sum)):
            if i > 0:
                offset = prefix_sum[i - 1]
            else:
                offset = 0
            shifted.append(actions[i] + offset)

        return shifted

    def rep_global_embed(self, graph_embed, v_p):
        prefix_sum = v_p.data.cpu().numpy()

        rep_idx = []
        for i in range(len(prefix_sum)):
            if i == 0:
                n_nodes = prefix_sum[i]
            else:
                n_nodes = prefix_sum[i] - prefix_sum[i - 1]
            rep_idx += [i] * n_nodes

        rep_idx = Variable(torch.LongTensor(rep_idx))
        if get_device_placement() == 'GPU':
            rep_idx = rep_idx.cuda()
        graph_embed = torch.index_select(graph_embed, 0, rep_idx)
        return graph_embed

    def prepare_node_features(self, batch_graph, picked_nodes):
        n_nodes = 0
        prefix_sum = []
        picked_ones = []
        for i in range(len(batch_graph)):
            if picked_nodes is not None and picked_nodes[i] is not None:
                assert picked_nodes[i] >= 0 and picked_nodes[i] < batch_graph[i].num_nodes
                picked_ones.append(n_nodes + picked_nodes[i])
            n_nodes += batch_graph[i].num_nodes
            prefix_sum.append(n_nodes)

        node_feat = torch.zeros(n_nodes, self.num_node_feats)
        node_feat[:, 0] = 1.0

        if len(picked_ones):
            node_feat.numpy()[picked_ones, 1] = 1.0
            node_feat.numpy()[picked_ones, 0] = 0.0

        return node_feat, torch.LongTensor(prefix_sum)

    def forward(self, states, actions, greedy_acts=False):
        batch_graph, picked_nodes, banned_list = zip(*states)

        node_feat, prefix_sum = self.prepare_node_features(batch_graph, picked_nodes)
        embed, graph_embed, prefix_sum = self.run_s2v_embedding(batch_graph, node_feat, prefix_sum)

        prefix_sum = Variable(prefix_sum)
        if actions is None:
            graph_embed = self.rep_global_embed(graph_embed, prefix_sum)
        else:
            shifted = self.add_offset(actions, prefix_sum)
            embed = embed[shifted, :]

        embed_s_a = torch.cat((embed, graph_embed), dim=1)
        embed_s_a = F.relu(self.linear_1(embed_s_a))
        raw_pred = self.linear_out(embed_s_a)

        if greedy_acts:
            actions, _ = greedy_actions(raw_pred, prefix_sum, banned_list)

        return actions, raw_pred, prefix_sum


class NStepQNet(nn.Module):
    def __init__(self, hyperparams, num_steps):
        super(NStepQNet, self).__init__()

        list_mod = [QNet(hyperparams, None)]

        for i in range(1, num_steps):
            list_mod.append(QNet(hyperparams, list_mod[0].s2v))

        self.list_mod = nn.ModuleList(list_mod)
        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts = False):
        assert time_t >= 0 and time_t < self.num_steps

        return self.list_mod[time_t](states, actions, greedy_acts)
