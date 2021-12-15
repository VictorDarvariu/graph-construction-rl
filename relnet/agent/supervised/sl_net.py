import torch
from pytorch_util import weights_init
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from relnet.agent.fnapprox.gnn_regressor import GNNRegressor
from relnet.common.modules.custom_mod import JaggedArgmaxModule
from relnet.utils.config_utils import get_device_placement

class SLNet(GNNRegressor, nn.Module):
    def __init__(self, hyperparams, s2v_module):
        super().__init__(hyperparams, s2v_module)

        embed_dim = hyperparams['latent_dim']

        self.linear_1 = nn.Linear(embed_dim, hyperparams['hidden'])
        self.linear_out = nn.Linear(hyperparams['hidden'], 1)
        weights_init(self)

        self.num_node_feats = 1
        self.num_edge_feats = 0

        if s2v_module is None:
            self.s2v = self.model(latent_dim=embed_dim,
                                  output_dim=0,
                                  num_node_feats=self.num_node_feats,
                                  num_edge_feats=self.num_edge_feats,
                                  max_lv=hyperparams['max_lv'])
        else:
            self.s2v = s2v_module

    def prepare_node_features(self, batch_graph):
        n_nodes = 0
        prefix_sum = []
        for i in range(len(batch_graph)):
            n_nodes += batch_graph[i].num_nodes
            prefix_sum.append(n_nodes)

        node_feat = torch.zeros(n_nodes, self.num_node_feats)
        node_feat[:, 0] = 1.0

        return node_feat, torch.LongTensor(prefix_sum)

    def forward(self, batch_graph, return_argmax=False):
        node_feat, prefix_sum = self.prepare_node_features(batch_graph)
        embed, graph_embed, prefix_sum = self.run_s2v_embedding(batch_graph, node_feat, prefix_sum)
        embed_s = F.relu(self.linear_1(graph_embed))
        raw_pred = self.linear_out(embed_s)

        if return_argmax:
            val_argmax = torch.argmax(raw_pred).item()
        else:
            val_argmax = None

        return val_argmax, raw_pred

