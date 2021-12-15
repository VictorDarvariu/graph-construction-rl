import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from relnet.agent.baseline.baseline_agent import BaselineAgent
from relnet.agent.supervised.sl_net import SLNet
from relnet.agent.pytorch_agent import PyTorchAgent
from relnet.utils.config_utils import get_device_placement

class SLAgent(PyTorchAgent, BaselineAgent):
    algorithm_name = "sl"
    is_deterministic = False
    is_trainable = True

    def __init__(self, environment):
        super().__init__(environment)
        self.obj_fun_scale_multiplier = 1

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)
        # overriding provided options
        self.validation_check_interval = 500
        self.max_validation_consecutive_steps = 10000

        self.setup_net()

    def train(self, train_g_list, validation_g_list, max_steps, **kwargs):
        self.setup_graphs(train_g_list, validation_g_list)
        self.setup_sample_idxes(len(train_g_list))

        self.setup_histories_file()
        self.setup_training_parameters(max_steps)

        pbar = tqdm(range(max_steps + 1), unit='steps', disable=None)
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        for self.step in pbar:
            self.logger.warn("starting validation loss check...")
            self.check_validation_loss(self.step, max_steps)
            self.logger.warn("completed validation loss check...")

            true_values, pred_values = self.run_training_batch()
            true_values = Variable(true_values.view(-1, 1))

            loss = F.mse_loss(Variable(true_values), pred_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('loss: %0.5f' % ( loss))
            should_stop = self.check_stopping_condition(self.step, max_steps)
            if should_stop:
                break

    def setup_net(self):
        self.net = SLNet(self.hyperparams, s2v_module=None)
        if get_device_placement() == 'GPU':
            self.net = self.net.cuda()
        if self.restore_model:
            self.restore_model_from_checkpoint()

    def setup_training_parameters(self, max_steps):
        self.learning_rate = self.hyperparams['learning_rate']

    def finalize(self):
        pass

    def pick_actions_using_strategy(self, t, i):
        g = self.environment.g_list[i]

        non_edges = list(self.environment.get_graph_non_edges(i))

        if len(non_edges) == 0:
            return (-1, -1)

        next_g_batch = []
        for first, second in non_edges:
            g_copy = g.copy()
            next_g, _ = g_copy.add_edge(first, second)
            next_g_batch.append(next_g)

        argmax_val_index, vals = self.net(next_g_batch, return_argmax=True)
        return non_edges[argmax_val_index]

    def run_training_batch(self):
        selected_idx = self.advance_pos_and_sample_indices()
        batch_graph = [self.train_g_list[idx] for idx in selected_idx]
        batch_true_obj_values = torch.Tensor([self.train_initial_obj_values[idx] for idx in selected_idx])
        batch_true_obj_values = torch.mul(batch_true_obj_values, self.obj_fun_scale_multiplier)
        if get_device_placement() == 'GPU':
            batch_true_obj_values = batch_true_obj_values.cuda()

        _, batch_pred_values = self.net(batch_graph)
        return batch_true_obj_values, batch_pred_values


    def post_env_setup(self):
        pass

    def get_default_hyperparameters(self):
        hyperparams = {'learning_rate': 0.0001,
                       'latent_dim': 64,
                       'hidden': 128,
                       'embedding_method': 'mean_field',
                       'max_lv': 3}
        return hyperparams

