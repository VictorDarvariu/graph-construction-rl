import random
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import torch

from relnet.evaluation.eval_utils import eval_on_dataset, get_values_for_g_list
from relnet.utils.config_utils import get_logger_instance

class Agent(ABC):
    def __init__(self, environment):
        self.environment = environment

    def train(self, train_g_list, validation_g_list, max_steps, **kwargs):
        pass

    def eval(self, g_list,
             initial_obj_values=None,
             validation=False,
             make_action_kwargs=None):

        eval_nets = [deepcopy(g) for g in g_list]
        initial_obj_values, final_obj_values = get_values_for_g_list(self, eval_nets, initial_obj_values, validation, make_action_kwargs)
        return eval_on_dataset(initial_obj_values, final_obj_values)

    @abstractmethod
    def make_actions(self, t, **kwargs):
        pass

    def setup(self, options, hyperparams):
        self.options = options
        if 'log_filename' in options:
            self.log_filename = options['log_filename']
        if 'log_progress' in options:
            self.log_progress = options['log_progress']
        else:
            self.log_progress = False
        if self.log_progress:
            self.logger = get_logger_instance(self.log_filename)
            self.environment.pass_logger_instance(self.logger)
        else:
            self.logger = None

        if 'random_seed' in options:
            self.set_random_seeds(options['random_seed'])
        else:
            self.set_random_seeds(42)
        self.hyperparams = hyperparams

    @abstractmethod
    def finalize(self):
        pass

    def pick_random_actions(self, i):
        g = self.environment.g_list[i]
        banned_first_nodes = g.banned_actions

        first_valid_acts = self.environment.get_valid_actions(g, banned_first_nodes)
        if len(first_valid_acts) == 0:
            return -1, -1

        first_node = self.local_random.choice(tuple(first_valid_acts))
        rem_budget = self.environment.get_remaining_budget(i)
        banned_second_nodes = g.get_invalid_edge_ends(first_node, rem_budget)
        second_valid_acts = self.environment.get_valid_actions(g, banned_second_nodes)

        if second_valid_acts is None or len(second_valid_acts) == 0:
            if self.logger is not None:
                self.logger.error(f"caught an illegal state: allowed first actions disagree with second")
                self.logger.error(f"first node valid acts: {first_valid_acts}")
                self.logger.error(f"second node valid acts: {second_valid_acts}")
                self.logger.error(f"the remaining budget: {rem_budget}")

                self.logger.error(f"first_node selection: {g.first_node}")

            return -1, -1
        else:
            second_node = self.local_random.choice(tuple(second_valid_acts))
            return first_node, second_node

    def set_random_seeds(self, random_seed):
        self.random_seed = random_seed
        self.local_random = random.Random()
        self.local_random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
