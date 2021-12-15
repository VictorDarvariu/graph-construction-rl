import os
from copy import deepcopy
from datetime import timedelta
from itertools import product

from projectconfig import ProjectConfig
from relnet.agent.baseline.baseline_agent import *
from relnet.agent.pytorch_agent import PyTorchAgent
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.agent.supervised.sl_agent import SLAgent
from relnet.objective_functions.objective_functions import *
from relnet.state.network_generators import NetworkGenerator, GNMNetworkGenerator, BANetworkGenerator, \
    EuroroadNetworkGenerator, ScigridNetworkGenerator


class ExperimentConditions(object):
    def __init__(self, possible_edge_percentage, train_individually):
        self.gen_params = {}
        self.base_n = 20
        self.possible_edge_percentage = possible_edge_percentage
        self.train_individually = train_individually

        self.gen_params['n'] = self.base_n
        self.gen_params['m_percentage_er'] = 20
        self.gen_params['m_ba'] = 2

        self.gen_params['m'] = NetworkGenerator.compute_number_edges(self.gen_params['n'], self.gen_params['m_percentage_er'])

        self.num_mc_sims_multiplier = 2
        self.num_mc_sims = self.gen_params['n'] * self.num_mc_sims_multiplier

        self.obj_fun_seed = 42

        self.greedy_size_threshold = 2.5

        self.agents_baseline = {
            CriticalFractionRandom.name: [
                RandomAgent,
                GreedyAgent,
                LowestDegreeProductAgent,
                FiedlerVectorAgent,
                EffectiveResistanceAgent
            ],
            CriticalFractionTargeted.name: [
                RandomAgent,
                GreedyAgent,
                LowestDegreeProductAgent,
                FiedlerVectorAgent,
                EffectiveResistanceAgent
            ],
        }

        self.objective_functions = [
            CriticalFractionRandom,
            CriticalFractionTargeted,
        ]


    def get_model_seed(self, run_number):
        return run_number * 42

    def update_size_dependant_params(self, multiplier):
        self.gen_params['n'] = int(self.base_n * multiplier)
        self.gen_params['m'] = NetworkGenerator.compute_number_edges(self.gen_params['n'], self.gen_params['m_percentage_er'])

        self.num_mc_sims = self.gen_params['n'] * self.num_mc_sims_multiplier
        self.gen_params['size_multiplier'] = multiplier

    def set_generator_seeds(self):
        self.train_seeds, self.validation_seeds, self.test_seeds = NetworkGenerator.construct_network_seeds(
            self.experiment_params['train_graphs'],
            self.experiment_params['validation_graphs'],
            self.experiment_params['test_graphs'])

    def set_generator_seeds_individually(self, g_num, num_graphs):
        self.validation_seeds = [g_num]
        self.test_seeds = [g_num]

        # TODO: may be replaced with a different strategy if considering rewiring...
        self.train_seeds = [g_num + (i * num_graphs) for i in range(1, PyTorchAgent.DEFAULT_BATCH_SIZE+1)]


    def update_relevant_agents(self, algorithm_class):
        self.relevant_agents = deepcopy(self.agents_models)

    def extend_seeds_to_skip(self, run_num_start, run_num_end):
        for net in self.network_generators:
            for obj in self.objective_functions:
                for agent in self.relevant_agents:
                    setting = (net.name, obj.name, agent.algorithm_name)
                    if setting not in self.model_seeds_to_skip:
                        self.model_seeds_to_skip[setting] = []

                    for run_num_before in range(0, run_num_start):
                        self.model_seeds_to_skip[setting].append(self.get_model_seed(run_num_before))

                    for run_num_after in range(run_num_end + 1, self.experiment_params['num_runs']):
                        self.model_seeds_to_skip[setting].append(self.get_model_seed(run_num_after))

    def __str__(self):
        as_dict = deepcopy(self.__dict__)
        del as_dict["agents_models"]
        del as_dict["agents_baseline"]
        del as_dict["objective_functions"]
        del as_dict["network_generators"]
        return str(as_dict)

    def __repr__(self):
        return self.__str__()


class SyntheticGraphsExperimentConditions(ExperimentConditions):
    def __init__(self, possible_edge_percentage, train_individually):
        super().__init__(possible_edge_percentage, train_individually)

        self.network_generators = [
            GNMNetworkGenerator,
            BANetworkGenerator,
        ]

        self.agents_models = [
            RNetDQNAgent,
            SLAgent,
        ]

        self.agent_budgets = {
            CriticalFractionRandom.name: {
                RNetDQNAgent.algorithm_name: int(4 * possible_edge_percentage * (10 ** 4)),
                SLAgent.algorithm_name: int(4 * possible_edge_percentage * (10 ** 4)),
            },
            CriticalFractionTargeted.name: {
                RNetDQNAgent.algorithm_name: int(4 * possible_edge_percentage * (10 ** 4)),
                SLAgent.algorithm_name: int(4 * possible_edge_percentage * (10 ** 4)),
            },
        }

        self.size_multipliers = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]


        self.experiment_params = {'train_graphs': 10000,
                                  'validation_graphs': 100,
                                  'test_graphs': 100,
                                  'num_runs': 50}

        self.experiment_params['model_seeds'] = [self.get_model_seed(run_num) for run_num in
                                                 range(self.experiment_params['num_runs'])]


        self.hyperparam_grids = self.create_hyperparam_grids()


        self.model_seeds_to_skip = {
            # Can be used to skip some random seeds in case e.g. training failed.
            #  (GNMNetworkGenerator.name, CriticalFractionTargeted.name, RNetDQNAgent.algorithm_name): [42],
        }


    def create_hyperparam_grids(self):
        hyperparam_grid_base =  {
            RNetDQNAgent.algorithm_name: {
                'learning_rate': [0.0001],
                'epsilon_start': [1],
                'latent_dim': [64],
                'hidden': [128],
                'embedding_method': ['mean_field'],
                'max_lv': [3],
                'mem_pool_to_steps_ratio': [1],
                'eps_step_denominator': [2]
            },

            SLAgent.algorithm_name: {
                'learning_rate': [0.0001],
                'latent_dim': [64],
                'hidden': [128],
                'embedding_method': ['mean_field'],
                'max_lv': [3],
            },

        }
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)
        return hyperparam_grids


class RealWorldGraphsExperimentConditions(ExperimentConditions):
    def __init__(self, possible_edge_percentage, train_individually):
        super().__init__(possible_edge_percentage, train_individually)

        self.network_generators = [
            EuroroadNetworkGenerator,
            ScigridNetworkGenerator,
        ]

        self.agents_models = [
            RNetDQNAgent,
        ]

        self.agent_budgets = {
            CriticalFractionRandom.name: {
                RNetDQNAgent.algorithm_name: int(4 * possible_edge_percentage * (10 ** 4)),
            },
            CriticalFractionTargeted.name: {
                RNetDQNAgent.algorithm_name: int(4 * possible_edge_percentage * (10 ** 4)),
            },
        }

        self.size_multipliers = [1]

        self.experiment_params = {'train_graphs': PyTorchAgent.DEFAULT_BATCH_SIZE,
                                  'validation_graphs': 1,
                                  'test_graphs': 1,
                                  'num_runs': 10}

        self.experiment_params['model_seeds'] = [self.get_model_seed(run_num) for run_num in
                                                 range(self.experiment_params['num_runs'])]

        self.hyperparam_grids = self.create_hyperparam_grids()

        self.model_seeds_to_skip = {
            # Can be used to skip some random seeds in case e.g. training failed.
            #  (GNMNetworkGenerator.name, CriticalFractionTargeted.name, RNetDQNAgent.algorithm_name): [42],
        }

    def create_hyperparam_grids(self):
        hyperparam_grid_base =  {
            RNetDQNAgent.algorithm_name: {
                'learning_rate': [0.0001],
                'epsilon_start': [1],
                'latent_dim': [64],
                'hidden': [32],
                'embedding_method': ['mean_field'],
                'max_lv': [5],
                'mem_pool_to_steps_ratio': [1],
                'eps_step_denominator': [10]
            },


        }
        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)

        return hyperparam_grids


def get_exp_conditions(which, possible_edge_percentage, train_individually):
    if which == 'synth':
        cond = SyntheticGraphsExperimentConditions(possible_edge_percentage, train_individually)
    else:
        cond = RealWorldGraphsExperimentConditions(possible_edge_percentage, train_individually)
    return cond

