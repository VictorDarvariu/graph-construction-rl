import numpy as np
import math
from copy import deepcopy

from relnet.state.graph_state import S2VGraph
from relnet.state.network_generators import NetworkGenerator

class GraphEdgeEnv(object):
    def __init__(self, objective_function, objective_function_kwargs,
                 edge_budget_percentage):
        self.objective_function = objective_function
        self.original_objective_function_kwargs = objective_function_kwargs
        self.objective_function_kwargs = deepcopy(self.original_objective_function_kwargs)

        self.edge_budget_percentage = edge_budget_percentage

        self.num_mdp_substeps = 2

        self.reward_eps = 1e-4
        self.reward_scale_multiplier = 100


    def setup(self, g_list, initial_objective_function_values, training=False):
        self.g_list = g_list
        self.n_steps = 0

        self.edge_budgets = self.compute_edge_budgets(self.g_list, self.edge_budget_percentage)
        self.used_edge_budgets = np.zeros(len(g_list), dtype=np.float)
        self.exhausted_budgets = np.zeros(len(g_list), dtype=np.bool)

        for i in range(len(self.g_list)):
            g = g_list[i]
            g.first_node = None
            g.populate_banned_actions(self.edge_budgets[i])

        self.training = training

        self.objective_function_values = np.zeros((2, len(self.g_list)), dtype=np.float)
        self.objective_function_values[0, :] = initial_objective_function_values
        self.objective_function_kwargs = deepcopy(self.original_objective_function_kwargs)
        self.rewards = np.zeros(len(g_list), dtype=np.float)

        if self.training:
            self.objective_function_values[0, :] = np.multiply(self.objective_function_values[0, :], self.reward_scale_multiplier)

    def pass_logger_instance(self, logger):
        self.logger_instance = logger

    def get_final_values(self):
        return self.objective_function_values[-1, :]

    def get_objective_function_value(self, s2v_graph):
        obj_function_value = self.objective_function.compute(s2v_graph, **self.objective_function_kwargs)
        return obj_function_value

    def get_objective_function_values(self, s2v_graphs):
        return np.array([self.get_objective_function_value(g) for g in s2v_graphs])

    def get_graph_non_edges(self, i):
        g = self.g_list[i]
        banned_first_nodes = g.banned_actions
        valid_acts = self.get_valid_actions(g, banned_first_nodes)

        budget = self.get_remaining_budget(i)
        non_edges = set()

        for first in valid_acts:
            banned_second_nodes = g.get_invalid_edge_ends(first, budget)
            valid_second_nodes = self.get_valid_actions(g, banned_second_nodes)

            for second in valid_second_nodes:
                if (first, second) in non_edges or (second, first) in non_edges:
                    continue
                non_edges.add((first, second))

        return non_edges

    def get_remaining_budget(self, i):
        return self.edge_budgets[i] - self.used_edge_budgets[i]

    @staticmethod
    def compute_edge_budgets(g_list, edge_budget_percentage):
        edge_budgets = np.zeros(len(g_list), dtype=np.float)

        for i in range(len(g_list)):
            g = g_list[i]
            n = g.num_nodes
            edge_budgets[i] = NetworkGenerator.compute_number_edges(n, edge_budget_percentage)

        return edge_budgets


    @staticmethod
    def get_valid_actions(g, banned_actions):
        all_nodes_set = g.all_nodes_set
        valid_first_nodes = all_nodes_set - banned_actions
        return valid_first_nodes

    @staticmethod
    def apply_action(g, action, remaining_budget, copy_state=False):
        if g.first_node is None:
            if copy_state:
                g_ref = g.copy()
            else:
                g_ref = g
            g_ref.first_node = action
            g_ref.populate_banned_actions(remaining_budget)
            # selection doesn't cost anything.
            return g_ref, remaining_budget
        else:
            new_g, edge_cost = g.add_edge(g.first_node, action)
            new_g.first_node = None

            updated_budget = remaining_budget - edge_cost
            new_g.populate_banned_actions(updated_budget)
            return new_g, updated_budget

    @staticmethod
    def apply_action_in_place(g, action, remaining_budget):
        if g.first_node is None:
            g.first_node = action
            g.populate_banned_actions(remaining_budget)
            return remaining_budget
        else:
            edge_cost = g.add_edge_dynamically(g.first_node, action)
            g.first_node = None

            updated_budget = remaining_budget - edge_cost
            g.populate_banned_actions(updated_budget)
            return updated_budget

    def step(self, actions):
        for i in range(len(self.g_list)):
            if not self.exhausted_budgets[i]:
                if actions[i] == -1:
                    if self.logger_instance is not None:
                        self.logger_instance.warn("budget not exhausted but trying to apply dummy action!")
                        self.logger_instance.error(f"the remaining budget: {self.get_remaining_budget(i)}")
                        g = self.g_list[i]
                        self.logger_instance.error(f"first_node selection: {g.first_node}")


                remaining_budget = self.get_remaining_budget(i)
                self.g_list[i], updated_budget = self.apply_action(self.g_list[i], actions[i], remaining_budget)
                self.used_edge_budgets[i] += (remaining_budget - updated_budget)

                if self.n_steps % 2 == 1:
                    if self.g_list[i].banned_actions == self.g_list[i].all_nodes_set:
                        self.exhausted_budgets[i] = True
                        objective_function_value = self.get_objective_function_value(self.g_list[i])
                        if self.training:
                            objective_function_value = objective_function_value * self.reward_scale_multiplier
                        self.objective_function_values[-1, i] = objective_function_value
                        reward = objective_function_value - self.objective_function_values[0, i]
                        if abs(reward) < self.reward_eps:
                            reward = 0.

                        self.rewards[i] = reward

        self.n_steps += 1

    def exploratory_actions(self, agent_exploration_policy):
        act_list_t0, act_list_t1 = [], []
        for i in range(len(self.g_list)):
            first_node, second_node = agent_exploration_policy(i)

            act_list_t0.append(first_node)
            act_list_t1.append(second_node)

        return act_list_t0, act_list_t1

    def get_max_graph_size(self):
        max_graph_size = np.max([g.num_nodes for g in self.g_list])
        return max_graph_size

    def is_terminal(self):
        return np.all(self.exhausted_budgets)

    def get_state_ref(self):
        cp_first = [g.first_node for g in self.g_list]
        b_list = [g.banned_actions for g in self.g_list]
        return zip(self.g_list, cp_first, b_list)

    def clone_state(self, indices=None):
        if indices is None:
            cp_first = [g.first_node for g in self.g_list][:]
            b_list = [g.banned_actions for g in self.g_list][:]
            return list(zip(deepcopy(self.g_list), cp_first, b_list))
        else:
            cp_g_list = []
            cp_first = []
            b_list = []

            for i in indices:
                cp_g_list.append(deepcopy(self.g_list[i]))
                cp_first.append(deepcopy(self.g_list[i].first_node))
                b_list.append(deepcopy(self.g_list[i].banned_actions))

            return list(zip(cp_g_list, cp_first, b_list))
