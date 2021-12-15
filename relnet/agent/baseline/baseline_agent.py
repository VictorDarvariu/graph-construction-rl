from abc import abstractmethod, ABC

import networkx as nx
import numpy as np

from relnet.agent.base_agent import Agent
from relnet.spectral_properties import *


class BaselineAgent(Agent, ABC):
    is_trainable = False

    def __init__(self, environment):
        super().__init__(environment)
        self.future_actions = None

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)

    def make_actions(self, t, **kwargs):
        if t % 2 == 0:
            first_actions, second_actions = [], []
            for i in range(len(self.environment.g_list)):
                first_node, second_node = self.pick_actions_using_strategy(t, i)
                first_actions.append(first_node)
                second_actions.append(second_node)

            self.future_actions = second_actions
            chosen_actions = first_actions

        else:
            chosen_actions = self.future_actions
            self.future_actions = None

        return chosen_actions

    def finalize(self):
        pass

    @abstractmethod
    def pick_actions_using_strategy(self, t, i):
        pass

class RandomAgent(BaselineAgent):
    algorithm_name = 'random'
    is_deterministic = False

    def __init__(self, environment):
        super().__init__(environment)

    def pick_actions_using_strategy(self, t, i):
        return self.pick_random_actions(i)

class GreedyAgent(BaselineAgent):
    algorithm_name = 'greedy'
    is_deterministic = True

    def __init__(self, environment):
        super().__init__(environment,)

    def pick_actions_using_strategy(self, t, i):
        first_node, second_node = None, None

        g = self.environment.g_list[i]
        non_edges = list(self.environment.get_graph_non_edges(i))
        if len(non_edges) == 0:
            return (-1, -1)
        elif len(non_edges) == 1:
            return non_edges[0][0], non_edges[0][1]

        initial_value = self.environment.objective_function_values[0, i]
        best_difference = float("-inf")

        for first, second in non_edges:
            g_copy = g.copy()
            next_g, _ = g_copy.add_edge(first, second)
            next_value = self.environment.get_objective_function_value(next_g)

            diff = next_value - initial_value
            if diff > best_difference:
                best_difference = diff
                first_node, second_node = first, second

        return first_node, second_node

class LowestToLowestDegreeAgent(BaselineAgent):
    algorithm_name = 'lowest_to_lowest'
    is_deterministic = True

    def __init__(self, environment):
        super().__init__(environment)

    def pick_actions_using_strategy(self, t, i):
        first_node, second_node = None, None

        g = self.environment.g_list[i]
        non_edges = list(self.environment.get_graph_non_edges(i))
        if len(non_edges) == 0:
            return (-1, -1)
        elif len(non_edges) == 1:
            return non_edges[0][0], non_edges[0][1]

        degrees = g.node_degrees
        two_smallest_degrees = np.partition(degrees, 2)[:2]
        sm1, sm2 = two_smallest_degrees[0], two_smallest_degrees[1]

        for first, second in non_edges:
            deg1 = degrees[first]
            deg2 = degrees[second]

            if (deg1 == sm1 and deg1 == sm2) or (deg2 == sm1 and deg1 == sm2):
                first_node, second_node = first, second
                break

        # failover strategy: if no such edge, pick a random edge.
        if first_node is None and second_node is None:
            return self.pick_random_actions(i)

        return first_node, second_node

class LowestToRandomDegreeAgent(BaselineAgent):
    algorithm_name = 'lowest_to_random'
    is_deterministic = False

    def __init__(self, environment):
        super().__init__(environment)

    def pick_actions_using_strategy(self, t, i):
        first_node, second_node = None, None

        g = self.environment.g_list[i]
        banned_first_nodes = g.banned_actions

        valid_first_nodes = self.environment.get_valid_actions(g, banned_first_nodes)
        if len(valid_first_nodes) == 0:
            return (-1, -1)

        degrees = g.node_degrees
        smallest_degree = min(degrees)
        smallest_degree_nodes = [n for n in valid_first_nodes if degrees[n] == smallest_degree]

        for first in smallest_degree_nodes:
            budget = self.environment.get_remaining_budget(i)
            valid_choices = list(self.environment.get_valid_actions(g, g.get_invalid_edge_ends(first, budget)))
            if len(valid_choices) > 0:
                first_node = first
                second_node = self.local_random.choice(valid_choices)
                break

        # failover strategy: if no such edge, pick a random edge.
        if first_node is None and second_node is None:
            return self.pick_random_actions(i)

        return first_node, second_node

class LowestDegreeProductAgent(BaselineAgent):
    algorithm_name = 'lowest_degree_product'
    is_deterministic = True

    def __init__(self, environment):
        super().__init__(environment)

    def pick_actions_using_strategy(self, t, i):
        g = self.environment.g_list[i]
        non_edges = list(self.environment.get_graph_non_edges(i))
        if len(non_edges) == 0:
            return (-1, -1)
        elif len(non_edges) == 1:
            return non_edges[0][0], non_edges[0][1]

        feature_products = self.get_local_feature_products(g, non_edges)
        first_node, second_node = non_edges[np.argpartition(feature_products, 1)[0]]
        return first_node, second_node

    def get_local_feature_products(self, g, non_edges):
        degrees = g.node_degrees
        degree_products = list(map(lambda pair: degrees[pair[0]] * degrees[pair[1]], non_edges))
        return degree_products

class LowestBetweennessProductAgent(LowestDegreeProductAgent):
    algorithm_name = 'lowest_betweenness_product'
    is_deterministic = True

    def get_local_feature_products(self, g, non_edges):
        betweeness = nx.algorithms.centrality.betweenness_centrality(g.to_networkx())
        degree_products = list(map(lambda pair: betweeness[pair[0]] * betweeness[pair[1]], non_edges))
        return degree_products

class FiedlerVectorAgent(BaselineAgent):
    algorithm_name = 'fiedler_vector'
    is_deterministic = True

    def __init__(self, environment):
        super().__init__(environment)


    def pick_actions_using_strategy(self, t, i):
        g = self.environment.g_list[i]
        non_edges = list(self.environment.get_graph_non_edges(i))
        if len(non_edges) == 0:
            return (-1, -1)
        elif len(non_edges) == 1:
            return non_edges[0][0], non_edges[0][1]

        A = g.get_adjacency_matrix()
        L = get_laplacian(A)
        fiedler_vector = compute_fiedler_vector(L)

        abs_differences = list(map(lambda pair: abs(fiedler_vector[pair[0]] - fiedler_vector[pair[1]]), non_edges))
        first_node, second_node = non_edges[np.argmax(abs_differences)]

        return first_node, second_node

class EffectiveResistanceAgent(BaselineAgent):
    algorithm_name = 'effective_resistance'
    is_deterministic = True

    def __init__(self, environment):
        super().__init__(environment)


    def pick_actions_using_strategy(self, t, i):
        g = self.environment.g_list[i]
        non_edges = list(self.environment.get_graph_non_edges(i))
        if len(non_edges) == 0:
            return (-1, -1)
        elif len(non_edges) == 1:
            return non_edges[0][0], non_edges[0][1]

        A = g.get_adjacency_matrix()
        L = get_laplacian(A)

        pinv = get_pseudoinverse(L)
        def get_pairwise_effective_resistance(edge):
            i, j = edge
            return pinv[i,i] + pinv[j,j] - 2 * pinv[i,j]

        pairwise_effective_resistances = list(map(get_pairwise_effective_resistance, non_edges))
        first_node, second_node = non_edges[np.argmax(pairwise_effective_resistances)]

        return first_node, second_node


