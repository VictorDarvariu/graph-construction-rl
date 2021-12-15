import math
from copy import deepcopy

import networkx as nx
import numpy as np
import xxhash
import warnings

budget_eps = 1e-5

class S2VGraph(object):
    def __init__(self, g):
        self.num_nodes = g.number_of_nodes()
        self.node_labels = np.arange(self.num_nodes)
        self.all_nodes_set = set(self.node_labels)

        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = np.ravel(self.edge_pairs)

        self.node_degrees = np.array([deg for (node, deg) in sorted(g.degree(), key=lambda deg_pair: deg_pair[0])])
        self.first_node = None
        self.dynamic_edges = None

    def add_edge(self, first_node, second_node):
        nx_graph = self.to_networkx()
        nx_graph.add_edge(first_node, second_node)
        s2v_graph = S2VGraph(nx_graph)
        return s2v_graph, 1

    def add_edge_dynamically(self, first_node, second_node):
        self.dynamic_edges.append((first_node, second_node))
        self.node_degrees[first_node] += 1
        self.node_degrees[second_node] += 1
        return 1

    def populate_banned_actions(self, budget=None):
        if budget is not None:
            if budget < budget_eps:
                self.banned_actions = self.all_nodes_set
                return

        if self.first_node is None:
            self.banned_actions = self.get_invalid_first_nodes(budget)
        else:
            self.banned_actions = self.get_invalid_edge_ends(self.first_node, budget)

    def get_invalid_first_nodes(self, budget=None):
        return set([node_id for node_id in self.node_labels if self.node_degrees[node_id] == (self.num_nodes - 1)])

    def get_invalid_edge_ends(self, query_node, budget=None):
        results = set()
        results.add(query_node)

        existing_edges = self.edge_pairs.reshape(-1, 2)
        existing_left = existing_edges[existing_edges[:,0] == query_node]
        results.update(np.ravel(existing_left[:,1]))

        existing_right = existing_edges[existing_edges[:,1] == query_node]
        results.update(np.ravel(existing_right[:,0]))

        if self.dynamic_edges is not None:
            dynamic_left = [entry[0] for entry in self.dynamic_edges if entry[0] == query_node]
            results.update(dynamic_left)
            dynamic_right = [entry[1] for entry in self.dynamic_edges if entry[1] == query_node]
            results.update(dynamic_right)
        return results

    def init_dynamic_edges(self):
        self.dynamic_edges = []

    def apply_dynamic_edges(self):
        nx_graph = self.to_networkx()
        for edge in self.dynamic_edges:
            nx_graph.add_edge(edge[0], edge[1])
        return S2VGraph(nx_graph)

    def to_networkx(self):
        edges = self.convert_edges()
        g = nx.Graph()
        g.add_edges_from(edges)
        return g

    def convert_edges(self):
        return np.reshape(self.edge_pairs, (self.num_edges, 2))

    def display(self, ax=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx.draw_shell(nx_graph, with_labels=True, ax=ax)

    def display_with_positions(self, node_positions, ax=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx.draw(nx_graph, pos=node_positions, with_labels=True, ax=ax)

    def draw_to_file(self, filename):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_size_length = self.num_nodes / 5
        figsize = (fig_size_length, fig_size_length)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        self.display(ax=ax)
        fig.savefig(filename)
        plt.close()

    def get_adjacency_matrix(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            adj_matrix = np.asarray(nx.convert_matrix.to_numpy_matrix(nx_graph, nodelist=self.node_labels))

        return adj_matrix

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        gh = get_graph_hash(self, size=32, include_first=True)
        return f"Graph State with hash {gh}"

def get_graph_hash(g, size=32, include_first=False):
    if size == 32:
        hash_instance = xxhash.xxh32()
    elif size == 64:
        hash_instance = xxhash.xxh64()
    else:
        raise ValueError("only 32 or 64-bit hashes supported.")

    if include_first:
        if g.first_node is not None:
            hash_instance.update(np.array([g.first_node]))
        else:
            hash_instance.update(np.zeros(g.num_nodes))

    hash_instance.update(g.edge_pairs)
    graph_hash = hash_instance.intdigest()
    return graph_hash
