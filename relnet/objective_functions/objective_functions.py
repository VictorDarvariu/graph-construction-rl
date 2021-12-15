import networkx as nx
from relnet.objective_functions.objective_functions_ext import *
from relnet.state.graph_state import get_graph_hash


def extract_kwargs(kwargs):
    num_mc_sims = 20
    random_seed = 42
    if 'num_mc_sims' in kwargs:
        num_mc_sims = kwargs['num_mc_sims']
    if 'random_seed' in kwargs:
        random_seed = kwargs['random_seed']
    return num_mc_sims, random_seed


class CriticalFractionRandom(object):
    name = "random_removal"
    upper_limit = 1.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        num_mc_sims, random_seed = extract_kwargs(kwargs)
        N, M, edges = s2v_graph.num_nodes, s2v_graph.num_edges, s2v_graph.edge_pairs
        graph_hash = get_graph_hash(s2v_graph)
        frac = critical_fraction_random(N, M, edges, num_mc_sims, graph_hash, random_seed)
        return frac


class CriticalFractionTargeted(object):
    name = "targeted_removal"
    upper_limit = 1.

    @staticmethod
    def compute(s2v_graph, **kwargs):
        num_mc_sims, random_seed = extract_kwargs(kwargs)
        N, M, edges = s2v_graph.num_nodes, s2v_graph.num_edges, s2v_graph.edge_pairs
        graph_hash = get_graph_hash(s2v_graph)
        frac = critical_fraction_targeted(N, M, edges, num_mc_sims, graph_hash, random_seed)
        return frac