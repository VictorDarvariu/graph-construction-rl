import json
import math
from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx

from relnet.data_wrangling.data_preprocessor import DataPreprocessor
from relnet.data_wrangling.euroroad_preprocessor import EuroroadDataPreprocessor
from relnet.data_wrangling.scigrid_preprocessor import ScigridDataPreprocessor

from relnet.evaluation.file_paths import FilePaths
from relnet.state.graph_state import S2VGraph
from relnet.utils.config_utils import get_logger_instance


class NetworkGenerator(ABC):
    enforce_connected = True

    def __init__(self, store_graphs=False, graph_storage_root=None, logs_file=None):
        super().__init__()
        self.store_graphs = store_graphs
        if self.store_graphs:
            self.graph_storage_root = graph_storage_root
            self.graph_storage_dir = graph_storage_root / self.name
            self.graph_storage_dir.mkdir(parents=True, exist_ok=True)

        if logs_file is not None:
            self.logger_instance = get_logger_instance(logs_file)
        else:
            self.logger_instance = None

    def generate(self, gen_params, random_seed):
        if self.store_graphs:
            filename = self.get_data_filename(gen_params, random_seed)
            filepath = self.graph_storage_dir / filename

            should_create = True
            if filepath.exists():
                try:
                    instance = self.read_graphml_with_ordered_int_labels(filepath)
                    state = self.post_generate_instance(instance)
                    should_create = False
                except Exception:
                    should_create = True

            if should_create:
                instance = self.generate_instance(gen_params, random_seed)
                state = self.post_generate_instance(instance)
                nx.readwrite.write_graphml(instance, filepath.resolve())

                drawing_filename = self.get_drawing_filename(gen_params, random_seed)
                drawing_path = self.graph_storage_dir / drawing_filename
                state.draw_to_file(drawing_path)
        else:
            instance = self.generate_instance(gen_params, random_seed)
            state = self.post_generate_instance(instance)

        return state

    def read_graphml_with_ordered_int_labels(self, filepath):
        instance = nx.readwrite.read_graphml(filepath.resolve())
        num_nodes = len(instance.nodes)
        relabel_map = {str(i): i for i in range(num_nodes)}
        nx.relabel_nodes(instance, relabel_map, copy=False)

        G = nx.Graph()
        G.add_nodes_from(sorted(instance.nodes(data=True)))
        G.add_edges_from(instance.edges(data=True))

        return G

    def generate_many(self, gen_params, random_seeds):
        return [self.generate(gen_params, random_seed) for random_seed in random_seeds]

    @abstractmethod
    def generate_instance(self, gen_params, random_seed):
        pass

    @abstractmethod
    def post_generate_instance(self, instance):
        pass

    def get_data_filename(self, gen_params, random_seed):
        n = gen_params['n']
        filename = f"{n}-{random_seed}.graphml"
        return filename

    def get_drawing_filename(self, gen_params, random_seed):
        n = gen_params['n']
        filename = f"{n}-{random_seed}.png"
        return filename

    @staticmethod
    def compute_number_edges(n, edge_percentage):
        total_possible_edges = (n * (n - 1)) / 2
        return int(math.ceil((total_possible_edges * edge_percentage / 100)))

    @staticmethod
    def construct_network_seeds(num_train_graphs, num_validation_graphs, num_test_graphs):
        train_seeds = list(range(0, num_train_graphs))
        validation_seeds = list(range(num_train_graphs, num_train_graphs + num_validation_graphs))
        offset = num_train_graphs + num_validation_graphs
        test_seeds = list(range(offset, offset + num_test_graphs))
        return train_seeds, validation_seeds, test_seeds


class OrdinaryGraphGenerator(NetworkGenerator, ABC):
    def post_generate_instance(self, instance):
        state = S2VGraph(instance)
        state.populate_banned_actions()
        return state


class GNMNetworkGenerator(OrdinaryGraphGenerator):
    name = 'random_network'
    num_tries = 10000

    def generate_instance(self, gen_params, random_seed):
        number_vertices = gen_params['n']
        number_edges = gen_params['m']

        if not self.enforce_connected:
            random_graph = nx.generators.random_graphs.gnm_random_graph(number_vertices, number_edges, seed=random_seed)
            return random_graph
        else:
            for try_num in range(0, self.num_tries):
                random_graph = nx.generators.random_graphs.gnm_random_graph(number_vertices, number_edges,
                                                                            seed=(random_seed + (try_num * 1000)))
                if nx.is_connected(random_graph):
                    return random_graph
                else:
                    continue
            raise ValueError("Maximum number of tries exceeded, giving up...")

class BANetworkGenerator(OrdinaryGraphGenerator):
    name = 'barabasi_albert'

    def generate_instance(self, gen_params, random_seed):
        n, m = gen_params['n'], gen_params['m_ba']
        ba_graph = nx.generators.random_graphs.barabasi_albert_graph(n, m, seed=random_seed)
        return ba_graph

class RealWorldNetworkGenerator(OrdinaryGraphGenerator, ABC):
    def __init__(self, store_graphs=False, graph_storage_root=None, logs_file=None, original_dataset_dir=None):
        super().__init__(store_graphs=store_graphs, graph_storage_root=graph_storage_root, logs_file=logs_file)

        if original_dataset_dir is None:
            raise ValueError(f"{original_dataset_dir} cannot be None")
        self.original_dataset_dir = original_dataset_dir

        graph_metadata_file = original_dataset_dir / self.name / DataPreprocessor.DATASET_METADATA_FILE_NAME
        with open(graph_metadata_file.resolve(), "r") as fh:
            graph_metadata = json.load(fh)
            self.num_graphs, self.graph_names = graph_metadata['num_graphs'], graph_metadata['graph_names']

    def generate_instance(self, gen_params, random_seed):
        graph_name = self.get_graph_name(random_seed)

        filepath = self.original_dataset_dir / self.name / f"{graph_name}.graphml"

        nx_graph = self.read_graphml_with_ordered_int_labels(filepath)

        return nx_graph

    def get_num_graphs(self):
        return self.num_graphs

    def get_graph_name(self, random_seed):
        graph_idx = random_seed % self.num_graphs
        graph_name = self.graph_names[graph_idx]
        return graph_name

    def get_data_filename(self, gen_params, random_seed):
        graph_name = self.get_graph_name(random_seed)
        filename = f"{random_seed}-{graph_name}.graphml"
        return filename

    def get_drawing_filename(self, gen_params, random_seed):
        graph_name = self.get_graph_name(random_seed)
        filename = f"{random_seed}-{graph_name}.png"
        return filename


class EuroroadNetworkGenerator(RealWorldNetworkGenerator):
    name = EuroroadDataPreprocessor.DS_NAME


class ScigridNetworkGenerator(RealWorldNetworkGenerator):
    name = ScigridDataPreprocessor.DS_NAME



def check_is_real_world(generator_class):
    if type(generator_class) == str:
        # given as a name from retrieved experiment conditions
        subclasses = [str(c.name) for c in RealWorldNetworkGenerator.__subclasses__() if hasattr(c, "name")]
        if generator_class in subclasses:
            return True
    else:
        return issubclass(generator_class, RealWorldNetworkGenerator)

def get_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass

def retrieve_generator_class(generator_class_name):
    subclass = [c for c in get_subclasses(NetworkGenerator) if hasattr(c, "name") and generator_class_name == c.name][0]
    return subclass


def create_generator_instance(generator_class, file_paths):
    processed_graph_dir = file_paths.processed_graphs_dir if isinstance(file_paths, FilePaths) else Path(file_paths['processed_graphs_dir'])
    graph_storage_dir = file_paths.graph_storage_dir if isinstance(file_paths, FilePaths) else Path(file_paths['graph_storage_dir'])
    log_filename = file_paths.construct_log_filepath() if isinstance(file_paths, FilePaths) else \
        Path(file_paths['logs_dir']) / FilePaths.construct_log_filename()

    if check_is_real_world(generator_class):
        original_dataset_dir = processed_graph_dir
        gen_kwargs = {'store_graphs': True, 'graph_storage_root': graph_storage_dir, 'logs_file': log_filename, 'original_dataset_dir': original_dataset_dir}
    else:
        gen_kwargs = {'store_graphs': True, 'graph_storage_root': graph_storage_dir, 'logs_file': log_filename}
    if type(generator_class) == str:
        generator_class = retrieve_generator_class(generator_class)
    gen_instance = generator_class(**gen_kwargs)
    return gen_instance


def get_graph_ids_to_iterate(train_individually, generator_class, file_paths):
    if train_individually:
        graph_ids = []
        network_generator_instance = create_generator_instance(generator_class, file_paths)
        num_graphs = network_generator_instance.get_num_graphs()
        for g_num in range(num_graphs):
            graph_id = network_generator_instance.get_graph_name(g_num)
            graph_ids.append(graph_id)
        return graph_ids
    else:
        return [None]