import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import networkx
import networkx as nx
import numpy as np
import json

from relnet.data_wrangling.geocoder import Geocoder


class DataPreprocessor(ABC):
    RAW_DATA_DIR_NAME = 'raw_data'
    CLEANED_DATA_DIR_NAME = 'cleaned_data'
    PROCESSED_DATA_DIR_NAME = 'processed_data'
    DATASET_METADATA_FILE_NAME = 'dataset_metadata.json'

    MIN_NETWORK_SIZE = 20
    MAX_NETWORK_SIZE = 50

    CANONICAL_LAT_ATTR_NAME = "lat"
    CANONICAL_LON_ATTR_NAME = "lon"

    CANONICAL_X_COORD_ATTR_NAME = "pos_x"
    CANONICAL_Y_COORD_ATTR_NAME = "pos_y"


    def __init__(self, root_dir_string):
        self.root_dir = Path(root_dir_string)
        self.raw_data_dir = self.root_dir / self.RAW_DATA_DIR_NAME
        self.cleaned_data_dir = self.root_dir / self.CLEANED_DATA_DIR_NAME
        self.processed_data_dir = self.root_dir / self.PROCESSED_DATA_DIR_NAME

        for d in [self.raw_data_dir, self.cleaned_data_dir, self.processed_data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.raw_dataset_dir = self.raw_data_dir / self.DS_NAME
        self.cleaned_dataset_dir = self.cleaned_data_dir / self.DS_NAME
        self.processed_dataset_dir = self.processed_data_dir / self.DS_NAME

        for d in [self.raw_dataset_dir, self.cleaned_dataset_dir, self.processed_dataset_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.geocoder = Geocoder(self.raw_dataset_dir, self.DS_NAME)

    def execute_task(self, task, **kwargs):
        if task == "clean":
            self.clean_data(**kwargs)
        elif task == "process":
            self.process_data(**kwargs)

    def check_connectedness(self, G):
        return nx.is_connected(G)

    def check_sizes(self, G):
        return len(G) > self.MIN_NETWORK_SIZE and len(G) < self.MAX_NETWORK_SIZE

    def extract_largest_cc(self, G):
        largest_cc = max(nx.connected_components(G), key=len)
        lcc_G = G.subgraph(largest_cc).copy()
        lcc_G_relabeled = nx.relabel.convert_node_labels_to_integers(lcc_G)
        return lcc_G_relabeled

    def check_graph_criteria(self, G):
        return self.check_sizes(G) and self.check_connectedness(G)

    @abstractmethod
    def clean_data(self, **kwargs):
        pass

    def process_data(self, **kwargs):
        cleaned_graph_files = sorted(self.cleaned_dataset_dir.glob("*.graphml"))
        dataset_metadata = {}

        num_graphs = 0
        graph_names = []

        for i, f in enumerate(cleaned_graph_files):
            cleaned_G = nx.readwrite.read_graphml(f.resolve())
            self.remove_location_attrs(cleaned_G)

            processed_filepath = self.processed_dataset_dir / f.name
            nx.readwrite.write_graphml(cleaned_G, processed_filepath.resolve())

            graph_name = f.stem
            num_graphs += 1
            graph_names.append(graph_name)

        dataset_metadata['num_graphs'] = num_graphs
        dataset_metadata['graph_names'] = graph_names

        metadata_filename = self.processed_dataset_dir / self.DATASET_METADATA_FILE_NAME

        with open(metadata_filename.resolve(), "w") as fp:
            json.dump(dataset_metadata, fp, indent=4)


    def remove_location_attrs(self, cleaned_G):
        self.remove_node_attrs(cleaned_G, self.CANONICAL_LAT_ATTR_NAME)
        self.remove_node_attrs(cleaned_G, self.CANONICAL_LON_ATTR_NAME)

    def partition_graph_by_attribute(self, G, attr, attr_possible_values):
        partitioned_graphs = {}
        for attr_value in attr_possible_values:
            nodes = (
                node
                for node, data
                in G.nodes(data=True)
                if data.get(attr) == attr_value
            )
            attr_subgraph = G.subgraph(nodes).copy()

            self.remove_node_attrs(attr_subgraph, attr)

            H = nx.relabel.convert_node_labels_to_integers(attr_subgraph)
            partitioned_graphs[attr_value] = H
        return partitioned_graphs

    def remove_node_attrs(self, G, attr):
        for (n, d) in G.nodes(data=True):
            del d[attr]

    def remove_edge_attrs(self, G, attr):
        for (t, f, d) in G.edges(data=True):
            del d[attr]

    def check_and_write_subgraph(self, country_code, subgraph):
        cleaned_filepath = self.cleaned_dataset_dir / f"{country_code}.graphml"
        is_connected = self.check_connectedness(subgraph)

        can_write = False
        if is_connected:
            if self.check_sizes(subgraph):
                can_write = True
        else:
            lcc = self.extract_largest_cc(subgraph)
            if self.check_sizes(lcc):
                can_write = True
                subgraph = lcc

        if can_write:
            nx.readwrite.write_graphml(subgraph, cleaned_filepath.resolve())