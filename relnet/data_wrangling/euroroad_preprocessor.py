import networkx as nx
import numpy as np

from relnet.data_wrangling.data_preprocessor import DataPreprocessor


class EuroroadDataPreprocessor(DataPreprocessor):
    DS_NAME = "euroroad"

    NODE_FILE_NAME = "ent.subelj_euroroad_euroroad.city.name"
    EDGE_FILE_NAME = "out.subelj_euroroad_euroroad"

    def clean_data(self, **kwargs):
        node_file = self.raw_dataset_dir / self.NODE_FILE_NAME
        nodes = []
        with open(node_file.resolve(), "r") as fh:
            for line in fh:
                node_name = line.strip()
                nodes.append(node_name)

        geocoded_info = self.get_geocoded_data(nodes)

        G = nx.Graph()
        for i, node in enumerate(nodes):
            if node in geocoded_info:
                location = geocoded_info[node]
                country_code = location['address']['country_code']
                G.add_node(i, lat=float(location['lat']), lon=float(location['lon']), country_code=country_code)

        edges_to_add = []
        edge_file = self.raw_dataset_dir / self.EDGE_FILE_NAME
        with open(edge_file.resolve(), "r") as fh:
            for _ in range(2):
                next(fh)
            for line in fh:
                edge_data = line.strip().split(sep=" ")
                edge_from, edge_to = int(edge_data[0]), int(edge_data[1])
                if edge_from in G and edge_to in G:
                    edges_to_add.append((edge_from, edge_to))

        G.add_edges_from(edges_to_add)

        all_appearing_countries = set([loc['address']['country_code'] for loc in geocoded_info.values()])
        all_subgraphs = self.partition_graph_by_attribute(G, "country_code", all_appearing_countries)

        for country_code, subgraph in all_subgraphs.items():
            self.check_and_write_subgraph(country_code, subgraph)

    def get_geocoded_data(self, nodes):
        if self.geocoder.exists_geocoded_data():
            return self.geocoder.read_geocoded_data()
        else:
            json_data = {}
            for i, city_name in enumerate(nodes):
                print(f"doing {i}/{len(nodes)}")
                location = self.geocoder.geocode_location(city_name)
                if location is not None:
                    json_data[city_name] = location.raw
                    print(f"geocoded {city_name}!")
            self.geocoder.write_geocoded_data(json_data)



