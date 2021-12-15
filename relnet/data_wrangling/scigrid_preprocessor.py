import csv

import networkx as nx

from relnet.data_wrangling.data_preprocessor import DataPreprocessor

class ScigridDataPreprocessor(DataPreprocessor):
    DS_NAME = "scigrid"

    NODE_FILE_NAME = "vertices_eu_power_160718.csvdata"
    EDGE_FILE_NAME = "links_eu_power_160718.csvdata"

    def clean_data(self, **kwargs):
        node_file = self.raw_dataset_dir / self.NODE_FILE_NAME
        edge_file = self.raw_dataset_dir / self.EDGE_FILE_NAME

        node_id_list = []
        coords_by_node = {}

        with open(node_file.resolve(), newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')

            for row in reader:
                nid = row['v_id']
                lat = float(row['lat'])
                lon = float(row['lon'])

                node_id_list.append(nid)
                coords_by_node[nid] = (lat, lon)

        geocoded_info = self.get_geocoded_data(coords_by_node)

        G = nx.Graph()
        for nid in node_id_list:
            if nid in geocoded_info:
                location = geocoded_info[nid]
                try:
                    country_code = location['address']['country_code']
                    G.add_node(nid, lat=coords_by_node[nid][0], lon=coords_by_node[nid][1], country_code=country_code)
                except BaseException:
                    print(f"couldn't find address for {nid}")
                    print(f"{location['address']}")


        edges_to_add = []
        with open(edge_file.resolve(), newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')

            for row in reader:
                edge_from = row['v_id_1']
                edge_to = row['v_id_2']

                if edge_from in G and edge_to in G:
                    edges_to_add.append((edge_from, edge_to))

        G.add_edges_from(edges_to_add)

        all_appearing_countries = set([loc['address']['country_code'] for loc in geocoded_info.values()])
        all_subgraphs = self.partition_graph_by_attribute(G, "country_code", all_appearing_countries)

        for country_code, subgraph in all_subgraphs.items():
            self.check_and_write_subgraph(country_code, subgraph)

    def get_geocoded_data(self, coords_by_node):
        if self.geocoder.exists_geocoded_data():
            geocoder_data = self.geocoder.read_geocoded_data()
            # setting country manually for two BorWin platforms
            geocoder_data['419']['address']['country_code'] = 'de'
            geocoder_data['420']['address']['country_code'] = 'de'
            return geocoder_data
        else:
            json_data = {}
            for nid, lat_lon_tuple in coords_by_node.items():
                print(f"doing node with id {nid}")
                location = self.geocoder.reverse(lat_lon_tuple)
                if location is not None:
                    json_data[nid] = location.raw
                    #print(f"geocoded {nid} as {json_data[nid]}!")
            self.geocoder.write_geocoded_data(json_data)



