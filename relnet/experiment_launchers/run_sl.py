import sys
from pathlib import Path
sys.path.append('/relnet')

from relnet.agent.supervised.sl_agent import SLAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.file_paths import FilePaths
from relnet.objective_functions.objective_functions import CriticalFractionTargeted
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator


def get_gen_params():
    gp = {}
    gp['n'] = 100
    gp['m_ba'] = 2
    gp['m_percentage_er'] = 20
    gp['m_ws'] = 2
    gp['p_ws'] = 0.1
    gp['d_reg'] = 2
    gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])

    gp['alpha_kh'] = 10
    gp['beta_kh'] = 0.001
    return gp

def get_options(file_paths):
    options = {"log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "models_path": file_paths.models_dir,
               "restore_model": False,
               "batch_size": 50,
               "validation_check_interval": 1000,
               "max_validation_consecutive_steps": 3000}

    return options

def get_file_paths():
    parent_dir = '/experiment_data'
    experiment_id = 'development'
    file_paths = FilePaths(parent_dir, experiment_id)
    return file_paths

if __name__ == '__main__':
    num_training_steps = 300000
    # num_train_graphs = 10000
    # num_validation_graphs = 100
    # num_test_graphs = 20
    #
    num_train_graphs = 100
    num_validation_graphs = 1
    num_test_graphs = 100

    gen_params = get_gen_params()
    file_paths = get_file_paths()

    options = get_options(file_paths)
    graph_seeds = NetworkGenerator.construct_network_seeds(num_train_graphs, num_validation_graphs, num_test_graphs)
    train_graph_seeds, validation_graph_seeds, test_graph_seeds = graph_seeds

    #storage_root = Path('/experiment_data/stored_graphs')
    #kwargs = {'store_graphs': True, 'graph_storage_root': storage_root}

    kwargs = {'store_graphs': False}

    gen = BANetworkGenerator(**kwargs)

    train_graphs = gen.generate_many(gen_params, train_graph_seeds)
    validation_graphs = gen.generate_many(gen_params, validation_graph_seeds)
    test_graphs = gen.generate_many(gen_params, test_graph_seeds)

    edge_percentage = 5
    obj_fun_kwargs = {"random_seed": 42, "num_mc_sims": gen_params['n'] * 2}
    obj = CriticalFractionTargeted()

    targ_env = GraphEdgeEnv(obj, obj_fun_kwargs, edge_percentage)

    agent = SLAgent(targ_env)
    agent.setup(options, agent.get_default_hyperparameters())
    agent.train(train_graphs, validation_graphs, num_training_steps)
    avg_perf = agent.eval(test_graphs)