import sys
from pathlib import Path

sys.path.append('/relnet')

from relnet.agent.pytorch_agent import PyTorchAgent
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.file_paths import FilePaths
from relnet.objective_functions.objective_functions import CriticalFractionTargeted, CriticalFractionRandom
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator, ScigridNetworkGenerator, \
    EuroroadNetworkGenerator

def get_gen_params():
    gp = {}
    gp['n'] = 20
    gp['m_ba'] = 2
    gp['m_percentage_er'] = 20
    gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
    return gp

def get_options(file_paths):
    options = {"log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "models_path": file_paths.models_dir,
               "restore_model": False}
    return options

def get_file_paths():
    parent_dir = '/experiment_data'
    experiment_id = 'development'
    file_paths = FilePaths(parent_dir, experiment_id)
    return file_paths

if __name__ == '__main__':
    num_training_steps = 5000
    num_train_graphs = 10000
    num_validation_graphs = 100
    num_test_graphs = 20

    gen_params = get_gen_params()
    file_paths = get_file_paths()

    options = get_options(file_paths)
    storage_root = Path('/experiment_data/stored_graphs')
    original_dataset_dir = Path('/experiment_data/real_world_graphs/processed_data')
    #kwargs = {'store_graphs': True, 'graph_storage_root': storage_root, 'original_dataset_dir': original_dataset_dir}
    kwargs = {'store_graphs': True, 'graph_storage_root': storage_root}

    gen = BANetworkGenerator(**kwargs)

    # g_num = 1
    # num_graphs = gen.get_num_graphs()
    # train_graph_seeds = [g_num + (i * num_graphs) for i in range(1, PyTorchAgent.DEFAULT_BATCH_SIZE+1)]
    # validation_graph_seeds = [g_num]
    # test_graph_seeds = [g_num]
    train_graph_seeds, validation_graph_seeds, test_graph_seeds = NetworkGenerator.construct_network_seeds(num_train_graphs, num_validation_graphs, num_test_graphs)

    train_graphs = gen.generate_many(gen_params, train_graph_seeds)
    validation_graphs = gen.generate_many(gen_params, validation_graph_seeds)
    test_graphs = gen.generate_many(gen_params, test_graph_seeds)

    edge_percentage = 1
    obj_fun_kwargs = {"random_seed": 42, "num_mc_sims": gen_params['n'] * 2}
    targ_env = GraphEdgeEnv(CriticalFractionTargeted(), obj_fun_kwargs, edge_percentage)

    agent = RNetDQNAgent(targ_env)
    agent.setup(options, agent.get_default_hyperparameters())
    agent.train(train_graphs, validation_graphs, num_training_steps)
    avg_perf = agent.eval(test_graphs)