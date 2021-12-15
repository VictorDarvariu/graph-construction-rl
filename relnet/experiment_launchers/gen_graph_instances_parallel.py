import argparse
import math
import traceback
from copy import deepcopy

import pandas as pd
from celery import group

import sys
sys.path.append('/relnet')

from relnet.evaluation.experiment_conditions import get_exp_conditions
from relnet.evaluation.file_paths import FilePaths

from billiard.pool import Pool
from psutil import cpu_count

def main():
    parser = argparse.ArgumentParser(description="Generate graph instances in parallel")
    parser.add_argument("--pool_size_multiplier", required=True, type=float,
                        help="Multiplier for worker pool size, applied to number of logical cores.",
                        )
    args = parser.parse_args()
    file_paths = FilePaths('/experiment_data', None, setup_directories=False)

    storage_root = file_paths.graph_storage_dir
    logs_file = str(file_paths.construct_log_filepath())
    kwargs = {'store_graphs': True, 'graph_storage_root': storage_root, 'logs_file': logs_file}


    experiment_conditions = get_exp_conditions('synth', 0, False)

    gen_params = experiment_conditions.gen_params
    all_seeds = []
    all_seeds.extend(experiment_conditions.train_seeds)
    all_seeds.extend(experiment_conditions.validation_seeds)
    all_seeds.extend(experiment_conditions.test_seeds)

    num_procs = math.ceil(cpu_count(logical=True) * args.pool_size_multiplier)
    worker_pool = Pool(processes=num_procs)

    tasks = []
    for network_generator_class in experiment_conditions.network_generators:
        gen_instance = network_generator_class(**kwargs)

        for net_seed in all_seeds:
            tasks.append((gen_instance, gen_params, net_seed))

    for net_seed in worker_pool.starmap(call_network_generator, tasks):
        pass

    worker_pool.close()

def call_network_generator(generator_instance, params, net_seed):
    generator_instance.generate(params, net_seed)
    return net_seed

if __name__ == "__main__":
    main()
