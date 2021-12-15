import sys


sys.path.append('/relnet')

from relnet.evaluation.file_paths import FilePaths
from relnet.evaluation.storage import EvaluationStorage
from relnet.state.network_generators import get_graph_ids_to_iterate

from itertools import product

import argparse
from tasks import app_settings

def main():
    parser = argparse.ArgumentParser(description="Plain utility to check best validation losses so far.")
    parser.add_argument("--experiment_id", required=True, help="experiment id to use")
    args = parser.parse_args()

    experiment_id = args.experiment_id
    storage = EvaluationStorage()
    experiment_details = storage.get_experiment_details('model', experiment_id)
    fp = FilePaths('/experiment_data', experiment_id, setup_directories=False)
    agent_names = list(experiment_details['agents'])
    experiment_conditions = experiment_details['experiment_conditions']
    objective_functions = experiment_details['objective_functions']
    network_generators = experiment_details['network_generators']
    model_seeds = experiment_conditions['experiment_params']['model_seeds']
    train_individually = experiment_conditions['train_individually']

    for agent_name in agent_names:
        print(f"<<{agent_name}>>")
        for objective_function in objective_functions:
            for network_generator in network_generators:
                agent_grid = experiment_conditions['hyperparam_grids'][objective_function][agent_name]
                num_hyperparam_combs = len(list(product(*agent_grid.values())))

                for comb in range(num_hyperparam_combs):
                    df = storage.fetch_eval_curves(agent_name, comb, fp, objective_function, network_generator, model_seeds, train_individually)

                    if len(df) > 0:
                        graph_ids = get_graph_ids_to_iterate(train_individually, network_generator, fp)

                        for g_id in graph_ids:
                            max_steps = experiment_conditions['agent_budgets'][objective_function][agent_name]
                            num_completed = 0
                            num_started = 0
                            num_not_started = 0
                            num_total = len(model_seeds)


                            out_str_completed = ""
                            out_str_started = ""
                            out_str_not_started = ""
                            for seed in model_seeds:
                                if g_id is None:
                                    df_subset = df[(df['model_seed'] == seed) &
                                                   (df['network_generator'] == network_generator) &
                                                   (df['objective_function'] == objective_function)]
                                else:
                                    df_subset = df[(df['model_seed'] == seed) &
                                                   (df['network_generator'] == network_generator) &
                                                   (df['objective_function'] == objective_function) &
                                                   (df['graph_id'] == g_id)]

                                if len(df_subset) > 0:
                                    best_perf = df_subset['perf'].max()
                                    training_step = df_subset['timestep'].max()
                                    if training_step == max_steps:
                                        out_str_completed += f"{best_perf:.3f} [{seed}],  "
                                        num_completed += 1
                                    else:
                                        out_str_started += f"{best_perf:.3f} [{training_step}; {seed}],  "
                                        num_started +=1
                                else:
                                    out_str_not_started += f"{seed}, "
                                    num_not_started +=1

                            print(f"=================")
                            if g_id is None:
                                print(f"{comb},{network_generator},{objective_function}")
                            else:
                                print(f"{comb},{network_generator}--{g_id},{objective_function}")
                            print(f"=================")
                            if len(out_str_completed) > 0:
                                print(f"-----------------")
                                print(f"training completed: {num_completed}/{num_total}")
                                print(f"-----------------")
                                print(f"{out_str_completed}")
                            if len(out_str_started) > 0:
                                print(f"-----------------")
                                print(f"training started: {num_started}/{num_total}")
                                print(f"-----------------")
                                print(f"{out_str_started}")
                            if len(out_str_not_started) > 0:
                                print(f"-----------------")
                                print(f"training not started: {num_not_started}/{num_total}")
                                print(f"-----------------")
                                print(f"{out_str_not_started}")

if __name__ == "__main__":
    main()
