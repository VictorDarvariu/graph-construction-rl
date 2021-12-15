import argparse
import random
import traceback
import uuid
import time
from copy import deepcopy
from datetime import datetime

from celery import group
from relnet.agent.supervised.sl_agent import SLAgent

from relnet.agent.baseline.baseline_agent import GreedyAgent, BaselineAgent
from relnet.evaluation.experiment_conditions import get_exp_conditions, SyntheticGraphsExperimentConditions
from relnet.evaluation.file_paths import FilePaths
from relnet.evaluation.eval_utils import generate_search_space, construct_search_spaces
from relnet.evaluation.storage import EvaluationStorage
from relnet.state.network_generators import RealWorldNetworkGenerator, create_generator_instance, \
    get_graph_ids_to_iterate
from relnet.utils.config_utils import get_logger_instance
from tasks import optimize_hyperparams_task, evaluate_for_network_seed_task


def run_hyperopt_part(experiment_conditions, algorithm_class, parent_dir, existing_experiment_id,
                      force_insert_details):
    storage = EvaluationStorage()

    experiment_started_datetime = datetime.now()
    started_str = experiment_started_datetime.strftime(FilePaths.DATE_FORMAT)
    started_millis = experiment_started_datetime.timestamp()

    if existing_experiment_id is not None:
        experiment_id = existing_experiment_id
    else:
        experiment_id = str(uuid.uuid4())
    file_paths = FilePaths(parent_dir, experiment_id)

    parameter_search_spaces = construct_search_spaces(experiment_conditions)

    if existing_experiment_id is None or force_insert_details:
        storage.insert_experiment_details(
            algorithm_class,
            file_paths,
            experiment_conditions,
            started_str,
            started_millis,
            parameter_search_spaces,
            experiment_id)

    logger = get_logger_instance(str(file_paths.construct_log_filepath()))
    logger.info(f"{datetime.now().strftime(FilePaths.DATE_FORMAT)} Started hyperparameter optimisations and training.")
    run_hyperparameter_optimisations(algorithm_class,
                                     file_paths,
                                     experiment_conditions)
    logger.info(
        f"{datetime.now().strftime(FilePaths.DATE_FORMAT)} Completed hyperparameter optimisations and training.")


def run_eval_part(experiment_conditions, algorithm_class, parent_dir, existing_experiment_id):
    storage = EvaluationStorage()
    if existing_experiment_id is not None:
        experiment_id = existing_experiment_id
    else:
        experiment_id = storage.find_latest_experiment_id(algorithm_class,
                                                          experiment_conditions.possible_edge_percentage)
    file_paths = FilePaths(parent_dir, experiment_id, setup_directories=False)

    logger = get_logger_instance(str(file_paths.construct_log_filepath()))

    eval_tasks = []
    for multiplier in experiment_conditions.size_multipliers:
        logger.info(f"working with size multiplier <<{multiplier}>>.")

        multiplier_tasks = construct_eval_tasks_for_multiplier(multiplier,
                                                               algorithm_class,
                                                               experiment_id,
                                                               file_paths,
                                                               experiment_conditions,
                                                               storage)

        eval_tasks.extend(multiplier_tasks)

    logger.info(f"about to run {len(eval_tasks)} evaluation tasks.")
    random.shuffle(eval_tasks)
    g = group(eval_tasks)
    try:
        results = g().get()
        for results_rows in results:
            storage.insert_evaluation_results(algorithm_class, experiment_id, results_rows)
    except Exception:
        logger.error("got an exception while processing evaluation results.")
        logger.error(traceback.format_exc())


def construct_eval_tasks_for_multiplier(multiplier,
                                        algorithm_class,
                                        experiment_id,
                                        file_paths,
                                        original_experiment_conditions,
                                        storage):
    experiment_conditions = deepcopy(original_experiment_conditions)
    experiment_conditions.update_size_dependant_params(multiplier)

    logger = get_logger_instance(str(file_paths.construct_log_filepath()))



    tasks = []

    train_individually = experiment_conditions.train_individually
    try:
        optimal_hyperparams = storage.retrieve_optimal_hyperparams(algorithm_class, experiment_id, experiment_conditions.model_seeds_to_skip, train_individually)
    except KeyError:
        logger.warn("no hyperparameters retrieved as no configured agents require them.")
        optimal_hyperparams = {}

    for network_generator in experiment_conditions.network_generators:
        for objective_function in experiment_conditions.objective_functions:
            relevant_agents = deepcopy(experiment_conditions.relevant_agents)
            relevant_agents.extend(experiment_conditions.agents_baseline[objective_function.name])
            for agent in relevant_agents:
                if agent.algorithm_name == GreedyAgent.algorithm_name:
                    if experiment_conditions.gen_params['size_multiplier'] > experiment_conditions.greedy_size_threshold:
                        logger.info(
                            f"Skipping greedy agent as we are above size modifier "
                            f"{experiment_conditions.greedy_size_threshold}")
                        continue

                is_baseline = issubclass(agent, BaselineAgent)
                is_sl = (agent == SLAgent)
                hyperparams_needed = (not is_baseline) or is_sl

                graph_ids_to_iterate = get_graph_ids_to_iterate(train_individually, network_generator, file_paths)
                for idx, g_id in enumerate(graph_ids_to_iterate):
                    if not train_individually:
                        setting = (network_generator.name, objective_function.name, agent.algorithm_name)
                    else:
                        setting = (network_generator.name, objective_function.name, agent.algorithm_name, g_id)

                    best_hyperparams, best_hyperparams_id = optimal_hyperparams[setting] if hyperparams_needed else ({}, -1)

                    if g_id is None:
                        experiment_conditions.set_generator_seeds()
                        test_seeds = experiment_conditions.test_seeds
                    else:
                        exp_copy = deepcopy(experiment_conditions)
                        exp_copy.set_generator_seeds_individually(idx, len(graph_ids_to_iterate))
                        test_seeds = [exp_copy.test_seeds[0]]

                    for net_seed in test_seeds:
                        tasks.append(evaluate_for_network_seed_task.s(agent,
                                                                      objective_function,
                                                                      network_generator,
                                                                      best_hyperparams,
                                                                      best_hyperparams_id,
                                                                      experiment_conditions,
                                                                      file_paths,
                                                                      net_seed,
                                                                      graph_id=g_id))


    return tasks


def run_hyperparameter_optimisations(algorithm_class,
                                     file_paths,
                                     experiment_conditions):
    relevant_agents = experiment_conditions.relevant_agents
    experiment_params = experiment_conditions.experiment_params
    model_seeds = experiment_params['model_seeds']

    hyperopt_tasks = []

    for network_generator in experiment_conditions.network_generators:
        for obj_fun in experiment_conditions.objective_functions:
            for agent in relevant_agents:
                agent_param_grid = experiment_conditions.hyperparam_grids[obj_fun.name][agent.algorithm_name]

                hyperopt_tasks.extend(
                    construct_parameter_search_tasks(
                        agent,
                        obj_fun,
                        network_generator,
                        experiment_conditions,
                        file_paths,
                        agent_param_grid,
                        model_seeds))


    logger = get_logger_instance(str(file_paths.construct_log_filepath()))
    logger.info(f"about to run {len(hyperopt_tasks)} hyperparameter optimisation tasks.")

    random.shuffle(hyperopt_tasks)
    g = group(hyperopt_tasks)
    results = g().get()
    return results


def construct_parameter_search_tasks(agent,
                                     objective_function,
                                     network_generator,
                                     experiment_conditions,
                                     file_paths,
                                     parameter_grid,
                                     model_seeds):
    keys = list(parameter_grid.keys())
    local_tasks = []
    search_space = generate_search_space(parameter_grid)

    additional_opts = {}

    for hyperparams_id, combination in search_space.items():
        hyperparams = {}
        for idx, param_value in enumerate(tuple(combination)):
            param_key = keys[idx]
            hyperparams[param_key] = param_value

        for model_seed in model_seeds:

            setting = (network_generator.name, objective_function.name, agent.algorithm_name)

            if setting in experiment_conditions.model_seeds_to_skip:
                if model_seed in experiment_conditions.model_seeds_to_skip[setting]:
                    print(f"skipping seed {model_seed} for setting {setting} as configured.")
                    continue

            graph_ids_to_iterate = get_graph_ids_to_iterate(experiment_conditions.train_individually, network_generator, file_paths)
            for idx, g_id in enumerate(graph_ids_to_iterate):
                exp_copy = deepcopy(experiment_conditions)

                if g_id is None:
                    exp_copy.set_generator_seeds()
                else:
                    exp_copy.set_generator_seeds_individually(idx, len(graph_ids_to_iterate))

                model_identifier_prefix = file_paths.construct_model_identifier_prefix(agent.algorithm_name,
                                                                                       objective_function.name,
                                                                                       network_generator.name,
                                                                                       model_seed, hyperparams_id,
                                                                                       graph_id=g_id)
                local_tasks.append(optimize_hyperparams_task.s(agent,
                                                               objective_function,
                                                               network_generator,
                                                               exp_copy,
                                                               file_paths,
                                                               hyperparams,
                                                               hyperparams_id,
                                                               model_seed,
                                                               model_identifier_prefix,
                                                               additional_opts=additional_opts))

    return local_tasks


def main():
    parser = argparse.ArgumentParser(description="Start running suite of experiments.")
    parser.add_argument("--which", required=True, type=str,
                        help="Which experiment to run.",
                        choices=["synth", "real_world"])
    parser.add_argument("--experiment_part", required=True, type=str,
                        help="Whether to run hyperparameter optimisation, evaluation, or both.",
                        choices=["hyperopt", "eval", "both"])
    parser.add_argument("--parent_dir", type=str, help="Root path for storing experiment data.")
    parser.add_argument("--experiment_id", required=False, help="experiment id to use")


    parser.add_argument("--run_num_start", type=int, required=False, help="Run number interval start [inclusive].\n"
                                                                          "If specified, together with run_num_end, "
                                                                          "restricts models to train/evaluate to a "
                                                                          "subset based on their run_number.")
    parser.add_argument("--run_num_end", type=int, required=False, help="Run number interval end [inclusive].")

    parser.add_argument("--edge_percentage", type=float, required=False, help="Percentage of possible edges to be added"
                                                                              " to graph. Default is 1%.")

    parser.add_argument('--force_insert_details', dest='force_insert_details', action='store_true', help="Whether to force insert experiment details, even if experiment id provided exists already.")
    parser.set_defaults(force_insert_details=False)


    parser.add_argument('--train_individually', dest='train_individually', action='store_true', help="Whether to train/validate/test on a single graph.")
    parser.set_defaults(train_individually=False)
    parser.set_defaults(edge_percentage=1.)
    parser.set_defaults(parent_dir="/experiment_data")
    args = parser.parse_args()

    experiment_conditions = get_exp_conditions(args.which, args.edge_percentage, args.train_individually)

    algorithm_class = "model"
    experiment_conditions.update_relevant_agents(algorithm_class)


    if args.run_num_start is not None:
        run_start = args.run_num_start
        run_end = args.run_num_end
        assert run_end is not None, "if run_num_start is defined, run_num_end must also be defined"
        num_runs = experiment_conditions.experiment_params['num_runs']
        assert 0 <= run_start <= run_end < num_runs, "run_num_start, run_num_end must satisfy 0 <= run_num_start " \
                                                     "<= run_num_end < num_runs"

        experiment_conditions.extend_seeds_to_skip(run_start, run_end)
        print(f"after updating exp conditions, seeds to skip are:")
        print(experiment_conditions.model_seeds_to_skip)

    if args.experiment_part == "both":
        run_hyperopt_part(experiment_conditions, algorithm_class, args.parent_dir, args.experiment_id, args.force_insert_details)
        time.sleep(60)
        run_eval_part(experiment_conditions, algorithm_class, args.parent_dir, args.experiment_id)
    elif args.experiment_part == "hyperopt":
        run_hyperopt_part(experiment_conditions, algorithm_class, args.parent_dir, args.experiment_id, args.force_insert_details)
    elif args.experiment_part == "eval":
        run_eval_part(experiment_conditions, algorithm_class, args.parent_dir, args.experiment_id)


if __name__ == "__main__":
    main()
