import argparse
import sys
from copy import copy

sys.path.append('/relnet')

from relnet.data_wrangling.data_preprocessor import DataPreprocessor
from relnet.data_wrangling.euroroad_preprocessor import EuroroadDataPreprocessor
from relnet.data_wrangling.scigrid_preprocessor import ScigridDataPreprocessor

supported_datasets = ["euroroad", "scigrid"]

def process_dataset(dataset, task, root_dir, additional_args):
    preprocessor_class = get_preprocessor_for_dataset(dataset)
    preprocessor = preprocessor_class(root_dir)
    preprocessor.execute_task(task, **additional_args)

def get_preprocessor_for_dataset(dataset):
    ds_preprocessors = {
                        EuroroadDataPreprocessor.DS_NAME: EuroroadDataPreprocessor,
                        ScigridDataPreprocessor.DS_NAME: ScigridDataPreprocessor,
                        }

    return ds_preprocessors[dataset]


def main():
    parser = argparse.ArgumentParser(description="Script to process raw real-world network data into canonical format.")
    parser.add_argument("--dataset", required=True, type=str,
                        help="Dataset to process.",
                        choices=supported_datasets + ["all"])

    parser.add_argument("--task", required=True, type=str,
                        help="Task to execute.",
                        choices=["clean", "process"])

    parser.add_argument("--root_dir", type=str, help="Root path where dataset is located.")
    parser.set_defaults(root_dir="/experiment_data/real_world_graphs")

    args = parser.parse_args()

    additional_args = {}

    dataset = args.dataset
    if dataset == "all":
        for ds in supported_datasets:
            process_dataset(ds, args.task, args.root_dir, additional_args)
    else:
        process_dataset(dataset,args.task,  args.root_dir, additional_args)

if __name__ == "__main__":
    main()