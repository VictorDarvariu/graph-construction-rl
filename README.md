# Code for "Goal-directed graph construction using reinforcement learning"
**NOTE.** A lightweight version of the code that removes the GPU dependency is available at [https://github.com/VictorDarvariu/graph-construction-rl-lite](this repository), which makes it easier to run the code in case you don't have a GPU. Also useful if you just want to try out the algorithm.

This is the code for the article [_Goal-directed graph construction using reinforcement learning_](https://royalsocietypublishing.org/doi/10.1098/rspa.2021.0168) by [Victor-Alexandru Darvariu](https://victor.darvariu.me), [Stephen Hailes](http://www.cs.ucl.ac.uk/drupalpeople/S.Hailes.html) and [Mirco Musolesi](https://mircomusolesi.org), Proc. R. Soc. A. **477**:20210168. If you use this code, please consider citing:

```
@article{darvariu2021goal, 
  author = {Darvariu, Victor-Alexandru and Hailes, Stephen and Musolesi, Mirco}, 
  title = {Goal-directed graph construction using reinforcement learning}, 
  journal = {Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences},
  year = 2021, 
  month = {oct}, 
  publisher = {The Royal Society Publishing}, 
  volume = {477}, 
  number = {2254}, 
  pages={20210168},
  doi = {10.1098/rspa.2021.0168}, 
  url = {https://doi.org/10.1098/rspa.2021.0168}, 
}
```

## License
MIT.

## Prerequisites
**Please ensure that you clone this repository under the `relnet` root directory**, e.g. 
```
git clone git@github.com:VictorDarvariu/graph-construction-rl.git relnet
```  

Currently tested on Linux and MacOS (specifically, CentOS 7.4.1708 and Mac OS Big Sur 11.2.3), can also be adapted to Windows through [WSL](https://docs.microsoft.com/en-us/windows/wsl/about). The host machine requires NVIDIA CUDA toolkit version 9.0 or above (tested with NVIDIA driver version 384.81).
Makes heavy use of Docker, see e.g. [here](https://docs.docker.com/engine/install/centos/) for how to install on CentOS. Tested with Docker 19.03. The use of Docker largely does away with dependency and setup headaches, making it significantly easier to reproduce the reported results.

## Configuration
The Docker setup uses Unix groups to control permissions. You can reuse an existing group that you are a member of, or create a new group `groupadd -g GID GNAME` and add your user to it `usermod -a -G GNAME MYUSERNAME`. 

Create a file `relnet.env` at the root of the project (see `relnet_example.env`) and adjust the paths within: this is where some data generated by the container will be stored. Also specify the group ID and name created / selected above.

Add the following lines to your `.bashrc`, replacing `/home/john/git/relnet` with the path where the repository is cloned. 

```bash
export RN_SOURCE_DIR='/home/john/git/relnet'
set -a
. $RN_SOURCE_DIR/relnet.env
set +a

export PATH=$PATH:$RN_SOURCE_DIR/scripts
```

Make the scripts executable (e.g. `chmod u+x scripts/*`) the first time after cloning the repository, and run `apply_permissions.sh` in order to create and permission the necessary directories.

## Managing the containers
Some scripts are provided for convenience. To build the containers (note, this will take a significant amount of time e.g. 2 hours, as some packages are built from source):
```bash
update_container.sh
```
To start them:
```bash
manage_container_gpu.sh up
manage_container.sh up
```
To stop them:
```bash
manage_container_gpu.sh stop
manage_container.sh stop
```
To purge the queue and restart the containers (useful for killing tasks that were launched):
```bash
purge_and_restart.sh
```

## Adjusting the number of workers and threads
To take maximum advantage of your machine's capacity, you may want to tweak the number of threads for the GPU and CPU workers. This configuration is provided in `projectconfig.py`.
Additionally, you may want to enforce certain memory limits for your workers to avoid OOM errors. This can be tweaked in `docker-compose.yml` and `manage_container_gpu.sh`.

It is also relatively straightforward to add more workers from different machines you control. For this, you will need to mount the volumes on networked-attached storage (i.e., make sure paths provided in `relnet.env` are network-accessible) and adjust the location of backend and queue in `projectconfig.py` to a network location instead of localhost. On the other machines, only start the worker container (see e.g. `manage_container.sh`).  

## Setting up graph data

### Real-world data
The real-world network data can be downloaded from Figshare at [this URL](https://rs.figshare.com/articles/dataset/Real-World_Network_Data_from_Goal-directed_graph_construction_using_reinforcement_learning/16843170?backTo=/collections/Supplementary_material_from_Goal-directed_graph_construction_using_reinforcement_learning_/5672367).

- Copy the `rw_network_data.zip` file provided to `$RN_EXPERIMENT_DATA_DIR/real_world_graphs/raw_data`. 
- Then, unzip it `unzip rw_network_data.zip` 
- Re-grant group permissions so they can be read/modified by container users: `chgrp -R $RN_GNAME $RN_EXPERIMENT_DATA_DIR/real_world_graphs/raw_data`.
- Delete zip file `rm rw_network_data.zip`
- Run the following commands to wrangle the data into the expected formats:
```
docker exec -it relnet-worker-cpu /bin/bash -c "python relnet/data_wrangling/process_networks.py --dataset scigrid --task clean"
docker exec -it relnet-worker-cpu /bin/bash -c "python relnet/data_wrangling/process_networks.py --dataset scigrid --task process"

docker exec -it relnet-worker-cpu /bin/bash -c "python relnet/data_wrangling/process_networks.py --dataset euroroad --task clean"
docker exec -it relnet-worker-cpu /bin/bash -c "python relnet/data_wrangling/process_networks.py --dataset euroroad --task process"

```
### Synthetic data

Synthetic data will be automatically generated when the experiments are ran and stored to `$RN_EXPERIMENT_DIR/stored_graphs`.

## Accessing the services
There are several services running on the `manager` node.
- Jupyter notebook server: `http://localhost:8888`
- Flower for queue statistics: `http://localhost:5555`
- Tensorboard (currently disabled due to its large memory footprint): `http://localhost:6006`
- RabbitMQ management: `http://localhost:15672`

The first time Jupyter is accessed it will prompt for a token to enable password configuration, it can be grabbed by running `docker exec -it relnet-manager /bin/bash -c "jupyter notebook list"`.

## Accessing experiment data and results database
Experiment data and results are stored in part as files (under your configured `$RN_EXPERIMENT_DATA_DIR`) as well as in a MongoDB database.
To access the MongoDB database with a GUI, you can use a MongoDB client such as [Robo3T](https://robomongo.org/download) and point it to `http://localhost:27017`.

Some functionality is provided in `relnet/evaluation/storage.py` to insert and retrieve data, you can use it in e.g. analysis notebooks.

## Running experiments
Experiments are launched from the manager container and processed (in a parallel way) by the workers.
The file `relnet/evaluation/experiment_conditions.py` contains the configuration for the experiments reported in the paper, but you may modify e.g. agents, objective functions, hyperparameters etc. to suit your needs.
Then, you can launch all the experiments as follows:

```bash
# synthetic graph experiments
docker exec -d relnet-manager /bin/bash -c "source activate ucfadar-relnet; python run_experiments.py --which synth --experiment_part both --edge_percentage 1 --experiment_id synth_1 --force_insert_details"
docker exec -d relnet-manager /bin/bash -c "source activate ucfadar-relnet; python run_experiments.py --which synth --experiment_part both --edge_percentage 2.5 --experiment_id synth_2_5 --force_insert_details"
docker exec -d relnet-manager /bin/bash -c "source activate ucfadar-relnet; python run_experiments.py --which synth --experiment_part both --edge_percentage 5 --experiment_id synth_5 --force_insert_details"

# real-world graph experiment
docker exec -d relnet-manager /bin/bash -c "source activate ucfadar-relnet; python run_experiments.py --which real_world --experiment_part both --edge_percentage 2.5 --train_individually --experiment_id real_world --force_insert_details"

```

To check the progress of running experiments, you can use the `peek_eval_losses.py` tool as follows:
```bash
docker exec -it relnet-manager /bin/bash -c "source activate ucfadar-relnet; python relnet/experiment_launchers/peek_val_losses.py --experiment_id synth_1"
```
It will print out the validation losses and step numbers of current training runs.

## Reproducing the results

Jupyter notebooks are used to perform the data analysis and produce tables and figures. Navigate to `http://localhost:8888`, then notebooks folder.

The relationships between notebooks and tables/figures are as follows:
- `Evaluation_Models.ipynb`: Table 1, Figure 2, Figure 3 
- `Evaluation_Models_RealWorld_Individual.ipynb`: Table 2

Open the notebook selecting the `Python (RelNET)` kernel and run all cells. Resulting .pdf figures and .tex tables can be found at `$RN_EXPERIMENT_DIR/aggregate`.

### Problems with jupyter kernel
In case the kernel is not found, try reinstalling the kernel by running `docker exec -it -u 0 relnet-manager /bin/bash -c "source activate ucfadar-relnet; python -m ipykernel install --user --name relnet --display-name "Python (RelNET)"`

## Contact

If you face any issues or have any queries feel free to contact `v.darvariu@ucl.ac.uk` and I will be happy to assist.