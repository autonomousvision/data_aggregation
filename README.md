## Exploring Data Aggregation for Urban Driving
This repository contains the code for the CVPR 2020 paper [Exploring Data Aggregation in Policy Learning for Vision-based Urban Autonomous Driving](http://www.cvlibs.net/publications/Prakash2020CVPR.pdf). It is built on top of the [COiLTRAiNE](https://github.com/felipecode/coiltraine) and [CARLA 0.8.4 data-collector](https://github.com/carla-simulator/data-collector) frameworks.

If you find this code useful, please cite:
```
  @inproceedings{Prakash2020CVPR,
        title = {Exploring Data Aggregation in Policy Learning for Vision-based Urban Autonomous Driving},
        author = {Prakash, Aditya and Behl, Aseem and Ohn-Bar, Eshed and Chitta, Kashyap and Geiger, Andreas},
        booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2020}
   }
  
  @inproceedings{Codevilla2019ICCV,
        title = {Exploring the Limitations of Behavior Cloning for Autonomous Driving},
        author = {Codevilla, Felipe and Santana, Eder and López, Antonio M. and Gaidon, Adrien},
        booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
        year = {2019}
   }
```

### Introduction
This work explores the limitaions of [DAgger](http://proceedings.mlr.press/v15/ross11a/ross11a.pdf) for urban autonomous driving in terms of inability to capture critical states, generalize to new environments and susceptability to high variance. It proposes simple modifications to the DAgger algorithm by incorporating critical states and a replay buffer mechanism which helps the driving policy to progressively focus on the high uncertainty regions of its state distribution, thereby leading to better empirical performance.

### Setup
The required dependencies can be installed by using the provided conda environment requirements file.
```
conda env create -f requirements.yaml
conda activate coiltraine
```

### Data Generation 
It is recommended to use the [CARLA Gear](https://drive.google.com/file/d/1X52PXqT0phEi5WEWAISAQYZs-Ivx4VoE/view) version of CARLA 0.8.4 with [Docker](https://carla.readthedocs.io/en/latest/build_docker/) for data generation. For more information on this version, refer to the [CARLA data-collector](https://github.com/carla-simulator/data-collector) framework.

#### Setting up CARLA Gear
Download the CARLA Gear server from [this link](https://drive.google.com/file/d/1X52PXqT0phEi5WEWAISAQYZs-Ivx4VoE/view) and install [Docker](https://carla.readthedocs.io/en/latest/build_docker/).

Clone the main CARLA repository.
```
git clone https://github.com/carla-simulator/carla.git <carla_folder>
```
Build a docker image of the CARLA Gear server.
```
docker image build -f <carla_folder>/Util/Docker/Release.Dockerfile -t carlagear <path-to-carlagear/>CarlaGear
```

#### Data Collection
Once CARLA Gear is setup, data collection can be run using `multi_gpu_collection.py` with the config file stored in the path `dataset_configurations/<config_file>.py`. There are 3 modes for data generation - expert, dagger and dart. Refer to the config file for requirements for each mode. 

- Off-policy data generation using expert policy:
```
python3 multi_gpu_collection.py -ids <gpu_ids> -n <num_collectors> -g <collectors_per_gpu> -e <num_episodes_per_collector> -pt <data_path> -d <config_file> -ct carlagear -m expert
```

- On-policy data generation using DAgger:
```
python3 multi_gpu_collection.py -ids <gpu_ids> -n <num_collectors> -g <collectors_per_gpu> -e <num_episodes_per_collector> -pt <data_path> -d <config_file> -ct carlagear -m dagger
```

- Off-policy data generation with noise injection using [DART](https://arxiv.org/pdf/1703.09327.pdf):
```
python3 multi_gpu_collection.py -ids <gpu_ids> -n <num_collectors> -g <collectors_per_gpu> -e <num_episodes_per_collector> -pt <data_path> -d <config_file> -ct carlagear -m dart
```

#### Data Processing
The data is generated in [this format](https://github.com/carla-simulator/data-collector/blob/master/docs/dataset_format_description.md) at a resolution of 800x600. It is then processed to a resolution of 200x88 using `tools/post_process.py`. The episode names follow the format `episode_<num>` and `post_process.py` operates on the episodes in the defined range between `<start_episode_num>` and `<end_episode_num>`.
```
python3 tools/post_process.py -pt <data_path> -e <start_episode_num> -t <end_episode_num> -ds -dd
```

#### Sampling Methods
The on-policy data can be further sampled using the sampling mechanism defined in Section 3.3 of the paper.

- For Task-based and Policy & Expert-based sampling, run `tools/filter_dagger_data.py`. The settings for each of the sampling methods can be defined individually in `filter_dagger_data.py`.
```
python3 tools/filter_dagger_data.py <source_dir> <target_dir> <start_episode_num> <end_episode_num>
```

- For Policy-based sampling, first save the prefinal layer activations using `coil_core/save_activations.py`, then compute the variance due to multiple runs with test-time dropout using `coil_core/run_entropy.py` and finally sample the on-policy data using `tools/filter_dagger_data_var.py`. Refer to `input/coil_dataset.py` regarding generation of the preload file.
```
python3 coil_core/save_activations.py --gpus <gpu_id> --dataset_name <preload_name_of_dataset> --config <path_to_yaml_config_file> --checkpoint <model_checkpoint> --save_path <path_to_save_activations>
python3 coil_core/run_entropy.py  --gpus <gpu_id> --dataset_name <preload_name_of_dataset> --config <path_to_yaml_config_file> --checkpoint <model_checkpoint> --save_path <path_to_save_computed_variance>
python3 tools/filter_dagger_data_var.py --source_dir <source_dir> --target_dir <target_dir> --preload <path_to_preload_file> --var_file <path_to_saved_variance>
```

### Training
Once the data is generated, set the environment variable `COIL_DATASET_PATH` to the location containing the training and validation datasets.
```
export COIL_DATASET_PATH = <path_to_data>
```

For standalone training:
```
python3 coiltraine.py --gpus <gpu_id> --folder <exp_batch> --exp <exp_alias> --single-process train
```
where `<exp_batch>` is the [experiment batch](https://github.com/felipecode/coiltraine/blob/master/docs/configuration.md) containing all the experiments to be trained or validated and `<exp_alias>` is the name of the config file containing details of the experiment (`configs/<exp_batch>/<exp_alias>.yaml`). For multi-gpu training, specify a string of gpu ids in `<gpu_ids>`.

For standalone validation:
```
python3 coiltraine.py --gpus <gpu_id> --folder <exp_batch> -vd <validation_dataset> --exp <exp_alias> --single-process validation --no-train
```
where `<validation_dataset>` must be in `COIL_DATASET_PATH` and the model checkpoints to be validated are specified in `configs/<exp_batch>/<exp_alias>.yaml`.

For joint training and validation:
```
python3 coiltraine.py --gpus <gpu_id> --folder <exp_batch> -vd <validation_dataset>
```
This results in a live terminal screen showing a running status of the on-going training or validation. To run standalone training and validation in this mode, remove `--exp` and `--single-process` flags while running `coiltraine.py`. For more details, refer to the [COiLTRAiNE](https://github.com/felipecode/coiltraine) framework.

### Evaluation
The model checkpoints to be evaluated are specified in `configs/<exp_batch>/<exp_alias>.yaml`. For a list of evaluation suites with different environmental conditions, see `drive/suites`.

Pretrained models:
We have provided pretrained models for CILRS+ and 3 iterations of DAgger+, DART and DA-RB+ (Table 1 and Fig. 2 of the main paper). To download the pretrained models, run:
```
wget https://s3.eu-central-1.amazonaws.com/avg-projects/data_aggregation/models.zip
unzip models.zip
```

For standalone evaluation:
```
python3 coiltraine.py --gpus <gpu_id> --folder <exp_batch> -de <evaluation_suite> --exp <exp_alias> --single-process drive --no-train --docker <docker_image>
```
where `<evaluation_suite>` is generally one of `NocrashTraining_Town01`, `NocrashNewWeather_Town01`, `NocrashNewTown_Town02` and `NocrashNewWeatherTown_Town02`. For evaluating an ensemble of models, refer to `drive/coil_agent.py`

For joint training, validation and evaluation:
```
python3 coiltraine.py --gpus <gpu_id> --folder <exp_batch> -vd <validation_dataset> -de <evaluation_suite> --docker <docker_image>
```

For generating GradCAM attention maps:
```
python3 coil_core/grad_cam.py --gpus <gpu_id> --dataset_path <path_to_data> --preload_name <npy_preload_file> --config <path_to_yaml_config_file> --checkpoint <model_checkpoint> --gradcam_path <path_to_save_gradcam_maps> --type <dir_name_to_save_gradcam_maps>
```

### Related Works on Online Learning
- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](http://proceedings.mlr.press/v15/ross11a/ross11a.pdf)
- [Efficient Reductions for Imitation Learning](https://ri.cmu.edu/pub_files/2010/5/Ross-AIStats10-paper.pdf)
- [Active Imitation Learning: Formal and Practical Reductions to I.I.D. Learning](http://www.jmlr.org/papers/volume15/judah14a/judah14a.pdf)
- [Experience Selection in Deep Reinforcement Learning for Control](http://www.jmlr.org/papers/volume19/17-131/17-131.pdf)
