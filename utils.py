import datetime
import os

import yaml


def get_unique_exp_name(exp_name: str):
    return f'{exp_name}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'


def setup_experiment_output(experiment_config: dict):
    experiment_name: str = get_unique_exp_name(experiment_config["experiment_name"])
    full_experiment_path: str = os.path.join(experiment_config["output_directory"], experiment_name)
    os.makedirs(full_experiment_path, exist_ok=True)
    with open(os.path.join(full_experiment_path, "config.json"), 'w') as stream:
        yaml.dump(experiment_config, stream)
    return full_experiment_path
