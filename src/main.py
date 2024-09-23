import os

import pandas as pd
import tensorflow as tf
import yaml

from src.data import prepare_data, show_sample_image
from src.binary_experiment import run_binary_experiment
from src.utils import setup_experiment_output


def main(experiment_config: dict):
    tf.random.set_seed(experiment_config["seed"])

    output_directory: str = setup_experiment_output(experiment_config)
    print(f"EXPERIMENT NAME: {os.path.basename(output_directory)}")
    metadata_filepath = os.path.join(experiment_config["data_directory"], experiment_config["metadata_file_name"])
    df: pd.DataFrame = pd.read_csv(metadata_filepath)
    # show_sample_image(df, experiment_config)
    # exit(0)
    data_generators, classes, datasets = prepare_data(df=df, experiment_config=experiment_config)

    if experiment_config["decision_class"] == "A":
        run_binary_experiment(data=data_generators,
                              experiment_config=experiment_config,
                              output_directory=output_directory,
                              classes=classes, dfs=datasets)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    # Read YAML file
    with open("src/config.yaml", 'r') as stream:
        experiment_config = yaml.safe_load(stream)

    main(experiment_config)