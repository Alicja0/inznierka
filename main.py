import os

import pandas as pd
import tensorflow as tf
import yaml

from data import prepare_data, show_sample_image
from experiment import run_binary_experiment
from utils import setup_experiment_output


def main(experiment_config: dict):
    tf.random.set_seed(experiment_config["seed"])

    output_directory: str = setup_experiment_output(experiment_config)

    metadata_filepath = os.path.join(experiment_config["data_directory"], experiment_config["metadata_file_name"])
    df: pd.DataFrame = pd.read_csv(metadata_filepath)
    # show_sample_image(df, experiment_config)

    train_ds, val_ds, test_ds = prepare_data(df=df, experiment_config=experiment_config)

    if experiment_config["decision_class"] == "A":
        run_binary_experiment(data={"train": train_ds, "val": val_ds, "test": test_ds},
                              experiment_config=experiment_config,
                              output_directory=output_directory)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    # Read YAML file
    with open("config.yaml", 'r') as stream:
        experiment_config = yaml.safe_load(stream)

    main(experiment_config)
