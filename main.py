import os

import pandas as pd
import yaml

from data import prepare_data, show_sample_image
from model import create_model

if __name__ == "__main__":
    # Read YAML file
    with open("config.yaml", 'r') as stream:
        experiment_config = yaml.safe_load(stream)

    metadata_filepath = os.path.join(experiment_config["data_directory"], experiment_config["metadata_file_name"])
    df = pd.read_excel(metadata_filepath)
    # show_sample_image(df, experiment_config)

    train_ds, val_ds, test_ds = prepare_data(df=df, experiment_config=experiment_config)

    model = create_model(experiment_config=experiment_config)
    import tensorflow as tf
    tf.keras.utils.plot_model(model, to_file='model.png')
    model.compile(loss="binary_crossentropy", metrics=["accuracy"])
    model.evaluate(test_ds, steps=2)
