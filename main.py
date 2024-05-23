import os

import pandas as pd
import tensorflow as tf
import yaml

from data import prepare_data, show_sample_image
from experiment import run_binary_experiment
from utils import setup_experiment_output


def main(experiment_config: dict):
    tf.random.set_seed(experiment_config["seed"])
    # Ustawia ziarno generatora liczb losowych TensorFlow dla powtarzalności wyników

    output_directory: str = setup_experiment_output(experiment_config)
    # Przygotowuje katalog do zapisywania wyników eksperymentu. Funkcja prawdopodobnie tworzy katalog na podstawie konfiguracji eksperymentu

    metadata_filepath = os.path.join(experiment_config["data_directory"], experiment_config["metadata_file_name"])
    df: pd.DataFrame = pd.read_csv(metadata_filepath)
    # show_sample_image(df, experiment_config)
    # Wczytuje plik CSV z metadanymi, zawierający informacje o obrazach

    train_ds, val_ds, test_ds = prepare_data(df=df, experiment_config=experiment_config)
    # Przygotowuje dane do trenowania, walidacji i testowania modelu, dzieląc je na odpowiednie zbiory.
    if experiment_config["decision_class"] == "A":
        run_binary_experiment(data={"train": train_ds, "val": val_ds, "test": test_ds},
                              experiment_config=experiment_config,
                              output_directory=output_directory)
        # Uruchamia eksperyment binarny, jeżeli klasa decyzyjna to "A".
    else:
        raise NotImplementedError
        # Wyjątek dla niezaimplementowanych klas decyzyjnych.

if __name__ == "__main__":
    # Read YAML file
    with open("config.yaml", 'r') as stream:
        experiment_config = yaml.safe_load(stream)
    # Wczytuje konfigurację eksperymentu z pliku YAML.
    main(experiment_config)
# Uruchamia główną funkcję programu z wczytaną konfiguracją.