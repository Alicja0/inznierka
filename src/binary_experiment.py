import json
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from src.model import create_model
from src.utils import calculate_metrics


def get_labels_from_generator(data: tf.data.Dataset, n: int, batch_size: int):
    # Extract labels from the dataset
    labels = []

    for X_batch, y_batch, sample_weights in data.take(math.ceil(n / batch_size)):
        labels.extend(y_batch.numpy())

    # Convert to a numpy array if needed
    labels = np.array(labels)
    return labels


def add_predictions_to_df(df, raw_predictions, predictions):
    df["prediction_sigmoid"] = raw_predictions
    df["prediction"] = predictions
    return df


def evaluate_binary_model_single_subset(model, subset, data, output_directory: str, batch_size: int, classes: list,
                                        df: pd.DataFrame):
    print(f"Evaluating on {subset} data.")
    raw_predictions = model.predict(data)
    predictions = raw_predictions > 0.5
    labels = get_labels_from_generator(data, n=len(predictions), batch_size=batch_size)
    conf_matrix = confusion_matrix(y_true=labels, y_pred=predictions, labels=classes)
    metrics = calculate_metrics(conf_matrix)
    df_with_predictions = add_predictions_to_df(df=df, raw_predictions=raw_predictions, predictions=predictions)
    df_with_predictions.to_csv(os.path.join(output_directory, f"{subset}_predictions.csv"), index=False)
    with open(os.path.join(output_directory, f"{subset}_metrics.json"), "w") as fp:
        json.dump(metrics, fp)
    # TODO sample errors (confusion report)


def evaluate_binary_model(model: tf.keras.Model, data: dict, output_directory: str, batch_size: int, classes: list,
                          dfs: list[pd.DataFrame]):
    for subset in ["test", "val", "train"]:
        evaluate_binary_model_single_subset(model=model, subset=subset, data=data[subset],
                                            output_directory=output_directory, batch_size=batch_size, classes=classes,
                                            df=dfs[subset])


def run_binary_experiment(data: dict, experiment_config: dict, output_directory: str, classes: list,
                          dfs: list[pd.DataFrame]):
    model = create_model(experiment_config=experiment_config)
    tf.keras.utils.plot_model(model, to_file=os.path.join(output_directory, 'model.png'))
    model.compile(loss="binary_crossentropy", metrics=["accuracy"])
    # tensorboard logs with plots
    log_dir = os.path.join(output_directory, "tensorboard")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # Initialize EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5,
                                                      # number of epochs with no improvement after
                                                      # which training will be stopped
                                                      monitor='val_loss',  # the metric to monitor
                                                      mode='min',
                                                      # 'min' for loss (lower is better),
                                                      # 'max' for accuracy (higher is better)
                                                      restore_best_weights=True,
                                                      # restore model weights from the epoch
                                                      # with the best value of the monitored metric
                                                      verbose=1,
                                                      )
    # TODO remove steps_per_epoch=2, validation_steps=1
    model.fit(data["train"], validation_data=data["val"], epochs=experiment_config["epochs"], steps_per_epoch=1,
              validation_steps=1, callbacks=[early_stopping, tensorboard_callback])
    model.save(os.path.join(output_directory, "model.keras"))

    evaluate_binary_model(model=model, data={"train": data["train_eval"], "val": data["val"], "test": data["test"]},
                          output_directory=output_directory,
                          batch_size=experiment_config["batch_size"], classes=classes, dfs=dfs)

