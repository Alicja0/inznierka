import os

import tensorflow as tf

from model import create_model


def evaluate_binary_model_single_subset(model, subset, data, output_directory):
    predictions = model.predict(data, steps=1)  # TODO remove
    print(predictions)
    # [[2.7285005e-18]
    #  [2.7273872e-18]]
    # TODO add dumping the predictions, metrics and sample errors (confusion report)


def evaluate_binary_model(model: tf.keras.Model, data: dict, output_directory: str):
    for subset in data.keys():
        evaluate_binary_model_single_subset(model=model, subset=subset, data=data[subset],
                                            output_directory=output_directory)


def run_binary_experiment(data: dict, experiment_config: dict, output_directory: str):
    model = create_model(experiment_config=experiment_config)
    tf.keras.utils.plot_model(model, to_file=os.path.join(output_directory, 'model.png'))
    model.compile(loss="binary_crossentropy", metrics=["accuracy"])
    # TODO remove steps_per_epoch=2, validation_steps=1
    model.fit(data["train"], validation_data=data["val"], epochs=experiment_config["epochs"], steps_per_epoch=1,
              validation_steps=1)
    model.save(os.path.join(output_directory, "model.h5"))

    evaluate_binary_model(model=model, data=data, output_directory=output_directory)
