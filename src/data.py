import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.augmentations import augment_image


def get_full_image_path(filename: str, experiment_config: dict):
    return os.path.join(experiment_config['data_directory'], experiment_config['images_path'],
                        filename)


# Display the original and augmented images using matplotlib
def display_image(image: np.ndarray):
    image_rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.suptitle('Image')
    plt.axis('off')

    plt.show()


def show_sample_image(df: pd.DataFrame, experiment_config: dict):
    sample_filename = df.sample(1)["Left-Fundus"].values[0]
    sample_filepath = get_full_image_path(sample_filename, experiment_config=experiment_config)
    print(sample_filepath)
    image = preprocess_image(path=sample_filepath, experiment_config=experiment_config)
    display_image(image=image)
    # cv2.imshow("Example", image)
    # # waits
    # # for user to press any key
    # # (this is necessary to avoid Python kernel form crashing)
    # cv2.waitKey(0)
    #
    # # closing all open windows
    # cv2.destroyAllWindows()


def print_data_statistics(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, experiment_config: dict):
    """
        Prints the class distribution statistics for the training, validation, and test datasets.

        Parameters:
        df_train (pd.DataFrame): The training dataset containing features and target.
        df_val (pd.DataFrame): The validation dataset containing features and target.
        df_test (pd.DataFrame): The test dataset containing features and target.
        experiment_config (dict): A configuration dictionary containing the key "decision_class" which specifies the
         target column for class distribution analysis.

        Returns:
        None: This function prints the class distributions of the specified target column
        in each of the provided datasets.
        """
    print("Train, ", df_train[experiment_config["decision_class"]].value_counts())
    print("Validation, ", df_val[experiment_config["decision_class"]].value_counts())
    print("Test, ", df_test[experiment_config["decision_class"]].value_counts())


def clean_eyes_df(df: pd.DataFrame, experiment_config: dict):
    clean_rows = []
    for _, patient_row in df.iterrows():
        left_eye_filename = get_full_image_path(filename=patient_row["Left-Fundus"],
                                                experiment_config=experiment_config)
        right_eye_filename = get_full_image_path(filename=patient_row["Right-Fundus"],
                                                 experiment_config=experiment_config)
        if os.path.exists(left_eye_filename) and os.path.exists(right_eye_filename):
            clean_rows.append(patient_row)
    return pd.DataFrame(clean_rows).reset_index()


def prepare_data(df: pd.DataFrame, experiment_config: dict):
    print(f"Preparing a set of {len(df)} images.")
    assert experiment_config['decision_class'] in df, f"Columns: {df.columns}"
    df_clean = clean_eyes_df(df=df, experiment_config=experiment_config)
    assert experiment_config['decision_class'] in df_clean, f"Columns: {df_clean.columns}"
    print(f"Removed missing images. Only {len(df_clean)} remained.")
    df_train, df_test = train_test_split(df_clean, test_size=0.1, random_state=experiment_config['seed'],
                                         stratify=df_clean[experiment_config['decision_class']])
    df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=experiment_config['seed'],
                                        stratify=df_train[experiment_config['decision_class']])

    print_data_statistics(df_train, df_val, df_test, experiment_config)

    classes = np.unique(df_train[experiment_config['decision_class']])
    class_weights = compute_class_weight('balanced', classes=classes,
                                         y=df_train[experiment_config['decision_class']])
    class_weights = dict(enumerate(class_weights))

    train_generator = get_data_generator(df_train, experiment_config, class_weights=class_weights, is_training=True)
    train_eval_generator = get_data_generator(df_train, experiment_config, class_weights=class_weights,
                                              is_training=False)
    val_generator = get_data_generator(df_val, experiment_config, class_weights=class_weights, is_training=False)
    test_generator = get_data_generator(df_test, experiment_config, class_weights=class_weights, is_training=False)

    data_generators = {"train": train_generator, "train_eval": train_eval_generator, "val": val_generator,
                       "test": test_generator}
    datasets = {
        "train": df_train, "val": df_val, "test": df_test}
    return data_generators, classes, datasets


def preprocess_image(path: str, experiment_config: dict):
    assert os.path.exists(path), f"{path} does not exist"
    image = cv2.imread(path)
    # cropped_image = remove_bg(image)
    augmented_image = augment_image(image=image, config=experiment_config["augmentations_config"])
    image_resized = cv2.resize(augmented_image,
                               (experiment_config["image_resolution"], experiment_config["image_resolution"]))
    return image_resized / 255


def remove_bg(image):
    # https://medium.com/@HeCanThink/rembg-effortlessly-remove-backgrounds-in-python-c2248501f992
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # get bounds of white pixels
    white = np.where(thresh == 255)
    xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
    # print(xmin, xmax, ymin, ymax)

    # crop the image at the bounds adding back the two blackened rows at the bottom
    return image[ymin:ymax, xmin:xmax]


def get_data_generator(df: pd.DataFrame, experiment_config: dict, class_weights: dict, is_training: bool):
    def generator():
        for _, raw in df.iterrows():
            left_eye_filename = get_full_image_path(filename=raw["Left-Fundus"], experiment_config=experiment_config)
            right_eye_filename = get_full_image_path(filename=raw["Right-Fundus"], experiment_config=experiment_config)
            left_eye_image = preprocess_image(path=left_eye_filename, experiment_config=experiment_config)
            right_eye_image = preprocess_image(path=right_eye_filename, experiment_config=experiment_config)
            y = raw[experiment_config["decision_class"]]
            yield {"input_L": left_eye_image, "input_R": right_eye_image}, (y,), (class_weights[y],)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {"input_L": tf.TensorSpec(shape=(
                experiment_config["image_resolution"], experiment_config["image_resolution"], 3),
                dtype=tf.float32),
                "input_R": tf.TensorSpec(
                    shape=(experiment_config["image_resolution"], experiment_config["image_resolution"], 3),
                    dtype=tf.float32)},
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)))
    if is_training:
        ds = ds.shuffle(experiment_config["shuffling_buffer"])
    ds = ds.batch(experiment_config["batch_size"])
    return ds