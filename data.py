import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def get_full_image_path(filename: str, experiment_config: dict):
    return os.path.join(experiment_config['data_directory'], experiment_config['images_path'],
                        filename)


def show_sample_image(df, experiment_config: dict):
    sample_filename = df.sample(1)["Left-Fundus"].values[0]
    sample_filepath = get_full_image_path(sample_filename, experiment_config=experiment_config)
    image = preprocess_image(path=sample_filepath, experiment_config=experiment_config)
    cv2.imshow("Example", image)
    # waits
    # for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


def print_data_statistics(df_train, df_val, df_test, experiment_config):
    print("Train, ", df_train[experiment_config["decision_class"]].value_counts())
    print("Validation, ", df_val[experiment_config["decision_class"]].value_counts())
    print("Test, ", df_test[experiment_config["decision_class"]].value_counts())


def prepare_data(df, experiment_config: dict):
    df_train, df_test = train_test_split(df, test_size=0.1)
    df_train, df_val = train_test_split(df_train, test_size=0.1)

    print_data_statistics(df_train, df_val, df_test, experiment_config)

    train_generator = get_data_generator(df_train, experiment_config, is_training=True)
    val_generator = get_data_generator(df_val, experiment_config, is_training=False)
    test_generator = get_data_generator(df_test, experiment_config, is_training=False)

    return train_generator, val_generator, test_generator


def preprocess_image(path: str, experiment_config: dict):
    image = cv2.imread(path)
    cropped_image = remove_bg(image)
    image_resized = cv2.resize(cropped_image,
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


def get_data_generator(df, experiment_config: dict, is_training: bool):
    def generator():
        for _, raw in df.iterrows():
            left_eye_filename = get_full_image_path(filename=raw["Left-Fundus"], experiment_config=experiment_config)
            right_eye_filename = get_full_image_path(filename=raw["Right-Fundus"], experiment_config=experiment_config)
            left_eye_image = preprocess_image(path=left_eye_filename, experiment_config=experiment_config)
            right_eye_image = preprocess_image(path=right_eye_filename, experiment_config=experiment_config)
            y = raw[experiment_config["decision_class"]]
            yield {"input_L": left_eye_image, "input_R": right_eye_image}, (y,)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {"input_L": tf.TensorSpec(shape=(
                experiment_config["image_resolution"], experiment_config["image_resolution"], 3),
                dtype=tf.float32),
                "input_R": tf.TensorSpec(
                    shape=(experiment_config["image_resolution"], experiment_config["image_resolution"], 3),
                    dtype=tf.float32)},
            tf.TensorSpec(shape=(1,), dtype=tf.int32)))
    if is_training:
        ds = ds.shuffle(experiment_config["shuffling_buffer"])
    ds = ds.batch(experiment_config["batch_size"])
    return ds
