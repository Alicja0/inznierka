import tensorflow as tf

net = tf.keras.applications.ConvNeXtTiny(
    model_name='convnext_tiny',
    include_top=False,
    include_preprocessing=True,
    weights='imagenet',
)


def backbone(input_image):
    return tf.keras.layers.Flatten()(net(input_image))


def classifier(concatenated_embedding, embedding_size):
    return tf.keras.layers.Dense(1, input_shape=(2 * embedding_size,), activation="sigmoid")(concatenated_embedding)


def create_model(experiment_config):
    image_shape = (experiment_config["image_resolution"], experiment_config["image_resolution"], 3)
    input_left = tf.keras.layers.Input(
        shape=image_shape, name='input_L')
    input_right = tf.keras.layers.Input(
        shape=image_shape, name='input_R')

    preprocessed_left_image = backbone(input_left)
    preprocessed_right_image = backbone(input_right)

    concatenated_embedding = tf.keras.layers.Concatenate()([preprocessed_left_image, preprocessed_right_image])

    # TODO change that
    embedding_size = 2 * 768
    output = classifier(concatenated_embedding, embedding_size)

    model = tf.keras.Model(inputs=[input_left, input_right], outputs=output)

    return model
