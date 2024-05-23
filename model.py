import tensorflow as tf

net = tf.keras.applications.ConvNeXtTiny(
    model_name='convnext_tiny',
    include_top=False,
    include_preprocessing=True,
    weights='imagenet',
)

# Ładuje model ConvNeXtTiny z wstępnie wytrenowanymi wagami z ImageNet. Ustawienia include_top=False powoduje,
# że ostatnie warstwy klasyfikacji nie są ładowane, co umożliwia wykorzystanie modelu jako ekstraktora cech.

def backbone(input_image):
    return tf.keras.layers.Flatten()(net(input_image))

# Funkcja backbone przyjmuje obraz wejściowy, przepuszcza go przez model ConvNeXtTiny i spłaszcza wyniki.
def classifier(concatenated_embedding, embedding_size):
    return tf.keras.layers.Dense(1, input_shape=(2 * embedding_size,), activation="sigmoid")(concatenated_embedding)

# Funkcja classifier przyjmuje połączone embeddingi i zwraca wynik klasyfikacji przez warstwę Dense z aktywacją sigmoid
def create_model(experiment_config):
    image_shape = (experiment_config["image_resolution"], experiment_config["image_resolution"], 3)
    input_left = tf.keras.layers.Input(
        shape=image_shape, name='input_L')
    input_right = tf.keras.layers.Input(
        shape=image_shape, name='input_R')
    # Tworzy wejścia dla lewego i prawego obrazu.
    preprocessed_left_image = backbone(input_left)
    preprocessed_right_image = backbone(input_right)
    # Przepuszcza obrazy przez model ConvNeXtTiny.
    concatenated_embedding = tf.keras.layers.Concatenate()([preprocessed_left_image, preprocessed_right_image])
    # Łączy wyniki przetworzenia obu obrazów.
    # TODO change that
    embedding_size = 2 * 768
    # Ustawia rozmiar embeddingu. Może być konieczna zmiana w zależności od architektury używanego modelu.
    output = classifier(concatenated_embedding, embedding_size)
    # Przepuszcza połączony embedding przez klasyfikator.
    model = tf.keras.Model(inputs=[input_left, input_right], outputs=output)
    # Tworzy model z dwoma wejściami (lewe i prawe obrazy) i jednym wyjściem (wynik klasyfikacji).
    return model
    # Funkcja create_model tworzy i zwraca kompletny model TensorFlow.