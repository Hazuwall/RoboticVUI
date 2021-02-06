import tensorflow as tf
from models_data import WeightsStorage, ReferenceWordsDictionary
import tf_utils


class AcousticModel(tf.keras.Model):
    def __init__(self, config, weights_storage: WeightsStorage, weights_step=None):
        super(AcousticModel, self).__init__()
        self.weights_storage = weights_storage

        def init_model():
            def cnn_block(x, filters, kernel_size):
                x = tf.keras.layers.Conv1D(
                    filters, kernel_size, padding="same", use_bias=False)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ReLU()(x)

                x = tf.keras.layers.Conv1D(
                    filters, kernel_size, padding="same", use_bias=False)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ReLU()(x)
                x = tf.keras.layers.MaxPooling1D(2)(x)
                return tf.keras.layers.Dropout(0.3)(x)

            def dense_block(x, units):
                x = tf.keras.layers.Dense(units, use_bias=False)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                return tf.keras.layers.ReLU()(x)

            # Input
            input = tf.keras.Input(
                shape=config.frontend_shape, dtype=tf.float64)
            x = tf.cast(input, tf.float32)

            # Phoneme Encoder
            x = cnn_block(x, 64, 3)
            x = cnn_block(x, 64, 3)
            x = cnn_block(x, 80, 3)
            x = cnn_block(x, 80, 3)

            # Word Encoder
            x = tf.keras.layers.Conv1D(
                80, 3, padding="same", use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.LSTM(256)(x)
            x = dense_block(x, 192)
            x = dense_block(x, 128)
            x = tf.keras.layers.Dense(60)(x)

            # Output
            config.set_embedding_size(x.shape[1])
            return tf.keras.Model(
                inputs=input, outputs=x, name=config.acoustic_model_name)

        self.encoder = init_model()
        self.checkpoint_step = self.weights_storage.load(
            self.encoder, weights_step)

    def save(self, step):
        self.weights_storage.save(
            self.encoder, step=step)
        self.checkpoint_step = step

    def encode(self, x, training=False):
        codes = self.encoder(x, training=training)
        return tf.linalg.normalize(codes, axis=1)[0]


class Classifier:
    def __init__(self, config, words_dictionary: ReferenceWordsDictionary):
        self.config = config
        self.words_dictionary = words_dictionary

    def update(self):
        self.words_dictionary.update()

    def classify(self, embeddings: tf.Tensor):
        ref_embeddings = self.words_dictionary.embeddings
        ref_embeddings = tf.reshape(
            ref_embeddings, [-1, self.config.embedding_features, 1])

        embeddings = tf.reshape(
            embeddings, [1, -1, self.config.embedding_features])
        similarity = tf_utils.cos_similarity(
            embeddings, ref_embeddings, axis=2)
        return similarity

    def get_word(self, index: int):
        return self.words_dictionary.words[index]
