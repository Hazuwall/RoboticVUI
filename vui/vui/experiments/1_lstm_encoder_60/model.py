import numpy as np
import tensorflow as tf
from typing import List, Optional
from vui.model.data_access import WeightsStorage, ReferenceWordsDictionary
import vui.model.tf_utils as tf_utils
from vui.model.abstract import AcousticModelBase, ClassifierBase
from vui.model.layers.Hypersphere import Hypersphere


class AcousticModel(AcousticModelBase):
    def __init__(self, config, weights_storage: WeightsStorage, stage_checkpoints: Optional[List[int]] = None) -> None:
        super(AcousticModel, self).__init__(
            config, weights_storage, stage_checkpoints)

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
                return tf.keras.layers.Dropout(0.5)(x)

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
            x = Hypersphere()(x)

            # Output
            return tf.keras.Model(
                inputs=input, outputs=x, name=config.acoustic_model_name)

        self._encoder: tf.keras.Model = init_model()
        self._stage_checkpoints[0] = self._weights_storage.load(
            self._encoder, self._stage_checkpoints[0])

    def save(self, step: int) -> None:
        self._weights_storage.save(
            self._encoder, step=step)
        self._stage_checkpoints[0] = step

    def encode(self, x: np.ndarray, training: bool = False) -> tf.Tensor:
        return self._encoder(x, training=training)

    @property
    def encoder(self):
        return self._encoder


class Classifier(ClassifierBase):
    def __init__(self, config, words_dictionary: ReferenceWordsDictionary) -> None:
        super(Classifier, self).__init__(config, words_dictionary)

    def classify(self, embeddings: tf.Tensor) -> tf.Tensor:
        ref_embeddings = self._words_dictionary.embeddings
        ref_embeddings = tf.reshape(
            ref_embeddings, [1, -1, self._config.embedding_size])

        embeddings = tf.reshape(
            embeddings, [-1, 1, self._config.embedding_size])
        similarity = tf_utils.cos_similarity(
            embeddings, ref_embeddings, axis=2)
        return similarity
