import tensorflow as tf


class Hypersphere(tf.keras.layers.Layer):
    def __init__(self, radius=1, **kwargs):
        self.radius = radius
        super(Hypersphere, self).__init__(**kwargs)

    def call(self, x):
        x = tf.linalg.normalize(x, axis=1)[0] * self.radius
        return tf.where(tf.math.is_nan(x), 0.0, x)

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("Incorrect number of dimensions.")
