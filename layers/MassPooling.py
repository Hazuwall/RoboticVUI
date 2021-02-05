import tensorflow as tf


class MassPooling1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MassPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this at the end
        super(MassPooling1D, self).build(input_shape)

    def call(self, x):
        time_len = x.shape[1]
        indices = tf.cast(tf.range(time_len), dtype=tf.float32)
        indices = tf.expand_dims(indices, axis=0)
        indices = tf.expand_dims(indices, axis=2)

        x_normed = tf.linalg.normalize(x+0.00001, axis=1)[0]
        center_mass = tf.reduce_sum(tf.multiply(x_normed, indices), axis=1)
        total_center = tf.reduce_mean(center_mass, axis=1, keepdims=True)
        center_mass = (center_mass-total_center)*2/time_len
        return center_mass

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
