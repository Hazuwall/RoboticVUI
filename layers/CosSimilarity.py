import tensorflow as tf


class CosSimilarity1D(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CosSimilarity1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(
                                          1, input_shape[2], self.output_dim),
                                      initializer='normal',
                                      trainable=True)

        # Be sure to call this at the end
        super(CosSimilarity1D, self).build(input_shape)

    def call(self, x):
        kernel_norm = tf.linalg.normalize(self.kernel, axis=1)[0]
        return tf.nn.conv1d(x, kernel_norm, 1, padding="SAME")

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)
