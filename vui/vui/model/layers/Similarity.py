import tensorflow as tf


class Similarity1D(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Similarity1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(
                                          1, 1, input_shape[2], self.output_dim),
                                      initializer='normal',
                                      trainable=True)
        self.gate = self.add_weight(name='gate',
                                    shape=(
                                        1, 1, input_shape[2], self.output_dim),
                                    initializer='normal',
                                    trainable=True)
        # Be sure to call this at the end
        super(Similarity1D, self).build(input_shape)

    def call(self, x):
        x = tf.expand_dims(x, axis=3)
        distance = (x - self.kernel)**2
        soft_gate = tf.sigmoid(self.gate)
        similarity = tf.reduce_mean(tf.exp(-distance), axis=2)
        return similarity

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)
