import tensorflow as tf


class Attention1D(tf.keras.layers.Layer):
    def __init__(self, channels, join_channels=True, **kwargs):
        self.channels = channels
        self.join_channels = join_channels
        super(Attention1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, input_shape[2], self.channels),
                                      initializer='normal',
                                      trainable=True)

        # Be sure to call this at the end
        super(Attention1D, self).build(input_shape)

    def call(self, x):
        W = tf.nn.conv1d(x, filters=self.kernel, stride=1,
                         padding="SAME")  # shape (N, T, C)
        W = tf.nn.sigmoid(W)

        W = tf.expand_dims(W, axis=2)  # shape (N, T, 1, C)
        x = tf.expand_dims(x, axis=3)  # shape (N, T, F, 1)

        y = tf.multiply(x, W)  # shape (N, T, F, C)

        if self.join_channels:
            return tf.reshape(y, [-1, x.shape[1], x.shape[2]*self.channels])
        else:
            return tf.reshape(y, [-1, x.shape[1], x.shape[2], self.channels])

    def compute_output_shape(self, input_shape):
        if self.join_channels:
            return (input_shape[0], input_shape[1], input_shape[2]*self.channels)
        else:
            return (input_shape[0], input_shape[1], input_shape[2], self.channels)
