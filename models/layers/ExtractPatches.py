import tensorflow as tf


class ExtractPatches(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        self.size = size
        super(ExtractPatches, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ExtractPatches, self).build(input_shape)

    def call(self, x):
        padding = self.size//2
        x = tf.pad(x, [[0, 0], [padding, padding], [0, 0]],
                   mode="CONSTANT", constant_values=0)
        x = tf.expand_dims(x, axis=3)
        patches = tf.image.extract_patches(x, sizes=[1, self.size, x.shape[2], 1], strides=[1, 1, 1, 1],
                                           rates=[1, 1, 1, 1], padding='VALID')
        return tf.squeeze(patches, axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2]*self.size)
