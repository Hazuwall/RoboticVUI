import tensorflow as tf
import sys
import numpy as np


def to_tiles(x):
    return tf.reshape(x, [-1, self.config.steps_per_embedding, self.config.preprocess_shape[1]])


def to_spectrograms(x):
    return tf.reshape(x, [-1, self.config.preprocess_shape[0], self.config.preprocess_shape[1]])


def print_stdout(op, tensors, message=None):
    def print_message(x):
        sys.stdout.write(message + " %s\n" % x)
        return x

    prints = [tf.py_func(print_message, [tensor], tensor.dtype)
              for tensor in tensors]
    with tf.control_dependencies(prints):
        op = tf.identity(op)
    return op


def cost_summary(cost_op):
    return tf.summary.scalar('cost', cost_op)


def validation_summary(validation_error, step):
    tf.print("Validation Error: ", validation_error,
             output_stream=sys.stdout)
    return tf.summary.scalar('validation_error', validation_error, step)


def test_summary(test_accuracy_op):
    test_accuracy_op = print_stdout(
        test_accuracy_op, [test_accuracy_op], "Test Accuracy")
    return tf.summary.scalar('test_accuracy', test_accuracy_op)


def spectrogram_summary(title, images, step, max_count):
    images = tf.reshape(
        images, [-1, self.config.preprocess_shape[0], self.config.preprocess_shape[1], 1])
    images = tf.pad(images, [[0, 0], [1, 0], [0, 0], [
                    0, 0]], mode="CONSTANT", constant_values=0)
    images = tf.pad(images, [[0, 0], [0, 1], [0, 0], [
                    0, 0]], mode="CONSTANT", constant_values=1)
    images = tf.transpose(images, (0, 2, 1, 3))
    images = tf.reverse(images, axis=[1])
    return tf.summary.image(title, images, step, max_outputs=max_count)


def wrong_images_summary(images, index_mask, step, max_count=10):
    images = tf.boolean_mask(images, index_mask)
    return spectrogram_summary('wrong_images', images, step, max_count=max_count)


def cos_similarity(x: tf.Tensor, y: tf.Tensor, axis=0, do_normalize=False):
    """Returns a vector-wise cosine similarity of tensors across a given axis.
    Tensors x, y must be of the same type.
    Operation supports broadcasting."""
    if do_normalize:
        x = tf.linalg.normalize(x, axis=axis)[0]
        y = tf.linalg.normalize(y, axis=axis)[0]
    return tf.reduce_sum(tf.multiply(x, y), axis=axis)


def ones_distance(x: tf.Tensor, y: tf.Tensor, axis=0):
    return tf.reduce_mean((x-y)**2, axis=axis)


def two_level_square_distance(output, y, true_level, false_level):
    true_output = tf.reduce_sum((output)*y, axis=1)
    true_loss = tf.reduce_mean((true_level - true_output)**2)
    false_output = output*(1-y)
    false_loss = tf.reduce_mean((false_output - false_level)**2)
    return true_loss + false_loss


def ones_similarity(x: tf.Tensor, y: tf.Tensor, axis=0):
    return 1 - tf.reduce_mean((x-y)**2, axis=axis)


def split_into_bands(x: tf.Tensor, axis, band_count, overlap=0):
    """Splits a tensor into overlapped subtensors of the same size across a given axis (1 and 2 supported)"""
    range_length = int(x.shape[axis].value * (1+overlap) / band_count)
    ksize = [1, 0, 0, 1]
    if axis == 1:
        ksize[1] = range_length
        ksize[2] = x.shape[2].value
    else:
        ksize[1] = x.shape[1].value
        ksize[2] = range_length

    stride = int((1 - overlap)*range_length)
    strides = [1, 1, 1, 1]
    strides[axis] = stride

    range_tensor = tf.extract_image_patches(
        x, ksizes=ksize, strides=strides, rates=[1, 1, 1, 1], padding='VALID')
    ranges_flat = tf.split(
        range_tensor, range_tensor.shape[axis].value, axis=axis)
    output = [tf.reshape(
        range, [-1, ksize[1], ksize[2], x.shape[3].value]) for range in ranges_flat]
    return output


def total_count(x: tf.Tensor):
    return tf.reduce_prod(tf.cast(tf.shape(x), dtype=tf.float32))


def stack_shifted_copies(x, step, step_count, shift_axis=1, stack_axis=0):
    x = tf.expand_dims(x, axis=0)
    factors = np.ones(len(x.shape.as_list()), dtype=np.int32)
    factors[0] = step_count
    copies = tf.tile(x, factors)
    copies = tf.unstack(copies, axis=0)
    zero_shift_index = step_count//2
    for i in range(step_count):
        shift_n = (i-zero_shift_index)*step
        copies[i] = tf.roll(copies[i], shift_n, axis=shift_axis)
    return tf.stack(copies, axis=stack_axis)
