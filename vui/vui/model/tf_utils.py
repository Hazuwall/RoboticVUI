import tensorflow as tf
import numpy as np


def cos_similarity(x: tf.Tensor, y: tf.Tensor, axis=0, do_normalize=False):
    """Returns a vector-wise cosine similarity of tensors across a given axis.
    Tensors x, y must be of the same type.
    Operation supports broadcasting."""
    if do_normalize:
        x = tf.linalg.normalize(x, axis=axis)[0]
        y = tf.linalg.normalize(y, axis=axis)[0]
    return tf.reduce_sum(tf.multiply(x, y), axis=axis)


def cos_similarity_triplet_loss(codes: tf.Tensor, max_positive_similarity: float = 1, min_negative_similarity: float = -1):
    coupled_codes = tf.reshape(codes, (-1, 2, codes.shape[1]))
    anchor, positive = tf.unstack(coupled_codes, axis=1)
    pos_distrib = cos_similarity(anchor, positive, axis=1)
    pos_cost = tf.reduce_mean(tf.minimum(
        max_positive_similarity, 1 - pos_distrib)**2)
    pos_similarity = tf.reduce_mean(pos_distrib)

    negative = tf.roll(positive, 1, axis=0)
    neg_distrib = cos_similarity(anchor, negative, axis=1)

    anchor = codes
    negative = tf.roll(codes, 2, axis=0)
    neg_distrib = tf.concat(
        [neg_distrib, cos_similarity(anchor, negative, axis=1)], axis=0)
    neg_cost = tf.reduce_mean(tf.maximum(
        min_negative_similarity, neg_distrib)**2)
    neg_similarity = tf.reduce_mean(neg_distrib)

    cost = tf.debugging.check_numerics(pos_cost + neg_cost, "Cost is NaN.")
    return cost, [pos_similarity, neg_similarity, pos_distrib, neg_distrib]


def cos_similarity_triplet_summary(cost: float, metrics: list) -> None:
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('pos_similarity', metrics[0])
    tf.summary.scalar('neg_similarity', metrics[1])
    tf.summary.histogram('pos_distrib', metrics[2])
    tf.summary.histogram('neg_distrib', metrics[3])


def coupled_cos_similarity_accuracy(codes):
    codes = tf.reshape(codes, (-1, 2, codes.shape[1]))
    anchor, positive = tf.unstack(codes, axis=1)
    anchor = tf.expand_dims(anchor, axis=1)
    positive = tf.expand_dims(positive, axis=0)
    similarity = cos_similarity(anchor, positive, axis=2)
    expected = tf.cast(tf.range(similarity.shape[0]), tf.int64)
    actual = tf.argmax(similarity, axis=0)
    incorrect_prediction = tf.not_equal(actual, expected)
    return 1 - tf.reduce_mean(tf.cast(incorrect_prediction, tf.float32))


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
