import datasets
import dsp_utils as p
import pipeline as sp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
import config as cfg
import math_utils
import random
import augmentation as aug
import pyaudio

storage = datasets.HdfStorage(cfg.get_dataset_path('r'), "raw/tree")


def main():
    frames = storage.fetch_subset("", 0, 2, "RANDOM")[0]
    make_spectrogram_test(frames)
    # blur_test(frames)
    # bulk_aug_test(frames)
    # encode_and_decode_test(frames)

    # word_similarity_test("zero","tree")
    # mean_word_predict_test(35)
    # recognize_words_from_stream_test(60)

# Decorators and handlers


def plot_vector(title, vector):
    print(title)
    plt.plot(vector)
    plt.show()


def plot_matrix(title, matrix):
    print(title)
    plt.pcolormesh(range(matrix.shape[1]), range(
        matrix.shape[0]), matrix, cmap=plt.cm.Blues)
    plt.show()


def fund_freq_mean_error(frames, error_seq):
    sg = sp.make_spectrogram(frames)
    std_sqrt = np.sqrt(np.std(sg, axis=1))
    mean_error = np.sqrt(
        np.sum(std_sqrt*np.square(error_seq)) / cfg.step_count)
    return mean_error


def before(fn):
    def wrapped(*args, **kwargs):
        p.write("before.wav", args[0], cfg.framerate)
        sg = sp.make_spectrogram(args[0])
        plot_matrix("Spectrogram before", sg.T)
        return fn(*args, **kwargs)
    return wrapped


def after(fn):
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        if out[0].ndim == 1:
            p.write("after.wav", out[0], out[1])
            seg_length = p.get_best_segment_length(out[1])
            sg = p.make_spectrogram(
                out[0], seg_length=seg_length, step_count=cfg.step_count)
            plot_matrix("Spectrogram after", sg.T)
        else:
            if out[0].dtype == np.complex_:
                plot_matrix("Spectrogram after", np.abs(out[0]).T)
            else:
                plot_matrix("Spectrogram after", out[0].T)
            frames = p.restore_frames(out[0], frame_count=out[2])
            p.write("after.wav", frames, out[1])
        return out
    return wrapped


def bulk_aug_test(frames):
    flatten = np.reshape(frames, [-1])
    p.write("before.wav", flatten, cfg.framerate)

    frames, _ = aug.process(frames, cfg.framerate, 1)
    flatten = np.reshape(frames, [-1])
    p.write("after.wav", flatten, cfg.framerate)


@after
@before
def encode_and_decode_test(frames):
    provider = datasets.DatasetProvider()
    harmonics = provider.get_batch(1)[0]
    #harmonics = sp.complete_preprocess(frames)
    plot_matrix("Harmonics before", np.abs(harmonics).T)
    codes = sp.encode(harmonics)
    harmonics = sp.decode(codes)
    plot_matrix("Harmonics after", np.abs(harmonics).T)
    sg = p.restore_spectrogram(harmonics, cfg.freq_count, cfg.freq_res)
    return sg, len(frames), cfg.framerate


def mean_word_predict_test(count):
    storage = datasets.HdfStorage(cfg.get_dataset_path(
        'e', label="s_en_SpeechCommands"), "embeddings")
    words = random.sample(storage.get_dataset_list(), count)
    weights = np.zeros((1, cfg.embedding_features, count))

    i = 0
    for word in words:
        codes = storage.fetch_subset(word, 0, 500, "RANDOM")
        codes = np.squeeze(codes, axis=1)
        weights[0, :, i] = np.mean(codes, axis=0)
        i += 1
    norm = np.linalg.norm(weights, axis=1, keepdims=True)
    weights /= norm

    x, y = datasets.DatasetProvider(
        label="s_en_SpeechCommands", embeddings_return=True).get_batch(1000, words)
    p = tf.nn.conv1d(x, weights, 1, padding="SAME")
    p = tf.squeeze(p, axis=1)
    incorrect_prediction = tf.not_equal(tf.argmax(p, 1), tf.argmax(y, 1))
    for w in words:
        print(w)
    print("Accuracy:")
    print(float(1 - tf.reduce_mean(tf.cast(incorrect_prediction, tf.float32))))


def pairwise_word_predict_test(count):
    storage = datasets.HdfStorage(cfg.get_dataset_path(
        'e', label="s_en_SpeechCommands"), "embeddings")
    words = random.sample(storage.get_dataset_list(), count)
    weights = np.zeros((1, cfg.embedding_features, count))

    x = []
    for word in words:
        codes = storage.fetch_subset(word, 0, 500, "RANDOM")
        codes = tf.squeeze(codes, axis=1)
        x.append(codes)
    x = tf.stack(x, axis=0)
    x = tf.reshape(x, [len(words), -1, 2, cfg.embedding_features])
    x = tf.transpose(x, [1, 0, 2, 3])

    anchor, positive = tf.unstack(x, axis=2)
    anchor = tf.expand_dims(anchor, axis=2)
    positive = tf.expand_dims(positive, axis=1)
    similarity = math_utils.cos_similarity(anchor, positive, axis=3)
    labels = tf.cast(tf.range(similarity.shape[1]), tf.int64)
    ps = tf.argmax(similarity, axis=1)
    #labels = tf.expand_dims(labels,axis=0)
    incorrect_prediction = tf.not_equal(ps, labels)
    for w in words:
        print(w)
    print("Accuracy:")
    print(float(1 - tf.reduce_mean(tf.cast(incorrect_prediction, tf.float32))))


def word_similarity_test(word1, word2):
    storage = datasets.HdfStorage(cfg.get_dataset_path(
        'e', label="s_en_SpeechCommands"), "embeddings")
    codes1 = storage.fetch_subset(word1, 0, 500, "RANDOM")
    codes2 = storage.fetch_subset(word2, 0, 500, "RANDOM")
    distrib = math_utils.cos_similarity(codes1, codes2, axis=2)
    print("Mean Similarity")
    print(float(tf.reduce_mean(distrib)))


def microphone_devices_test():
    audio = pyaudio.PyAudio()
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ",
                  audio.get_device_info_by_host_api_device_index(0, i).get('name'))


def detect_words_test(frames):
    sg = sp.make_spectrogram(frames)
    plot_matrix("Spectrogram", sg.T)
    indices = sp.detect_words(sg)
    print(indices)


def recognize_words_from_stream_test(time):
    sp.recognize_words_from_stream(time)


if __name__ == "__main__":
    main()
