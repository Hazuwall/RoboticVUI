import matplotlib.pyplot as plt
import numpy as np
import vui.infrastructure.locator as locator
import vui.dataset.pipeline as pipeline
import vui.frontend.dsp as dsp

config = locator.get_config()
filesystem = locator.get_filesystem_provider()


def input_local():
    frames = dsp.read("input.wav")
    config.framerate = dsp.get_framerate("input.wav")
    config.seg_length = dsp.get_best_segment_length(config.framerate)
    config.freq_count = dsp.get_freq_count(config.seg_length)
    config.freq_res = dsp.get_freq_resolution(
        config.framerate, config.freq_count)
    return frames


def input() -> np.ndarray:
    storage = pipeline.get_hdf_storage('r', "s_en_SpeechCommands")
    return np.reshape(storage.fetch_subset("tree", 0, 1), [-1])


def input_bulk(n: int) -> np.ndarray:
    storage = pipeline.get_hdf_storage('r', "t_mx_Mix")
    return storage.fetch_subset("", 0, n)


def output(frames_before: np.ndarray, frames_after: np.ndarray):
    dsp.write("input.wav", frames_before, config.framerate)
    dsp.write("output.wav", frames_after, config.framerate)


def plot_vector(title, vector):
    plt.plot(vector)
    plt.title = title
    plt.show()


def plot_matrix(title, matrix):
    print(title)
    plt.pcolormesh(range(matrix.shape[1]), range(
        matrix.shape[0]), matrix, cmap=plt.cm.Blues)
    plt.show()
