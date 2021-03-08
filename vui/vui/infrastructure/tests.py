import matplotlib.pyplot as plt
import numpy as np
import vui.infrastructure.locator as locator
from vui.dataset.pipeline import RANDOM_FETCH_MODE, HdfStorage
import vui.frontend.dsp as dsp

config = locator.get_config()
filesystem = locator.get_filesystem_provider()


def input() -> np.ndarray:
    frames = dsp.read("input.wav")
    return frames


def input_bulk(n: int) -> np.ndarray:
    dataset_path = filesystem.get_dataset_path(
        "r", label="t_mx_Mix")
    storage = HdfStorage(dataset_path, "raw")
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
