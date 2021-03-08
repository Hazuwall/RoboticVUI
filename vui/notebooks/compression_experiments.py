from matplotlib import pyplot as plt
import vui.frontend.dsp as dsp
import numpy as np
import vui.infrastructure.tests as tests


def main():
    remove_repeat_compression_test(0.98, 0.04)


def remove_silence_test(n: int, threshold: float):
    frames_arr = tests.input_bulk(n)
    frames_before = np.reshape(frames_arr, [-1])
    frames_after = dsp.remove_silence(frames_before, threshold)
    tests.output(frames_before, frames_after)


def remove_silence_compression_test(threshold: float):
    frames_arr = tests.input_bulk(1000)
    total_length = 0
    total_compressed_length = 0
    for frames in frames_arr:
        total_length += len(frames)
        total_compressed_length += len(dsp.remove_silence(frames, threshold))
    compression = 1-(total_compressed_length/total_length)

    print("Threshold: {}, Compression: {}".format(threshold, compression))
    return compression


def remove_silence_compression_test2(resolution: int):
    compression = np.zeros((resolution))
    threshold = np.array(range(resolution)) / resolution
    for i in range(resolution):
        compression[i] = remove_silence_compression_test(i/resolution)

    plt.plot(threshold, compression)
    plt.show()


def remove_repeat_test(n: int, threshold: float, silence_threshold: float = 0):
    frames_arr = tests.input_bulk(n)
    frames_before = np.reshape(frames_arr, [-1])
    frames_after = dsp.remove_silence(frames_before, silence_threshold)
    frames_after = dsp.remove_repeat(
        frames_after, tests.config.framerate, 200, 5, max_correlation=threshold)
    tests.output(frames_before, frames_after)


def remove_repeat_compression_test(threshold: float, silence_threshold: float = 0):
    frames_arr = tests.input_bulk(50)
    total_length = 0
    total_compressed_length = 0
    for frames in frames_arr:
        frames = dsp.remove_silence(frames, silence_threshold)
        total_length += len(frames)
        total_compressed_length += len(dsp.remove_repeat(
            frames, tests.config.framerate, 200, 5, max_correlation=threshold))
    compression = 1-(total_compressed_length/total_length)

    print("Threshold: {}, Compression: {}".format(threshold, compression))
    return compression


def remove_repeat_compression_test2(resolution: int, silence_threshold: float = 0):
    compression = np.zeros((resolution))
    threshold = 0.6 + np.array(range(resolution)) / (resolution-1) * 0.4
    for i in range(resolution):
        compression[i] = remove_repeat_compression_test(
            i/(resolution-1)*0.4 + 0.6, silence_threshold)

    plt.plot(threshold, compression)
    plt.show()


if __name__ == "__main__":
    main()
