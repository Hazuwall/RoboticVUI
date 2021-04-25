import numpy as np
from cProfile import Profile
import vui.frontend.dsp as dsp
import vui.dataset.augmentation as augmentation
import vui.infrastructure.tests as tests


def main():
    apply_some_filters_test()


def change_pitch_test():
    frames_before = tests.input_bulk(10)

    frames_after = augmentation.change_pitch(
        frames_before, tests.config.framerate)

    frames_before = np.reshape(frames_before, [-1])
    frames_after = np.reshape(frames_after, [-1])
    sg = dsp.make_spectrogram(
        frames_before, seg_length=tests.config.seg_length)
    tests.plot_matrix("Spectrogram before", sg)
    sg = dsp.make_spectrogram(
        frames_after, seg_length=tests.config.seg_length)
    tests.plot_matrix("Spectrogram after", sg)
    tests.output(frames_before, frames_after)


def change_speed_test():
    frames_before = tests.input_bulk(10)

    frames_after = augmentation.change_speed(
        frames_before, tests.config.framerate)

    frames_before = np.reshape(frames_before, [-1])
    frames_after = np.reshape(frames_after, [-1])
    sg = dsp.make_spectrogram(
        frames_before, seg_length=tests.config.seg_length)
    tests.plot_matrix("Spectrogram before", sg)
    sg = dsp.make_spectrogram(
        frames_after, seg_length=tests.config.seg_length)
    tests.plot_matrix("Spectrogram after", sg)
    tests.output(frames_before, frames_after)


def add_noise_test():
    frames_before = tests.input_bulk(10)

    frames_after = augmentation.add_noise(frames_before)

    frames_before = np.reshape(frames_before, [-1])
    frames_after = np.reshape(frames_after, [-1])
    sg = dsp.make_spectrogram(
        frames_before, seg_length=tests.config.seg_length)
    tests.plot_matrix("Spectrogram before", sg)
    sg = dsp.make_spectrogram(
        frames_after, seg_length=tests.config.seg_length)
    tests.plot_matrix("Spectrogram after", sg)
    tests.output(frames_before, frames_after)


def apply_some_filters_test():
    frames_before = tests.input_bulk(10)

    frames_after = augmentation.apply_some_filters(
        frames_before, tests.config.framerate)

    frames_before = np.reshape(frames_before, [-1])
    frames_after = np.reshape(frames_after, [-1])
    sg = dsp.make_spectrogram(
        frames_before, seg_length=tests.config.seg_length)
    tests.plot_matrix("Spectrogram before", sg)
    sg = dsp.make_spectrogram(
        frames_after, seg_length=tests.config.seg_length)
    tests.plot_matrix("Spectrogram after", sg)
    tests.output(frames_before, frames_after)


def apply_some_filters_profile():
    frames = tests.input_bulk(200)

    profiler = Profile()
    profiler.enable()

    augmentation.apply_some_filters(
        frames, tests.config.framerate)

    profiler.disable()
    profiler.print_stats()


if __name__ == "__main__":
    main()
