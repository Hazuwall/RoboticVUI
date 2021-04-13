from matplotlib import pyplot as plt
import vui.frontend.dsp as dsp
import numpy as np
import vui.infrastructure.tests as tests


def main():
    norm_harmonics_test()


def make_spectrogram_test():
    frames = tests.input()
    sg = dsp.make_spectrogram(
        frames, seg_length=tests.config.seg_length, keep_phase=False)

    tests.output(frames, frames)
    tests.plot_matrix("Spectrogram", sg)


def restore_frames_test():
    frames_before = tests.input()
    sg = dsp.make_spectrogram(
        frames_before, seg_length=tests.config.seg_length, keep_phase=False)
    restored = dsp.restore_frames(
        sg, seg_length=tests.config.seg_length, frame_count=len(frames_before))

    tests.output(frames_before, restored)


def norm_harmonics_test():
    frames = tests.input()
    spectrogram = dsp.make_spectrogram(
        frames, step_count=tests.config.spectrums_per_sec, seg_length=tests.config.seg_length, keep_phase=False)
    fund_freqs = dsp.get_fund_freq(frames, tests.config.framerate, tests.config.spectrums_per_sec,
                                   min_freq=tests.config.min_fund_freq, max_freq=tests.config.max_fund_freq)

    _, amps = dsp.get_harmonics(
        fund_freqs, tests.config.freq_res, tests.config.harmonics_count, spectrogram=spectrogram)

    normed = dsp.norm_harmonics(spectrogram, amps)

    tests.output(frames, frames)
    tests.plot_matrix("Harmonics' amps normed", np.abs(normed).T)


if __name__ == "__main__":
    main()
