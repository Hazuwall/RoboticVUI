import numpy as np
import dsp_utils


class FrontendProcessor:
    def __init__(self, config):
        self.config = config

    def make_spectrogram(self, frames):
        return dsp_utils.make_spectrogram(frames, seg_length=self.config.seg_length, step_count=self.config.step_count, keep_phase=False)

    def get_fund_freq(self, frames):
        return dsp_utils.get_fund_freq(frames, self.config.framerate, self.config.step_count, min_freq=self.config.min_fund_freq, max_freq=self.config.max_fund_freq)

    def get_harmonics(self, fund_freqs, spectrogram):
        return dsp_utils.get_harmonics(fund_freqs, self.config.freq_res, self.config.harmonics_count, spectrogram=spectrogram)

    def process(self, frames):
        if len(frames) < self.config.framerate:
            frames = np.pad(frames, (0, self.config.framerate-len(frames)))
        else:
            frames = frames[:self.config.framerate]

        sg = self.make_spectrogram(frames)
        fund_freqs = self.get_fund_freq(frames)
        # sg = dsp_utils.blur(sg,c.blur_size)
        _, amps = self.get_harmonics(fund_freqs, sg)
        # amps = dsp_utils.norm_harmonics(sg, amps)
        return amps.astype(np.float32)
