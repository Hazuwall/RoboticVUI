import numpy as np
import vui.frontend.dsp as dsp
from vui.frontend.abstract import FrontendProcessorBase


class FrontendProcessor(FrontendProcessorBase):
    def __init__(self, config):
        super(FrontendProcessor, self).__init__(config)

    def process_core(self, frames):
        frames = dsp.remove_silence(frames, self._config.silence_threshold)
        frames = np.pad(frames, (0, self._config.framerate-len(frames)))

        spectrogram = dsp.make_spectrogram(
            frames, seg_length=self._config.seg_length, step_count=self._config.spectrums_per_sec, keep_phase=False)

        fund_freqs = dsp.get_fund_freq(frames, self._config.framerate, self._config.spectrums_per_sec,
                                       min_freq=self._config.min_fund_freq, max_freq=self._config.max_fund_freq)

        _, amps = dsp.get_harmonics(
            fund_freqs, self._config.freq_res, self._config.harmonics_count, spectrogram=spectrogram)

        return amps
