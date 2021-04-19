import vui.frontend.dsp as dsp
from vui.frontend.abstract import FrontendProcessorBase


class FrontendProcessor(FrontendProcessorBase):
    def __init__(self, config):
        super(FrontendProcessor, self).__init__(config)

    def process_core(self, frames):
        spectrogram = dsp.make_spectrogram(
            frames, seg_length=self._config.seg_length, step_count=self._config.spectrums_per_sec, keep_phase=False)

        return spectrogram
