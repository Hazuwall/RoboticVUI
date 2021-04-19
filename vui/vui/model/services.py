import numpy as np
from typing import Optional
from vui.frontend.abstract import FrontendProcessorBase
from vui.model.abstract import AcousticModelBase


class FramesToEmbeddingService:
    def __init__(self, config, frontend: FrontendProcessorBase, acoustic_model: AcousticModelBase):
        self.config = config
        self.frontend = frontend
        self.acoustic_model = acoustic_model

    def encode(self, frames: np.ndarray):
        if len(frames.shape) == 1:
            frames = np.expand_dims(frames, axis=0)

        harmonics = self.frontend.process(frames)
        return self.acoustic_model.encode(harmonics)
