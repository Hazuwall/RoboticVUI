import numpy as np
from typing import Optional
from frontend.abstract import FrontendProcessorBase
from models.abstract import AcousticModelBase


class FramesToEmbeddingService:
    def __init__(self, config, frontend: FrontendProcessorBase, acoustic_model: AcousticModelBase):
        self.config = config
        self.frontend = frontend
        self.acoustic_model = acoustic_model

    def encode(self, frames: np.ndarray, indices: Optional[np.ndarray] = None):
        if indices is None:
            harmonics = np.expand_dims(self.frontend.process(frames), axis=0)
        else:
            harmonics = np.zeros(
                shape=(len(indices), self.config.frontend_shape[0], self.config.frontend_shape[1]))
            for i in range(len(indices)):
                start = indices[i]*self.config.seg_length//2
                end = start + self.config.framerate
                if end > len(frames):
                    data = frames[start:]
                    data = np.pad(data, (0, end-len(frames)))
                else:
                    data = frames[start:end]
                harmonics[i] = self.frontend.process(data)

        return self.acoustic_model.encode(harmonics)
