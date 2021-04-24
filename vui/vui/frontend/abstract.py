from abc import ABC, abstractmethod
import numpy as np


class FrontendProcessorBase(ABC):
    def __init__(self, config):
        self._config = config

    def process(self, frames: np.ndarray) -> np.ndarray:
        frames = frames.copy()
        if len(frames.shape) == 1:
            return self._process_one(frames)

        elif len(frames.shape) == 2:
            batch = np.zeros(
                [len(frames), self._config.frontend_shape[0], self._config.frontend_shape[1]])

            for i in range(len(batch)):
                batch[i] = self._process_one(frames[i])
            return batch

        else:
            raise ValueError("Too many dimensions.")

    @abstractmethod
    def process_core(self, frames: np.ndarray) -> np.ndarray:
        pass

    def _process_one(self, frames: np.ndarray) -> np.ndarray:
        if len(frames) < self._config.framerate:
            frames = np.pad(frames, (0, self._config.framerate-len(frames)))
        else:
            frames = frames[:self._config.framerate]

        return self.process_core(frames)
