from abc import ABC, abstractmethod
import numpy as np


class FrontendProcessorBase(ABC):
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def process(self, frames: np.ndarray):
        pass
