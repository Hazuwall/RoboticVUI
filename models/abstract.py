from typing import List, Optional
import tensorflow as tf
from abc import ABC, abstractmethod
import numpy as np
from models.data_access import ReferenceWordsDictionary, WeightsStorage


class AcousticModelBase(ABC):
    def __init__(self, config, weights_storage: WeightsStorage, stage_checkpoints: Optional[List[Optional[int]]] = None) -> None:
        self._config = config
        self._weights_storage = weights_storage
        if stage_checkpoints is None:
            self._stage_checkpoints: list = [None]*config.stages
        else:
            self._stage_checkpoints: list = stage_checkpoints.copy()

    def get_checkpoint_step(self, stage: int = 0) -> int:
        return self._stage_checkpoints[stage]

    @abstractmethod
    def save(self, step: int, stage: int = 0) -> None:
        pass

    @abstractmethod
    def encode(self, x: np.ndarray, training: bool = False) -> tf.Tensor:
        pass

    @property
    @abstractmethod
    def encoder(self):
        return self._encoder


class ClassifierBase(ABC):
    def __init__(self, config, words_dictionary: ReferenceWordsDictionary) -> None:
        self._config = config
        self._words_dictionary = words_dictionary

    def get_word(self, index: int) -> List[str]:
        return self._words_dictionary.words[index]

    @abstractmethod
    def classify(self, embeddings: tf.Tensor) -> tf.Tensor:
        pass
