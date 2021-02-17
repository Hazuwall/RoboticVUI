import tensorflow as tf
from abc import ABC, abstractmethod
from vui.dataset.pipeline import DatasetPipelineFactory
from vui.infrastructure.filesystem import FilesystemProvider
from vui.model.abstract import AcousticModelBase


class TrainerBase(ABC):
    def __init__(self, config, filesystem: FilesystemProvider, acoustic_model: AcousticModelBase, dataset_pipeline_factory: DatasetPipelineFactory) -> None:
        self._config = config
        self._filesystem = filesystem
        self._acoustic_model = acoustic_model
        self._dataset_pipeline_factory = dataset_pipeline_factory

    @property
    def model(self):
        return self._acoustic_model

    @abstractmethod
    def run_step(self, step: tf.Tensor) -> None:
        pass


class TrainerFactoryBase(ABC):
    def __init__(self, config, filesystem: FilesystemProvider, acoustic_model: AcousticModelBase, dataset_pipeline_factory: DatasetPipelineFactory) -> None:
        self._config = config
        self._filesystem = filesystem
        self._acoustic_model = acoustic_model
        self._dataset_pipeline_factory = dataset_pipeline_factory

    @abstractmethod
    def get_trainer(self, stage: int) -> TrainerBase:
        pass
