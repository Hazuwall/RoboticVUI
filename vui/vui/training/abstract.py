from typing import Callable
import tensorflow as tf
from abc import ABC, abstractmethod, abstractproperty
from vui.dataset.pipeline import DatasetPipelineFactory
from vui.infrastructure.filesystem import FilesystemProvider
from vui.model.abstract import AcousticModelBase
from vui.model.metrics import Evaluator


class TrainerBase(ABC):
    @abstractproperty
    def model(self):
        pass

    @abstractmethod
    def run_step(self, step: tf.Tensor) -> None:
        pass


class AcousticModelTrainer(TrainerBase):
    def __init__(self, config, filesystem: FilesystemProvider, acoustic_model: AcousticModelBase, dataset_pipeline_factory: DatasetPipelineFactory, evaluator: Evaluator) -> None:
        self._config = config
        self._filesystem = filesystem
        self._acoustic_model = acoustic_model
        self._dataset_pipeline_factory = dataset_pipeline_factory
        self._evaluator = evaluator

        logs_path = filesystem.get_logs_dir()
        self._summary_writer = tf.summary.create_file_writer(logs_path)

    @property
    def model(self):
        return self._acoustic_model

    @abstractmethod
    def run_step_core(self, step: tf.Tensor) -> None:
        pass

    def run_step(self, step: tf.Tensor) -> None:
        with self._summary_writer.as_default(step):
            self.retry_on_error(lambda: self.run_step_core(step), 5)

            if (step % self._config.checkpoint_interval) == 0:
                self.model.save(int(step))
                if self._config.verbose:
                    print(int(step))

                test_accuracy = self._evaluator.evaluate()
                tf.summary.scalar("test/accuracy", test_accuracy)

        self._summary_writer.flush()

    def retry_on_error(self, func: Callable, attempts: int):
        counter = 0
        while True:
            try:
                func()
            except:
                counter += 1
                if counter >= attempts:
                    raise
                else:
                    print("Non-fatal error has occured.")
                    continue
            else:
                break


class TrainerFactoryBase(ABC):
    def __init__(self, config, filesystem: FilesystemProvider, acoustic_model: AcousticModelBase, dataset_pipeline_factory: DatasetPipelineFactory, evaluator: Evaluator) -> None:
        self._config = config
        self._filesystem = filesystem
        self._acoustic_model = acoustic_model
        self._dataset_pipeline_factory = dataset_pipeline_factory
        self._evaluator = evaluator

    @abstractmethod
    def get_trainer(self, stage: int) -> TrainerBase:
        pass
