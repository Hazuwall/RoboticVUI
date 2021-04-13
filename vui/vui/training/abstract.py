from typing import Callable
import tensorflow as tf
from abc import ABC, abstractmethod, abstractproperty
from vui.dataset.pipeline import COUPLED_FETCH_MODE, DatasetPipelineFactory
from vui.infrastructure.filesystem import FilesystemProvider
from vui.model.abstract import AcousticModelBase
from vui.model.metrics import AveragingTimer, Evaluator
import vui.model.tf_utils as tf_utils


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

        self._validation_dataset = dataset_pipeline_factory.get_builder().from_labeled_storage(
            "s_en_SpeechCommands").with_size(config.validation_size).with_fetch_mode(COUPLED_FETCH_MODE).build()

        self._timer = AveragingTimer()

    @property
    def model(self):
        return self._acoustic_model

    @abstractmethod
    def run_step_core(self, step: tf.Tensor) -> None:
        pass

    def run_step(self, step: tf.Tensor) -> None:
        try:
            with self._summary_writer.as_default(step):
                with self._timer:
                    self._retry_on_error(lambda: self.run_step_core(step), 5)

                if step % self._config.display_interval == 0:
                    tf.summary.scalar("training/avg_step_time",
                                      self._timer.reset())

                if (step % self._config.checkpoint_interval) == 0:
                    self.model.save(int(step))
                    if self._config.verbose:
                        print(int(step))

                    self._run_validation()
                    self._run_test()
            self._summary_writer.flush()

        except KeyboardInterrupt:
            self.model.save(int(step))
            raise

    def _retry_on_error(self, func: Callable, attempts: int):
        counter = 0
        while True:
            try:
                func()
            except KeyboardInterrupt:
                raise
            except:
                counter += 1
                if counter >= attempts:
                    raise
                else:
                    print("Non-fatal error has occured.")
                    continue
            else:
                break

    def _run_validation(self):
        with tf.name_scope("validation/speech_commands"):
            try:
                x, _ = self._validation_dataset.get_batch()
                codes = self.model.encode(x, training=False)
                cost, triplet_metrics = tf_utils.cos_similarity_triplet_loss(
                    codes)
                tf_utils.cos_similarity_triplet_summary(
                    cost, triplet_metrics)
            except:
                pass

    def _run_test(self):
        with tf.name_scope("test"):
            test_accuracy, silence_slice, unknown_slice = self._evaluator.evaluate()
            tf.summary.scalar("accuracy", test_accuracy)
            tf.summary.scalar("silence_slice", silence_slice)
            tf.summary.scalar("unknown_slice", unknown_slice)


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
