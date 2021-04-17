import tensorflow as tf
from abc import ABC, abstractmethod, abstractproperty
import vui.dataset.pipeline as pipeline
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

    @abstractmethod
    def run(self, start_step: int, end_step: int) -> None:
        pass


class AcousticModelTrainer(TrainerBase):
    def __init__(self, config, filesystem: FilesystemProvider, acoustic_model: AcousticModelBase, evaluator: Evaluator) -> None:
        self._config = config
        self._filesystem = filesystem
        self._acoustic_model = acoustic_model
        self._evaluator = evaluator

        logs_path = filesystem.get_logs_dir()
        self._summary_writer = tf.summary.create_file_writer(logs_path)

        self._validation_dataset = self.create_validation_pipeline()

        self._timer = AveragingTimer()

    def create_validation_pipeline(self):
        storage = pipeline.get_hdf_storage('h', "s_en_SpeechCommands")
        x = pipeline.LabeledStorage(self._config.frontend_shape, storage,
                                    batch_size=self._config.validation_size, fetch_mode=pipeline.COUPLED_FETCH_MODE,
                                    use_max_classes_per_batch=True)
        return pipeline.Shuffle(group_size=4)(x)

    @property
    def model(self):
        return self._acoustic_model

    @abstractmethod
    def run_step_core(self, step: tf.Tensor) -> None:
        pass

    def run_step(self, step: tf.Tensor) -> None:
        with self._summary_writer.as_default(step):
            with self._timer:
                self.run_step_core(step)

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

    def run(self, start_step: int, end_step: int):
        for step in tf.range(start_step, end_step, dtype=tf.int64):
            try:
                self.run_step(step)
            except KeyboardInterrupt:
                self.model.save(int(step))
                print("Stopped by user.")
                break

    def _run_validation(self):
        with tf.name_scope("validation/speech_commands"):
            x, _ = self._validation_dataset.get_batch()
            codes = self.model.encode(x, training=False)
            cost, triplet_metrics = tf_utils.cos_similarity_triplet_loss(
                codes)
            tf_utils.cos_similarity_triplet_summary(
                cost, triplet_metrics)

    def _run_test(self):
        with tf.name_scope("test"):
            test_accuracy, silence_slice, unknown_slice, wrong_word_slice, correct_weight, incorrect_weight = self._evaluator.evaluate()
            tf.summary.scalar("accuracy", test_accuracy)
            tf.summary.scalar("silence_slice", silence_slice)
            tf.summary.scalar("unknown_slice", unknown_slice)
            tf.summary.scalar("wrong_word_slice", wrong_word_slice)
            tf.summary.scalar("correct_weight", correct_weight)
            tf.summary.scalar("incorrect_weight", incorrect_weight)


class TrainerFactoryBase(ABC):
    @abstractmethod
    def get_trainer(self, stage: int) -> TrainerBase:
        pass
