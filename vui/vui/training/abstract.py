import tensorflow as tf
from abc import ABC, abstractmethod, abstractproperty
import vui.dataset.pipeline as pipeline
from vui.infrastructure.filesystem import FilesystemProvider
from vui.model.abstract import AcousticModelBase
from vui.model.metrics import AveragingTimer, Evaluator
import vui.model.tf_utils as tf_utils
from vui.frontend.abstract import FrontendProcessorBase


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
    def __init__(self, config, filesystem: FilesystemProvider, acoustic_model: AcousticModelBase,
                 evaluator: Evaluator, frontend: FrontendProcessorBase, stage: int) -> None:
        self._config = config
        self._filesystem = filesystem
        self._acoustic_model = acoustic_model
        self._evaluator = evaluator
        self._stage = stage
        self._frontend = frontend

        logs_path = "{}\\stage{}".format(filesystem.get_logs_dir(), stage)
        self._summary_writer = tf.summary.create_file_writer(logs_path)

        self._validation_dataset = self.create_validation_pipeline()

        self._timer = AveragingTimer()

    def create_validation_pipeline(self):
        storage = pipeline.get_hdf_storage('h', "s_en_SpeechCommands")
        x = pipeline.LabeledSource(self._config.frontend_shape, storage,
                                   batch_size=self._config.validation_size, fetch_mode=pipeline.COUPLED_FETCH_MODE,
                                   use_max_classes_per_batch=True)
        return pipeline.Shuffle(patch_size=4)(x)

    def merge_fine_tuning_dataset(self, x: pipeline.Pipe, size: int):
        storage = pipeline.get_wav_folder_storage(
            self._config.ref_dataset_name)
        y = pipeline.LabeledSource([self._config.framerate], storage,
                                   batch_size=size, start_index=self._config.test_size, fetch_mode=pipeline.COUPLED_FETCH_MODE)
        y = pipeline.FramesAugment(0.8, self._config.framerate)(y)
        y = pipeline.Frontend(self._config.frontend_shape, self._frontend)(y)

        return pipeline.Merge(y)(x)

    @ property
    def model(self):
        return self._acoustic_model

    @ abstractmethod
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
            result = self._evaluator.evaluate().get_summary()
            tf.summary.scalar("correct", result.correct)
            tf.summary.scalar("incorrect", result.incorrect)
            tf.summary.scalar("false_silence", result.false_silence)
            tf.summary.scalar("false_unknown", result.false_unknown)
            tf.summary.scalar("false_word", result.false_word)
            tf.summary.scalar("correct_weight", result.correct_weight)
            tf.summary.scalar("incorrect_weight", result.incorrect_weight)


class TrainerFactoryBase(ABC):
    @ abstractmethod
    def get_trainer(self, stage: int) -> TrainerBase:
        pass
