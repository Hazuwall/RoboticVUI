import tensorflow as tf
import vui.model.tf_utils as tf_utils
from vui.dataset.pipeline import DatasetPipelineFactory, COUPLED_FETCH_MODE
from vui.infrastructure.filesystem import FilesystemProvider
from vui.model.abstract import AcousticModelBase
from vui.training.abstract import AcousticModelTrainer, TrainerFactoryBase
from vui.model.metrics import Evaluator


class TrainerFactory(TrainerFactoryBase):
    def __init__(self, config, filesystem: FilesystemProvider, acoustic_model: AcousticModelBase, dataset_pipeline_factory: DatasetPipelineFactory, evaluator: Evaluator):
        super(TrainerFactory, self).__init__(
            config, filesystem, acoustic_model, dataset_pipeline_factory, evaluator)

    def get_trainer(self, stage: int):
        return Trainer(self._config, self._filesystem, self._acoustic_model, self._dataset_pipeline_factory, self._evaluator)


class Trainer(AcousticModelTrainer):
    def __init__(self, config, filesystem: FilesystemProvider, acoustic_model: AcousticModelBase, pipeline_factory: DatasetPipelineFactory, evaluator: Evaluator) -> None:
        super(Trainer, self).__init__(
            config, filesystem, acoustic_model, pipeline_factory, evaluator)

        # Classes initialization
        line1 = pipeline_factory.get_builder().from_labeled_storage(
            "s_en_SpeechCommands").with_size(config.batch_size//3).with_fetch_mode(COUPLED_FETCH_MODE).cache().build()

        line2 = pipeline_factory.get_builder().from_unlabeled_storage(
            "t_mx_Mix").with_fetch_mode(COUPLED_FETCH_MODE).shuffle().augment().cache().build()

        self._dataset = pipeline_factory.get_builder().merge(line1, line2).build()
        self._validation_dataset = pipeline_factory.get_builder().from_labeled_storage(
            "s_en_SpeechCommands").with_size(config.validation_size).with_fetch_mode(COUPLED_FETCH_MODE).build()

        optimizer = tf.keras.optimizers.Adam()
        model = self._acoustic_model

        # Graph initialization
        @tf.function
        def train_step(x):
            with tf.GradientTape() as tape:
                codes = model.encode(x, training=True)
                cost, metrics = tf_utils.cos_similarity_triplet_loss(
                    codes, 0.95, 0.3)
                accuracy = tf_utils.coupled_cos_similarity_accuracy(
                    codes[:self._config.batch_size])

                vars = model.encoder.trainable_variables
                gradients = tape.gradient(cost, vars)
                optimizer.apply_gradients(zip(gradients, vars))

                return accuracy, cost, metrics

        self.train_step = train_step

    def run_step_core(self, step: tf.Tensor):
        x, _ = self._dataset.get_batch()
        accuracy, cost, triplet_metrics = self.train_step(x)

        if step % self._config.display_interval == 0:
            with tf.name_scope("training"):
                tf.summary.scalar(
                    "accuracy/{}words".format(self._config.batch_size // 2), accuracy)
                tf_utils.cos_similarity_triplet_summary(
                    cost, triplet_metrics)

        if step % self._config.checkpoint_interval == 0:
            with tf.name_scope("validation/speech_commands"):
                x, _ = self._validation_dataset.get_batch()
                codes = self.model.encode(x, training=False)
                cost, triplet_metrics = tf_utils.cos_similarity_triplet_loss(
                    codes)
                tf_utils.cos_similarity_triplet_summary(
                    cost, triplet_metrics)
