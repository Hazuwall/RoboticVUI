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

        summary_writer = self._summary_writer
        training_accuracy_label = str.format(
            "training/accuracy/{}words", config.batch_size // 2)

        optimizer = tf.keras.optimizers.Adam()
        model = self._acoustic_model

        # Graph initialization
        @tf.function
        def train_step(x, step):
            with summary_writer.as_default():
                with tf.GradientTape() as tape:
                    codes = model.encode(x, training=True)
                    cost, metrics = tf_utils.cos_similarity_triplet_loss(
                        codes, 0.95, 0.3)
                    training_accuracy = tf_utils.coupled_cos_similarity_accuracy(
                        codes[:self._config.batch_size])
                    if step % config.display_interval == 0:
                        tf.summary.scalar(
                            training_accuracy_label, training_accuracy, step)
                        tf.summary.scalar('training/cost', cost, step)
                        tf.summary.scalar(
                            'training/pos_similarity', metrics[0], step)
                        tf.summary.scalar(
                            'training/neg_similarity', metrics[1], step)
                        tf.summary.histogram(
                            'training/pos_distrib', metrics[2], step)
                        tf.summary.histogram(
                            'training/neg_distrib', metrics[3], step)

                    vars = model.encoder.trainable_variables
                    gradients = tape.gradient(cost, vars)
                    optimizer.apply_gradients(zip(gradients, vars))

        self.train_step = train_step

    def run_step_core(self, step: tf.Tensor):
        x, _ = self._dataset.get_batch()
        self.train_step(x, step)
