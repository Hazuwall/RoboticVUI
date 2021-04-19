import tensorflow as tf
import vui.model.tf_utils as tf_utils
from vui.infrastructure.filesystem import FilesystemProvider
from vui.model.abstract import AcousticModelBase
from vui.training.abstract import AcousticModelTrainer, TrainerFactoryBase
from vui.frontend.abstract import FrontendProcessorBase
from vui.model.metrics import Evaluator
import vui.dataset.pipeline as pipeline
import vui.infrastructure.locator as locator


class TrainerFactory(TrainerFactoryBase):
    def get_trainer(self, stage: int):
        return Trainer(locator.get_config(), locator.get_filesystem_provider(),
                       locator.get_acoustic_model(), locator.get_evaluator(), locator.get_frontend_processor())


class Trainer(AcousticModelTrainer):
    def __init__(self, config, filesystem: FilesystemProvider,
                 acoustic_model: AcousticModelBase, evaluator: Evaluator, frontend: FrontendProcessorBase) -> None:
        super(Trainer, self).__init__(
            config, filesystem, acoustic_model, evaluator)

        self._frontend = frontend

        # Classes initialization
        self._dataset = self.create_training_pipeline()
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
                    codes[-self._config.batch_size:])

                vars = model.encoder.trainable_variables
                gradients = tape.gradient(cost, vars)
                optimizer.apply_gradients(zip(gradients, vars))

                return accuracy, cost, metrics

        self.train_step = train_step

    def create_training_pipeline(self):
        storage = pipeline.get_hdf_storage('h', "s_en_SpeechCommands")
        x = pipeline.LabeledStorage(self._config.frontend_shape, storage,
                                    batch_size=self._config.cache_size*5, fetch_mode=pipeline.COUPLED_FETCH_MODE)
        x = pipeline.Shuffle(patch_size=4)(x)
        x = pipeline.Cache(self._config.batch_size // 3)(x)

        storage = pipeline.get_hdf_storage('h', "t_mx_Mix")
        z = pipeline.UnlabeledStorage(self._config.frontend_shape, storage,
                                      batch_size=self._config.cache_size, fetch_mode=pipeline.COUPLED_FETCH_MODE)
        z = pipeline.Shuffle(patch_size=4)(z)
        storage = pipeline.get_hdf_storage('r', "t_mx_Mix")
        z = pipeline.UnlabeledAugment(
            storage, self._frontend, self._config.aug_rate, self._config.framerate)(z)
        z = pipeline.Cache(self._config.batch_size)(z)

        return pipeline.Merge(z)(x)

    def run_step_core(self, step: tf.Tensor):
        x, _ = self._dataset.get_batch()
        accuracy, cost, triplet_metrics = self.train_step(x)

        if step % self._config.display_interval == 0:
            with tf.name_scope("training"):
                tf.summary.scalar(
                    "accuracy/{}words".format(self._config.batch_size // 2), accuracy)
                tf_utils.cos_similarity_triplet_summary(
                    cost, triplet_metrics)
