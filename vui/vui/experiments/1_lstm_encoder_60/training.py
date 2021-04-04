import tensorflow as tf
import vui.model.tf_utils as tf_utils
from vui.dataset.pipeline import DatasetPipelineFactory, COUPLED_FETCH_MODE
from vui.infrastructure.filesystem import FilesystemProvider
from vui.model.abstract import AcousticModelBase
from vui.training.abstract import TrainerBase, TrainerFactoryBase


class TrainerFactory(TrainerFactoryBase):
    def __init__(self, config, filesystem: FilesystemProvider, acoustic_model: AcousticModelBase, dataset_pipeline_factory: DatasetPipelineFactory):
        super(TrainerFactory, self).__init__(
            config, filesystem, acoustic_model, dataset_pipeline_factory)

    def get_trainer(self, stage: int):
        return Trainer(self._config, self._filesystem, self._acoustic_model, self._dataset_pipeline_factory)


class Trainer(TrainerBase):
    def __init__(self, config, filesystem: FilesystemProvider, acoustic_model: AcousticModelBase, pipeline_factory: DatasetPipelineFactory) -> None:
        super(Trainer, self).__init__(
            config, filesystem, acoustic_model, pipeline_factory)

        # Classes initialization
        line1 = pipeline_factory.get_builder().from_labeled_storage(
            "s_en_SpeechCommands").with_size(config.batch_size//3).with_fetch_mode(COUPLED_FETCH_MODE).cache().build()

        line2 = pipeline_factory.get_builder().from_unlabeled_storage(
            "t_mx_Mix").with_fetch_mode(COUPLED_FETCH_MODE).shuffle().augment().cache().build()

        self._dataset = pipeline_factory.get_builder().merge(line1, line2).build()

        logs_path = filesystem.get_model_dir(config.acoustic_model_name).logs
        self._summary_writer = tf.summary.create_file_writer(logs_path)

        optimizer = tf.keras.optimizers.Adam()
        model = self._acoustic_model

        # Graph initialization
        @tf.function
        def train_step(x, step):
            with self._summary_writer.as_default():
                with tf.GradientTape() as tape:
                    codes = model.encode(x, training=True)
                    cost, metrics = self.compute_loss(codes)
                    if step % config.display_interval == 0:
                        tf.summary.scalar(
                            'accuracy36', self.evaluate(codes), step)
                        tf.summary.scalar('cost', cost, step)
                        tf.summary.scalar('pos_similarity', metrics[0], step)
                        tf.summary.scalar('neg_similarity', metrics[1], step)
                        tf.summary.histogram('pos_distrib', metrics[2], step)
                        tf.summary.histogram('neg_distrib', metrics[3], step)

                    vars = model.encoder.trainable_variables
                    gradients = tape.gradient(cost, vars)
                    optimizer.apply_gradients(zip(gradients, vars))

        self.train_step = train_step

    def compute_loss(self, codes):
        coupled_codes = tf.reshape(codes, (-1, 2, self._config.embedding_size))
        anchor, positive = tf.unstack(coupled_codes, axis=1)
        pos_distrib = tf_utils.cos_similarity(anchor, positive, axis=1)
        pos_cost = tf.reduce_mean(tf.minimum(0.95, 1 - pos_distrib)**2)
        pos_similarity = tf.reduce_mean(pos_distrib)

        negative = tf.roll(positive, 1, axis=0)
        neg_distrib = tf_utils.cos_similarity(anchor, negative, axis=1)

        anchor = codes
        negative = tf.roll(codes, 2, axis=0)
        neg_distrib = tf.concat(
            [neg_distrib, tf_utils.cos_similarity(anchor, negative, axis=1)], axis=0)
        neg_cost = tf.reduce_mean(tf.maximum(0.3, neg_distrib)**2)
        neg_similarity = tf.reduce_mean(neg_distrib)

        cost = tf.debugging.check_numerics(pos_cost + neg_cost, "Cost is NaN.")
        return cost, [pos_similarity, neg_similarity, pos_distrib, neg_distrib]

    def evaluate(self, codes):
        codes = codes[:self._config.batch_size]
        codes = tf.reshape(codes, (-1, 2, self._config.embedding_size))
        anchor, positive = tf.unstack(codes, axis=1)
        anchor = tf.expand_dims(anchor, axis=1)
        positive = tf.expand_dims(positive, axis=0)
        similarity = tf_utils.cos_similarity(anchor, positive, axis=2)
        incorrect_prediction = tf.not_equal(tf.argmax(
            similarity, axis=0), tf.cast(tf.range(similarity.shape[0]), tf.int64))
        return 1 - tf.reduce_mean(tf.cast(incorrect_prediction, tf.float32))

    def run_step(self, step: tf.Tensor):
        x, _ = self._dataset.get_batch()
        self.train_step(x, step)
        self._summary_writer.flush()
