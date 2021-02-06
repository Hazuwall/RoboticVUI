import tensorflow as tf
import models.tf_utils as tf_utils


def get_trainer(config, locator, stage: int):
    return Trainer(config, locator)


class Trainer:
    def __init__(self, config, locator) -> None:
        self.config = config

        # Classes initialization
        line1 = locator.get_dataset_pipeline_builder().from_labeled_storage(
            "s_en_SpeechCommands").build()
        line2 = locator.get_dataset_pipeline_builder().from_unlabeled_storage(
            "t_mx_Mix").cache(1000).augment().with_size(config.training_batch_size//3).build()
        self.dataset = locator.get_dataset_pipeline_builder().merge(line1, line2).build()

        logs_path = locator.get_filesystem_provider(
        ).get_model_dir(config.acoustic_model_name).logs
        self.summary_writer = tf.summary.create_file_writer(logs_path)

        optimizer = tf.keras.optimizers.Adam()
        self.model = locator.get_acoustic_model()

        # Graph initialization
        @tf.function
        def train_step(x, step):
            with self.summary_writer.as_default():
                with tf.GradientTape() as tape:
                    codes = self.model.encode(x, training=True)
                    cost, metrics = self.compute_loss(codes)
                    if step % config.display_interval == 0:
                        tf.summary.scalar(
                            'accuracy36', self.evaluate(codes), step)
                        tf.summary.scalar('cost', cost, step)
                        tf.summary.scalar('pos_similarity', metrics[0], step)
                        tf.summary.scalar('neg_similarity', metrics[1], step)
                        tf.summary.histogram('pos_distrib', metrics[2], step)
                        tf.summary.histogram('neg_distrib', metrics[3], step)

                    vars = self.model.encoder.trainable_variables
                    gradients = tape.gradient(cost, vars)
                    optimizer.apply_gradients(zip(gradients, vars))

        self.train_step = train_step

    def compute_loss(self, codes):
        coupled_codes = tf.reshape(codes, (-1, 2, self.config.embedding_size))
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

        cost = pos_cost + neg_cost
        return cost, [pos_similarity, neg_similarity, pos_distrib, neg_distrib]

    def evaluate(self, codes):
        codes = codes[:self.config.training_batch_size]
        codes = tf.reshape(codes, (-1, 2, self.config.embedding_size))
        anchor, positive = tf.unstack(codes, axis=1)
        anchor = tf.expand_dims(anchor, axis=1)
        positive = tf.expand_dims(positive, axis=0)
        similarity = tf_utils.cos_similarity(anchor, positive, axis=2)
        incorrect_prediction = tf.not_equal(tf.argmax(
            similarity, axis=0), tf.cast(tf.range(similarity.shape[0]), tf.int64))
        return 1 - tf.reduce_mean(tf.cast(incorrect_prediction, tf.float32))

    def run_step(self, step):
        x = self.dataset.get_batch()
        self.train_step(x, step)
        self.summary_writer.flush()
