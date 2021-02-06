import tensorflow as tf
import core.training
import infrastructure.locator as locator


def main(stage=1):
    config = locator.get_config()
    trainer = core.training.get_trainer(config, locator, stage)

    if config.verbose:
        trainer.model.summary()
    start_step = trainer.model.checkpoint_step + 1
    end_step = start_step + config.training_steps

    locator.get_filesystem_provider().store_core_modules()
    print("Optimization Started!")

    for step in tf.range(start_step, end_step, dtype=tf.int64):
        trainer.run_step(step)

        if (step % config.checkpoint_interval) == 0:
            trainer.model.save(int(step))
            if config.verbose:
                print(int(step))

    print("Optimization Finished!")


if __name__ == "__main__":
    main()
