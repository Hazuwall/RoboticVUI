from typing import Callable
import tensorflow as tf
import vui.infrastructure.locator as locator
import vui.model.metrics as metrics


def main(stage: int = 0):
    config = locator.get_config()
    trainer = locator.get_trainer_factory().get_trainer(stage)

    info = metrics.get_structure_info(trainer.model.encoder)
    locator.get_model_info_saver().save_structure_info(info)

    start_step = trainer.model.get_checkpoint_step(stage) + 1
    end_step = start_step + config.training_steps

    print("Optimization Started!")

    for step in tf.range(start_step, end_step, dtype=tf.int64):
        retry_on_error(lambda: trainer.run_step(step), 5)

        if (step % config.checkpoint_interval) == 0:
            trainer.model.save(int(step))
            if config.verbose:
                print(int(step))

    print("Optimization Finished!")


def retry_on_error(func: Callable, attempts: int):
    counter = 0
    while True:
        try:
            func()
        except:
            counter += 1
            if counter >= attempts:
                raise
            else:
                continue
        else:
            break


if __name__ == "__main__":
    main()
