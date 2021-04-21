from argparse import ArgumentParser
import vui.infrastructure.locator as locator
import vui.model.metrics as metrics


def main(stage: int):
    config = locator.get_config()
    trainer = locator.get_trainer_factory().get_trainer(stage)

    info = metrics.get_structure_info(trainer.model.encoder)
    locator.get_model_info_saver().save_structure_info(info)

    start_step = trainer.model.get_checkpoint_step(stage) + 1
    end_step = start_step + config.training_steps

    print("Optimization Started!")
    trainer.run(start_step, end_step)
    print("Optimization Finished!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--stage", dest="stage",
                        help="training stage", default=0)

    main(parser.parse_args().stage)
