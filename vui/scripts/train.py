from argparse import ArgumentParser
from termcolor import colored
import vui.infrastructure.locator as locator
import vui.model.metrics as metrics


def main(stage: int, steps: int):
    status("Loading model...")
    trainer = locator.get_trainer_factory().get_trainer(stage)

    status("Saving model info...")
    info = metrics.get_structure_info(trainer.model.encoder)
    locator.get_model_info_saver().save_structure_info(info)

    start_step = trainer.model.get_checkpoint_step(stage) + 1
    end_step = start_step + steps

    status("Started! Stage {}, {} steps...".format(stage, steps))
    trainer.run(start_step, end_step)
    status("Finished!")


def status(text: str):
    print(colored("[Training]: " + text, 'green'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--stage", dest="stage",
                        help="training stage", default=0, type=int)
    parser.add_argument("-n", "--steps", dest="steps",
                        help="training steps", default=1000, type=int)

    main(parser.parse_args().stage, parser.parse_args().steps)
