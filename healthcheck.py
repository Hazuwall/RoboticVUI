#from ._setup import *
from termcolor import colored
import os
import sys
import infrastructure.locator as locator


def main():
    status("Arranging...")
    override_config()
    locator.get_filesystem_provider().clear_experiment()

    status("Running training...")
    run_training()

    status("Running inference...")
    run_inference()

    status("Reloading...")
    locator.reset()
    override_config()

    status("Continuing training...")
    run_training()

    status("All right!")


def status(text):
    print(colored("[Healthcheck]: " + text, 'green'))


def override_config():
    config = locator.get_config()
    config.experiment_name = "healthcheck"
    config.build = 0
    config.training_steps = 2
    config.display_interval = 1
    config.checkpoint_interval = 2


def run_training():
    with HiddenPrints():
        import train
        stages = locator.get_config().stages
        for stage in range(stages):
            train.main(stage)


def run_inference():
    with HiddenPrints():
        import inference
        inference.main(3)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


if __name__ == "__main__":
    main()
