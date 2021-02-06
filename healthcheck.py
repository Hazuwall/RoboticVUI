import os
import sys
from termcolor import colored
import locator


def main():
    status("Arranging...")
    arrange()

    status("Running...")
    run()

    status("Running once more...")
    run()

    status("All right!")


def status(text):
    print(colored("[Healthcheck]: " + text, 'green'))


def arrange():
    config = locator.get_config()
    config.experiment_name = "healthcheck"
    config.build = 0
    config.training_steps = 2
    config.display_interval = 1
    config.checkpoint_interval = 2
    locator.get_filesystem_provider().clear_experiment()


def run():
    with HiddenPrints():
        import training
        training.main()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


if __name__ == "__main__":
    main()
