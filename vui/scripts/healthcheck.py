from termcolor import colored
import os
import sys
import vui.infrastructure.locator as locator


def main():
    status("Starting...")

    status("Arranging test experiment...")
    experiment_dir = locator.get_filesystem_provider().get_experiment_dir()
    with locator.create_isolated_scope():
        override_config()
        clone_experiment_core(experiment_dir)

        status("Running training...")
        run_training()

        status("Running inference...")
        run_inference()

    status("Reloading...")
    with locator.create_isolated_scope():
        override_config()

        status("Continuing training...")
        run_training()

    status("All right!")


def status(text: str):
    print(colored("[Healthcheck]: " + text, 'green'))


def override_config():
    config = locator.get_config()
    config.build = 1000
    config.training_steps = 2
    config.display_interval = 1
    config.checkpoint_interval = 2
    config.cache_size = config.batch_size * 2
    config.validation_size = config.batch_size
    config.test_size = config.batch_size


def clone_experiment_core(source_dir: str):
    filesystem = locator.get_filesystem_provider()
    filesystem.clear_experiment()
    filesystem.clone_core_modules(source_dir)


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
