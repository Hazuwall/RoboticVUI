import os
import shutil
import coolname
from types import SimpleNamespace
from typing import Optional


class FilesystemProvider:
    def __init__(self, config):
        self.config = config

    def get_experiment_dir(self) -> str:
        experiment_name = self.get_experiment_name(self.config.build)

        return "{dir}{build}_{name}\\".format(
            dir=self.config.experiments_dir, build=self.config.build, name=experiment_name)

    def get_experiment_name(self, build: int) -> Optional[str]:
        experiment_dir_names = os.listdir(self.config.experiments_dir)
        for dir_name in experiment_dir_names:
            if dir_name.startswith("{}_".format(build)):
                experiment_name_start = dir_name.find("_") + 1
                return dir_name[experiment_name_start:]
        return coolname.generate_slug(2)

    def get_logs_dir(self) -> str:
        return self.get_experiment_dir() + "logs\\"

    def get_model_dir(self, model_name: str) -> SimpleNamespace:
        model_dir = self.get_experiment_dir() + model_name + "\\"
        return SimpleNamespace(
            plugin_modules=model_dir,
            weights=model_dir,
            checkpoint=lambda step: model_dir + self.config.checkpoint_prefix + str(step))

    def get_core_module_path(self, module_name: str) -> str:
        return self.get_experiment_dir() + module_name + ".py"

    def clone_core_modules(self, source_dir: str) -> None:
        current_experiment_dir = self.get_experiment_dir()

        os.makedirs(current_experiment_dir, exist_ok=True)
        file_list = os.listdir(source_dir)
        for file_name in file_list:
            src_path = source_dir + file_name
            if os.path.isfile(src_path):
                shutil.copy(src_path, current_experiment_dir + file_name)

    def get_checkpoint_step(self, checkpoint_path: str) -> int:
        filename = os.path.basename(checkpoint_path)
        prefix_length = len(self.config.checkpoint_prefix)
        return int(filename[prefix_length:])

    def get_dataset_path(self, type_letter: str, label: str, extension: str = "") -> str:
        return self.config.datasets_dir + label + '_' + type_letter + extension

    def clear_experiment(self) -> None:
        dir = self.get_experiment_dir()
        shutil.rmtree(dir, ignore_errors=True)
