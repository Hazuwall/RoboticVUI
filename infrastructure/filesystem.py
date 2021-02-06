import os
import shutil
from types import SimpleNamespace
from typing import Optional


class FilesystemProvider:
    def __init__(self, config):
        self.config = config

    def get_experiment_dir(self):
        return "{dir}{build}_{name}\\".format(
            dir=self.config.experiments_dir, build=self.config.build, name=self.config.experiment_name)

    def get_model_dir(self, model_name: str) -> SimpleNamespace:
        model_dir = self.get_experiment_dir() + model_name + "\\"
        return SimpleNamespace(
            plugin_modules=model_dir,
            weights=model_dir,
            logs=model_dir + "logs\\",
            checkpoint=lambda step: model_dir + self.config.checkpoint_prefix + str(step))

    def get_checkpoint_step(self, checkpoint_path: str) -> int:
        filename = os.path.basename(checkpoint_path)
        prefix_length = len(self.config.checkpoint_prefix)
        return int(filename[prefix_length:])

    def get_dataset_path(self, type_letter: str, label: Optional[str] = None):
        if label is None:
            label = self.config.dataset_label
        return self.config.datasets_dir + label + '_' + type_letter + ".hdf5"

    def get_reference_word_paths(self):
        dir_path = self.config.ref_words_dir
        file_list = os.listdir(dir_path)
        ref_words = {}
        for filename in file_list:
            word = filename[:-4]
            file_path = os.path.join(dir_path, filename)
            ref_words[word] = file_path
        return ref_words

    def store_core_modules(self):
        src_dir = self.config.core_dir
        dest_dir = self.get_experiment_dir()
        os.makedirs(dest_dir, exist_ok=True)
        file_list = os.listdir(src_dir)
        for filename in file_list:
            src_path = src_dir + filename
            if os.path.isfile(src_path):
                shutil.copy(src_path, dest_dir + filename)

    def clear_experiment(self):
        dir = self.get_experiment_dir()
        shutil.rmtree(dir, ignore_errors=True)
