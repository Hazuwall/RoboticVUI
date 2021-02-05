import os
import shutil
from types import SimpleNamespace

CHECKPOINT_PREFIX = "checkpoint-"


class FilesystemProvider:
    def __init__(self, config):
        self.config = config

    def get_experiment_dir(self):
        return "{dir}{build}_{name}\\".format(
            dir=self.config.models_root, build=self.config.build, name=self.config.experiment_name)

    def get_model_dir(self, model_name: str) -> SimpleNamespace:
        model_dir = self.get_experiment_dir() + model_name + "\\"
        return SimpleNamespace(
            plugin_modules=model_dir,
            weights=model_dir,
            logs=model_dir + "logs\\",
            checkpoint=lambda step: model_dir + CHECKPOINT_PREFIX + str(step))

    def get_checkpoint_step(self, checkpoint_path: str) -> int:
        filename = os.path.basename(checkpoint_path)
        return int(filename[len(CHECKPOINT_PREFIX):])

    def get_dataset_path(self, type_letter, label=None):
        if label is None:
            label = self.config.dataset_label
        return self.config.datasets_root + label + '_' + type_letter + ".hdf5"

    def get_reference_word_paths(self):
        dir_path = self.config.ref_words_dir
        file_list = os.listdir(dir_path)
        ref_words = {}
        for filename in file_list:
            word = filename[:-4]
            file_path = os.path.join(dir_path, filename)
            ref_words[word] = file_path
        return ref_words

    def store_plugin_modules(self):
        dest_dir = self.get_experiment_dir()
        os.makedirs(dest_dir, exist_ok=True)
        for module in self.config.plugin_modules:
            file_name = "{}.py".format(module)
            shutil.copy(file_name, dest_dir + file_name)
