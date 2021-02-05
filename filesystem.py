import os
import random
import string
from types import SimpleNamespace

CHECKPOINT_PREFIX = "checkpoint-"


class FilesystemProvider:
    def __init__(self, config):
        self.config = config

    def get_model_dir(self, model_name: str, experiment_name=None) -> SimpleNamespace:
        if experiment_name is None:
            experiment_name = self.config.experiment_name
        experiment_path = self.config.models_root + \
            str(self.config.build) + '_' + experiment_name + "\\"
        model_path = experiment_path + model_name + "\\"
        return SimpleNamespace(
            weights=model_path + "weights\\",
            logs=model_path + "logs\\",
            checkpoint=lambda step: model_path + CHECKPOINT_PREFIX + str(step))

    def get_checkpoint_step(self, checkpoint_path: str) -> int:
        filename = os.path.basename(checkpoint_path)
        return int(filename[len(CHECKPOINT_PREFIX):])

    def get_dataset_path(self, type_letter, label=None):
        if label is None:
            label = self.config.dataset_label
        return self.config.datasets_root + label + '_' + type_letter + ".hdf5"

    def get_reference_word_paths(self):
        dir_path = self.config.words_dir
        file_list = os.listdir(dir_path)
        ref_words = {}
        for filename in file_list:
            word = filename[:-4]
            file_path = os.path.join(dir_path, filename)
            ref_words[word] = file_path
        return ref_words
