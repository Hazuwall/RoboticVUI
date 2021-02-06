import tensorflow as tf
from typing import Callable, Optional
import numpy as np
import dsp_utils
from filesystem import FilesystemProvider


class WeightsStorage:
    def __init__(self, filesystem: FilesystemProvider):
        self.filesystem = filesystem

    def load(self, model: tf.keras.Model, step: Optional[int] = None):
        model_dir = self.filesystem.get_model_dir(model.name)
        if step is None:
            start_step = 0
            path = tf.train.latest_checkpoint(model_dir.weights)
            if path is not None:
                start_step = self.filesystem.get_checkpoint_step(path)
                model.load_weights(path)
            else:
                print("Checkpoint is not found. New weights has been used.")
        else:
            start_step = step
            path = self.filesystem.get_checkpoint_path(model_dir.weights, step)
            model.load_weights(path)
        return start_step

    def save(self, model: tf.keras.Model, step: int):
        model_dir = self.filesystem.get_model_dir(model.name)
        path = model_dir.checkpoint(step)
        model.save_weights(path)


class ReferenceWordsDictionary():
    def __init__(self, config, filesystem: FilesystemProvider, frames_encoding_handler: Callable):
        self.config = config
        self.filesystem = filesystem
        self.frames_encoding_handler = frames_encoding_handler

        words, embeddings = self.get_ref_words()
        self.words = words
        self.embeddings = embeddings

    def get_ref_words(self):
        ref_word_paths = self.filesystem.get_reference_word_paths()
        words = []
        embeddings = []
        for word, path in ref_word_paths.items():
            frames = dsp_utils.read(path)
            embedding = self.frames_encoding_handler(frames)
            words.append(word)
            embeddings.append(embedding)
        return words, np.stack(embeddings, axis=0)

    def update(self):
        words, embeddings = self.get_ref_words()
        self.words = words
        self.embeddings = embeddings
