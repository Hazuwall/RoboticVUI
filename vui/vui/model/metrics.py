import tensorflow as tf
import numpy as np
import time
from keras.utils.layer_utils import count_params
from keras.utils.vis_utils import model_to_dot
from keras_flops import get_flops
import IPython.core.magics.namespace  # not used here, but need for tensorflow
from types import SimpleNamespace
from typing import Tuple
import vui.frontend.dsp as dsp
from vui.infrastructure.filesystem import FilesystemProvider
from vui.model.services import FramesToEmbeddingService
from vui.model.data_access import ReferenceWordsDictionary
from vui.recognition import WordRecognizer


class StructureInfo:
    def __init__(self):
        self._variable_count = 0
        self._weights_size = 0
        self._flops = 0
        self._svg = None

    @property
    def variable_count(self) -> int:
        return self._variable_count

    @variable_count.setter
    def variable_count(self, value: int):
        self._variable_count = value

    @property
    def weights_size(self) -> int:
        return self._weights_size

    @weights_size.setter
    def weights_size(self, value: int):
        self._weights_size = value

    @property
    def flops(self) -> int:
        return self._flops

    @flops.setter
    def flops(self, value: int):
        self._flops = value

    @property
    def svg(self) -> bytearray:
        return self._svg

    @svg.setter
    def svg(self, value: bytearray):
        self._svg = value


def get_structure_info(model: tf.keras.Model) -> StructureInfo:
    trainable_count = count_params(model.trainable_weights)
    non_trainable_count = count_params(model.non_trainable_weights)

    info = StructureInfo()
    info.variable_count = trainable_count
    info.weights_size = (trainable_count + non_trainable_count) * 4
    info.svg = model_to_dot(model, show_layer_names=True,
                            show_shapes=True, dpi=None).create(prog='dot', format='svg')
    info.flops = get_flops(model, batch_size=1)
    return info


class AveragingTimer:
    def __init__(self) -> None:
        self._count = 0
        self._total_time = 0
        self._last_start_time = 0

    def __enter__(self) -> None:
        self._last_start_time = time.monotonic()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._count += 1
        self._total_time += time.monotonic() - self._last_start_time
        self._last_start_time = 0

    def reset(self) -> float:
        avg_time = 0 if self._count == 0 else self._total_time / self._count
        self._count = 0
        self._total_time = 0
        return avg_time


class Evaluator:
    def __init__(self, config, filesystem: FilesystemProvider, ref_word_dictionary: ReferenceWordsDictionary,
                 f2e_service: FramesToEmbeddingService, word_recognizer: WordRecognizer):
        self._config = config
        self._filesystem = filesystem
        self._f2e_service = f2e_service
        self._ref_word_dictionary = ref_word_dictionary
        self._word_recognizer = word_recognizer

    def evaluate(self) -> Tuple[float, float, float]:
        word_paths = self._filesystem.get_test_word_paths()
        if len(word_paths.keys()) == 0:
            return np.NaN
        word_samples = self._get_word_samples(word_paths)

        total_count = 0
        correct_count = 0
        silence_count = 0
        unknown_count = 0
        correct_weight = 0
        incorrect_weight = 0
        word_samples_iterator = iter(word_samples.values())
        first_word_samples = next(word_samples_iterator)
        dictor_count = len(first_word_samples)
        for ref_dictor_i in range(dictor_count):
            self._set_ref_samples(word_samples, ref_dictor_i)

            for expected_word in word_samples.keys():
                for dictor_i, sample in enumerate(word_samples[expected_word]):
                    if dictor_i != ref_dictor_i:
                        actual_word, weight = self._word_recognizer.recognize(
                            sample.frames)
                        if actual_word == expected_word:
                            correct_count += 1
                            correct_weight += weight
                        else:
                            incorrect_weight += weight
                        silence_count += 1 if actual_word == self._config.silence_word else 0
                        unknown_count += 1 if actual_word == self._config.unknown_word else 0
                        total_count += 1

        self._ref_word_dictionary.force_load()
        return np.true_divide(correct_count, total_count), np.true_divide(silence_count, total_count), np.true_divide(unknown_count, total_count), np.true_divide(total_count - unknown_count - silence_count, total_count), np.true_divide(correct_weight, correct_count), np.true_divide(incorrect_weight, total_count - correct_count)

    def _get_word_samples(self, word_paths: dict) -> dict:
        word_embeddings = {}
        for word in word_paths:
            samples = []
            for path in word_paths[word]:
                frames = dsp.read(path)
                embedding = self._f2e_service.encode(frames)
                sample = SimpleNamespace(frames=frames, embedding=embedding)
                samples.append(sample)
            word_embeddings[word] = samples
        return word_embeddings

    def _set_ref_samples(self, word_samples: dict, dictor_i: int) -> None:
        words = list(word_samples.keys())
        embeddings = [word_samples[word][dictor_i].embedding for word in words]
        embeddings = np.stack(embeddings, axis=0)
        self._ref_word_dictionary.update(words, embeddings)
