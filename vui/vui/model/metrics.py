import tensorflow as tf
import numpy as np
import time
from keras.utils.layer_utils import count_params
from keras.utils.vis_utils import model_to_dot
from tensorflow.keras import Sequential, Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import IPython.core.magics.namespace  # not used here, but need for tensorflow
from types import SimpleNamespace
from typing import Optional, Tuple, Union
from vui.infrastructure.filesystem import FilesystemProvider
from vui.model.services import FramesToEmbeddingService
from vui.model.data_access import ReferenceWordsDictionary
from vui.recognition import WordRecognizer
from vui.dataset.storage import ROW_FETCH_MODE, Storage, get_storage_from_wav_folder


def get_flops(model: Union[Model, Sequential], batch_size: Optional[int] = None) -> int:
    """
    https://pypi.org/project/keras-flops/

    Calculate FLOPS for tf.keras.Model or tf.keras.Sequential .
    Ignore operations used in only training mode such as Initialization.
    Use tf.profiler of tensorflow v1 api.
    """
    if not isinstance(model, (Sequential, Model)):
        raise KeyError(
            "model arguments must be tf.keras.Model or tf.keras.Sequential instanse"
        )

    if batch_size is None:
        batch_size = 1

    # convert tf.keras model into frozen graph to count FLOPS about operations used at inference
    # FLOPS depends on batch size
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs
    ]
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPS with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts["output"] = "none"
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts,
    )
    return flops.total_float_ops


class StructureInfo:
    def __init__(self):
        self.variable_count: int = 0
        self.weights_size: int = 0
        self.flops: int = 0
        self.svg: Optional[bytearray] = None


class EvaluationSummary:
    def __init__(self) -> None:
        self.total = 0
        self.correct = 0
        self.incorrect = 0
        self.false_silence = 0
        self.false_unknown = 0
        self.false_word = 0
        self.correct_weight = 0
        self.incorrect_weight = 0

    def normalize(self):
        self.correct_weight = np.true_divide(self.correct_weight, self.correct)
        self.incorrect_weight = np.true_divide(
            self.incorrect_weight, self.incorrect)

        self.correct = np.true_divide(self.correct, self.total)
        self.incorrect = np.true_divide(self.incorrect, self.total)
        self.false_silence = np.true_divide(self.false_silence, self.total)
        self.false_unknown = np.true_divide(self.false_unknown, self.total)
        self.false_word = np.true_divide(self.false_word, self.total)
        self.total = 1


class ConfusionMatrix:
    def __init__(self, labels: list, silence_word: str, unknown_word: str) -> None:
        all_labels = labels.copy()
        all_labels.append(silence_word)
        all_labels.append(unknown_word)

        self.matrix = self._create_confusion_matrix(all_labels)
        self.silence_word = silence_word
        self.unknown_word = unknown_word

    def add(self, expected: str, actual: str, weight: float):
        entry = self.matrix[expected][actual]
        entry[0] += 1
        entry[1] += weight

    def get_summary(self):
        summary = EvaluationSummary()
        for expected, actual_vector in self.matrix.items():
            for actual, entry in actual_vector.items():
                count, weight = entry

                summary.total += count
                if actual == expected:
                    summary.correct += count
                    summary.correct_weight += weight
                else:
                    summary.incorrect += count
                    summary.incorrect_weight += weight

                    if actual == self.silence_word:
                        summary.false_silence += count
                    elif actual == self.unknown_word:
                        summary.false_unknown += count
                    else:
                        summary.false_word += count

        summary.normalize()
        return summary

    def _create_confusion_matrix(self, labels: list) -> dict:
        matrix = {}
        for expected_word in labels:
            actual_vector = {}
            for actual_word in labels:
                actual_vector[actual_word] = [0, 0]
            matrix[expected_word] = actual_vector
        return matrix


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

    def evaluate(self) -> ConfusionMatrix:
        path = self._filesystem.get_dataset_path(
            "r", self._config.ref_dataset_name)

        samples_by_words = get_storage_from_wav_folder(path).as_dict()
        words = list(samples_by_words.keys())
        samples_by_dictors = self._get_samples_by_dictors(samples_by_words)

        matrix = ConfusionMatrix(
            words, self._config.silence_word, self._config.unknown_word)
        for ref_dictor_i, ref_sample in enumerate(samples_by_dictors):
            self._ref_word_dictionary.update(words, ref_sample.embeddings)

            for dictor_i, sample in enumerate(samples_by_dictors):
                if dictor_i != ref_dictor_i:
                    for word_i, frames in enumerate(sample.frames):
                        expected = words[word_i]
                        actual, weight = self._word_recognizer.recognize(
                            frames)
                        matrix.add(expected, actual, weight)

        self._ref_word_dictionary.force_load()
        return matrix

    def _get_samples_by_dictors(self, samples_by_words: dict) -> list:
        samples_by_dictors = []
        words = list(samples_by_words.keys())
        frames_length = self._config.framerate

        for i in range(self._config.test_size):
            frames = np.zeros([len(words), frames_length])
            for j, samples_per_word in enumerate(samples_by_words.values()):
                if len(samples_per_word[i]) < frames_length:
                    frames[j] = np.pad(
                        samples_per_word[i], (0, frames_length-len(samples_per_word[i])))
                else:
                    frames[j] = samples_per_word[i, :frames_length]

            embeddings = self._f2e_service.encode(frames)

            samples_per_dictor = SimpleNamespace(
                frames=frames, embeddings=embeddings)
            samples_by_dictors.append(samples_per_dictor)

        return samples_by_dictors
