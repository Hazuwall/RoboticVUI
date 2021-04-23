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
        path = self._filesystem.get_dataset_path(
            "r", self._config.ref_dataset_name)

        storage = get_storage_from_wav_folder(path)
        word_samples = self._get_word_samples(storage)

        total_count = 0
        correct_count = 0
        silence_count = 0
        unknown_count = 0
        correct_weight = 0
        incorrect_weight = 0
        for ref_dictor_i in range(self._config.test_size):
            self._set_ref_samples(word_samples, ref_dictor_i)

            for expected_word in word_samples.keys():
                frames = word_samples[expected_word].frames
                for dictor_i in range(len(frames)):
                    if dictor_i != ref_dictor_i:
                        actual_word, weight = self._word_recognizer.recognize(
                            frames[dictor_i])
                        if actual_word == expected_word:
                            correct_count += 1
                            correct_weight += weight
                        else:
                            incorrect_weight += weight
                        silence_count += 1 if actual_word == self._config.silence_word else 0
                        unknown_count += 1 if actual_word == self._config.unknown_word else 0
                        total_count += 1

        self._ref_word_dictionary.force_load()
        wrong_word_count = total_count - correct_count - unknown_count - silence_count
        return np.true_divide(correct_count, total_count), np.true_divide(silence_count, total_count), np.true_divide(unknown_count, total_count), np.true_divide(wrong_word_count, total_count), np.true_divide(correct_weight, correct_count), np.true_divide(incorrect_weight, total_count - correct_count)

    def _get_word_samples(self, storage: Storage) -> dict:
        word_embeddings = {}
        for word in storage.get_dataset_list():
            frames = storage.fetch_subset(
                word, start=0, size=self._config.test_size, mode=ROW_FETCH_MODE)
            embeddings = self._f2e_service.encode(frames)
            word_embeddings[word] = SimpleNamespace(
                frames=frames, embeddings=embeddings)
        return word_embeddings

    def _set_ref_samples(self, word_samples: dict, dictor_i: int) -> None:
        words = list(word_samples.keys())
        embeddings = [word_samples[word].embeddings[dictor_i]
                      for word in words]
        embeddings = np.stack(embeddings, axis=0)
        self._ref_word_dictionary.update(words, embeddings)
