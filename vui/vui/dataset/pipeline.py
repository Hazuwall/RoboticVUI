from abc import ABC, abstractmethod, abstractproperty
import random
from typing import Optional, Tuple
from vui.frontend.abstract import FrontendProcessorBase
import numpy as np
from vui.dataset.storage import Storage, HdfStorage, WavFolderStorage, COUPLED_FETCH_MODE, RANDOM_FETCH_MODE, PATCH_FETCH_MODE
from vui.dataset import augmentation
import vui.infrastructure.locator as locator


def expand_patch_indices_to_item_indices(patch_indices: list, patch_size: int) -> list:
    indices = []
    for i in patch_indices:
        for j in range(patch_size):
            indices.append(i*patch_size+j)
    return indices


def get_hdf_storage(type_letter: str, dataset_name: str) -> HdfStorage:
    path = locator.get_filesystem_provider().get_dataset_path(
        type_letter, dataset_name, ".hdf5")
    return HdfStorage(path)


def get_wav_folder_storage(dataset_name: str) -> WavFolderStorage:
    path = locator.get_filesystem_provider().get_dataset_path(
        "r", dataset_name)
    return WavFolderStorage(path)


class Pipe(ABC):
    @abstractmethod
    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get 3D-array of data X and vector of indices y."""
        pass

    @abstractproperty
    def output_shape(self) -> list:
        """A shape of 3D-array without first axis."""
        pass


class SourcePipe(Pipe):
    def __init__(self, output_shape: list) -> None:
        self._output_shape = output_shape

    @property
    def output_shape(self) -> list:
        return self._output_shape


class TransformPipe(Pipe):
    def __init__(self, output_shape: Optional[list] = None) -> None:
        self._input_shape = None
        self._output_shape = output_shape
        self._previous_pipe: Optional[Pipe] = None
        self._do_clone_shape = output_shape is None

    @property
    def input_shape(self) -> list:
        return self._input_shape

    @property
    def output_shape(self) -> list:
        return self._output_shape

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        batch = self._previous_pipe.get_batch()
        return self.process(*batch)

    @abstractmethod
    def process(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def __call__(self, previous_pipe: Pipe) -> Pipe:
        self._previous_pipe = previous_pipe
        self._input_shape = previous_pipe.output_shape
        if self._do_clone_shape:
            self._output_shape = self._input_shape
        return self


class LabeledStorage(SourcePipe):
    def __init__(self, output_shape: list, storage: Storage, batch_size: int, start_index: int = 0,
                 fetch_mode: str = RANDOM_FETCH_MODE, labels: Optional[list] = None, use_max_classes_per_batch: bool = False) -> None:
        super().__init__(output_shape)

        labels = storage.get_dataset_list() if labels is None else labels
        if fetch_mode == COUPLED_FETCH_MODE:
            self.inner_fetch_mode = RANDOM_FETCH_MODE
            self.max_classes_per_batch = len(labels) if len(
                labels) % 2 == 0 else len(labels) - 1
            self.classes_per_batch = self.get_greatest_devisor(
                number=batch_size//2, max_divisor=self.max_classes_per_batch)
        else:
            self.inner_fetch_mode = fetch_mode
            self.max_classes_per_batch = len(labels)
            self.classes_per_batch = self.get_greatest_devisor(
                number=batch_size, max_divisor=self.max_classes_per_batch)

        self.storage = storage
        self.labels = labels
        self.batch_size = batch_size
        self.start_index = start_index
        self.fetch_mode = fetch_mode
        self.use_max_classes_per_batch = use_max_classes_per_batch
        self.validate()

    def validate(self):
        if self.use_max_classes_per_batch and (self.classes_per_batch != self.max_classes_per_batch):
            raise ValueError("{} of {} classes are used per batch. Try to change batch size.".format(
                self.classes_per_batch, self.max_classes_per_batch))

        if (self.fetch_mode == COUPLED_FETCH_MODE) and (self.classes_per_batch % 2 != 0):
            raise ValueError("{} classes are used per batch. The coupled fetch mode require even class count.".format(
                self.classes_per_batch))

        if (self.fetch_mode == COUPLED_FETCH_MODE) and (self.batch_size % 4 != 0):
            raise ValueError(
                "A batch size = {} is not divisible by 4. The coupled fetch mode can not be used.".format(self.batch_size))

    def get_greatest_devisor(self, number: int, max_divisor: int):
        for divisor in range(max_divisor, 0, -1):
            if number % divisor == 0:
                return divisor

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        class_labels = sorted(random.sample(
            self.labels, self.classes_per_batch))
        class_size = self.batch_size//self.classes_per_batch
        shape = self.output_shape

        x = np.zeros(
            [self.classes_per_batch, class_size, shape[0], shape[1]])
        y = np.zeros([self.classes_per_batch, class_size])

        for class_i, label in enumerate(class_labels):
            x[class_i, :] = self.storage.fetch_subset(
                label, self.start_index, class_size, mode=self.inner_fetch_mode)
            y[class_i, :] = class_i

        if self.fetch_mode == COUPLED_FETCH_MODE:
            x = np.reshape(
                x, [self.classes_per_batch, -1, 2, shape[0], shape[1]])
            x = np.transpose(x, [1, 0, 2, 3, 4])
            y = np.reshape(y, [self.classes_per_batch, -1, 2])
            y = np.transpose(y, [1, 0, 2])

        x = np.reshape(x, [-1, shape[0], shape[1]])
        y = np.reshape(y, [-1])
        return x, y


class UnlabeledStorage(SourcePipe):
    def __init__(self, output_shape: list, storage: Storage, batch_size: int,
                 start_index: int = 0, fetch_mode: str = RANDOM_FETCH_MODE) -> None:
        super().__init__(output_shape)

        self.storage = storage
        self.batch_size = batch_size
        self.start_index = start_index
        self.fetch_mode = fetch_mode
        self.inner_fetch_mode = PATCH_FETCH_MODE if fetch_mode == COUPLED_FETCH_MODE else fetch_mode
        self.validate()

    def validate(self):
        if self.fetch_mode == COUPLED_FETCH_MODE and (self.batch_size % 4 != 0):
            raise ValueError(
                "Batch size = {} is not divisible by 4.".format(self.batch_size))

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        x, indices = self.storage.fetch_subset(
            '', self.start_index, self.batch_size, mode=self.inner_fetch_mode, return_indices=True, patch_size=4)
        return x, np.asarray(indices)


class UnlabeledAugment(TransformPipe):
    def __init__(self, storage: Storage, frontend: FrontendProcessorBase, rate: float, framerate: float) -> None:
        super().__init__()
        self.storage = storage
        self.rate = rate
        self.framerate = framerate
        self.frontend = frontend

    def process(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        aug_size = int(self.rate*len(x))

        y_indices = range(len(y))
        y_aug_indices = random.sample(y_indices, aug_size)
        y_aug = y[y_aug_indices]

        x_aug = self.storage.fetch_subset_from_indices("", y_aug)
        x_aug = augmentation.apply_some_filters(
            x_aug, self.framerate)

        x[y_aug_indices] = self.frontend.process(x_aug)

        return x, y


class Cache(TransformPipe):
    def __init__(self, batch_size: int) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.x_cache = None
        self.y_cache = None
        self.index = 0

    def fill_if_needed(self):
        if (self.x_cache is None) or (self.index + self.batch_size > len(self.x_cache)):
            self.x_cache, self.y_cache = self._previous_pipe.get_batch()
            self.index = 0

    def get_batch(self):
        self.fill_if_needed()

        x = self.x_cache[self.index:self.index + self.batch_size]
        y = self.y_cache[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return x, y

    def process(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, y


class Shuffle(TransformPipe):
    def __init__(self, patch_size: int = 1) -> None:
        super().__init__()
        self.patch_size = patch_size

    def process(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        group_indices = np.arange(len(x)//self.patch_size)
        np.random.shuffle(group_indices)
        indices = expand_patch_indices_to_item_indices(
            group_indices, self.patch_size)

        return x[indices], y[indices]


class Merge(TransformPipe):
    def __init__(self, second_pipe: Pipe) -> None:
        super().__init__()
        self.second_pipe = second_pipe

    def process(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x2, y2 = self.second_pipe.get_batch()

        x = np.concatenate([x, x2], axis=0)
        y = np.concatenate([y, y2], axis=0)
        return x, y
