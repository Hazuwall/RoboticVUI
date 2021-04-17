from abc import ABC, abstractmethod, abstractproperty
import random
from typing import Optional, Tuple
from vui.frontend.abstract import FrontendProcessorBase
import numpy as np
from vui.dataset.storage import HdfStorage, COUPLED_FETCH_MODE, RANDOM_FETCH_MODE
from vui.dataset import augmentation
import vui.infrastructure.locator as locator


def expand_group_indices_to_item_indices(group_indices: list, group_size: int) -> list:
    indices = []
    for i in group_indices:
        for j in range(group_size):
            indices.append(i*group_size+j)
    return indices


def get_hdf_storage(type_letter: str, label: str) -> HdfStorage:
    mapping = {
        "r": "raw",
        "h": "harmonics",
        "e": "embeddings"
    }

    path = locator.get_filesystem_provider().get_dataset_path(type_letter, label)
    return HdfStorage(path, mapping[type_letter])


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
    def __init__(self, output_shape: list, storage: HdfStorage, batch_size: int, start_index: int = 0,
                 fetch_mode: str = RANDOM_FETCH_MODE, labels: Optional[list] = None, use_all_classes_per_batch: bool = False) -> None:
        super().__init__(output_shape)

        labels = storage.get_dataset_list() if labels is None else labels
        if fetch_mode == COUPLED_FETCH_MODE:
            self.inner_fetch_mode = RANDOM_FETCH_MODE
            self.classes_per_batch = self.get_greatest_devisor(
                number=batch_size//2, max_divisor=len(labels))
        else:
            self.inner_fetch_mode = fetch_mode
            self.classes_per_batch = self.get_greatest_devisor(
                number=batch_size, max_divisor=len(labels))

        self.storage = storage
        self.labels = labels
        self.batch_size = batch_size
        self.start_index = start_index
        self.fetch_mode = fetch_mode
        self.use_all_classes_per_batch = use_all_classes_per_batch
        self.validate()

    def validate(self):
        if self.use_all_classes_per_batch and (self.classes_per_batch != len(self.labels)):
            raise ValueError("{} of {} classes are used per batch. Try to change batch size.".format(
                self.classes_per_batch, len(self.labels)))

        if (self.fetch_mode == COUPLED_FETCH_MODE) and (self.classes_per_batch < 2):
            raise ValueError("{} classes are used per batch. The coupled fetch mode require >= 2 classes. Try to change batch size.".format(
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
    def __init__(self, output_shape: list, storage: HdfStorage, batch_size: int,
                 start_index: int = 0, fetch_mode: str = RANDOM_FETCH_MODE) -> None:
        super().__init__(output_shape)

        self.storage = storage
        self.batch_size = batch_size
        self.start_index = start_index
        self.fetch_mode = fetch_mode
        self.validate()

    def validate(self):
        if self.fetch_mode == COUPLED_FETCH_MODE and (self.batch_size % 4 != 0):
            raise ValueError(
                "Batch size = {} is not divisible by 4.".format(self.batch_size))

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        x, indices = self.storage.fetch_subset(
            '', self.start_index, self.batch_size, mode=self.fetch_mode, return_indices=True)
        return x, np.asarray(indices)


class UnlabeledAugment(TransformPipe):
    def __init__(self, storage: HdfStorage, frontend: FrontendProcessorBase, rate: float, framerate: float) -> None:
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

        j = 0
        for i in y_aug_indices:
            x[i] = self.frontend.process(x_aug[j])
            j += 1

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
    def __init__(self, group_size: int = 1) -> None:
        super().__init__()
        self.group_size = group_size

    def process(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        group_indices = np.arange(len(x)//self.group_size)
        np.random.shuffle(group_indices)
        indices = expand_group_indices_to_item_indices(
            group_indices, self.group_size)

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
