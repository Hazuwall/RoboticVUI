import random
import numpy as np
import h5py
import math
from typing import Callable, Optional
from vui.infrastructure.filesystem import FilesystemProvider
import vui.dataset.augmentation as augmentation
from vui.frontend.abstract import FrontendProcessorBase

RANDOM_FETCH_MODE = "RANDOM"
ROW_FETCH_MODE = "ROW"
COUPLED_FETCH_MODE = "COUPLED"


def convert_group_indices_to_item_indices(group_indices: list, group_size: int) -> list:
    indices = []
    for i in group_indices:
        for j in range(group_size):
            indices.append(i*group_size+j)
    return indices


class HdfStorage:
    def __init__(self, path, group_label):
        self.path = path
        self.group_label = group_label + '/' if group_label else ''

    def get_dataset_list(self):
        """Get the list of all saved datasets and groups"""
        items = []
        with h5py.File(self.path, 'a') as f:
            f[self.group_label].visititems(lambda t, _: items.append(t))
        return items

    def delete(self, label: str):
        """Delete dataset or group"""
        with h5py.File(self.path, 'a') as f:
            del f[self.group_label + label]

    def clear_dataset(self, label: str):
        with h5py.File(self.path, 'a') as f:
            dset = f[self.group_label + label]
            mask = np.ones(len(dset), np.bool)
            for i in range(len(dset)):
                if np.max(dset[i]) < 0.0001 or np.isnan(dset[i]).any():
                    mask[i] = 0
            count = np.count_nonzero(mask)
            dset[:count, :] = dset[mask, :]
            shape = list(dset.shape)
            shape[0] = count
            dset.resize(shape)

    def fetch_subset(self, label: str, start: int, size: int, mode=RANDOM_FETCH_MODE, return_indices=False):
        """Fetch a subset from the dataset.
        Returns:
            If mode=RANDOM, a random subset of the given size from the dataset
            skipping specified number of items
            If mode=ROW, a subset of items in a row"""
        with h5py.File(self.path, 'r') as f:
            dset = f[self.group_label + label]
            if mode == ROW_FETCH_MODE:
                X = dset[start:start+size]
                indices = range(start, start+size)
            else:
                if mode == COUPLED_FETCH_MODE:
                    group_size = 4
                    group_indices = sorted(random.sample(
                        range(math.ceil(start/group_size), len(dset)//group_size), size//group_size))
                    indices = convert_group_indices_to_item_indices(
                        group_indices, group_size)
                    X = dset[indices]
                else:
                    indices = sorted(random.sample(
                        range(start, len(dset)), size))
                    X = dset[indices]
            X = np.nan_to_num(X)
            if return_indices:
                return X, indices
            else:
                return X

    def fetch_subset_from_indices(self, label: str, indices):
        with h5py.File(self.path, 'r') as f:
            return np.nan_to_num(f[self.group_label + label][sorted(indices)])


class DatasetPipeline():
    def __init__(self, get_subset_handler: Callable, start_index: int, size: int, fetch_mode: str):
        self._get_subset = get_subset_handler
        self._start_index = start_index
        self._size = size
        self._fetch_mode = fetch_mode

    def get_batch(self):
        return self._get_subset(self._size, self._start_index, self._fetch_mode)


class DatasetPipelineBuilder():
    def __init__(self, config, filesystem: FilesystemProvider, frontend: FrontendProcessorBase):
        self._config = config
        self._filesystem = filesystem
        self._frontend = frontend

        self._last_pipe: Optional[Callable] = None
        self._is_finalized = False

        self._labeled: Optional[bool] = None
        self._dataset_name: Optional[str] = None
        self._start_index = config.validation_size + config.test_size
        self._size = config.batch_size
        self._fetch_mode = RANDOM_FETCH_MODE

    def _create_pipe(self, handler: Callable, previous_pipe: Callable):
        def pipe(size: int, start_index: int, fetch_mode: str):
            return handler(size, start_index, fetch_mode, previous_pipe)
        return pipe

    def attach_handler(self, handler: Callable):
        if self._is_finalized:
            raise ValueError("Pipeline has been finalized!")
        self._last_pipe = self._create_pipe(handler, self._last_pipe)

    def from_labeled_storage(self, dataset_name: Optional[str] = None, labels: Optional[list] = None):
        self._labeled = True
        self._dataset_name = dataset_name
        storage_path = self._filesystem.get_dataset_path('h', dataset_name)
        storage = HdfStorage(storage_path, 'harmonics')
        if labels is None:
            labels = storage.get_dataset_list()

        def storage_handler(size: int, start_index: int, fetch_mode: str, handler: Callable):
            nonlocal labels

            if fetch_mode == COUPLED_FETCH_MODE:
                fetch_mode = RANDOM_FETCH_MODE
                do_split_into_couples = True
                labels = random.sample(labels, 2)
            else:
                do_split_into_couples = False

            group_count = len(labels)
            group_size = size//group_count
            size = group_count*group_size

            shape = self._config.frontend_shape

            x = np.zeros([size, shape[0], shape[1]])
            y = np.zeros([size])

            index = 0
            for group_i in range(group_count):
                x[index:index+group_size] = storage.fetch_subset(
                    labels[group_i], start_index, group_size, mode=fetch_mode)
                y[index:index+group_size] = group_i
                index += group_size

            if do_split_into_couples:
                x = np.reshape(x, [2, -1, 2, shape[0], shape[1]])
                x = np.transpose(x, [1, 0, 2, 3, 4])
                x = np.reshape(x, [-1, shape[0], shape[1]])
                y = np.reshape(y, [2, -1, 2])
                y = np.transpose(y, [1, 0, 2])
                y = np.reshape(y, [-1])
            return x, y

        self.attach_handler(storage_handler)
        return self

    def from_unlabeled_storage(self, dataset_name: Optional[str] = None):
        self._labeled = False
        self._dataset_name = dataset_name
        storage_path = self._filesystem.get_dataset_path('h', dataset_name)
        storage = HdfStorage(storage_path, 'harmonics')

        def storage_handler(size: int, start_index: int, fetch_mode: str, handler: Callable):
            x, indices = storage.fetch_subset(
                '', start_index, size, mode=fetch_mode, return_indices=True)
            return x, np.asarray(indices)

        self.attach_handler(storage_handler)
        return self

    def shuffle(self):
        def shuffle_handler(size: int, start_index: int, fetch_mode: str, handler: Callable):
            shape = self._config.frontend_shape

            x, y = handler(size, start_index, fetch_mode)

            group_size = 4
            group_indices = np.arange(x.shape[0]//group_size)
            np.random.shuffle(group_indices)
            indices = convert_group_indices_to_item_indices(
                group_indices, group_size)

            x = x[indices]
            y = y[indices]

            return x, y

        self.attach_handler(shuffle_handler)
        return self

    def cache(self, cache_size: Optional[int] = None):
        if cache_size is None:
            cache_size = self._config.cache_size
        x_cache = None
        y_cache = None
        cache_index = 0

        def cache_handler(size: int, start_index: int, fetch_mode: str, handler: Callable):
            nonlocal cache_size, cache_index, x_cache, y_cache

            batch_end_index = cache_index + size
            if (x_cache is None) or (batch_end_index >= x_cache.shape[0]):
                x_cache, y_cache = handler(cache_size, start_index, fetch_mode)
                batch_end_index = size

            x = x_cache[cache_index:batch_end_index]
            y = y_cache[cache_index:batch_end_index]
            cache_index = batch_end_index
            return x, y

        self.attach_handler(cache_handler)
        return self

    def augment(self):
        if (self._labeled is None) or (self._labeled is None):
            raise NotImplementedError()

        storage_path = self._filesystem.get_dataset_path(
            'r', self._dataset_name)
        storage = HdfStorage(storage_path, 'raw')

        def augment_handler(size: int, start_index: int, fetch_mode: str, handler: Callable):
            x, y = handler(size, start_index, fetch_mode)

            aug_size = int(self._config.aug_rate*len(x))

            y_indices = range(len(y))
            y_aug_indices = random.sample(y_indices, aug_size)
            y_aug = y[y_aug_indices]

            x_aug = storage.fetch_subset_from_indices("", y_aug)
            x_aug = augmentation.apply_some_filters(
                x_aug, self._config.framerate)

            j = 0
            for i in y_aug_indices:
                x[i] = self._frontend.process(x_aug[j])
                j += 1

            return x, y

        self.attach_handler(augment_handler)
        return self

    def merge(self, pipeline1: DatasetPipeline, pipeline2: DatasetPipeline):
        def merge_handler(size: int, start_index: int, fetch_mode: str, handler: Callable):
            x1, y1 = pipeline1.get_batch()
            x2, y2 = pipeline2.get_batch()
            x = np.concatenate([x1, x2], axis=0)
            y = np.concatenate([y1, y2], axis=0)
            return x, y

        self.attach_handler(merge_handler)
        self._is_finalized = True
        return self

    def with_size(self, size: int):
        self._size = size
        return self

    def with_start_index(self, start_index: int):
        self._start_index = start_index
        return self

    def with_fetch_mode(self, fetch_mode: str):
        self._fetch_mode = fetch_mode
        return self

    def build(self):
        return DatasetPipeline(self._last_pipe, self._start_index, self._size, self._fetch_mode)


class DatasetPipelineFactory():
    def __init__(self, config, filesystem: FilesystemProvider, frontend: FrontendProcessorBase) -> None:
        self._config = config
        self._filesystem = filesystem
        self._frontend = frontend

    def get_builder(self) -> DatasetPipelineBuilder:
        return DatasetPipelineBuilder(self._config, self._filesystem, self._frontend)
