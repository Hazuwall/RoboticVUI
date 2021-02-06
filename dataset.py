import random
import numpy as np
import h5py
from typing import Callable, Optional

from filesystem import FilesystemProvider
import augmentation as aug

RANDOM_FETCH_MODE = "RANDOM"
ROW_FETCH_MODE = "ROW"
COUPLED_FETCH_MODE = "COUPLED"


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
                    indices0 = sorted(random.sample(
                        range(start, len(dset)//4), size//4))
                    indices = []
                    for i in indices0:
                        for j in range(4):
                            indices.append(i*4+j)
                    X = dset[indices]
                else:
                    indices = sorted(random.sample(
                        range(start, len(dset)), size))
                    X = dset[indices]
            if return_indices:
                return X, indices
            else:
                return X

    def fetch_subset_from_indices(self, label, indices):
        with h5py.File(self.path, 'r') as f:
            return f[self.group_label + label][indices]


class DatasetPipeline():
    def __init__(self, get_subset_handler: Callable, start_index: int, size: int, fetch_mode: str):
        self.__get_subset = get_subset_handler
        self.__start_index = start_index
        self.__size = size
        self.__fetch_mode = fetch_mode

    def get_batch(self):
        return self.__get_subset(self.__size, self.__start_index, self.__fetch_mode)


class DatasetPipelineBuilder():
    def __init__(self, config, filesystem: FilesystemProvider):
        self.config = config
        self.filesystem = filesystem
        self.__last_pipe: Optional[Callable] = None

        self.__labeled = False
        self.__start_index = config.validation_size + config.test_size
        self.__size = config.training_batch_size
        self.__fetch_mode = RANDOM_FETCH_MODE

    def __create_pipe(self, handler: Callable, previous_pipe: Callable):
        def pipe(size: int, start_index: int, fetch_mode: str):
            return handler(size, start_index, fetch_mode, previous_pipe)
        return pipe

    def attach_handler(self, handler: Callable):
        self.__last_pipe = self.__create_pipe(handler, self.__last_pipe)

    def from_labeled_storage(self, dataset_name: Optional[str] = None, labels: Optional[list] = None):
        self.__labeled = True
        storage_path = self.filesystem.get_dataset_path('h', dataset_name)
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

            out_shape = self.config.frontend_shape

            x = np.zeros([size, out_shape[0], out_shape[1]])
            y = np.zeros([size])

            index = 0
            for group_i in range(group_count):
                x[index:index+group_size] = storage.fetch_subset(
                    labels[group_i], start_index, group_size, mode=fetch_mode)
                y[index:index+group_size] = group_i
                index += group_size

            if do_split_into_couples:
                x = np.reshape(x, [2, -1, 2, out_shape[0], out_shape[1]])
                x = np.transpose(x, [1, 0, 2, 3, 4])
                x = np.reshape(x, [-1, out_shape[0], out_shape[1]])
                y = np.reshape(y, [2, -1, 2, group_count])
                y = np.transpose(y, [1, 0, 2, 3])
                y = np.reshape(y, [-1, group_count])
            return x, y

        self.attach_handler(storage_handler)
        return self

    def from_unlabeled_storage(self, dataset_name: Optional[str] = None):
        storage_path = self.filesystem.get_dataset_path('h', dataset_name)
        storage = HdfStorage(storage_path, 'harmonics')

        def storage_handler(size: int, start_index: int, fetch_mode: str, handler: Callable):
            out_shape = self.config.preprocess_shape

            x, indices = storage.fetch_subset(
                '', start_index, size, mode=fetch_mode, return_indices=True)

            if fetch_mode == COUPLED_FETCH_MODE:
                x = np.reshape(x, (-1, 4, out_shape[0], out_shape[1]))
                np.random.shuffle(x)
                x = np.reshape(x, (-1, out_shape[0], out_shape[1]))

            return x, np.asarray(indices)

        self.attach_handler(storage_handler)
        return self

    def cache(self, size: int):
        if self.__labeled:
            raise NotImplementedError()
        # TODO

        """
        if self.__embeddings_return:
            out_shape = self.config.embedding_shape
            storage = self.storages[2]
        else:
            out_shape = self.config.preprocess_shape
            storage = self.storages[1]

        self.__cache_index = 0
        self.__cache, indices = storage.fetch_subset(
            '', storage_start_index, self.config.dataset_cache, mode=fetch_mode, return_indices=True)

        if fetch_mode == "COUPLED":
            self.__cache = np.reshape(
                self.__cache, (-1, 4, out_shape[0], out_shape[1]))
            np.random.shuffle(self.__cache)
            self.__cache = np.reshape(
                self.__cache, (-1, out_shape[0], out_shape[1]))
        end_index = self.__cache_index + size
            if (self.__cache is None) or (end_index >= self.__cache.shape[0]):
                self.fill_batch_cache(storage_start_index, fetch_mode)
                end_index = self.__cache_index + size
            x = self.__cache[self.__cache_index:end_index]
            self.__cache_index = end_index"""

        return self

    def augment(self):
        if self.__labeled:
            raise NotImplementedError()

        """data = self.storages[0].fetch_subset_from_indices("", indices)
            data, aug_indices = aug.process(
                data, self.config.framerate, self.config.aug_rate)
            data = sp.complete_preprocess(data)
            if self.__embeddings_return:
                data = sp.encode(data)
            self.__cache[aug_indices] = data"""

        # TODO
        return self

    def merge(self, pipeline1: DatasetPipeline, pipeline2: DatasetPipeline):
        def merge_handler(size: int, start_index: int, fetch_mode: str, handler: Callable):
            x1, y1 = pipeline1.get_batch()
            x2, y2 = pipeline1.get_batch()
            x = np.concatenate([x1, x2], axis=0)
            y = np.concatenate([y1, y2], axis=0)
            return x, y

        self.attach_handler(merge_handler)
        return self

    def with_size(self, size: int):
        self.__size = size
        return self

    def with_start_index(self, start_index: int):
        self.__start_index = start_index
        return self

    def with_fetch_mode(self, fetch_mode: str):
        self.__fetch_mode = fetch_mode
        return self

    def build(self):
        return DatasetPipeline(self.__last_pipe, self.__start_index, self.__size, self.__fetch_mode)
