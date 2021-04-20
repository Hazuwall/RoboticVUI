import os
import random
import numpy as np
import h5py
import math
from abc import ABC, abstractmethod
import vui.frontend.dsp as dsp

RANDOM_FETCH_MODE = "RANDOM"
ROW_FETCH_MODE = "ROW"
PATCH_FETCH_MODE = "PATCH"
COUPLED_FETCH_MODE = "COUPLED"


class Storage(ABC):
    @abstractmethod
    def get_dataset_list(self) -> list:
        """Get the list of all saved datasets and groups"""
        pass

    @abstractmethod
    def fetch_subset(self, label: str, start: int, size: int, mode: str = RANDOM_FETCH_MODE,
                     return_indices: bool = False, patch_size: int = 2):
        """Fetch a subset from the dataset.
        Returns:
            If mode=RANDOM_FETCH_MODE, a random subset of the given size from the dataset
            skipping specified number of items
            If mode=ROW_FETCH_MODE, a subset of items in a row
            If mode=PATCH_FETCH_MODE, a subset of patches."""
        pass

    @abstractmethod
    def fetch_subset_from_indices(self, label: str, indices: list):
        pass

    def _expand_patch_indices_to_item_indices(self, patch_indices: list, patch_size: int) -> list:
        indices = []
        for i in patch_indices:
            for j in range(patch_size):
                indices.append(i*patch_size+j)
        return indices

    def _get_sorted_indices(self, start: int, size: int, dataset_length: int, mode: str, patch_size: int) -> list:
        if mode == ROW_FETCH_MODE:
            return range(start, start+size)

        elif mode == PATCH_FETCH_MODE:
            all_allowed_patch_indices = range(
                math.ceil(start/patch_size), dataset_length//patch_size)
            patch_indices = random.sample(
                all_allowed_patch_indices, size//patch_size)
            patch_indices = sorted(patch_indices)
            return self._expand_patch_indices_to_item_indices(
                patch_indices, patch_size)

        elif mode == RANDOM_FETCH_MODE:
            all_allowed_indices = range(start, dataset_length)
            indices = random.sample(all_allowed_indices, size)
            return sorted(indices)

        else:
            raise NotImplementedError()


class HdfStorage(Storage):
    def __init__(self, path, group_label="data") -> None:
        self.path = path
        self.group_label = group_label + '/' if group_label else ''

    def get_dataset_list(self) -> str:
        items = []
        with h5py.File(self.path, 'r') as f:
            f[self.group_label].visititems(lambda t, _: items.append(t))
        return items

    def delete(self, label: str) -> None:
        """Delete dataset or group"""
        with h5py.File(self.path, 'a') as f:
            del f[self.group_label + label]

    def clear_dataset(self, label: str) -> None:
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

    def fetch_subset(self, label: str, start: int, size: int, mode: str = RANDOM_FETCH_MODE,
                     return_indices: bool = False, patch_size: int = 2):
        with h5py.File(self.path, 'r') as f:
            dset = f[self.group_label + label]
            indices = self._get_sorted_indices(
                start, size, len(dset), mode, patch_size)
            X = dset[indices]

            if return_indices:
                return X, indices
            else:
                return X

    def fetch_subset_from_indices(self, label: str, indices: list):
        with h5py.File(self.path, 'r') as f:
            return f[self.group_label + label][sorted(indices)]


class WavFolderStorage(Storage):
    def __init__(self, path: str) -> None:
        self.path = path
        self.cache = {}
        self.load()

    def load(self) -> None:
        labels = os.listdir(self.path)
        self.cache = {}
        for label in labels:
            label_dir = os.path.join(self.path, label)
            frames_list = []

            for filename in os.listdir(label_dir):
                file_path = os.path.join(label_dir, filename)
                frames_list.append(dsp.read(file_path))

            max_length = max([len(frames) for frames in frames_list])
            frames_array = np.zeros([len(frames_list), max_length])
            for i, frames in enumerate(frames_list):
                frames_array[i, :len(frames)] = frames

            self.cache[label] = frames_array

    def get_dataset_list(self) -> list:
        return self.cache.keys()

    def fetch_subset(self, label: str, start: int, size: int, mode: str = RANDOM_FETCH_MODE,
                     return_indices: bool = False, patch_size: int = 2):
        dset = self.cache[label]
        indices = self._get_sorted_indices(
            start, size, len(dset), mode, patch_size)
        X = dset[indices]

        if return_indices:
            return X, indices
        else:
            return X

    def fetch_subset_from_indices(self, label: str, indices: list):
        return self.cache[label][indices]
