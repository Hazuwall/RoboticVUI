import random
import numpy as np
import h5py
import math

RANDOM_FETCH_MODE = "RANDOM"
ROW_FETCH_MODE = "ROW"
COUPLED_FETCH_MODE = "COUPLED"


def expand_group_indices_to_item_indices(group_indices: list, group_size: int) -> list:
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
                    indices = expand_group_indices_to_item_indices(
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
