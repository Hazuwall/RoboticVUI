import numpy as np
import vui.dataset.pipeline as pipeline
import vui.frontend.dsp as dsp
import vui.dataset.augmentation as augmentation
import vui.infrastructure.tests as tests
from vui.frontend.abstract import FrontendProcessorBase


def main():
    unlabeled_augment_test()


def coupled_labeled_source_test():
    storage = pipeline.get_hdf_storage('h', "s_en_SpeechCommands")
    dataset = pipeline.LabeledSource(
        tests.config.frontend_shape, storage, batch_size=12, fetch_mode=pipeline.COUPLED_FETCH_MODE)

    _, y = dataset.get_batch()

    print("Classes: {}".format(y))


def coupled_unlabeled_source_test():
    storage = pipeline.get_hdf_storage('h', "t_mx_Mix")
    dataset = pipeline.UnlabeledSource(
        tests.config.frontend_shape, storage, batch_size=12, fetch_mode=pipeline.COUPLED_FETCH_MODE)

    _, y = dataset.get_batch()

    print("Classes: {}".format(y))


def shuffle_test():
    storage = pipeline.get_hdf_storage('h', "t_mx_Mix")
    dataset = pipeline.UnlabeledSource(
        tests.config.frontend_shape, storage, batch_size=12, fetch_mode=pipeline.COUPLED_FETCH_MODE)
    dataset = pipeline.Shuffle(4)(dataset)

    _, y = dataset.get_batch()

    print("Classes: {}".format(y))


def unlabeled_augment_test():
    class FrontendMock(FrontendProcessorBase):
        def __init__(self):
            super().__init__(tests.config)

        def process_core(self, frames):
            return np.zeros(tests.config.frontend_shape)

    harmonics_storage = pipeline.get_hdf_storage('h', "t_mx_Mix")
    dataset = pipeline.UnlabeledSource(
        tests.config.frontend_shape, harmonics_storage, batch_size=12, fetch_mode=pipeline.COUPLED_FETCH_MODE)
    raw_storage = pipeline.get_hdf_storage('r', "t_mx_Mix")
    dataset = pipeline.UnlabeledSortedAugment(
        raw_storage, FrontendMock(), tests.config.aug_rate, tests.config.framerate)(dataset)

    x, y = dataset.get_batch()

    zero_indices = np.max(x, axis=(1, 2)) == 0
    print("Classes: {}".format(y))
    print("Augmented: {}".format(y[zero_indices]))


if __name__ == "__main__":
    main()
