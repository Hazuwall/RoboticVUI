from termcolor import colored
import h5py
import vui.infrastructure.locator as locator
import vui.dataset.pipeline as pipeline

config = locator.get_config()
filesystem = locator.get_filesystem_provider()
frontend = locator.get_frontend_processor()


def main():
    status("Caching Mix...")
    cache("t_mx_Mix")

    status("Caching Speech Commands...")
    labels = pipeline.get_hdf_storage(
        'r', "s_en_SpeechCommands").get_dataset_list()
    for label in labels:
        cache("s_en_SpeechCommands", label)
        status("✓ - {}".format(label))

    status("Finished!")


def cache(dataset_name: str, label: str = ""):
    raw_dataset_path = filesystem.get_dataset_path('r', dataset_name)
    harmonics_dataset_path = filesystem.get_dataset_path('h', dataset_name)

    with h5py.File(raw_dataset_path, 'r') as f1:
        with h5py.File(harmonics_dataset_path, 'a') as f2:
            dset1 = f1["data/" + label]
            length = len(dset1)
            dset2 = f2.create_dataset("data/" + label, dtype="float32",
                                      shape=(
                                          length, config.frontend_shape[0], config.frontend_shape[1]),
                                      compression="lzf")
            dset2[:] = frontend.process(dset1)


def status(text: str):
    print(colored(text, 'green'))


if __name__ == "__main__":
    main()
