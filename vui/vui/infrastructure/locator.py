import importlib
from typing import Callable, List, Optional
import vui.config as config


class ServiceCollection:
    def singleton(self, factory: Callable):
        def wrapped(*args, **kwargs):
            key = factory.__name__
            if key not in self.__dict__:
                self.__dict__[key] = factory(*args, **kwargs)
            return self.__dict__[key]
        return wrapped


services = ServiceCollection()


@services.singleton
def get_config():
    return config


def get_frontend_processor():
    return get_core_module("frontend").FrontendProcessor(config)


def get_filesystem_provider():
    import vui.infrastructure.filesystem as module
    return module.FilesystemProvider(config)


def get_weights_storage():
    import vui.model.data_access as module
    return module.WeightsStorage(get_filesystem_provider())


@services.singleton
def get_reference_words_dictionary():
    import vui.model.data_access as module
    return module.ReferenceWordsDictionary(config, get_filesystem_provider(), get_frames_to_embedding_service().encode)


def get_dataset_pipeline_factory():
    import vui.dataset.pipeline as module
    return module.DatasetPipelineFactory(config, get_filesystem_provider(), get_frontend_processor())


@services.singleton
def get_acoustic_model(stage_checkpoints: Optional[List[Optional[int]]] = None):
    return get_core_module("model").AcousticModel(config, get_weights_storage(), stage_checkpoints=stage_checkpoints)


@services.singleton
def get_classifier():
    return get_core_module("model").Classifier(config, get_reference_words_dictionary())


def get_frames_to_embedding_service():
    import vui.model.services as module
    return module.FramesToEmbeddingService(config, get_frontend_processor(), get_acoustic_model())


def get_word_recognizer():
    import vui.recognition as module
    return module.WordRecognizer(config, get_frames_to_embedding_service(), get_classifier())


def get_voice_user_interface(word_handler):
    import vui.recognition as module
    return module.VoiceUserInterface(config, get_word_recognizer(), word_handler)


def get_trainer_factory():
    return get_core_module("training").TrainerFactory(config, get_filesystem_provider(), get_acoustic_model(), get_dataset_pipeline_factory())


def get_core_module(module_name: str):
    module_path = get_filesystem_provider().get_core_module_path(module_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_core_config():
    return get_core_module("config")


def create_overriden_scope(override_func: Callable):
    class OverrideScope:
        def __enter__(self):
            self.temp_config_dict = config.__dict__.copy()
            self.temp_services_dict = services.__dict__.copy()
            override_func()

        def __exit__(self, exc_type, exc_val, exc_tb):
            config.__dict__.clear()
            config.__dict__.update(self.temp_config_dict)
            services.__dict__.clear()
            services.__dict__.update(self.temp_services_dict)

    return OverrideScope()
