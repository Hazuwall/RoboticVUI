import importlib
from typing import Callable
import vui.config as config


class Lazy():
    def __init__(self, factory: Callable):
        self._instance = None
        self._factory = factory

    @property
    def instance(self):
        if self._instance is None:
            self._instance = self._factory()
        return self._instance


class ServiceCollection:
    def transient(self, factory: Callable) -> Callable:
        key = factory.__name__
        self.register_transient(key, factory)
        return self.get_factory_proxy(key)

    def singleton(self, factory: Callable) -> Callable:
        key = factory.__name__
        self.register_singleton(factory.__name__, factory)
        return self.get_factory_proxy(key)

    def register_transient(self, key: str, factory: Callable) -> None:
        self.__dict__[key] = factory

    def register_singleton(self, key: str, factory: Callable) -> None:
        lazy = Lazy(factory)
        def singleton_factory(*args, **kwargs): return lazy.instance
        self.__dict__[key] = singleton_factory

    def get_factory_proxy(self, key: str) -> Callable:
        def factory_proxy(*args, **kwargs):
            return self.__dict__[key](*args, **kwargs)
        return factory_proxy


services = ServiceCollection()


@services.singleton
def get_config():
    return config


@services.transient
def get_frontend_processor():
    return get_core_module("frontend").FrontendProcessor(config)


@services.transient
def get_filesystem_provider():
    import vui.infrastructure.filesystem as module
    return module.FilesystemProvider(config)


@services.transient
def get_weights_storage():
    import vui.model.data_access as module
    return module.WeightsStorage(get_filesystem_provider())


@services.singleton
def get_reference_words_dictionary():
    import vui.model.data_access as module
    return module.ReferenceWordsDictionary(config, get_filesystem_provider(), get_frames_to_embedding_service().encode)


@services.transient
def get_model_info_saver():
    import vui.model.data_access as module
    return module.ModelInfoSaver(get_filesystem_provider())


@services.transient
def get_evaluator():
    import vui.model.metrics as module
    return module.Evaluator(config, get_filesystem_provider(), get_reference_words_dictionary(),
                            get_frames_to_embedding_service(), get_word_recognizer())


@services.singleton
def get_acoustic_model():
    return get_core_module("model").AcousticModel(config, get_weights_storage())


@services.singleton
def get_classifier():
    return get_core_module("model").Classifier(config, get_reference_words_dictionary())


@services.transient
def get_frames_to_embedding_service():
    import vui.model.services as module
    return module.FramesToEmbeddingService(config, get_frontend_processor(), get_acoustic_model())


@services.transient
def get_word_recognizer():
    import vui.recognition as module
    return module.WordRecognizer(config, get_frames_to_embedding_service(), get_classifier())


@services.transient
def get_voice_user_interface(word_handler):
    import vui.recognition as module
    if config.use_test_recordings:
        return module.VoiceUserInterfaceStub(config, get_filesystem_provider(), get_word_recognizer(), word_handler)
    else:
        return module.VoiceUserInterface(config, get_word_recognizer(), word_handler)


@services.transient
def get_trainer_factory():
    return get_core_module("training").TrainerFactory()


@services.transient
def get_core_module(module_name: str):
    module_path = get_filesystem_provider().get_core_module_path(module_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@services.singleton
def get_core_config():
    return get_core_module("config")


def create_isolated_scope():
    class IsolatedScope:
        def __enter__(self):
            self.temp_config_dict = config.__dict__.copy()
            self.temp_services_dict = services.__dict__.copy()

        def __exit__(self, exc_type, exc_val, exc_tb):
            config.__dict__.clear()
            config.__dict__.update(self.temp_config_dict)
            services.__dict__.clear()
            services.__dict__.update(self.temp_services_dict)

    return IsolatedScope()
