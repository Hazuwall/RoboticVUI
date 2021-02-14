import importlib
from typing import Callable, List, Optional
import config


def get_config():
    return config


def get_frontend_processor():
    return get_core_module("frontend").FrontendProcessor(config)


def get_filesystem_provider():
    import infrastructure.filesystem as module
    return module.FilesystemProvider(config)


def get_weights_storage():
    import model.data_access as module
    return module.WeightsStorage(get_filesystem_provider())


reference_words_dictionary = None


def get_reference_words_dictionary():
    global reference_words_dictionary
    if reference_words_dictionary is None:
        import model.data_access as module
        reference_words_dictionary = module.ReferenceWordsDictionary(
            config, get_filesystem_provider(), get_frames_to_embedding_service().encode)
    return reference_words_dictionary


def get_dataset_pipeline_factory():
    import dataset.pipeline as module
    return module.DatasetPipelineFactory(config, get_filesystem_provider())


acoustic_model = None


def get_acoustic_model(stage_checkpoints: Optional[List[Optional[int]]] = None):
    global acoustic_model
    if acoustic_model is None:
        acoustic_model = get_core_module("model").AcousticModel(
            config, get_weights_storage(), stage_checkpoints=stage_checkpoints)
    return acoustic_model


classifier = None


def get_classifier():
    global classifier
    if classifier is None:
        classifier = get_core_module("model").Classifier(
            config, get_reference_words_dictionary())
    return classifier


def get_frames_to_embedding_service():
    import model.services as module
    return module.FramesToEmbeddingService(config, get_frontend_processor(), get_acoustic_model())


def get_word_recognizer():
    import recognition.services as module
    return module.WordRecognizer(config, get_frames_to_embedding_service(), get_classifier())


def get_voice_user_interface(word_handler):
    import recognition.services as module
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


def reset():
    global acoustic_model
    global classifier
    global reference_words_dictionary
    acoustic_model = None
    classifier = None
    reference_words_dictionary = None

    importlib.reload(config)
    import __init__
    importlib.reload(__init__)


def create_overrided_scope(override_func: Callable):
    class OverrideScope:
        def __enter__(self):
            reset()
            override_func()

        def __exit__(self, exc_type, exc_val, exc_tb):
            reset()
    return OverrideScope()
