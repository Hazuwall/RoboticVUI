import config
from typing import List, Optional


def get_config():
    return config


def get_frontend_processor():
    import core.frontend as module
    return module.FrontendProcessor(config)


def get_filesystem_provider():
    import infrastructure.filesystem as module
    return module.FilesystemProvider(config)


def get_weights_storage():
    import models.data_access as module
    return module.WeightsStorage(get_filesystem_provider())


reference_words_dictionary = None


def get_reference_words_dictionary():
    global reference_words_dictionary
    if reference_words_dictionary is None:
        import models.data_access as module
        reference_words_dictionary = module.ReferenceWordsDictionary(
            config, get_filesystem_provider(), get_frames_to_embedding_service().encode)
    return reference_words_dictionary


def get_dataset_pipeline_builder():
    import dataset.pipeline as module
    return module.DatasetPipelineBuilder(config, get_filesystem_provider())


acoustic_model = None


def get_acoustic_model(stage_checkpoints: Optional[List[Optional[int]]] = None):
    global acoustic_model
    if acoustic_model is None:
        import core.models as module
        acoustic_model = module.AcousticModel(
            config, get_weights_storage(), stage_checkpoints=stage_checkpoints)
    return acoustic_model


classifier = None


def get_classifier():
    global classifier
    if classifier is None:
        import core.models as module
        classifier = module.Classifier(
            config, get_reference_words_dictionary())
    return classifier


def get_frames_to_embedding_service():
    import models.services as module
    return module.FramesToEmbeddingService(config, get_frontend_processor(), get_acoustic_model())


def get_word_recognizer():
    import recognition.services as module
    return module.WordRecognizer(config, get_frames_to_embedding_service(), get_classifier())


def get_voice_user_interface(word_handler):
    import recognition.services as module
    return module.VoiceUserInterface(config, get_word_recognizer(), word_handler)


def reset():
    import importlib
    importlib.reload(config)

    global acoustic_model
    global classifier
    global reference_words_dictionary
    acoustic_model = None
    classifier = None
    reference_words_dictionary = None
