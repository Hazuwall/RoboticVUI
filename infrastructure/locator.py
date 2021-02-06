import config
from typing import Optional


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


def get_reference_words_dictionary():
    import models.data_access as module
    return module.ReferenceWordsDictionary(config, get_filesystem_provider(), get_frames_to_embedding_service().encode)


def get_dataset_pipeline_builder():
    import dataset.pipeline as module
    return module.DatasetPipelineBuilder(config, get_filesystem_provider())


acoustic_model = None


def get_acoustic_model(weights_step: Optional[int] = None):
    global acoustic_model
    if acoustic_model is None:
        import core.models as module
        acoustic_model = module.AcousticModel(
            config, get_weights_storage(), weights_step=weights_step)
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
    acoustic_model = None
    classifier = None
