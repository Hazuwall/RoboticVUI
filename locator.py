import config
from typing import Optional


def get_config():
    return config


def get_frontend_processor():
    import frontend
    return frontend.FrontendProcessor(config)


def get_filesystem_provider():
    import filesystem
    return filesystem.FilesystemProvider(config)


def get_weights_storage():
    import models_data
    return models_data.WeightsStorage(get_filesystem_provider())


def get_reference_words_dictionary():
    import models_data
    return models_data.ReferenceWordsDictionary(config, get_filesystem_provider(), get_frames_to_embedding_service().encode)


def get_dataset_provider(label: Optional[str] = None, embeddings_return=False, do_augment=False):
    import dataset
    return dataset.DatasetProvider(config, get_filesystem_provider(), label=label, embeddings_return=embeddings_return, do_augment=do_augment)


def get_dataset_pipeline_builder():
    import dataset
    return dataset.DatasetPipelineBuilder(config, get_filesystem_provider())


acoustic_model = None


def get_acoustic_model(weights_step: Optional[int] = None):
    global acoustic_model
    if acoustic_model is None:
        import models
        acoustic_model = models.AcousticModel(
            config, get_weights_storage(), weights_step=weights_step)
    return acoustic_model


classifier = None


def get_classifier():
    global classifier
    if classifier is None:
        import models
        classifier = models.Classifier(
            config, get_reference_words_dictionary())
    return classifier


def get_frames_to_embedding_service():
    import services
    return services.FramesToEmbeddingService(config, get_frontend_processor(), get_acoustic_model())


def get_word_recognizer():
    import services
    return services.WordRecognizer(config, get_frames_to_embedding_service(), get_classifier())


def get_voice_user_interface(word_handler):
    import services
    return services.VoiceUserInterface(config, get_word_recognizer(), word_handler)


def reset():
    import importlib
    importlib.reload(config)

    global acoustic_model
    global classifier
    acoustic_model = None
    classifier = None
