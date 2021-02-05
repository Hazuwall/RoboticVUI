import numpy as np
import pyaudio
import time
from typing import Callable
import dsp_utils
from models import AcousticModel, Classifier
from models_data import ReferenceWordsDictionary
from frontend import FrontendProcessor


class FramesToEmbeddingService:
    def __init__(self, config, frontend: FrontendProcessor, acoustic_model: AcousticModel):
        self.config = config
        self.frontend = frontend
        self.acoustic_model = acoustic_model

    def encode(self, frames: np.ndarray, indices=None):
        if indices is None:
            harmonics = np.expand_dims(self.frontend.process(frames), axis=0)
        else:
            harmonics = np.zeros(
                shape=(len(indices), self.config.preprocess_shape[0], self.config.preprocess_shape[1]))
            for i in range(len(indices)):
                start = indices[i]*self.config.seg_length//2
                end = start + self.config.framerate
                if end > len(frames):
                    data = frames[start:]
                    data = np.pad(data, (0, end-len(frames)))
                else:
                    data = frames[start:end]
                harmonics[i] = self.frontend.process(data)

        return self.acoustic_model.encode(harmonics)


class WordRecognizer():
    def __init__(self, config, frames_to_embedding_service: FramesToEmbeddingService, classifier: Classifier):
        self.config = config
        self.frames_to_embedding_service = frames_to_embedding_service
        self.classifier = classifier

    def recognize(self, frames):
        word = "_silence"
        weight = 0
        sg = dsp_utils.make_spectrogram(frames, self.config.seg_length)
        indices = dsp_utils.detect_words(sg)
        if len(indices) > 0:
            embedding = self.frames_to_embedding_service.encode(
                frames, indices)
            logits = self.classifier.classify(embedding)

            best_fragment_index = np.argmax(np.max(logits, axis=1), axis=0)
            best_fragment_logits = logits[best_fragment_index]

            weight = np.max(best_fragment_logits)
            if weight > 0.75:
                word_index = np.argmax(best_fragment_logits)
                word = self.classifier.get_word(int(word_index))
            else:
                word = "_unknown"
        return word, weight


class VoiceUserInterface:
    def __init__(self, config, word_recognizer: WordRecognizer, word_handler: Callable):
        self.config = config
        self.word_recognizer = word_recognizer
        self.frames_buffer = None
        self.word_handler = word_handler

    def run(self):
        buffer_length = self.config.framerate//2
        self.frames_buffer = np.zeros(buffer_length*3)
        audio = pyaudio.PyAudio()

        def callback(in_data, frame_count, time_info, status):
            frames = np.fromstring(in_data, dtype=np.int16)[::2]
            high, low = abs(max(frames)), abs(min(frames))
            self.frames_buffer[2*frame_count:] = frames / max(high, low)

            word, weight = self.word_recognizer.recognize(self.frames_buffer)
            self.word_handler(word, weight)

            self.frames_buffer = np.roll(
                self.frames_buffer, -frame_count, axis=0)
            #flag = pyaudio.paContinue if step < duration*2 else pyaudio.paAbort
            return (in_data, pyaudio.paContinue)

        stream = audio.open(format=pyaudio.paInt16, channels=2,
                            rate=self.config.framerate, input=True,
                            input_device_index=1,
                            frames_per_buffer=buffer_length,
                            stream_callback=callback)

        print("Start recording...")
        stream.start_stream()
        while stream.is_active():
            time.sleep(0.1)
        print("Stop recording...")

        stream.stop_stream()
        stream.close()
        audio.terminate()
