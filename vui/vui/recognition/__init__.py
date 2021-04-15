import numpy as np
import pyaudio
import time
from typing import Callable, Optional, Tuple
import vui.frontend.dsp as dsp
from vui.model.abstract import ClassifierBase
from vui.model.services import FramesToEmbeddingService


class WordRecognizer():
    def __init__(self, config, frames_to_embedding_service: FramesToEmbeddingService, classifier: ClassifierBase):
        self.config = config
        self.frames_to_embedding_service = frames_to_embedding_service
        self.classifier = classifier

    def recognize(self, frames: np.ndarray) -> Tuple[str, float]:
        weight = 0
        embedding = self.frames_to_embedding_service.encode(frames)
        logits = self.classifier.classify(embedding)

        best_fragment_index = np.argmax(np.max(logits, axis=1), axis=0)
        best_fragment_logits = logits[best_fragment_index]

        weight = np.max(best_fragment_logits)
        if weight > self.config.min_word_weight:
            word_index = np.argmax(best_fragment_logits)
            word = self.classifier.get_word(int(word_index))
        else:
            word = self.config.unknown_word
        return word, weight


class VoiceUserInterface:
    def __init__(self, config, word_recognizer: WordRecognizer, word_handler: Optional[Callable] = None):
        self.config = config
        self.word_recognizer = word_recognizer
        self.frames_buffer: Optional[np.ndarray] = None
        self.word_handler = word_handler if word_handler is not None else self.__dummy_handler

    def __dummy_handler(word: str, weight: float):
        pass

    def __convert_to_frames_array(self, data):
        frames = np.fromstring(data, dtype=np.int16)[::self.config.channels]
        high, low = abs(max(frames)), abs(min(frames))
        return frames / max(high, low)

    def __should_continue(self, start_time: float, max_duration: Optional[float]):
        flag = pyaudio.paContinue
        if (max_duration is not None) and (time.monotonic() - start_time > max_duration):
            flag = pyaudio.paAbort
        return flag

    def run(self, duration: Optional[float] = None, filter_words=True):
        buffer_length = self.config.framerate//2
        self.frames_buffer = np.zeros(buffer_length*3)
        audio = pyaudio.PyAudio()

        start_time = time.monotonic()

        def callback(in_data, frame_count, time_info, status):
            self.frames_buffer[2 *
                               frame_count:] = self.__convert_to_frames_array(in_data)

            word, weight = self.word_recognizer.recognize(self.frames_buffer)
            if (not filter_words) or (weight > self.config.min_word_weight):
                self.word_handler(word, weight)

            self.frames_buffer = np.roll(
                self.frames_buffer, -frame_count, axis=0)

            return (in_data, self.__should_continue(start_time, duration))

        stream = audio.open(format=pyaudio.paInt16, channels=self.config.channels,
                            rate=self.config.framerate, input=True,
                            input_device_index=1,
                            frames_per_buffer=buffer_length,
                            stream_callback=callback)

        try:
            print("Start recording...")
            stream.start_stream()
            while stream.is_active():
                time.sleep(0.1)
            print("Stop recording...")
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
