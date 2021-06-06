import numpy as np
import pyaudio
import time
from playsound import playsound
from typing import Callable, Optional, Tuple
from vui.model.abstract import ClassifierBase
from vui.model.services import FramesToEmbeddingService
from vui.infrastructure.filesystem import FilesystemProvider
import vui.frontend.dsp as dsp


class WordRecognizer():
    def __init__(self, config, f2e_service: FramesToEmbeddingService, classifier: ClassifierBase):
        self.config = config
        self.f2e_service = f2e_service
        self.classifier = classifier

    def recognize(self, frames: np.ndarray) -> Tuple[str, float]:
        weight = 0
        embedding = self.f2e_service.encode(frames)
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
        word_picker = BestWordPicker(
            self.config.silence_word, self.config.unknown_word, buffer_length=4)

        start_time = time.monotonic()

        def callback(in_data, frame_count, time_info, status):
            self.frames_buffer[2 *
                               frame_count:] = self.__convert_to_frames_array(in_data)

            word, weight = self.word_recognizer.recognize(self.frames_buffer)
            word, weight = word_picker.pick(word, weight)
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


class VoiceUserInterfaceStub:
    def __init__(self, config, filesystem: FilesystemProvider, word_recognizer: WordRecognizer, word_handler: Optional[Callable] = None):
        self.config = config
        self.filesystem = filesystem
        self.word_recognizer = word_recognizer
        self.word_handler = word_handler if word_handler is not None else self.__dummy_handler

    def __dummy_handler(word: str, weight: float):
        pass

    def run(self, duration: Optional[float] = None, filter_words=True):
        try:
            print("Start recording...")

            play_count = 0
            while (play_count == 0) or (duration is None):
                file_path = self.filesystem.get_test_recording_path()
                self.play_and_recognize(file_path, filter_words)
                play_count += 1

            print("Stop recording...")
        except KeyboardInterrupt:
            pass

    def play_and_recognize(self, file_path: str, filter_words: bool):
        recognitions_per_sec = 4
        frames_buffer = dsp.read(file_path)
        word_picker = BestWordPicker(
            self.config.silence_word, self.config.unknown_word, buffer_length=recognitions_per_sec)

        playsound(file_path, block=False)
        time.sleep(1 / recognitions_per_sec)

        start_time = time.monotonic()
        fragment_index = 0
        while True:
            word, weight = self.word_recognizer.recognize(frames_buffer)
            word, weight = word_picker.pick(word, weight)
            if (not filter_words) or (weight > self.config.min_word_weight):
                self.word_handler(word, weight)

            if len(frames_buffer) < self.config.framerate // recognitions_per_sec:
                break
            frames_buffer = frames_buffer[self.config.framerate //
                                          recognitions_per_sec:]

            fragment_index += 1
            while time.monotonic() - start_time < (1 / recognitions_per_sec) * fragment_index:
                time.sleep(0.05)


class BestWordPicker:
    def __init__(self, silence_word: str, unknown_word: str, buffer_length: int) -> None:
        self.is_waiting_silence = False
        self.silence_word = silence_word
        self.unknown_word = unknown_word
        self.buffer = []
        self.buffer_length = buffer_length

    def pick(self, word: str, weight: float):
        is_silence = word == self.silence_word

        if self.is_waiting_silence:
            if is_silence:
                self.is_waiting_silence = False
            else:
                return self.unknown_word, 0

        if is_silence:
            if len(self.buffer) == 0:
                return word, weight
        else:
            self.buffer.append((word, weight))
            if len(self.buffer) < self.buffer_length:
                return self.unknown_word, 0

        high_priority_items = [
            item for item in self.buffer if item[0] != self.unknown_word]
        if len(high_priority_items) == 0:
            return self.buffer.pop(0)

        best_item = max(high_priority_items, key=lambda x: x[1])
        self.buffer.clear()
        self.is_waiting_silence = True
        return best_item
