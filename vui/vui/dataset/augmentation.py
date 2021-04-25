import random
from typing import Optional, Sequence
import numpy as np
import librosa


def shift(frames: np.ndarray, max_n: int, axis: Optional[int] = None) -> np.ndarray:
    n1 = random.randint(1, max_n)
    n2 = random.randint(-max_n, -1)
    return np.concatenate((frames, np.roll(frames, n1, axis=axis), np.roll(frames, n2, axis=axis)))


def change_pitch(frames: np.ndarray, framerate: int) -> np.ndarray:
    frames_out = np.zeros(frames.shape)

    for i in range(len(frames)):
        pitch_change = 3*np.random.uniform() - 2
        frames_out[i] = librosa.effects.pitch_shift(
            frames[i], framerate, n_steps=pitch_change)
    return frames_out


def add_noise(frames: np.ndarray) -> np.ndarray:
    noise_amp = 0.02 * \
        np.random.uniform(
            size=(frames.shape[0], 1))*np.amax(frames, axis=1, keepdims=True)
    return frames + noise_amp * np.random.normal(size=frames.shape)


def sample_with_rate(x: Sequence, rate: float) -> list:
    return random.sample(x, int(rate*len(x)))


def apply_some_filters(frames: np.ndarray, framerate: int) -> np.ndarray:
    frames = add_noise(frames)

    indices = range(len(frames))
    pitch_indices = sample_with_rate(indices, 0.5)
    frames[pitch_indices] = change_pitch(frames[pitch_indices], framerate)

    return frames
