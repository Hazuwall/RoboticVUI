import random
import numpy as np
import librosa

def shift(x, max_n, axis=None):
    n1 = random.randint(1,max_n)
    n2 = random.randint(-max_n,-1)
    return np.concatenate((x, np.roll(x, n1, axis=axis), np.roll(x, n2, axis=axis)))

def change_pitch(frames, framerate):
    for i in range(len(frames)):
        pitch_change = 3*np.random.uniform() - 2
        frames[i] = librosa.effects.pitch_shift(frames[i], framerate, n_steps=pitch_change)

def add_noise(frames):
    noise_amp = 0.01*np.random.uniform(size=(frames.shape[0], 1))*np.amax(frames, axis=1, keepdims=True)
    frames += noise_amp * np.random.normal(size=frames.shape)

def process(frames, framerate, aug_rate):
    indices = random.sample(range(0, len(frames)), int(aug_rate*len(frames)))
    
    X = frames[indices]
    add_noise(X)

    pitch_group_size = len(X)//2
    pitch_indices = random.sample(range(0, len(X)), pitch_group_size)
    temp = X[pitch_indices]
    change_pitch(temp, framerate)
    X[pitch_indices] = temp
    return X, indices