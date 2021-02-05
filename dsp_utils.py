import numpy as np
import scipy.signal
import math
import soundfile as sf


def get_framerate(filepath):
    """Get a framerate of the given audiofile. MP3 is not supported."""
    with sf.SoundFile(filepath, mode='r') as f:
        return f.samplerate


def read(filepath):
    """Get frames of the given audiofile first channel. MP3 is not supported."""
    return sf.read(filepath, dtype='float32', always_2d=True)[0][:, 0]


def write(filepath, frames, framerate):
    """Save frames to the audiofile of the arbitrary format. MP3 is not supported."""
    sf.write(filepath, frames, framerate)


def make_spectrogram(frames, seg_length, step_count=None, win_flag=True, keep_phase=False):
    if step_count is None:
        step = seg_length//2
        step_count = len(frames)//step
    else:
        step = math.ceil(len(frames)/step_count)
    if win_flag:
        window = np.hamming(seg_length)

    freq_count = (seg_length//2) + 1
    start, end = 0, seg_length
    sg = np.zeros([step_count, freq_count], dtype=np.complex_)

    i = 0
    while end <= len(frames):
        segment = frames[start: end]
        if win_flag:
            segment *= window
        sg[i, :] = np.fft.rfft(segment)
        start += step
        end += step
        i += 1
    if not keep_phase:
        return np.abs(sg)
    return sg


def restore_frames(spectrogram, seg_length=None, frame_count=None, win_flag=True):
    step_count = len(spectrogram)
    if (seg_length is None) and (frame_count is None):
        raise ValueError
    if frame_count is None:
        step = seg_length//2
        frame_count = step*step_count
    else:
        step = frame_count//len(spectrogram)
    if win_flag:
        window = 1 / np.hamming((spectrogram.shape[1] - 1)*2)

    start, end = 0, step
    frames = np.zeros(frame_count)
    for i in range(step_count):
        segment = np.fft.irfft(spectrogram[i, :])
        if win_flag:
            segment *= window
        frames[start:end] = segment[:step]
        start += step
        end += step
    return frames


def blur(spectrogram, size):
    return scipy.ndimage.medfilt(spectrogram, [1, size])


def get_freq_resolution(framerate, freq_count):
    return framerate / 2 / (freq_count - 1)


def get_freq_count(seg_length):
    return (seg_length//2) + 1


def get_best_segment_length(framerate):
    return int(2 ** round(np.log2(framerate / 32)))


def fft_autocorr(x):
    xp = np.fft.ifftshift((x - np.average(x))/np.std(x))
    n, = xp.shape
    xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
    f = np.fft.fft(xp)
    p = np.absolute(f)**2
    pi = np.fft.ifft(p)
    return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)


def get_fund_freq(frames, framerate, step_count, min_freq=80, max_freq=300):
    max_freq_offset = framerate // max_freq
    sample_length = framerate // min_freq
    step = math.ceil(len(frames)/step_count)
    funds = np.zeros((step_count))
    for i in range(step_count):
        start_index = i*step
        corr = fft_autocorr(frames[start_index:start_index + sample_length])
        max_i = np.argmax(corr[max_freq_offset:], axis=0) + max_freq_offset
        funds[i] = framerate / max_i
    return funds


def get_harmonics(fund_freqs, freq_res, count, spectrogram=None):
    harmonics = np.broadcast_to(
        fund_freqs, (count, len(fund_freqs))).T * np.arange(1, count+1)
    if spectrogram is None:
        return harmonics
    step_count = spectrogram.shape[0]
    freq_count = spectrogram.shape[1]
    harmonic_indices = np.rint(np.ndarray.flatten(
        harmonics) / freq_res).astype(np.int)
    harmonic_indices = np.minimum(harmonic_indices, freq_count-1)
    step_indices = np.repeat(np.arange(step_count, dtype=np.int), count)
    amps = np.reshape(spectrogram[step_indices, harmonic_indices], [-1, count])
    return harmonics, amps


def restore_spectrogram(harmonics_amps, freq_count, freq_res, fund_freq=200):
    harmonics_count = harmonics_amps.shape[1]
    step_count = harmonics_amps.shape[0]
    harmonics = np.tile(np.arange(1, harmonics_count+1),
                        step_count) * fund_freq
    harmonic_indices = np.rint(harmonics / freq_res).astype(np.int)
    harmonic_indices = np.minimum(harmonic_indices, freq_count-1)
    step_indices = np.repeat(
        np.arange(step_count, dtype=np.int), harmonics_count)

    spectrogram = np.zeros((step_count, freq_count))
    spectrogram[step_indices, harmonic_indices] = np.ndarray.flatten(
        harmonics_amps)
    return spectrogram


def norm_harmonics(spectrogram, harmonics_amps):
    amps = spectrogram
    std = np.std(amps, axis=1, keepdims=True)
    mean = np.mean(amps, axis=1, keepdims=True)

    signal_i = amps > (std**2)+mean
    #signal = amps[np.nonzero(signal_i)]
    signal_mean = np.mean(amps*signal_i, axis=1, keepdims=True)
    signal_std = np.std(amps*signal_i, axis=1, keepdims=True)

    signal_steps_i = signal_mean > 0.000005
    fund_amps = harmonics_amps[:, np.newaxis, 0]
    harmonics_amps *= (signal_steps_i)/(fund_amps+0.00001)*2/3
    harmonics_amps = np.clip(harmonics_amps, 0, 1) * signal_std
    return harmonics_amps


def detect_words(sg) -> np.ndarray:
    dev = np.std(sg, axis=1)**2
    #diff_kernel = np.array([-1,0,1])
    #d_dev = scipy.signal.convolve(diff_kernel,k,mode="same")
    gaussian = scipy.signal.gaussian(16, 6)
    #d_dev = scipy.signal.convolve(dev,gaussian,mode="same")
    #mean_kernel = np.ones((8))*0.125
    dev = scipy.signal.convolve(dev, gaussian, mode="valid")[::4]
    indices = np.where(dev > 40)[0]*4
    return indices
