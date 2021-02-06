import os
import frontend.dsp as dsp
from core.config import *

# Paths
experiments_dir = "experiments\\"
ref_words_dir = "resources\\ref_words\\"
core_dir = "core\\"
datasets_dir = "C:\\Datasets\\Audio\\"
checkpoint_prefix = "checkpoint-"

# Recognition
min_word_weight = 0.75
silence_word = "_silence"
unknown_word = "_unknown"

# Training
verbose = False

if not verbose:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Frontend
seg_length = dsp.get_best_segment_length(framerate)
freq_count = dsp.get_freq_count(seg_length)
freq_res = dsp.get_freq_resolution(framerate, freq_count)
frontend_shape = (spectrums_per_sec, harmonics_count)
