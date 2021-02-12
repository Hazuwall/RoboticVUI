build = 1

# Paths
experiments_dir = "experiments\\"
ref_words_dir = "resources\\ref_words\\"
datasets_dir = "C:\\Datasets\\Audio\\"
checkpoint_prefix = "checkpoint-"

# Recognition
min_word_weight = 0.75
silence_word = "_silence"
unknown_word = "_unknown"

# Training
training_steps = 100
display_interval = 100
checkpoint_interval = 100
verbose = False


def _generate_params():
    import frontend.dsp as dsp

    seg_length = dsp.get_best_segment_length(framerate)
    freq_count = dsp.get_freq_count(seg_length)
    freq_res = dsp.get_freq_resolution(framerate, freq_count)

    return {
        "seg_length": seg_length,
        "freq_count": freq_count,
        "freq_res": freq_res,
        "frontend_shape": (spectrums_per_sec, harmonics_count),
    }


def init(core_config):
    import sys
    current_module = sys.modules[__name__]
    current_module.__dict__.update(core_config.__dict__)

    generated_params = _generate_params()
    current_module.__dict__.update(generated_params)
