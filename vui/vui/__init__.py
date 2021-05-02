import os
from typing import Callable, Optional
import vui.infrastructure.locator as locator


def with_core_params(config):
    core_config = locator.get_core_config()
    # core config module transforms to extended common config module
    core_config.__dict__.update(config.__dict__)
    # original common config module extends
    config.__dict__.update(core_config.__dict__)
    return config


def with_generated_params(config):
    import vui.frontend.dsp as dsp

    config.seg_length = dsp.get_best_segment_length(config.framerate)
    config.freq_count = dsp.get_freq_count(config.seg_length)
    config.freq_res = dsp.get_freq_resolution(
        config.framerate, config.freq_count)
    config.frontend_shape = (config.spectrums_per_sec, config.harmonics_count)

    return config


config = locator.get_config()
config = with_core_params(config)
config = with_generated_params(config)

if not config.verbose:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Build {}".format(config.build))


def run(word_handler: Callable, duration: Optional[float] = None):
    vui = locator.get_voice_user_interface(word_handler)
    vui.run(duration)
