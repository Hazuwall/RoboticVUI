import os
from typing import Callable, Optional
import vui.infrastructure.locator as locator


def generate_params(config):
    import vui.frontend.dsp as dsp

    seg_length = dsp.get_best_segment_length(config.framerate)
    freq_count = dsp.get_freq_count(seg_length)
    freq_res = dsp.get_freq_resolution(config.framerate, freq_count)

    return {
        "seg_length": seg_length,
        "freq_count": freq_count,
        "freq_res": freq_res,
        "frontend_shape": (config.spectrums_per_sec, config.harmonics_count),
    }


config = locator.get_config()
core_config = locator.get_core_config()
config.__dict__.update(core_config.__dict__)

generated_params = generate_params(config)
config.__dict__.update(generated_params)


if not config.verbose:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run(word_handler: Callable, duration: Optional[float] = None):
    vui = locator.get_voice_user_interface(word_handler)
    vui.run(duration)


def reset():
    import importlib
    import vui
    importlib.reload(vui)
