import os
import infrastructure.locator as locator

config = locator.get_config()
core_config = locator.get_core_config()

config.init(core_config)

if not config.verbose:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
