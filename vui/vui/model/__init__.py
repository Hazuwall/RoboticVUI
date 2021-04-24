import tensorflow
import vui.infrastructure.locator

if not vui.infrastructure.locator.get_config().verbose:
    tensorflow.get_logger().setLevel('ERROR')
