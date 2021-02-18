import tensorflow as tf
from keras.utils.layer_utils import count_params
from keras.utils.vis_utils import model_to_dot
from keras_flops import get_flops
import IPython.core.magics.namespace  # not used here, but need for tensorflow


class StructureInfo:
    def __init__(self):
        self._variable_count = 0
        self._weights_size = 0
        self._flops = 0
        self._svg = None

    @property
    def variable_count(self) -> int:
        return self._variable_count

    @variable_count.setter
    def variable_count(self, value: int):
        self._variable_count = value

    @property
    def weights_size(self) -> int:
        return self._weights_size

    @weights_size.setter
    def weights_size(self, value: int):
        self._weights_size = value

    @property
    def flops(self) -> int:
        return self._flops

    @flops.setter
    def flops(self, value: int):
        self._flops = value

    @property
    def svg(self) -> bytearray:
        return self._svg

    @svg.setter
    def svg(self, value: bytearray):
        self._svg = value


def get_structure_info(model: tf.keras.Model) -> StructureInfo:
    trainable_count = count_params(model.trainable_weights)
    non_trainable_count = count_params(model.non_trainable_weights)

    info = StructureInfo()
    info.variable_count = trainable_count
    info.weights_size = (trainable_count + non_trainable_count) * 4
    info.svg = model_to_dot(model, show_layer_names=True,
                            show_shapes=True, dpi=None).create(prog='dot', format='svg')
    info.flops = get_flops(model, batch_size=1)
    return info
