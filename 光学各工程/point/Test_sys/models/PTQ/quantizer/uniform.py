import torch
import torch.nn as nn

from .base import BaseQuantizer


class UniformQuantizer(BaseQuantizer):
    def __init__(
            self,
            bit_type,
            observer,
            module_type):
        super(UniformQuantizer, self).__init__(
            bit_type,
            observer,
            module_type)
        self.scale = None
        self.zero_point = None
        self.dequant_scale = None

    def update_quantization_params(self, *args, **kwargs):
        self.scale, self.zero_point = self.observer.get_quantization_params(
            *args, **kwargs)
        if 'weight' in self.module_type and 'before_scale' in kwargs.keys():
            self.dequant_scale = self.scale * kwargs['before_scale']


    def quant(self, inputs, alpha=None, beta=None):
        scale = self.scale if alpha is None else self.scale * alpha
        zero_point = self.zero_point if beta is None else self.zero_point + beta
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = inputs / scale + zero_point
        outputs = outputs.round().clamp(
            self.bit_type.lower_bound, self.bit_type.upper_bound)
        return outputs

    def dequantize(self, inputs, alpha=None, beta=None):
        scale = self.scale if alpha is None else self.scale * alpha
        zero_point = self.zero_point if beta is None else self.zero_point + beta
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs
