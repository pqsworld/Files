from sklearn.utils import resample
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import os

class BaseQuantizer(nn.Module):
    def __init__(
            self,
            bit_type,
            observer,
            module_type):
        super(BaseQuantizer, self).__init__()
        self.bit_type = bit_type
        self.observer = observer
        self.module_type = module_type

    def get_reshape_range(self, inputs):
        range_shape = None
        if self.module_type == "conv_weight":
            if len(inputs.shape) == 3:
                range_shape = (-1, 1, 1)
            else:
                range_shape = (-1, 1, 1, 1)
        elif self.module_type == "linear_weight":
            range_shape = (-1, 1)
        elif self.module_type == "activation":
            if len(inputs.shape) == 2:
                # range_shape = (1, -1)
                range_shape = (-1, 1)
            elif len(inputs.shape) == 3:
                # range_shape = (1, 1, -1)
                range_shape = (1, -1, 1)
            elif len(inputs.shape) == 4:
                range_shape = (1, -1, 1, 1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return range_shape

    def get_quant_values(self, output):
        output = output.detach().cpu().numpy().flatten()
        res = 'static signed char ' + \
            'feature' + \
            '[{0}] = '.format(
                output.flatten().shape[0]) + '{ \n'
        output = np.concatenate((output[:10], output[-10:]), axis=0)
        if output.shape[0] % 3 == 0:
            np.savetxt('param.txt', output.reshape(-1, 3), fmt='%d',
                        delimiter=', ', newline=',\n')
        else:
            np.savetxt('param.txt', output.reshape(-1, 2), fmt='%d',
                        delimiter=', ', newline=',\n')
        with open('param.txt', 'r') as f:
            res += f.read()
        res += '};\n\n'
        
        path = '/hdd/file-input/linwc/match/datasets/supermatch/output/GNN_feature.h'
        if os.path.exists(path):
            with open(path, 'a') as f:
                f.write(res)
        else:
            # res = '#if 1\n' + res
            with open(path, 'w') as f:
                f.write(res)
        # res += '#endif\n'
        Path(r'param.txt').unlink()

    def update_quantization_params(self, *args, **kwargs):
        pass

    def quant(self, inputs, alpha=None, beta=None):
        raise NotImplementedError

    def dequantize(self, inputs, alpha=None, beta=None):
        raise NotImplementedError

    def forward(self, inputs, dequant=True, alpha=None):
        outputs = self.quant(inputs, alpha=alpha)
        # # save quant feature
        # if self.module_type != 'conv_weight':
        #     self.get_quant_values(outputs)
        if dequant:
            outputs = self.dequantize(outputs)
        return outputs
