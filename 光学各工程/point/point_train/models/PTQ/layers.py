import torch
import torch.nn as nn
from torch.nn import functional as F

from .bit_type import BIT_TYPE_DICT
from .observer import build_observer
from .quantizer import build_quantizer
from .ptq_config import Config

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class QSequential(nn.Sequential):
    def __init__(self, *args):
        super(QSequential, self).__init__(*args)
        self.out_scale = 1
    
    def update_out_scale(self, scale):
        self.out_scale = scale

    def forward(self, input, scale):
        i = 0
        for module in self:
            if type(module) in [QConv1d, QConv2d, QConvTranspose2d, QBlock, QSeModule, QLinear]:
                # print(module)
                if i > 0: 
                    middle_scale = self[i-1].quantizer.scale if type(self[i-1]) == QAct else self[i-1].out_scale              
                    input = module(input, middle_scale) 
                else:
                    input = module(input, scale)
                self.update_out_scale(module.quantizer.scale if type(module) not in [QBlock, QSeModule] else module.out_scale)
            else:
                input = module(input)
            i += 1
        return input


class QBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride, cfg: Config):
        super(QBlock, self).__init__()
        self.stride = stride
        self.out_scale = 1

        self.qconv1 = QConv2d(
            in_size,
            expand_size,
            kernel_size=1,
            bias=False,
            quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_W,
            calibration_mode=cfg.CALIBRATION_MODE_W,
            observer_str=cfg.OBSERVER_W,
            quantizer_str=cfg.QUANTIZER_W
        )
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.qconv1_act = QAct(
            quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
        )
        self.nolinear1 = nolinear
        self.nolinear1_act = QAct(
            quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
        )
        self.qconv2 = QConv2d(
            expand_size,
            expand_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expand_size,
            bias=False,
            quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_W,
            calibration_mode=cfg.CALIBRATION_MODE_W,
            observer_str=cfg.OBSERVER_W,
            quantizer_str=cfg.QUANTIZER_W
        )
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.qconv2_act = QAct(
            quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
        )
        self.nolinear2 = nolinear
        self.nolinear2_act = QAct(
                quant=False,
                calibrate=False,
                bit_type=cfg.BIT_TYPE_A,
                calibration_mode=cfg.CALIBRATION_MODE_A,
                observer_str=cfg.OBSERVER_A,
                quantizer_str=cfg.QUANTIZER_A
        )
        self.qconv3 = QConv2d(
                expand_size,
                out_size,
                kernel_size=1,
                bias=False,
                quant=False,
                calibrate=False,
                bit_type=cfg.BIT_TYPE_W,
                calibration_mode=cfg.CALIBRATION_MODE_W,
                observer_str=cfg.OBSERVER_W,
                quantizer_str=cfg.QUANTIZER_W
        )
        self.bn3 = nn.BatchNorm2d(out_size)
        self.qconv3_act = QAct(
                quant=False,
                calibrate=False,
                bit_type=cfg.BIT_TYPE_A,
                calibration_mode=cfg.CALIBRATION_MODE_A,
                observer_str=cfg.OBSERVER_A,
                quantizer_str=cfg.QUANTIZER_A
        )
        self.se = semodule
        # self.shortcut = nn.Sequential()
        self.shortcut = QSequential()
        # if stride == 1 and in_size != out_size:
        if stride == 1 and in_size == out_size:
            #self.shortcut = nn.Sequential(
            self.shortcut = QSequential(
                QConv2d(
                    in_size,
                    out_size,
                    kernel_size=1,
                    bias=False,
                    quant=False,
                    calibrate=False,
                    bit_type=cfg.BIT_TYPE_W,
                    calibration_mode=cfg.CALIBRATION_MODE_W,
                    observer_str=cfg.OBSERVER_W,
                    quantizer_str=cfg.QUANTIZER_W
                    
                ),
                nn.BatchNorm2d(out_size),
                QAct(
                    quant=False,
                    calibrate=False,
                    bit_type=cfg.BIT_TYPE_A,
                    calibration_mode=cfg.CALIBRATION_MODE_A,
                    observer_str=cfg.OBSERVER_A,
                    quantizer_str=cfg.QUANTIZER_A 
                )
            )
        self.add_act = QAct(
            quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A 
        )

    def update_out_scale(self, scale):
        self.out_scale = scale

    def forward(self, x, scale):
        out = self.nolinear1_act(self.nolinear1(self.qconv1_act(self.bn1(self.qconv1(x, scale)))))
        out = self.nolinear2_act(self.nolinear2(self.qconv2_act(self.bn2(self.qconv2(out, self.nolinear1_act.quantizer.scale)))))
        out = self.qconv3_act(self.bn3(self.qconv3(out, self.nolinear2_act.quantizer.scale)))
        out_scale = self.qconv3_act.quantizer.scale
        if self.se != None:
            out = self.se(out, out_scale)
            out_scale = self.se.out_scale
        out = self.add_act(out + self.shortcut(x, scale)) if self.stride == 1 and (x.shape[1] == out.shape[1]) else out
        out_scale = self.add_act.quantizer.scale if self.stride == 1 and (x.shape[1] == out.shape[1]) else out_scale
        self.update_out_scale(out_scale)
        return out


class QSeModule(nn.Module):
    def __init__(self, in_size, cfg: Config, reduction=4):
        super(QSeModule, self).__init__()
        self.out_scale = 1
        # self.se = nn.Sequential(
        self.se = QSequential(
            nn.AdaptiveAvgPool2d(1),
            QAct(
                quant=False,
                calibrate=False,
                bit_type=cfg.BIT_TYPE_A,
                calibration_mode=cfg.CALIBRATION_MODE_A,
                observer_str=cfg.OBSERVER_A,
                quantizer_str=cfg.QUANTIZER_A
            ),
            QConv2d(
                in_size,
                in_size // reduction,
                kernel_size=1,
                bias=False,
                quant=False,
                calibrate=False,
                bit_type=cfg.BIT_TYPE_W,
                calibration_mode=cfg.CALIBRATION_MODE_W,
                observer_str=cfg.OBSERVER_W,
                quantizer_str=cfg.QUANTIZER_W
            ),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            QAct(
                quant=False,
                calibrate=False,
                bit_type=cfg.BIT_TYPE_A,
                calibration_mode=cfg.CALIBRATION_MODE_A,
                observer_str=cfg.OBSERVER_A,
                quantizer_str=cfg.QUANTIZER_A
            ),
            QConv2d(
                in_size // reduction,
                in_size,
                kernel_size=1,
                bias=False,
                quant=False,
                calibrate=False,
                bit_type=cfg.BIT_TYPE_W,
                calibration_mode=cfg.CALIBRATION_MODE_W,
                observer_str=cfg.OBSERVER_W,
                quantizer_str=cfg.QUANTIZER_W
            ),
            nn.BatchNorm2d(in_size),
            QAct(
                quant=False,
                calibrate=False,
                bit_type=cfg.BIT_TYPE_A,
                calibration_mode=cfg.CALIBRATION_MODE_A,
                observer_str=cfg.OBSERVER_A,
                quantizer_str=cfg.QUANTIZER_A
            ),
            hsigmoid(),
            QAct(
                quant=False,
                calibrate=False,
                bit_type=cfg.BIT_TYPE_A,
                calibration_mode=cfg.CALIBRATION_MODE_A,
                observer_str=cfg.OBSERVER_A,
                quantizer_str=cfg.QUANTIZER_A
            )
        )
        self.mul_act = QAct(
            quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
        )

    def update_out_scale(self, scale):
        self.out_scale = scale

    def forward(self, x, scale):
        out = x * self.se(x, scale)
        out = self.mul_act(out)
        self.update_out_scale(self.mul_act.quantizer.scale)
        return out


class QConv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT["int8"],
            calibration_mode="layer_wise",
            observer_str="minmax",
            quantizer_str="uniform",
            hist_mode=False):
        super(QConv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.hist_mode = hist_mode

        self.module_type = "conv_weight"
        self.observer = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)
    
    def forward(self, x, scale):
        if self.calibrate:
            if self.hist_mode:
                self.quantizer.observer.update_histc(self.weight)
            else:
                self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(before_scale=scale)
        if not self.quant:
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        weight = self.quantizer(self.weight)
        return F.conv1d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class QConv2d(nn.Conv2d):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT["int8"],
            calibration_mode="layer_wise",
            observer_str="minmax",
            quantizer_str="uniform",
            hist_mode=False):
        super(QConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.hist_mode = hist_mode

        self.module_type = "conv_weight"
        self.observer = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x, scale):
        if self.calibrate:
            if self.hist_mode:
                self.quantizer.observer.update_histc(self.weight)
            else:
                self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(before_scale=scale)
        if not self.quant:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        weight = self.quantizer(self.weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class QConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            dilation=1,
            groups=1,
            bias=True,
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT["int8"],
            calibration_mode="layer_wise",
            observer_str="minmax",
            quantizer_str="uniform",
            hist_mode=False):
        super(QConvTranspose2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.hist_mode = hist_mode

        self.module_type = "conv_weight"
        self.observer = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x, scale):
        if self.calibrate:
            if self.hist_mode:
                self.quantizer.observer.update_histc(self.weight)
            else:
                self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(before_scale=scale)
        if not self.quant:
            return F.conv_transpose2d(
                input=x,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups
            )
        weight = self.quantizer(self.weight)
        return F.conv_transpose2d(
            input=x, weight=weight, bias=self.bias, stride=self.stride, padding=self.padding, output_padding=self.output_padding, dilation=self.dilation, groups=self.groups
        )


class QLinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT["int8"],
            calibration_mode="layer_wise",
            observer_str="minmax",
            quantizer_str="uniform",
            hist_mode=False):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.hist_mode = hist_mode

        self.module_type = "linear_weight"
        self.observer = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            if self.hist_mode:
                self.quantizer.observer.update_histc(self.weight)
            else:
                self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        weight = self.quantizer(self.weight)
        return F.linear(x, weight, self.bias)


class QAct(nn.Module):
    def __init__(
            self,
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT["int8"],
            calibration_mode="layer_wise",
            observer_str="minmax",
            quantizer_str="uniform",
            hist_mode=False):
        super(QAct, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.hist_mode = hist_mode

        self.module_type = "activation"
        self.observer = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            if self.hist_mode:
                self.quantizer.observer.update_histc(x)
            else:
                self.quantizer.observer.update(x)
            if self.last_calibrate:
                # import ipdb;ipdb.set_trace()
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return x
        x = self.quantizer(x)
        return x


class QIntLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(QIntLayerNorm, self).__init__(
            normalized_shape, eps, elementwise_affine)
        assert isinstance(normalized_shape, int)
        self.mode = 'ln'

    def get_MN(self, x):
        bit = 7
        N = torch.clamp(bit - torch.floor(torch.log2(x)), 0, 31)
        M = torch.clamp(torch.floor(x * torch.pow(2, N)), 0, 2**(bit+1)-1)
        return M, N

    def forward(self, x, in_quantizer=None, out_quantizer=None, in_scale_expand=1):
        if self.mode == 'ln':
            x = F.layer_norm(x, self.normalized_shape,
                             self.weight, self.bias, self.eps)
        elif self.mode == 'int':
            in_scale = in_quantizer.scale
            if in_scale_expand != 1:
                in_scale = in_scale.unsqueeze(-1).expand(-1,
                                                         in_scale_expand).T.reshape(-1)
            out_scale = out_quantizer.scale
            assert in_scale is not None and out_scale is not None
            channel_nums = x.shape[-1]
            in_scale = in_scale.reshape(1, 1, -1)
            out_scale = out_scale.reshape(1, 1, -1)
            x_q = (x / in_scale).round()
            in_scale1 = in_scale.min()
            in_scale_mask = (in_scale / in_scale1).round()

            x_q = x_q * in_scale_mask

            mean_x_q = x_q.mean(dim=-1) * in_scale1
            std_x_q = (in_scale1 / channel_nums) * torch.sqrt(channel_nums *
                                                              (x_q ** 2).sum(dim=-1) - x_q.sum(dim=-1) ** 2)

            A = (in_scale1 / std_x_q).unsqueeze(-1) * \
                self.weight.reshape(1, 1, -1) / out_scale
            A_sign = A.sign()
            M, N = self.get_MN(A.abs())
            B = ((self.bias.reshape(1, 1, -1) - (mean_x_q / std_x_q).unsqueeze(-1)
                 * self.weight.reshape(1, 1, -1)) / out_scale * torch.pow(2, N)).round()

            x_q = ((A_sign * M * x_q + B) / torch.pow(2, N)).round()
            x = x_q * out_scale
        else:
            raise NotImplementedError
        return x


class QIntSoftmax(nn.Module):
    def __init__(
            self,
            log_i_softmax=False,
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT["int8"],
            calibration_mode="layer_wise",
            observer_str="minmax",
            quantizer_str="uniform",
            hist_mode=False,
            dim=-1):
        super(QIntSoftmax, self).__init__()

        self.log_i_softmax = log_i_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.hist_mode = hist_mode
        self.dim = dim

        self.module_type = "activation"
        self.observer = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor):
        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # a0(x+b0)**2+c0 = ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / scaling_factor ** 2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor ** 2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.6931  # -ln2
            n = 30  # sufficiently large integer
            x0_int = torch.floor(x0 / scaling_factor)
            x_int = torch.max(x_int, n * x0_int)
            q = torch.floor(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = torch.clamp(torch.floor(exp_int * 2 ** (n - q)), min=0)
            scaling_factor = exp_scaling_factor / 2 ** n
            return exp_int, scaling_factor

        x_int = x/scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        return exp_int, exp_int_sum

    def forward(self, x, scale=None):
        if self.log_i_softmax and scale is not None and not self.last_calibrate:
            exp_int, exp_int_sum = self.int_softmax(x, scale)
            softmax_out = torch.round(exp_int_sum / exp_int)
            rounds = self.log_round(softmax_out)
            mask = rounds >= 2**self.bit_type.bits
            qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits-1)
            deq_softmax = 2**(-qlog)
            deq_softmax[mask] = 0
            return deq_softmax
        else:
            x = x.softmax(dim=self.dim)
            if self.calibrate:
                if self.hist_mode:
                    self.quantizer.observer.update_histc(x)
                else:
                    self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            x = self.quantizer(x)
            return x


# def softmax_ps(x):
#     ln2_param = 0.6931
#     q = torch.floor(x / ln2_param)
#     r = x - ln2_param * q
#     exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
#     exp_int = torch.clamp(torch.floor(exp_int * 2 ** (n - q)), min=0)
#     scaling_factor = exp_scaling_factor / 2 ** n
#     return exp_int, scaling_factor

# def exp_ps(input):
#     pass