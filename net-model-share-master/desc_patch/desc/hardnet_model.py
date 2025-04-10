import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.autograd import Variable
import torchvision
import math
from typing import Tuple
# from desc.RotEqNet.layers_2D import *

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1,
                      stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()
        # if stride == 1 and in_size != out_size:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_size, out_size, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.shortcut_flag else out
        return out

class Block_cRelu(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_cRelu, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.conv1 = nn.Conv2d(in_size, expand_size//2,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size//2)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size//2, expand_size//2, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size//2)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()
        # if stride == 1 and in_size != out_size:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_size, out_size, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.cat([self.nolinear2(out),self.nolinear2(-out)],dim=1)
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.shortcut_flag else out
        return out


class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SeModule(out_channels)
        else:
            self.se = nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class MobileOne(nn.Module):

    def __init__(self,in_planes, planes, stride=1, use_se=False, num_conv_branches=4, inference_mode=False):
        super(MobileOne, self).__init__()
        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches
        self.in_planes = in_planes
       
        blocks = []

        blocks.append(MobileOneBlock(in_channels=self.in_planes,
                            out_channels=planes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                            inference_mode=self.inference_mode,
                            use_se=False,
                            num_conv_branches=self.num_conv_branches))

        # Depthwise conv
        blocks.append(MobileOneBlock(in_channels=planes,
                            out_channels=planes,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            groups=planes,
                            inference_mode=self.inference_mode,
                            use_se=False,
                            num_conv_branches=self.num_conv_branches))
        # Pointwise conv
        blocks.append(MobileOneBlock(in_channels=planes,
                            out_channels=planes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                            inference_mode=self.inference_mode,
                            use_se=use_se,
                            num_conv_branches=self.num_conv_branches))
        self.in_planes = planes
           
        self.conv = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass.  """
        x = self.conv(x)
        return x

class Double_linear(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Double_linear, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)


    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.bn3(self.conv3(out))

        return out


class PconvBlock(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride, n_div=4):
        super(PconvBlock, self).__init__()
        assert stride == 1
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.dim_conv = in_size // n_div
        self.conv1 = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.dim_conv)
        # self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()


    def forward(self, x):
        x[:,:self.dim_conv,:,:] = self.conv1(x[:,:self.dim_conv,:,:])
        out = self.nolinear2(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.shortcut_flag else out
        return out

class Block_SA(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_SA, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()

    def forward(self, x):
        B, c, h, w = x.size()
        Q = x.view(B,c,1,-1)
        K = Q
        V = self.nolinear1(self.bn1(self.conv1(x)))
        V = V.view(B,c,1,-1)
        out, _ = attention(Q, K, V)
        out = out.view(B,c,h,w)
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.shortcut_flag else out
        return out

class Block_PCA8(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_PCA8, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.features = nn.Sequential(
            nn.Conv2d(in_size, 8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(8, expand_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expand_size),
            nolinear,
            nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False),
            nn.BatchNorm2d(expand_size),
            nolinear,
            nn.Conv2d(expand_size, 8,kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(8, out_size,kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_size)
        )
        self.se = semodule
        self.shortcut = nn.Sequential()
        # if stride == 1 and in_size != out_size:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_size, out_size, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )

    def forward(self, x):
        out = self.features(x)
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.shortcut_flag else out
        return out

class Block_simple(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_simple, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear

        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.shortcut_flag else out
        return out

class Block_pad(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride, pad=1):
        super(Block_pad, self).__init__()
        self.stride = stride
        self.pad = pad
        self.shortcut_flag = (self.stride == 1 and in_size == out_size and pad == 1)

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=pad, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()
        # if stride == 1 and in_size != out_size:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_size, out_size, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        if self.pad == 0:
            out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.shortcut_flag else out
        return out


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            hswish() if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            hswish() if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=hswish(), se_ratio=4):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_chs == out_chs)

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SeModule(mid_chs, reduction=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # # shortcut
        # if (in_chs == out_chs and self.stride == 1):
        #     self.shortcut = nn.Sequential()
        # # else:
        # #     self.shortcut = nn.Sequential(
        # #         nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
        # #                padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
        # #         nn.BatchNorm2d(in_chs),
        # #         nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
        # #         nn.BatchNorm2d(out_chs),
        # #     )

        self.shortcut = nn.Sequential()



    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        if self.shortcut_flag:
            x += self.shortcut(residual)
        return x

   
class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,mode='original',args=None):
        super(GhostModuleV2, self).__init__()
        self.mode=mode
        self.gate_fn=hsigmoid()

        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                hsigmoid() if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                hsigmoid()if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']: 
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            ) 
            self.short_conv = nn.Sequential( 
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,5), stride=1, padding=(0,2), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5,1), stride=1, padding=(2,0), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
            ) 
      
    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]         
        elif self.mode in ['attn']:  
            res=self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))  
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]*F.interpolate(self.gate_fn(res),size=(out.shape[-2],out.shape[-1]),mode='nearest') 


class GhostBottleneckV2(nn.Module): 

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=hsigmoid(), se_ratio=4,layer_id=0,args=None):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_chs == out_chs)

        # Point-wise expansion
        if layer_id<=1:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True,mode='original',args=args)
        else:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True,mode='attn',args=args) 

        # Depth-wise convolution
        # if self.stride > 1:
        self.dw = nn.Sequential(
            nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                    padding=(dw_kernel_size-1)//2,groups=mid_chs, bias=False),
            nn.BatchNorm2d(mid_chs)
        )

        # Squeeze-and-excitation
        if has_se:
            self.se = SeModule(mid_chs, reduction=se_ratio)
        else:
            self.se = None
            
        self.ghost2 = GhostModuleV2(mid_chs, out_chs, relu=False,mode='original',args=args)
        
        # shortcut
        self.shortcut = nn.Sequential()
    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        # if self.stride > 1:
        #     x = self.conv_dw(x)
        #     x = self.bn_dw(x)
        x = self.dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        if self.shortcut_flag:
            x += self.shortcut(residual)
        return x

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # nn.init.orthogonal(m.weight.data, gain=0.6)
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return

class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )
                
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_small(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_small, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=2),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=2),
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 8, kernel_size=1, padding=0)
        )

        # self.features = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, padding=1, bias = False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #     Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
        #              nolinear=hswish(), semodule=SeModule(int(64)), stride=2),
        #     Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
        #              nolinear=hswish(), semodule=SeModule(int(128)), stride=2),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(128, 256, kernel_size=1, padding=0)
        # )
                
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class HardNet_tiny(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_tiny, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=2),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=2),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1),
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 8, kernel_size=1, padding=0)
        )
                
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output

class HardNet_fast(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            nn.Conv2d(64, 16, kernel_size=1, padding=0)
        )
 
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class HardNet_fast_inpaint(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_inpaint, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1),
            nn.Conv2d(128, 16, kernel_size=1, padding=0)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                    nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                    nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Block(kernel_size=3, in_size=int(16), expand_size=int(8), out_size=int(8),
                    nolinear=hswish(), semodule=SeModule(int(8)), stride=1),
            Block(kernel_size=3, in_size=int(8), expand_size=int(8), out_size=int(8),
                    nolinear=hswish(), semodule=SeModule(int(8)), stride=1),
            nn.Conv2d(8, 1, kernel_size=1, padding=0),
            hsigmoid()
        )
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(16, 16, kernel_size=1, padding=0, bias = False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 8, kernel_size=1, padding=0, bias = False),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        #     nn.Conv2d(8, 8, kernel_size=1, padding=0, bias = False),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        #     nn.Conv2d(8, 1, kernel_size=1, padding=0),
        #     hsigmoid()
        # )
 
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # patch_inpaint = self.decoder(x_features)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class HardNet_fast_expand(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_expand, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128
        self.expand = nn.Sequential(
            nn.ZeroPad2d((0,0,1,1)),
            Block(kernel_size=3, in_size=int(1), expand_size=int(4), out_size=int(4),
                     nolinear=hswish(), semodule=SeModule(int(4)), stride=2),
            Block(kernel_size=3, in_size=int(4), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            nn.ZeroPad2d((2,2,1,1)),
            Block(kernel_size=3, in_size=int(8), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=1),
            Block(kernel_size=3, in_size=int(8), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=1),
            Block(kernel_size=3, in_size=int(8), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Block(kernel_size=3, in_size=int(8), expand_size=int(8), out_size=int(4),
                    nolinear=hswish(), semodule=SeModule(int(4)), stride=1),
            Block(kernel_size=3, in_size=int(4), expand_size=int(4), out_size=int(4),
                    nolinear=hswish(), semodule=SeModule(int(4)), stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Block(kernel_size=3, in_size=int(4), expand_size=int(4), out_size=int(4),
                    nolinear=hswish(), semodule=SeModule(int(4)), stride=1),
            nn.Conv2d(4, 1, kernel_size=1, padding=0),
            hsigmoid()
        )

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            nn.Conv2d(64, 16, kernel_size=1, padding=0)
        )

        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):

        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # patch_inpaint = self.decoder(x_features)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)
    
    def expand_forward(self, input):
        #输入为128*52
        input = input[:,:,3:-3,8:-8] #128*52=>122*36
        input_expand = self.expand(input) #122*36=>132*52
        input_expand[:,:,5:-5,8:-8] = input #中心图像不改变
        input = input_expand[:,:,2:-2,:] #135*52=>128*52
        return input


class HardNet_fast_t(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_t, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
            #          nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
            #          nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1),
            nn.Conv2d(128, 16, kernel_size=1, padding=0)
        )
 
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_s(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_s, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 16, kernel_size=1, padding=0)
        )

        # self.features = nn.Sequential(
        #     Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
        #              nolinear=hswish(), semodule=None, stride=2),
        #     Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
        #              nolinear=hswish(), semodule=None, stride=2),
        #     Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
        #              nolinear=hswish(), semodule=None, stride=1),
        #     Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
        #              nolinear=hswish(), semodule=None, stride=1),
        #     Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
        #              nolinear=hswish(), semodule=None, stride=1),
        #     # Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
        #     #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #     # Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
        #     #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #     Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=None, stride=1),
        #     nn.Conv2d(32, 16, kernel_size=1, padding=0)
        # )
 
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_s1(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_s1, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 16, kernel_size=1, padding=0)
        )
 
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_s_mobileone(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_s_mobileone, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            MobileOne(in_planes=16,planes=16,stride=1),
            MobileOne(in_planes=16,planes=16,stride=1),             
            MobileOne(in_planes=16,planes=16,stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0)
        )

        
        # self.features = nn.Sequential(
        #     MobileOne(in_planes=1, planes=8,stride=2),
        #     MobileOne(in_planes=8, planes=16,stride=2), 
        #     MobileOne(in_planes=16,planes=16,stride=1),
        #     MobileOne(in_planes=16,planes=16,stride=1),             
        #     MobileOne(in_planes=16,planes=16,stride=1, use_se=True),
        #     MobileOne(in_planes=16,planes=32,stride=1, use_se=True),
        #     nn.Conv2d(32, 16, kernel_size=1, padding=0)
        # )
 
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class HardNet_fast_MS(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_MS, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(2), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=1),
            nn.AdaptiveAvgPool2d(1),
            Block(kernel_size=3, in_size=int(8), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1)
        )
        self.ms0 = Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1)
        self.ms1 = Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1)
        self.ms2 = Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1)
 
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        ms0 = self.ms0(x_features)
        ms1 = self.ms1(ms0)
        ms2 = self.ms2(ms1)
        x_features = torch.cat([ms0,ms1,ms2],dim=1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_np(nn.Module):
    """HardNet model definition
    with non linear projection
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_np, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(2), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            nn.Conv2d(64, 8, kernel_size=1, padding=0)
        )
        self.nonlinear_projection = nn.Sequential(
            nn.Conv2d(8, 256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 8, kernel_size=1, padding=0)
        )
 
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        x_features = self.nonlinear_projection(x_features)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_pconv(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_pconv, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(2), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            PconvBlock(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            PconvBlock(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            PconvBlock(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            nn.Conv2d(64, 16, kernel_size=1, padding=0)
        )
 
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class HardNet_fast_cat(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_cat, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.ds = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2)
        )
        self.mbn0 = Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1)
        self.mbn1 = Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1)
        self.mbn2 = Block(kernel_size=3, in_size=int(48), expand_size=int(32), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1)
        self.mbn3 = Block(kernel_size=3, in_size=int(64), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1)
        self.post = nn.Conv2d(96, 8, kernel_size=1, padding=0)
 
        if self.train_flag:
            self.ds.apply(weights_init)
            self.mbn0.apply(weights_init)
            self.mbn1.apply(weights_init)
            self.mbn2.apply(weights_init)
            self.mbn3.apply(weights_init)
            self.post.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x0 = self.ds(input)
        x1 = self.mbn0(x0) #16=>16
        x1_cat = torch.cat([x0,x1],dim=1) #32
        x2 = self.mbn1(x1_cat) #32=>16
        x2_cat = torch.cat([x1_cat,x2],dim=1) #48
        x3 = self.mbn2(x2_cat) #48=>16
        x3_cat = torch.cat([x2_cat,x3],dim=1) #64
        x4 = self.mbn3(x3_cat) #64=>32
        x4_cat = torch.cat([x3_cat,x4],dim=1) #96
        x_features = self.post(x4_cat) #96=>8
        # x_features = self.post(x4) #96=>8

        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


# class HardNet_fast_cat(nn.Module):
#     """HardNet model definition
#     """
#     def __init__(self, train_flag=False):
#         super(HardNet_fast_cat, self).__init__()
#         self.train_flag = train_flag
       
#         # version 1, 2
#         # 16 128

#         self.ds = nn.Sequential(
#             Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
#                      nolinear=hswish(), semodule=None, stride=2),
#             Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
#                      nolinear=hswish(), semodule=None, stride=2)
#         )
#         self.mbn0 = Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(16),
#                      nolinear=hswish(), semodule=None, stride=1)
#         self.mbn1 = Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(16),
#                      nolinear=hswish(), semodule=None, stride=1)
#         self.mbn2 = Block(kernel_size=3, in_size=int(48), expand_size=int(32), out_size=int(16),
#                      nolinear=hswish(), semodule=None, stride=1)
#         self.mbn3 = Block(kernel_size=3, in_size=int(64), expand_size=int(32), out_size=int(64),
#                      nolinear=hswish(), semodule=None, stride=1)
#         self.post = nn.Conv2d(64, 8, kernel_size=1, padding=0)
 
#         if self.train_flag:
#             self.ds.apply(weights_init)
#             self.mbn0.apply(weights_init)
#             self.mbn1.apply(weights_init)
#             self.mbn2.apply(weights_init)
#             self.mbn3.apply(weights_init)
#             self.post.apply(weights_init)
#         return
    
#     def input_norm(self,x):
#         flat = x.view(x.size(0), -1)
#         mp = torch.mean(flat, dim=1)
#         sp = torch.std(flat, dim=1) + 1e-7
#         return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
#     def forward(self, input):
#         # x_features = self.features(self.input_norm(input))
#         x0 = self.ds(input)
#         x1 = self.mbn0(x0) #16=>16
#         x1_cat = torch.cat([x0,x1],dim=1) #32
#         x2 = self.mbn1(x1_cat) #32=>16
#         x2_cat = torch.cat([x1_cat,x2],dim=1) #48
#         x3 = self.mbn2(x2_cat) #48=>16
#         x3_cat = torch.cat([x2_cat,x3],dim=1) #64
#         x4 = self.mbn3(x3_cat) #64=>32
#         # x4_cat = torch.cat([x3_cat,x4],dim=1) #96
#         # x_features = self.post(x4_cat) #96=>8
#         x_features = self.post(x4) #96=>8

#         # print('x_features.shape: ', x_features.shape)
#         x = x_features.view(x_features.size(0), -1)
       
#         # x_features = self.input_norm(input)
#         # # print('x_features.shape: ', x_features.shape)
#         # x = x_features.view(x_features.size(0), -1)
#         return L2Norm()(x)


class HardNet_fast_PCA(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_PCA, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block_PCA8(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_PCA8(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_PCA8(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_PCA8(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            nn.Conv2d(64, 8, kernel_size=1, padding=0)
        )
 
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_subpixel(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_subpixel, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=2),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            DepthToSpace(2)
        )

        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_fix180(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_fix180, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block_pad(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2, pad=0),
            Block_pad(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2, pad=0),
            Block_pad(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1, pad=0),
            Block_pad(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_pad(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_pad(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            nn.Conv2d(64, 128, kernel_size=1, padding=0)
        )


                
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(input)
        x = x_features.view(x_features.size(0), -1)
       
        return L2Norm()(x)

class HardNet_fast_ghost(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_ghost, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            GhostBottleneckV2(in_chs=int(32), mid_chs=int(32), out_chs=int(32), stride=1),
            GhostBottleneckV2(in_chs=int(32), mid_chs=int(32), out_chs=int(32), stride=1),
            GhostBottleneckV2(in_chs=int(32), mid_chs=int(64), out_chs=int(64), stride=1),
            nn.Conv2d(64, 8, kernel_size=1, padding=0)
        )
 
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)



class HardNet_dense(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_dense, self).__init__()
        self.train_flag = train_flag

        # self.features = nn.Sequential(
        #     nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True),
        #     Block_simple(in_size=1, expand_size=16, out_size=16,
        #              nolinear=hswish(), semodule=SeModule(16), stride=1),
        #     nn.AvgPool2d(kernel_size=2,stride=2),
        #     Block_simple(in_size=16, expand_size=32, out_size=32,
        #              nolinear=hswish(), semodule=SeModule(32), stride=1),
        #     nn.AvgPool2d(kernel_size=2,stride=2),
        #     Block_simple(in_size=32, expand_size=32, out_size=32,
        #              nolinear=hswish(), semodule=SeModule(32), stride=1),
        #     Block_simple(in_size=32, expand_size=64, out_size=64,
        #              nolinear=hswish(), semodule=SeModule(64), stride=1),
        #     nn.Conv2d(64, 8, kernel_size=1, padding=0)
        # )

        self.features = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True),
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=2),
            nn.ZeroPad2d((3,3,3,3)),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            nn.Conv2d(64, 16, kernel_size=1, padding=0)
        )


        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        # print('x_features.shape: ', x_features.shape)
        x = x_features
        return x


class HardNet_dense_test(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_dense_test, self).__init__()
        self.train_flag = train_flag

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            nn.ZeroPad2d((3,3,3,3)),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            nn.Conv2d(64, 8, kernel_size=1, padding=0)
        )


        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        input_rot90 = input.permute(0,1,3,2).flip(dims=[-2]) #逆时针旋转
        input_rot180 = input_rot90.permute(0,1,3,2).flip(dims=[-2]) #逆时针旋转
        input_rot270 = input_rot180.permute(0,1,3,2).flip(dims=[-2]) #逆时针旋转
        x_features = self.features(self.input_norm(input))
        x_features_rot90 = self.features(self.input_norm(input_rot90))
        x_features_rot180 = self.features(self.input_norm(input_rot180))
        x_features_rot270 = self.features(self.input_norm(input_rot270))
        
        #方向变换
        x_features_rot180 = x_features_rot180.flip(dims=[-1,-2])
        x_features_rot270 = x_features_rot270.flip(dims=[-1,-2])

        x_features = x_features + x_features_rot180
        x_features_rot90 = x_features_rot90 + x_features_rot270
        x_features_rot90 = x_features_rot90.permute(0,1,3,2).flip(dims=[-1])
        x_features = x_features + x_features_rot90
        

        x = x_features
        return x


class HardNet_dense_Re(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_dense_Re, self).__init__()
        self.train_flag = train_flag

        self.kernel_size = 9
        self.features = nn.Sequential(
            RotConv(1, 8, [self.kernel_size, self.kernel_size], 1, self.kernel_size // 2, n_angles=17, mode=1),
            VectorMaxPool(2),
            VectorBatchNorm(8),

            RotConv(8, 16, [self.kernel_size, self.kernel_size], 1, self.kernel_size // 2, n_angles=17, mode=2),
            VectorMaxPool(2),
            VectorBatchNorm(16),

            RotConv(16, 32, [self.kernel_size, self.kernel_size], 1, self.kernel_size // 2, n_angles=17, mode=2),
            # VectorMaxPool(2),
            VectorBatchNorm(32),

            RotConv(32, 64, [self.kernel_size, self.kernel_size], 1, self.kernel_size // 2, n_angles=17, mode=2),
            # VectorMaxPool(2),
            # VectorBatchNorm(64),

            Vector2Magnitude(),

            nn.Conv2d(64, 8, kernel_size=1, padding=0)

        )

        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        # print('x_features.shape: ', x_features.shape)
        x = x_features
        # print(x)
        return x

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(hsigmoid())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=1)
        )
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        # query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
        #                      for l, x in zip(self.proj, (query, key, value))]
        # x, _ = attention(query, key, value)
        # return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

        query, key, value = [l(x).view(batch_dim, self.num_heads,self.dim, -1).permute(0,2,1,3)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.permute(0,2,1,3).contiguous().view(batch_dim, self.dim*self.num_heads, -1))



class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, 8, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.layer = AttentionalPropagation(feature_dim, 1)

    def forward(self, desc0):
        B, c, Hc, Wc = desc0.size()
        desc0 = desc0.view(B, c, -1)
        delta0 = self.layer(desc0, desc0)
        desc0  = (desc0 + delta0)
        desc0 = desc0.view(B, c, Hc, Wc)

        return desc0

class HardNet_Linear(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_Linear, self).__init__()
        self.train_flag = train_flag
        
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            nn.Conv2d(16, 8, kernel_size=1, padding=0, bias = False),
        )

        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        batch_size, _, H, W = input.size()
        unfold = nn.Unfold(kernel_size=4, dilation=1, padding=0, stride=4)
        input_unfold = unfold(self.input_norm(input)).view(batch_size, 16, 4, 4)
        x_features = self.features(input_unfold)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        return L2Norm()(x)



class HardNet_fast_GNN(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_GNN, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_SA(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_SA(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_SA(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0)
        )
 
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.reshape(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


def test_model(model: torch.nn.Module):
    # evaluate the `model` on 8 rotated versions of the input image `x`
    model.eval()
    
    wrmup = (torch.randint(0,255,(1, 1, 136, 40))).to(device) / 255
    wrmup_90 = wrmup.permute(0,1,3,2).flip(dims=[-2]) #逆时针旋转

    
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(16):
            x_transformed = torchvision.transforms.functional.rotate(wrmup,r*22.5)
            x_transformed = torchvision.transforms.functional.resize(x_transformed, [137,41])
            x_transformed = x_transformed.to(device)

            x_transformed_90 = torchvision.transforms.functional.rotate(wrmup_90,r*22.5)
            x_transformed_90 = torchvision.transforms.functional.resize(x_transformed_90, [41,137])
            x_transformed_90 = x_transformed_90.to(device)

            y = model(x_transformed)
            y_90 = model(x_transformed_90)
    
            y = y[0, ...].detach().unsqueeze(0)
            y = torchvision.transforms.functional.rotate(y,-r*22.5)
            y = y[:,:,8,8]
            y = y.view(-1)[:16].cpu().numpy()

            # y_90 = y_90[0, ...].detach().unsqueeze(1)
            # y_90 = torchvision.transforms.functional.rotate(y_90,-r*45)
            # y_90 = y_90.permute(0,1,3,2).flip(dims=[-1]) #逆时针旋转
            # y_90 = y_90[:,:,8,8]
            # y_90 = y_90.view(-1)[:8].cpu().numpy()
            
            angle = r * 22.5
            print("{:5f} : {}".format(angle, y))
            # print("{:5d}_90 : {}".format(angle, y_90))
    print('##########################################################################################')
    print()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HardNet_dense(train_flag=False).to(device)

    test_model(model)

# if __name__ == '__main__':
#     '''print the number of parameters '''
#     net = HardNet_dense_test()
#     total = 0
#     for name, parameters in net.named_parameters():
#         total += parameters.nelement()
#         print(name, ":", parameters.size())
#     print("Number of parameter: %.4fM" % (total / 1e6))

#     input_tensor = torch.randn(32, 1, 136, 40)
#     import torchvision

#     input_tensor_90 = torchvision.transforms.functional.rotate(input_tensor,180)
#     y = net(input_tensor)
#     y_90 = net(input_tensor_90)
#     y_90 = torchvision.transforms.functional.rotate(y_90.view(-1,8,4,4),180).view(y_90.size(0),-1)



