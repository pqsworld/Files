import torch
import torch.nn as nn
import torch.nn.functional as F

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Module, Tensor]`.
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Module, Tensor]`.
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)

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

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class CReLU(nn.Module):
    '''
    note: replace=False
    '''
    def __init__(self):
        super(CReLU, self).__init__()
    def forward(self, x):
        return torch.cat((F.relu6(x) / 6, F.relu6(-x) / 6), 1)

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
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: torch.Tensor) -> torch.Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x

class PconvBlock(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(PconvBlock, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear

        self.conv2 = Partial_conv3(expand_size, n_div=4, forward='split_cat')

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
        if stride == 1 and in_size == out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        # out = out + self.shortcut(x) if self.stride == 1 else out
        out = out + self.shortcut(x) if (self.stride == 1 and out.shape[1] == x.shape[1]) else out
        return out

class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride

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
        if stride == 1 and in_size == out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        # out = out + self.shortcut(x) if self.stride == 1 else out
        out = out + self.shortcut(x) if (self.stride == 1 and out.shape[1] == x.shape[1]) else out
        return out

class Block_CRelu(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_CRelu, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size // 2,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size // 2)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size // 2, expand_size // 2, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size // 2)
        self.nolinear2 = CReLU()
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
        if stride == 1 and in_size == out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        # out = out + self.shortcut(x) if self.stride == 1 else out
        out = out + self.shortcut(x) if (self.stride == 1 and out.shape[1] == x.shape[1]) else out
        return out

class HardNet_fast_256(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=True):
        super(HardNet_fast_256, self).__init__()
        self.train_flag = train_flag

        # 原始
        # self.features = nn.Sequential(
        #     Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
        #              nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #     Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
        #              nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
        #     Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #     Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #     Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #     Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
        #              nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
        #     nn.Conv2d(64, 16, kernel_size=1, padding=0)
        # )

        # 所有升通道的block均替换为CReLU操作
        # self.features = nn.Sequential(
        #     Block_CRelu(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
        #              nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #     Block_CRelu(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
        #              nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
        #     Block_CRelu(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #     Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #     Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #     Block_CRelu(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
        #              nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
        #     nn.Conv2d(64, 16, kernel_size=1, padding=0)
        # )


        # half
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
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
    
    # def forward(self, input):
    #     x_features = self.features(self.input_norm(input))
    #     # print('x_features.shape: ', x_features.shape)
    #     x = x_features.view(x_features.size(0), -1)
       
    #     # x_features = self.input_norm(input)
    #     # # print('x_features.shape: ', x_features.shape)
    #     # x = x_features.view(x_features.size(0), -1)
    #     return L2Norm()(x)
    def forward(self, input, flip_flag=False):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        bs, bins, h, w = x_features.shape

        x_features_deform = x_features.permute(0, 2, 3, 1).contiguous().view(bs, -1, bins)
        if flip_flag:
            x_features_flip = torch.flip(x_features_deform, dims=[1])
            x_features_flip = x_features_flip.view(bs, h * w, 2, bins // 2)   # [bs, 16, 2, 8]
            x = x_features_flip.permute(0, 2, 1, 3).contiguous().view(bs, -1)
            # x = x_features_flip.view(x_features_flip.size(0), -1)
        else:
            x_features_deform = x_features_deform.view(bs, h * w, 2, bins // 2)   # [bs, 16, 2, 8]
            x = x_features_deform.permute(0, 2, 1, 3).contiguous().view(bs, -1)
        # print(x.shape)
        return L2Norm()(x)
        # return x
        # return x.view(bs, h, w, bins).permute(0, 3, 1, 2)

class HardNet_fast_Pconv(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=True):
        super(HardNet_fast_Pconv, self).__init__()
        self.train_flag = train_flag

        self.features = nn.Sequential(
            PconvBlock(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            PconvBlock(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            PconvBlock(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            PconvBlock(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            PconvBlock(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            PconvBlock(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
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
    
    # def forward(self, input):
    #     x_features = self.features(self.input_norm(input))
    #     # print('x_features.shape: ', x_features.shape)
    #     x = x_features.view(x_features.size(0), -1)
       
    #     # x_features = self.input_norm(input)
    #     # # print('x_features.shape: ', x_features.shape)
    #     # x = x_features.view(x_features.size(0), -1)
    #     return L2Norm()(x)
    def forward(self, input, flip_flag=False):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        bs, bins, h, w = x_features.shape

        if flip_flag:
            x_features_flip = x_features.permute(0, 2, 3, 1).contiguous().view(bs, -1, bins)
            x_features_flip = torch.flip(x_features_flip, dims=[1])
            x = x_features_flip.view(x_features_flip.size(0), -1)
        else:
            x = x_features.permute(0, 2, 3, 1).contiguous().view(x_features.size(0), -1)
        # print(x.shape)
        return L2Norm()(x)


# The following is Over-Parametrization Net
from typing import Optional, List, Tuple
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
                 use_activation: bool = True,
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
        if use_activation:
            self.activation = hswish() # nn.ReLU()
        else:
            self.activation = nn.Identity()

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
            # self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
            #     if out_channels == in_channels and stride == 1 else None
            self.rbr_skip = None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if 0:
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

class Block_OverParam(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_size, expand_size, out_size, nolinear, semodule, stride,
                inference_mode: bool = False,
                num_conv_branches: int = 2):
        super(Block_OverParam, self).__init__()
        self.stride = stride
        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileOneBlock(in_channels=in_size, out_channels=expand_size,
                                     kernel_size=1, stride=1, padding=0,
                                     inference_mode=self.inference_mode)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(expand_size, out_size, self.stride, se_blocks=True)

        self.shortcut = None
        if stride == 1 and in_size == out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def _make_stage(self,
                    in_planes: int,
                    out_planes: int,
                    stride: int,
                    se_blocks: bool) -> nn.Sequential:
        """ Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param stride: Stride size.
        :param se_blocks: Whether use SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        # strides = [2] + [1]*(num_blocks-1)
        strides = [int(stride)]
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = se_blocks

            # Depthwise conv
            blocks.append(MobileOneBlock(in_channels=in_planes,
                                         out_channels=out_planes,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         groups=in_planes,
                                         inference_mode=self.inference_mode,
                                         use_se=False,
                                         use_activation=True,
                                         num_conv_branches=self.num_conv_branches))
            # Pointwise conv
            blocks.append(MobileOneBlock(in_channels=in_planes,
                                         out_channels=out_planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         use_activation=False,
                                         num_conv_branches=self.num_conv_branches))
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        
        return out

class Block_OverParam_V2(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_size, expand_size, out_size, nolinear, semodule, stride,
                inference_mode: bool = False,
                num_conv_branches: int = 6):
        super(Block_OverParam_V2, self).__init__()
        self.stride = stride
        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileOneBlock(in_channels=in_size, out_channels=expand_size,
                                     kernel_size=1, stride=1, padding=0,
                                     inference_mode=self.inference_mode)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(expand_size, out_size, self.stride, se_blocks=True)

        self.shortcut = None
        if stride == 1 and in_size == out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def _make_stage(self,
                    in_planes: int,
                    out_planes: int,
                    stride: int,
                    se_blocks: bool) -> nn.Sequential:
        """ Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param stride: Stride size.
        :param se_blocks: Whether use SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        # strides = [2] + [1]*(num_blocks-1)
        strides = [int(stride)]
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = se_blocks

            # Depthwise conv
            blocks.append(RepVGGBlock(in_channels=in_planes,
                                         out_channels=out_planes,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         groups=in_planes,
                                         inference_mode=self.inference_mode,
                                         use_se=False))
            # Pointwise conv
            blocks.append(MobileOneBlock(in_channels=in_planes,
                                         out_channels=out_planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         use_activation=False,
                                         num_conv_branches=self.num_conv_branches))
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        
        return out

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', inference_mode=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = hswish()

        if use_se:
            #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
            self.se = SeModule(out_channels)
        else:
            self.se = nn.Identity()

        if inference_mode:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = self.conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = self.conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    #   Optional. This may improve the accuracy and facilitates quantization in some cases.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle


#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if branch is None:
            return 0, 0
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

    def conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        result = nn.Sequential()
        result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
        result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
        return result

    def reparameterize(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.inference_mode = True


class HardNet_fast_256_Overparam(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=True):
        super(HardNet_fast_256_Overparam, self).__init__()
        self.train_flag = train_flag

        # half
        self.features = nn.Sequential(
            Block_OverParam(in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_OverParam(in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block_OverParam(in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block_OverParam(in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block_OverParam(in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_OverParam(in_size=int(32), expand_size=int(32), out_size=int(32),
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
    
    # def forward(self, input):
    #     x_features = self.features(self.input_norm(input))
    #     # print('x_features.shape: ', x_features.shape)
    #     x = x_features.view(x_features.size(0), -1)
       
    #     # x_features = self.input_norm(input)
    #     # # print('x_features.shape: ', x_features.shape)
    #     # x = x_features.view(x_features.size(0), -1)
    #     return L2Norm()(x)
    def forward(self, input, flip_flag=False):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        bs, bins, h, w = x_features.shape

        x_features_deform = x_features.permute(0, 2, 3, 1).contiguous().view(bs, -1, bins)
        if flip_flag:
            x_features_flip = torch.flip(x_features_deform, dims=[1])
            x_features_flip = x_features_flip.view(bs, h * w, 2, bins // 2)   # [bs, 16, 2, 8]
            x = x_features_flip.permute(0, 2, 1, 3).contiguous().view(bs, -1)
            # x = x_features_flip.view(x_features_flip.size(0), -1)
        else:
            x_features_deform = x_features_deform.view(bs, h * w, 2, bins // 2)   # [bs, 16, 2, 8]
            x = x_features_deform.permute(0, 2, 1, 3).contiguous().view(bs, -1)
        # print(x.shape)
        return L2Norm()(x)
        # return x
        # return x.view(bs, h, w, bins).permute(0, 3, 1, 2)

class HardNet_fast_256_double(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=True):
        super(HardNet_fast_256_double, self).__init__()
        self.train_flag = train_flag

        # double the middle model
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=2),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
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
    
    # def forward(self, input):
    #     x_features = self.features(self.input_norm(input))
    #     # print('x_features.shape: ', x_features.shape)
    #     x = x_features.view(x_features.size(0), -1)
       
    #     # x_features = self.input_norm(input)
    #     # # print('x_features.shape: ', x_features.shape)
    #     # x = x_features.view(x_features.size(0), -1)
    #     return L2Norm()(x)
    def forward(self, input, flip_flag=False):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        bs, bins, h, w = x_features.shape

        x_features_deform = x_features.permute(0, 2, 3, 1).contiguous().view(bs, -1, bins)
        if flip_flag:
            x_features_flip = torch.flip(x_features_deform, dims=[1])
            x_features_flip = x_features_flip.view(bs, h * w, 2, bins // 2)   # [bs, 16, 2, 8]
            x = x_features_flip.permute(0, 2, 1, 3).contiguous().view(bs, -1)
            # x = x_features_flip.view(x_features_flip.size(0), -1)
        else:
            x_features_deform = x_features_deform.view(bs, h * w, 2, bins // 2)   # [bs, 16, 2, 8]
            x = x_features_deform.permute(0, 2, 1, 3).contiguous().view(bs, -1)
        # print(x.shape)
        return L2Norm()(x)
        # return x
        # return x.view(bs, h, w, bins).permute(0, 3, 1, 2)

class SELayer(nn.Module):
    def __init__(self, in_channel=4, output_channel=2, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, output_channel, bias=False),
            nn.Softmax()
        )

        for module in self.fc:
            if isinstance(module, nn.Linear):
                c2_xavier_fill(module)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, -1, 1, 1)
        return y

class HardNet_fast_256_double_MSAD_2T(nn.Module):
    """
    需要2 teacher网络(HardNet_fast_256_double),单独训练第三个teacher(+CFF)
    """
    def __init__(self, train_flag=True):
        super(HardNet_fast_256_double_MSAD_2T, self).__init__()
        self.train_flag = train_flag

        # double the middle model
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=2),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1),
            nn.Conv2d(128, 16, kernel_size=1, padding=0)
        )
        self.SE1 = SELayer(32, reduction=4)
        self.SE2 = SELayer(64, reduction=4)
        self.SE3 = SELayer(128, reduction=4)
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input, hr_features:dict=None, lr_features:dict=None, flip_flag:bool=False):
        
        feature_name = ['F0', 'F1', 'F2']
        fr_features = []
        if self.train_flag:
            if hr_features is not None and lr_features is not None:
                hr_features = [hr_features[f] for f in feature_name]
                lr_features = [lr_features[f] for f in feature_name]
                SEs = [self.SE1, self.SE2, self.SE3]
                for hr_feature, lr_feature, SE in zip(hr_features, lr_features, SEs):
                    fr_feature = torch.cat([hr_feature, lr_feature], dim=1)
                    # SE = SELayer(fr_feature.shape[1], reduction=4)
                    fusion_score = SE(fr_feature)
                    fr_features.append(fusion_score[:,0].unsqueeze(-1) * hr_feature + fusion_score[:,1].unsqueeze(-1) * lr_feature) 

        out = {}
        # feature0
        x = self.features[0](input) + 0.5 * fr_features[0] if len(fr_features) != 0 else self.features[0](input)
        out["F0"] = x
        # feature1
        x = self.features[1](x) + 0.5 * fr_features[1] if len(fr_features) != 0 else self.features[1](x)
        out["F1"] = x
        # feature2
        x = self.features[2](x) + 0.5 * fr_features[2] if len(fr_features) != 0 else self.features[2](x)
        out["F2"] = x
        # feature3
        x = self.features[3](x)
        out["F3"] = x
        # feature4
        x = self.features[4](x)
        out["F4"] = x
        # feature5
        x = self.features[5](x)
        out["F5"] = x
        # feature6
        x = self.features[6](x)
        # out["F6"] = x
        
        bs, bins, h, w = x.shape

        x_features_deform = x.permute(0, 2, 3, 1).contiguous().view(bs, -1, bins)
        if flip_flag:
            x_features_flip = torch.flip(x_features_deform, dims=[1])
            x_features_flip = x_features_flip.view(bs, h * w, 2, bins // 2)   # [bs, 16, 2, 8]
            x = x_features_flip.permute(0, 2, 1, 3).contiguous().view(bs, -1)
            # x = x_features_flip.view(x_features_flip.size(0), -1)
        else:
            x_features_deform = x_features_deform.view(bs, h * w, 2, bins // 2)   # [bs, 16, 2, 8]
            x = x_features_deform.permute(0, 2, 1, 3).contiguous().view(bs, -1)
        # print(x.shape)

        out["put"] = L2Norm()(x)

        return out["put"]

class HardNet_fast_256_double_MSAD(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=True):
        super(HardNet_fast_256_double_MSAD, self).__init__()
        self.train_flag = train_flag

        # double the middle model
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=2),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1),
            nn.Conv2d(128, 16, kernel_size=1, padding=0)
        )
        # self.SE1 = SELayer(32, reduction=4)
        # self.SE2 = SELayer(64, reduction=4)
        # self.SE3 = SELayer(128, reduction=4)
        self.SE = SELayer(128, reduction=4)
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def feature_collect(self, x):
        out = {}
        # feature0
        y = self.features[0](x)
        out["F0"] = y
        # feature1
        y = self.features[1](y)
        out["F1"] = y
        # feature2
        y = self.features[2](y)
        out["F2"] = y

        return out

    def forward(self, input, flip_flag:bool=False):
        '''
        @input: dict['hr_img': xx, 'lr_img': xx]
        '''
        fr_features = []
        feature_name = ['F0', 'F1', 'F2']
        
        if self.train_flag:
            if input['hr_img'] is not None:
                hr_features = self.feature_collect(input['hr_img'])     # dict
                lr_features = self.feature_collect(input['lr_img'])

                # 3layers
                # hr_features = [hr_features[f] for f in feature_name]    # list
                # lr_features = [lr_features[f] for f in feature_name]
                # SEs = [self.SE1, self.SE2, self.SE3]
                # for hr_feature, lr_feature, SE in zip(hr_features, lr_features, SEs):
                #     fr_feature = torch.cat([hr_feature, lr_feature], dim=1)
                #     fusion_score = SE(fr_feature)
                #     fr_features.append(fusion_score[:,0].unsqueeze(-1) * hr_feature + fusion_score[:,1].unsqueeze(-1) * lr_feature) 

                fr_feature = torch.cat([hr_features['F2'], lr_features['F2']], dim=1)
                fusion_score = self.SE(fr_feature)
                fr_features.append(fusion_score[:,0].unsqueeze(-1) * hr_features['F2'] + fusion_score[:,1].unsqueeze(-1) * lr_features['F2'])
            else:   # 针对mesh
                fea = self.feature_collect(input['lr_img'])
                fr_features.append(fea['F2'])
        else:
            fea = self.feature_collect(input)
            fr_features.append(fea['F2'])

        out = {}
        # feature3
        x = self.features[3](fr_features[0])
        out["F3"] = x
        # feature4
        x = self.features[4](x)
        out["F4"] = x
        # feature5
        x = self.features[5](x)
        out["F5"] = x
        # feature6
        x = self.features[6](x)
        # out["F6"] = x
        
        bs, bins, h, w = x.shape

        x_features_deform = x.permute(0, 2, 3, 1).contiguous().view(bs, -1, bins)
        if flip_flag:
            x_features_flip = torch.flip(x_features_deform, dims=[1])
            x_features_flip = x_features_flip.view(bs, h * w, 2, bins // 2)   # [bs, 16, 2, 8]
            x = x_features_flip.permute(0, 2, 1, 3).contiguous().view(bs, -1)
            # x = x_features_flip.view(x_features_flip.size(0), -1)
        else:
            x_features_deform = x_features_deform.view(bs, h * w, 2, bins // 2)   # [bs, 16, 2, 8]
            x = x_features_deform.permute(0, 2, 1, 3).contiguous().view(bs, -1)
        # print(x.shape)

        out["put"] = L2Norm()(x)

        return out["put"]
class HardNet_fast_128_double_rect(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=True):
        super(HardNet_fast_128_double_rect, self).__init__()
        self.train_flag = train_flag

        # double the middle model
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=2),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1),
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
    
    def forward(self, input, flip_flag=False):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        bs, bins, h, w = x_features.shape

        if flip_flag:
            x_features_flip = x_features.permute(0, 2, 3, 1).contiguous().view(bs, -1, bins)
            x_features_flip = torch.flip(x_features_flip, dims=[1])
            x = x_features_flip.view(x_features_flip.size(0), -1)
        else:
            x = x_features.permute(0, 2, 3, 1).contiguous().view(x_features.size(0), -1)
        # print(x.shape)
        return L2Norm()(x)
        # return x
        # return x.view(bs, h, w, bins).permute(0, 3, 1, 2)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # nn.init.orthogonal(m.weight.data, gain=0.6)
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return

if __name__ == '__main__':
    # net = HardNet_fast_256(train_flag=True)
    net = HardNet_fast_256_Overparam(train_flag=True)
    # from torchstat import stat
    net.eval()
    # x = torch.rand(16,1,16,16)
    # net(x)
    # from torchsummary import summary
    # summary(x, input_size=(1, 136, 32))
    print('# net parameters:', sum(param.numel() for param in net.parameters()))
    print('# net parameters memory:', sum(param.numel()*4/1024 for param in net.parameters()), 'kb')

    import copy
    model = copy.deepcopy(net)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize() # 不需要指定推理开关：inference_mode， 当调用reparameterize方法时就会执行重参数化操作
    print('# After reparameterize net parameters:', sum(param.numel() for param in model.parameters()))
    print('# After reparameterize net parameters memory:', sum(param.numel()*4/1024 for param in model.parameters()), 'kb')

    # torchstat
    from torchstat import stat
    stat(net, (1, 16, 16))

    # torchsummary
    # from torchsummary import summary
    # summary(net.cuda(), (1, 16, 16), device="cuda")

    # torchinfo
    # from torchinfo import summary
    # summary(net, (1, 1, 16, 16))

