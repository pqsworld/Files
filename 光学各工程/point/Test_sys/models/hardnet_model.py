from this import d
from simplejson import OrderedDict
from typing import Tuple
#from e2cnn import gspaces
#from e2cnn import nn as enn
#from .utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
# from models.PTQ.layers import QConv2d, QAct, QLinear, QSequential, QBlock, QSeModule
# from models.PTQ import ptq_config
import numpy as np

# from models.hardnet_model import Block_Deform


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, groups=1, modulation=False):
        """
        Args:
            moduleation(bool, optional): If True, Modulated Defromable Convolution(Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias, groups=groups)
        
        # self.p_conv偏置层，学习公式（2）中的偏移量。
        # 2*kernel_size*kernel_size：代表了卷积核中所有元素的偏移坐标，因为同时存在x和y的偏移，故要乘以2。
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)

        # nn.init.constant_(self.p_conv.weight, 0)
        # # register_backward_hook是为了方便查看这几层学出来的结果，对网络结构无影响。
        # self.p_conv.register_backward_hook(self._set_lr)
        
        self.modulation = modulation
        if modulation:
            # self.m_conv权重学习层，是后来提出的第二个版本的卷积也就是公式（3）描述的卷积。
            # kernel_size*kernel_size：代表了卷积核中每个元素的权重。
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            # nn.init.constant_(self.m_conv.weight, 0)
            # register_backward_hook是为了方便查看这几层学出来的结果，对网络结构无影响。
            # self.m_conv.register_backward_hook(self._set_lr)
            
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
    
    # 生成卷积核的邻域坐标
    def _get_p_n(self, N, dtype):
        """
        torch.meshgrid():Creates grids of coordinates specified by the 1D inputs in attr:tensors.
        功能是生成网格，可以用于生成坐标。
        函数输入两个数据类型相同的一维张量，两个输出张量的行数为第一个输入张量的元素个数，
        列数为第二个输入张量的元素个数，当两个输入张量数据类型不同或维度不是一维时会报错。
        
        其中第一个输出张量填充第一个输入张量中的元素，各行元素相同；
        第二个输出张量填充第二个输入张量中的元素各列元素相同。
        """
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        
        # p_n ===>offsets_x(kernel_size*kernel_size,) concat offsets_y(kernel_size*kernel_size,) 
        #     ===> (2*kernel_size*kernel_size,)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        # （1， 2*kernel_size*kernel_size, 1, 1）
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n
    
    # 获取卷积核在feature map上所有对应的中心坐标，也就是p0
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # (b, 2*kernel_size, h, w)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0
    
    # 将获取的相对坐标信息与中心坐标相加就获得了卷积核的所有坐标。
    # 再加上之前学习得到的offset后，就是加上了偏移量后的坐标信息。
    # 即对应论文中公式(2)中的(p0+pn+Δpn)
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        # p_n ===> (1, 2*kernel_size*kernel_size, 1, 1)
        p_n = self._get_p_n(N, dtype).to(offset.device)
        # p_0 ===> (1, 2*kernel_size*kernel_size, h, w)
        p_0 = self._get_p_0(h, w, N, dtype).to(offset.device)
        # (1, 2*kernel_size*kernel_size, h, w)
        # print(p_0.device, p_n.device, offset.device)
        p = p_0 + p_n + offset
        return p
    
    def _get_x_q(self, x, q, N):
        # b, h, w, 2*kerel_size*kernel_size
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # x ===> (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        # 因为x是与h轴方向平行，y是与w轴方向平行。故将2D卷积核投影到1D上，位移公式如下：
        # 各个卷积核中心坐标及邻域坐标的索引 offsets_x * w + offsets_y
        # (b, h, w, kernel_size*kernel_size)
        index = q[..., :N] * padded_w + q[..., N:]
        # (b, c, h, w, kernel_size*kernel_size) ===> (b, c, h*w*kernel_size*kernel_size)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        # (b, c, h*w)
        # x_offset[0][0][0] = x[0][0][index[0][0][0]]
        # index[i][j][k]的值应该是一一对应着输入x的(h*w)的坐标，且在之前将index[i][j][k]的值clamp在[0, h]及[0, w]范围里。
        # (b, c, h, w, kernel_size*kernel_size)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset
    
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        # (b, c, h, w, kernel_size*kernel_size)
        b, c, h, w, N = x_offset.size()
        # (b, c, h, w*kernel_size)
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        # (b, c, h*kernel_size, w*kernel_size)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
        
        return x_offset
    
    def forward(self, x):
        # (b, c, h, w) ===> (b, 2*kernel_size*kernel_size, h, w)
        offset = self.p_conv(x)

        if self.modulation:
            # (b, c, h, w) ===> (b, kernel_size*kernel_size, h, w)
            m = torch.sigmoid(self.m_conv(x))
        
        dtype = offset.data.type()
        ks = self.kernel_size
        # kernel_size*kernel_size
        N = offset.size(1) // 2
        
        if self.padding:
            x = self.zero_padding(x)
        # (b, 2*kernel_size*kernel_size, h, w)
        p = self._get_p(offset, dtype)
        # (b, h, w, 2*kernel_size*kernel_size)
        p = p.contiguous().permute(0, 2, 3, 1)
        # 将p从tensor的前向计算中取出来，并向下取整得到左上角坐标q_lt。
        q_lt = p.detach().floor()
        # 将p向上再取整，得到右下角坐标q_rb。
        q_rb = q_lt + 1
        
        # 学习的偏移量是float类型，需要用双线性插值的方法去推算相应的值。
        # 同时防止偏移量太大，超出feature map，故需要torch.clamp来约束。
        # Clamps all elements in input into the range [ min, max ].
        # torch.clamp(a, min=-0.5, max=0.5)
        
        # p左上角x方向的偏移量不超过h,y方向的偏移量不超过w。
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # p右下角x方向的偏移量不超过h,y方向的偏移量不超过w。
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # p左上角的x方向的偏移量和右下角y方向的偏移量组合起来，得到p左下角的值。
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        # p右下角的x方向的偏移量和左上角y方向的偏移量组合起来，得到p右上角的值。
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        
        # clip p。
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
        
        # 双线性插值公式里的四个系数。即bilinear kernel。
        # 作者代码为了保持整齐，每行的变量计算形式一样，所以计算需要做一点对应变量的对应变化。
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        
        # 计算双线性插值的四个坐标对应的像素值。
        # (b, c, h, w, kernel_size*kernel_size)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        
        # 双线性插值的最后计算
        # (b, c, h, w, kernel_size*kernel_size)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                    g_rb.unsqueeze(dim=1) * x_q_rb + \
                    g_lb.unsqueeze(dim=1) * x_q_lb + \
                    g_rt.unsqueeze(dim=1) * x_q_rt
        
        # modulation
        if self.modulation:
            # (b, kernel_size*kernel_size, h, w) ===> (b, h, w, kernel_size*kernel_size)
            m = m.contiguous().permute(0, 2, 3, 1)
            # (b, h, w, kernel_size*kernel_size) ===>  (b, 1, h, w, kernel_size*kernel_size)
            m = m.unsqueeze(dim=1)
            # (b, c, h, w, kernel_size*kernel_size)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        # x_offset: (b, c, h, w, kernel_size*kernel_size)
        # x_offset: (b, c, h*kernel_size, w*kernel_size)
        x_offset = self._reshape_x_offset(x_offset, ks)
        # out: (b, c, h, w)
        out = self.conv(x_offset)
        # print(x_offset.shape, out.shape)
        # exit()
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
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class SeModule_standard(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule_standard, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            nn.Hardsigmoid(inplace=True)
            # hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class SeModule_short(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule_short, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1,
                      stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1,
                      stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


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

        self.reparameterize()
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


class MobileOneBlock_short(nn.Module):
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
        super(MobileOneBlock_short, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SeModule_short(out_channels)
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

        # self.reparameterize()
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


class MobileOne_short(nn.Module):

    def __init__(self,in_planes, planes, stride=1, use_se=False, num_conv_branches=4, inference_mode=False):
        super(MobileOne_short, self).__init__()
        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches
        self.in_planes = in_planes
       
        blocks = []

        blocks.append(MobileOneBlock_short(in_channels=self.in_planes,
                            out_channels=planes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                            inference_mode=self.inference_mode,
                            use_se=False,
                            num_conv_branches=self.num_conv_branches))

        # Depthwise conv
        blocks.append(MobileOneBlock_short(in_channels=planes,
                            out_channels=planes,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            groups=planes,
                            inference_mode=self.inference_mode,
                            use_se=False,
                            num_conv_branches=self.num_conv_branches))
        # Pointwise conv
        blocks.append(MobileOneBlock_short(in_channels=planes,
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


class MobileOne_shortcut(nn.Module):

    def __init__(self, in_planes, planes, stride=1, use_se=False, num_conv_branches=4, inference_mode=False, nolinear=nn.ReLU()):
        super(MobileOne_shortcut, self).__init__()
        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches
        self.in_planes = in_planes
        self.stride = stride
       
        blocks = []

        blocks.append(MobileOneBlock_new(in_channels=self.in_planes,
                            out_channels=planes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                            inference_mode=self.inference_mode,
                            use_se=False,
                            num_conv_branches=self.num_conv_branches,
                            use_nolinear=True,
                            nolinear=nolinear))

        # Depthwise conv
        blocks.append(MobileOneBlock_new(in_channels=planes,
                            out_channels=planes,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            groups=planes,
                            inference_mode=self.inference_mode,
                            use_se=False,
                            num_conv_branches=self.num_conv_branches,
                            use_nolinear=True,
                            nolinear=nolinear))
        # Pointwise conv
        blocks.append(MobileOneBlock_new(in_channels=planes,
                            out_channels=planes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                            inference_mode=self.inference_mode,
                            use_se=use_se,
                            num_conv_branches=self.num_conv_branches,
                            use_nolinear=False))
        # self.in_planes = planes
           
        self.conv = nn.Sequential(*blocks)

        self.shortcut = nn.Sequential()
        # if stride == 1 and in_size != out_size:

        if stride == 1 and in_planes == planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass.  """
        out = self.conv(x)
        # shortcut
        out = out + self.shortcut(x) if (self.stride == 1 and x.shape[1] == out.shape[1]) else out
        return out


class MobileOneBlock_new(nn.Module):
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
                 num_conv_branches: int = 1,
                 use_nolinear: bool = True,
                 nolinear: nn.Module = nn.ReLU()) -> None:
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
        super(MobileOneBlock_new, self).__init__()
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
        
        # Check if nolinear is requested
        if use_nolinear:
            self.activation = nolinear
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
        self.reparameterize()
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

class MobileOneBlock_new_short(nn.Module):
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
                 num_conv_branches: int = 1,
                 use_nolinear: bool = True,
                 nolinear: nn.Module = nn.ReLU()) -> None:
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
        super(MobileOneBlock_new_short, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SeModule_short(out_channels)
        else:
            self.se = nn.Identity()
        
        # Check if nolinear is requested
        if use_nolinear:
            self.activation = nolinear
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
        # self.reparameterize()
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



class MobileOne_new(nn.Module):

    def __init__(self,in_planes, planes, stride=1, use_se=False, num_conv_branches=4, inference_mode=False, nolinear=nn.ReLU()):
        super(MobileOne_new, self).__init__()
        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches
        self.in_planes = in_planes
       
        blocks = []

        blocks.append(MobileOneBlock_new(in_channels=self.in_planes,
                            out_channels=planes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                            inference_mode=self.inference_mode,
                            use_se=False,
                            num_conv_branches=self.num_conv_branches,
                            use_nolinear=True,
                            nolinear=nolinear))

        # Depthwise conv
        blocks.append(MobileOneBlock_new(in_channels=planes,
                            out_channels=planes,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            groups=planes,
                            inference_mode=self.inference_mode,
                            use_se=False,
                            num_conv_branches=self.num_conv_branches,
                            use_nolinear=True,
                            nolinear=nolinear))
        # Pointwise conv
        blocks.append(MobileOneBlock_new(in_channels=planes,
                            out_channels=planes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                            inference_mode=self.inference_mode,
                            use_se=use_se,
                            num_conv_branches=self.num_conv_branches,
                            use_nolinear=False))
        self.in_planes = planes
           
        self.conv = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass.  """
        x = self.conv(x)
        return x

class MobileOne_new_short(nn.Module):

    def __init__(self,in_planes, planes, stride=1, use_se=False, num_conv_branches=4, inference_mode=False, nolinear=nn.ReLU()):
        super(MobileOne_new_short, self).__init__()
        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches
        self.in_planes = in_planes
       
        blocks = []

        blocks.append(MobileOneBlock_new_short(in_channels=self.in_planes,
                            out_channels=planes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                            inference_mode=self.inference_mode,
                            use_se=False,
                            num_conv_branches=self.num_conv_branches,
                            use_nolinear=True,
                            nolinear=nolinear))

        # Depthwise conv
        blocks.append(MobileOneBlock_new_short(in_channels=planes,
                            out_channels=planes,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            groups=planes,
                            inference_mode=self.inference_mode,
                            use_se=False,
                            num_conv_branches=self.num_conv_branches,
                            use_nolinear=True,
                            nolinear=nolinear))
        # Pointwise conv
        blocks.append(MobileOneBlock_new_short(in_channels=planes,
                            out_channels=planes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                            inference_mode=self.inference_mode,
                            use_se=use_se,
                            num_conv_branches=self.num_conv_branches,
                            use_nolinear=False))
        self.in_planes = planes
           
        self.conv = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass.  """
        x = self.conv(x)
        return x


# class ESeModule(enn.EquivariantModule):
#     def __init__(self, in_type: enn.FieldType, in_fiber: str, in_stage: int, F: float = 1., sigma: float = 0.45, fixparams: bool = True, reduction: int = 4):
#         super(ESeModule, self).__init__()
#         self.in_type = in_type
#         self.inner_type = FIBERS[in_fiber](in_type.gspace, in_stage // reduction, fixparams=fixparams)
#         self.out_type = in_type
#         self.se = enn.SequentialModule(
#             enn.PointwiseAdaptiveAvgPool(self.in_type, 1),
#             conv1x1(self.in_type, self.inner_type, 
#                     stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False),
#             # conv1x1(in_size, in_size // reduction, kernel_size=1,
#             #           stride=1, padding=0, bias=False),
#             enn.InnerBatchNorm(self.inner_type),
#             # nn.BatchNorm2d(in_size // reduction),
#             enn.ReLU(self.inner_type, inplace=True),
#             conv1x1(self.inner_type, self.out_type, 
#                     stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False),
#             # nn.Conv2d(in_size // reduction, in_size, kernel_size=1,
#             #           stride=1, padding=0, bias=False),
#             enn.InnerBatchNorm(self.out_type),
#             Ehsigmoid(self.out_type, inplace=True)
#         )

#     def forward(self, x):
#         return x * self.se(x).tensor
    
#     def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
#         assert len(input_shape) == 4
#         assert input_shape[1] == self.in_type.size
    
#         b, _, hi, wi = input.shape

#         return b, self.out_type.size, hi, wi

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

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()
    def forward(self, x):
        return torch.cat((F.relu6(x) / 6, F.relu6(-x) / 6), 1)


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
        # print(self.shortcut(x)[0, 0, :, :], out[0, 0, :, :])
        out = out + self.shortcut(x) if (self.stride == 1 and x.shape[1] == out.shape[1]) else out
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
        out = out + self.shortcut(x) if (self.stride == 1 and x.shape[1] == out.shape[1]) else out
        return out

class Block_short(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_short, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=True)
        # self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()
        # if stride == 1 and in_size != out_size:
        if stride == 1 and in_size == out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=True),
                # nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.conv1(x))
        out = self.nolinear2(self.conv2(out))
        out = self.conv3(out)
        if self.se != None:
            out = self.se(out)
        # out = out + self.shortcut(x) if self.stride == 1 else out
        out = out + self.shortcut(x) if (self.stride == 1 and x.shape[1] == out.shape[1]) else out
        return out

class Block_noshortcut(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_noshortcut, self).__init__()
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

        # if stride == 1 and in_size == out_size:
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
        out = out + self.shortcut(x) if (self.stride == 1 and x.shape[1] == out.shape[1]) else out
        return out

class Block_noshortcut_short(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_noshortcut_short, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=True)
        # self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()
        # if stride == 1 and in_size != out_size:
        # if stride == 1 and in_size == out_size:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_size, out_size, kernel_size=1,
        #                   stride=1, padding=0, bias=True),
        #         # nn.BatchNorm2d(out_size),
        #     )

    def forward(self, x):
        out = self.nolinear1(self.conv1(x))
        out = self.nolinear2(self.conv2(out))
        out = self.conv3(out)
        if self.se != None:
            out = self.se(out)
        # out = out + self.shortcut(x) if self.stride == 1 else out
        out = out + self.shortcut(x) if (self.stride == 1 and x.shape[1] == out.shape[1]) else out
        return out


class Block_Pconv(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_Pconv, self).__init__()
        self.stride = stride

        # self.bn1 = nn.BatchNorm2d(expand_size)
        # self.nolinear1 = nolinear
        self.conv1 = Partial_conv3(in_size, 4, 'split_cat')
        self.conv2 = nn.Conv2d(in_size, expand_size,
                        kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()
        # if stride == 1 and in_size != out_size:
        if stride == 1 and in_size == out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if (self.stride == 1 and x.shape[1] == out.shape[1]) else out
        return out


class Block_Deform(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_Deform, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        # self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
        #                        stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)

        self.conv2 = DeformConv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)

        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()
        # if stride == 1 and in_size != out_size:
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
        out = out + self.shortcut(x) if (self.stride == 1 and x.shape[1] == out.shape[1]) else out
        return out

# class EConstantPad2d(enn.EquivariantModule):
#     def __init__(self, in_type, padding: Tuple = None, value: float = None):
#         super(EConstantPad2d, self).__init__()
        
#         self.in_type = in_type
#         self.out_type = in_type
#         self.padding = padding
#         self.value = value

#     def forward(self, input: enn.GeometricTensor) -> enn.GeometricTensor:
#         assert input.type == self.in_type, "Error! the type of the input does not match the input type of this module"
#         return enn.GeometricTensor(F.pad(input.tensor, self.padding, 'constant', self.value), self.out_type)

#     def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
#         assert len(input_shape) == 4
#         assert input_shape[1] == self.in_type.size
    
#         b, _, hi, wi = input.shape
#         ho = hi + self.padding[2] + self.padding[3]
#         wo = wi + self.padding[0] + self.padding[1]

#         return b, self.out_type.size, ho, wo

# class EWHSA(enn.EquivariantModule):
#     def __init__(self, in_type, win_size: int=4):
#         super(EWHSA, self).__init__()
        
#         self.win_size = win_size
#         self.out_type = in_type

#     def forward(self, q: enn.GeometricTensor, k: enn.GeometricTensor, v: enn.GeometricTensor) -> enn.GeometricTensor:
#         _, qc, qhi, qwi = q.tensor.shape
#         assert qhi % self.win_size == 0 and qwi % self.win_size == 0, "Error! Q:H x W is not win_size multiple!"
#         _, kc, khi, kwi = k.tensor.shape
#         assert khi % self.win_size == 0 and kwi % self.win_size == 0, "Error! K:H x W is not win_size multiple!"
#         _, vc, vhi, vwi = v.tensor.shape
#         assert vhi % self.win_size == 0 and vwi % self.win_size == 0, "Error! V:H x W is not win_size multiple!"
#         # Partition
#         q_win = q.tensor.reshape(-1, qc, qhi // self.win_size, self.win_size, qwi // self.win_size, self.win_size).transpose(3, 4).reshape(-1, qc, qhi // self.win_size * qwi // self.win_size, self.win_size*self.win_size)
#         k_win = k.tensor.reshape(-1, kc, khi // self.win_size, self.win_size, kwi // self.win_size, self.win_size).transpose(3, 4).reshape(-1, kc, khi // self.win_size * kwi // self.win_size, self.win_size*self.win_size)
#         v_win = v.tensor.reshape(-1, vc, vhi // self.win_size, self.win_size, vwi // self.win_size, self.win_size).transpose(3, 4).reshape(-1, vc, vhi // self.win_size * vwi // self.win_size, self.win_size*self.win_size)
#         # qk_win = torch.einsum('bdhn,bdhm->bhnm', q_win, k_win)
#         qk_win = torch.einsum('bdhn,bdhm->bhnm', q_win, k_win) / qc**.5
#         scores = F.softmax(qk_win, dim=-1)
#         outV = torch.einsum('bhnm,bdhm->bdhn', scores, v_win)
#         out = outV.reshape(-1, vc, vhi // self.win_size, vwi // self.win_size, self.win_size, self.win_size).transpose(3, 4).reshape(-1, vc, vhi, vwi)
#         return enn.GeometricTensor(out[:, :, :-3, :], self.out_type)

#     def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
#         assert len(input_shape) == 4
#         assert input_shape[1] == self.in_type.size
    
#         b, c, hi, wi = input.shape

#         return b, c, hi, wi

# class EWHSA_HiLo(enn.EquivariantModule):
#     def __init__(self, in_type, win_size_Hi: int=4, win_size_Lo: int=4):
#         super(EWHSA_HiLo, self).__init__()
        
#         self.win_size_Hi = win_size_Hi
#         self.win_size_Lo = win_size_Lo
#         self.out_type = in_type
#         self.channel_ratio = 0.5
#         self.avgpoolK = nn.AvgPool2d(kernel_size=self.win_size_Lo, stride=self.win_size_Lo, padding=0)
#         self.avgpoolV = nn.AvgPool2d(kernel_size=self.win_size_Lo, stride=self.win_size_Lo, padding=0)

#     def forward(self, q: enn.GeometricTensor, k: enn.GeometricTensor, v: enn.GeometricTensor) -> enn.GeometricTensor:
#         qb, qc, qhi, qwi = q.tensor.shape
#         assert qhi % self.win_size_Hi == 0 and qwi % self.win_size_Hi == 0, "Error! Q:H x W is not win_size multiple!"
#         kb, kc, khi, kwi = k.tensor.shape
#         assert khi % self.win_size_Hi == 0 and kwi % self.win_size_Hi == 0, "Error! K:H x W is not win_size multiple!"
#         vb, vc, vhi, vwi = v.tensor.shape
#         assert vhi % self.win_size_Hi == 0 and vwi % self.win_size_Hi == 0, "Error! V:H x W is not win_size multiple!"
        
#         high_qc = int(qc * self.channel_ratio)
#         high_kc = int(kc * self.channel_ratio)
#         high_vc = int(vc * self.channel_ratio)
#         # High Frequency
#         # Partition
#         q_win = q.tensor[:, :high_qc, :, :].reshape(qb, -1, qhi // self.win_size_Hi, self.win_size_Hi, qwi // self.win_size_Hi, self.win_size_Hi).transpose(3, 4).reshape(qb, -1, qhi // self.win_size_Hi * qwi // self.win_size_Hi, self.win_size_Hi*self.win_size_Hi)
#         k_win = k.tensor[:, :high_kc, :, :].reshape(kb, -1, khi // self.win_size_Hi, self.win_size_Hi, kwi // self.win_size_Hi, self.win_size_Hi).transpose(3, 4).reshape(kb, -1, khi // self.win_size_Hi * kwi // self.win_size_Hi, self.win_size_Hi*self.win_size_Hi)
#         v_win = v.tensor[:, :high_vc, :, :].reshape(vb, -1, vhi // self.win_size_Hi, self.win_size_Hi, vwi // self.win_size_Hi, self.win_size_Hi).transpose(3, 4).reshape(vb, -1, vhi // self.win_size_Hi * vwi // self.win_size_Hi, self.win_size_Hi*self.win_size_Hi)
#         qk_win = torch.einsum('bdhn,bdhm->bhnm', q_win, k_win) / high_qc**.5
#         scores = F.softmax(qk_win, dim=-1)
#         outV = torch.einsum('bhnm,bdhm->bdhn', scores, v_win)
#         high_out = outV.reshape(-1, high_vc, vhi // self.win_size_Hi, vwi // self.win_size_Hi, self.win_size_Hi, self.win_size_Hi).transpose(3, 4).reshape(-1, high_vc, vhi, vwi)
        
#         # Low Frequency
#         low_q = q.tensor[:, high_qc:, :, :].reshape(qb, qc - high_qc, -1)  # [b, 16, H, W]
#         low_k = self.avgpoolK(k.tensor[:, high_kc:, :, :]).reshape(kb, kc - high_kc, -1) # [b, 16, H/ks, W/ks]
#         low_v = self.avgpoolV(v.tensor[:, high_vc:, :, :]).reshape(vb, vc - high_vc, -1) # [b, 32, H/ks, W/ks]
#         low_qk_win = torch.einsum('bdn,bdm->bnm', low_q, low_k) / (qc - high_qc)**.5
#         low_scores = F.softmax(low_qk_win, dim=-1)
#         low_outV = torch.einsum('bnm,bdm->bdn', low_scores, low_v)
#         low_out = low_outV.reshape(-1, vc - high_vc, vhi, vwi)
#         out = torch.cat((high_out, low_out), dim=1)
        
#         return enn.GeometricTensor(out[:, :, :-3, :], self.out_type)
    
#     def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
#         assert len(input_shape) == 4
#         assert input_shape[1] == self.in_type.size
    
#         b, c, hi, wi = input.shape

#         return b, c, hi, wi


# class EBlock(enn.EquivariantModule):
#     '''expand + depthwise + pointwise'''

#     def __init__(self, kernel_size, in_fiber, inner_fiber, inner_stages, out_fiber, stride, semodule_reduce: int = 0,
#                  F: float = 1., sigma: float = 0.45, fixparams: bool = True):
#         super(EBlock, self).__init__()
#         self.stride = stride
#         self.kernel_size = kernel_size
#         self.in_type = in_fiber
#         self.out_type = out_fiber
#         self.semodule_reduce = semodule_reduce
#         self.inner_type = [FIBERS[inner_fiber](self.in_type.gspace, p, fixparams=fixparams) for p in inner_stages]

#         self.conv1 = conv1x1(self.in_type, self.inner_type[0], 
#                              stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False)

#         self.bn1 = enn.InnerBatchNorm(self.inner_type[0])
#         self.relu1 = enn.ReLU(self.inner_type[0], inplace=True)

#         # self.bn1 = nn.BatchNorm2d(expand_size)
#         # self.nolinear1 = nolinear
#         if kernel_size == 3:
#             conv = conv3x3
#         else:
#             conv = conv5x5
#         self.conv2 = conv(self.inner_type[0], self.inner_type[1], 
#                           stride=stride, padding=kernel_size // 2, groups=len(self.inner_type[0].representations), bias=False, sigma=sigma, F=F, initialize=False)
#         # nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
#         #                        stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)

#         self.bn2 = enn.InnerBatchNorm(self.inner_type[1])
#         self.relu2 = enn.ReLU(self.inner_type[1], inplace=True)

#         # self.bn2 = nn.BatchNorm2d(expand_size)
#         # self.nolinear2 = nolinear
        
#         self.conv3 = conv1x1(self.inner_type[1], self.out_type, 
#                              stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False)

#         # self.conv3 = nn.Conv2d(expand_size, out_size,
#         #                        kernel_size=1, stride=1, padding=0, bias=False)

#         self.bn3 = enn.InnerBatchNorm(self.out_type)

#         # self.bn3 = nn.BatchNorm2d(out_size)
    
#         if semodule_reduce > 0:
#             self.se = ESeModule(self.out_type, inner_fiber, inner_stages[-1], F, sigma, fixparams, semodule_reduce)
#         # self.se = semodule

#         self.shortcut = enn.SequentialModule()

#         # if stride != 1 or self.in_type != self.out_type:
#         if stride == 1 and self.in_type == self.out_type:
#             self.shortcut = enn.SequentialModule(
#                 conv1x1(self.in_type, self.out_type,
#                         stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False),
#                 enn.InnerBatchNorm(self.out_type)
#             )

#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(x)))
#         out = self.relu2(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         if self.semodule_reduce > 0:
#             out = self.se(out)
#         out = out + self.shortcut(x) if self.stride == 1 and self.in_type == self.out_type else out
#         return out
    
#     def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
#         assert len(input_shape) == 4
#         assert input_shape[1] == self.in_type.size
    
#         b, c, hi, wi = input.shape
#         ho = math.floor((hi + 2 * (self.kernel_size // 2) - (self.kernel_size - 1) - 1) / self.stride + 1)
#         wo = math.floor((wi + 2 * (self.kernel_size // 2) - (self.kernel_size - 1) - 1) / self.stride + 1)

#         if self.stride == 1 and self.in_type == self.out_type:
#             return self.shortcut.evaluate_output_shape(input_shape)
#         else:
#             return b, self.out_type.size, ho, wo

# class EBlock_SA(enn.EquivariantModule):
#     '''expand + depthwise + pointwise'''

#     def __init__(self, kernel_size, in_fiber, inner_fiber, inner_stages, out_fiber, stride, semodule_reduce: int = 0,
#                  F: float = 1., sigma: float = 0.45, fixparams: bool = True, win_size: int=4):
#         super(EBlock_SA, self).__init__()
#         self.stride = stride
#         self.kernel_size = kernel_size
#         self.in_type = in_fiber
#         self.out_type = out_fiber
#         self.semodule_reduce = semodule_reduce
#         self.win_size = win_size
#         self.inner_type = [FIBERS[inner_fiber](self.in_type.gspace, p, fixparams=fixparams) for p in inner_stages]

#         self.conv1 = conv1x1(self.in_type, self.inner_type[0], 
#                              stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False)

#         self.bn1 = enn.InnerBatchNorm(self.inner_type[0])
#         self.relu1 = enn.ReLU(self.inner_type[0], inplace=True)

#         self.padQ = EConstantPad2d(self.in_type, [0, 0, 0, 3], 0)
#         self.padV = EConstantPad2d(self.inner_type[0], [0, 0, 0, 3], 0)
#         self.wmhsa = EWHSA(self.inner_type[0], self.win_size)


#         # self.bn1 = nn.BatchNorm2d(expand_size)
#         # self.nolinear1 = nolinear
#         if kernel_size == 3:
#             conv = conv3x3
#         else:
#             conv = conv5x5
#         self.conv2 = conv(self.inner_type[0], self.inner_type[1], 
#                           stride=stride, padding=kernel_size // 2, groups=len(self.inner_type[0].representations), bias=False, sigma=sigma, F=F, initialize=False)
#         # nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
#         #                        stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)

#         self.bn2 = enn.InnerBatchNorm(self.inner_type[1])
#         self.relu2 = enn.ReLU(self.inner_type[1], inplace=True)

#         # self.bn2 = nn.BatchNorm2d(expand_size)
#         # self.nolinear2 = nolinear
        
#         self.conv3 = conv1x1(self.inner_type[1], self.out_type, 
#                              stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False)

#         # self.conv3 = nn.Conv2d(expand_size, out_size,
#         #                        kernel_size=1, stride=1, padding=0, bias=False)

#         self.bn3 = enn.InnerBatchNorm(self.out_type)

#         # self.bn3 = nn.BatchNorm2d(out_size)
    
#         if semodule_reduce > 0:
#             self.se = ESeModule(self.out_type, inner_fiber, inner_stages[-1], F, sigma, fixparams, semodule_reduce)
#         # self.se = semodule

#         self.shortcut = enn.SequentialModule()

#         # if stride != 1 or self.in_type != self.out_type:
#         if stride == 1 and self.in_type == self.out_type:
#             self.shortcut = enn.SequentialModule(
#                 conv1x1(self.in_type, self.out_type,
#                         stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False),
#                 enn.InnerBatchNorm(self.out_type)
#             )

#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(x)))
#         # WMHSA
#         x_pad = self.padQ(x)
#         out_pad = self.padV(out)
#         out1 = self.wmhsa(x_pad, x_pad, out_pad)
#         out = self.relu2(self.bn2(self.conv2(out1)))
#         # skip
#         out = out + out1
#         out = self.bn3(self.conv3(out))
#         if self.semodule_reduce > 0:
#             out = self.se(out)
#         out = out + self.shortcut(x) if self.stride == 1 and self.in_type == self.out_type else out
#         return out
    
#     def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
#         assert len(input_shape) == 4
#         assert input_shape[1] == self.in_type.size
    
#         b, c, hi, wi = input.shape
#         ho = math.floor((hi + 2 * (self.kernel_size // 2) - (self.kernel_size - 1) - 1) / self.stride + 1)
#         wo = math.floor((wi + 2 * (self.kernel_size // 2) - (self.kernel_size - 1) - 1) / self.stride + 1)

#         if self.stride == 1 and self.in_type == self.out_type:
#             return self.shortcut.evaluate_output_shape(input_shape)
#         else:
#             return b, self.out_type.size, ho, wo


# class EBlock_HiLo_SA(enn.EquivariantModule):
#     '''expand + depthwise + pointwise'''

#     def __init__(self, kernel_size, in_fiber, inner_fiber, inner_stages, out_fiber, stride, semodule_reduce: int = 0,
#                  F: float = 1., sigma: float = 0.45, fixparams: bool = True, win_size_Hi: int=4, win_size_Lo: int=4):
#         super(EBlock_HiLo_SA, self).__init__()
#         self.stride = stride
#         self.kernel_size = kernel_size
#         self.in_type = in_fiber
#         self.out_type = out_fiber
#         self.semodule_reduce = semodule_reduce
#         self.win_size_Hi = win_size_Hi
#         self.win_size_Lo = win_size_Lo
#         self.inner_type = [FIBERS[inner_fiber](self.in_type.gspace, p, fixparams=fixparams) for p in inner_stages]

#         self.conv1 = conv1x1(self.in_type, self.inner_type[0], 
#                              stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False)

#         self.bn1 = enn.InnerBatchNorm(self.inner_type[0])
#         self.relu1 = enn.ReLU(self.inner_type[0], inplace=True)

#         self.padQ = EConstantPad2d(self.in_type, [0, 0, 0, 3], 0)
#         self.padV = EConstantPad2d(self.inner_type[0], [0, 0, 0, 3], 0)
#         self.wmhsa = EWHSA_HiLo(self.inner_type[0], self.win_size_Hi, self.win_size_Lo)


#         # self.bn1 = nn.BatchNorm2d(expand_size)
#         # self.nolinear1 = nolinear
#         if kernel_size == 3:
#             conv = conv3x3
#         else:
#             conv = conv5x5
#         self.conv2 = conv(self.inner_type[0], self.inner_type[1], 
#                           stride=stride, padding=kernel_size // 2, groups=len(self.inner_type[0].representations), bias=False, sigma=sigma, F=F, initialize=False)
#         # nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
#         #                        stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)

#         self.bn2 = enn.InnerBatchNorm(self.inner_type[1])
#         self.relu2 = enn.ReLU(self.inner_type[1], inplace=True)

#         # self.bn2 = nn.BatchNorm2d(expand_size)
#         # self.nolinear2 = nolinear
        
#         self.conv3 = conv1x1(self.inner_type[1], self.out_type, 
#                              stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False)

#         # self.conv3 = nn.Conv2d(expand_size, out_size,
#         #                        kernel_size=1, stride=1, padding=0, bias=False)

#         self.bn3 = enn.InnerBatchNorm(self.out_type)

#         # self.bn3 = nn.BatchNorm2d(out_size)
    
#         if semodule_reduce > 0:
#             self.se = ESeModule(self.out_type, inner_fiber, inner_stages[-1], F, sigma, fixparams, semodule_reduce)
#         # self.se = semodule

#         self.shortcut = enn.SequentialModule()

#         # if stride != 1 or self.in_type != self.out_type:
#         if stride == 1 and self.in_type == self.out_type:
#             self.shortcut = enn.SequentialModule(
#                 conv1x1(self.in_type, self.out_type,
#                         stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False),
#                 enn.InnerBatchNorm(self.out_type)
#             )

#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(x)))
#         # WMHSA
#         x_pad = self.padQ(x)
#         out_pad = self.padV(out)

#         out1 = self.wmhsa(x_pad, x_pad, out_pad)
#         out = self.relu2(self.bn2(self.conv2(out1)))
#         # skip
#         out = out + out1
#         out = self.bn3(self.conv3(out))
#         if self.semodule_reduce > 0:
#             out = self.se(out)
#         out = out + self.shortcut(x) if self.stride == 1 and self.in_type == self.out_type else out
#         return out
    
#     def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
#         assert len(input_shape) == 4
#         assert input_shape[1] == self.in_type.size
    
#         b, c, hi, wi = input.shape
#         ho = math.floor((hi + 2 * (self.kernel_size // 2) - (self.kernel_size - 1) - 1) / self.stride + 1)
#         wo = math.floor((wi + 2 * (self.kernel_size // 2) - (self.kernel_size - 1) - 1) / self.stride + 1)

#         if self.stride == 1 and self.in_type == self.out_type:
#             return self.shortcut.evaluate_output_shape(input_shape)
#         else:
#             return b, self.out_type.size, ho, wo


# class EBlock_Branch(enn.EquivariantModule):
#     '''expand + depthwise + pointwise'''

#     def __init__(self, kernel_size, in_fiber, inner_fiber, inner_stages, out_fiber, stride, semodule_reduce: int = 0,
#                  F: float = 1., sigma: float = 0.45, fixparams: bool = True):
#         super(EBlock_Branch, self).__init__()
#         self.stride = stride
#         self.kernel_size = kernel_size
#         self.in_type = in_fiber
#         self.out_type = out_fiber
#         self.semodule_reduce = semodule_reduce
#         self.inner_type = [FIBERS[inner_fiber](self.in_type.gspace, p, fixparams=fixparams) for p in inner_stages]

#         self.conv1 = conv1x1(self.in_type, self.inner_type[0], 
#                              stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False)

#         self.bn1 = enn.InnerBatchNorm(self.inner_type[0])
#         self.relu1 = enn.ReLU(self.inner_type[0], inplace=True)

#         # self.bn1 = nn.BatchNorm2d(expand_size)
#         # self.nolinear1 = nolinear
#         # if kernel_size == 3:
#         #     conv = conv3x3
#         # else:
#         #     conv = conv5x5
#         self.conv2_1 = conv3x3(self.inner_type[0], self.inner_type[1], 
#                     stride=stride, padding=kernel_size // 2, groups=len(self.inner_type[0].representations), bias=False, sigma=sigma, F=F, initialize=False)
#         self.conv2_2 = conv5x5(self.inner_type[0], self.inner_type[1], 
#                     stride=stride, padding=kernel_size // 2, groups=len(self.inner_type[0].representations), bias=False, sigma=sigma, F=F, initialize=False)
#         # nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
#         #                        stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)

#         self.bn2 = enn.InnerBatchNorm(self.inner_type[1])
#         self.relu2 = enn.ReLU(self.inner_type[1], inplace=True)

#         # self.bn2 = nn.BatchNorm2d(expand_size)
#         # self.nolinear2 = nolinear
        
#         self.conv3 = conv1x1(self.inner_type[1], self.out_type, 
#                              stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False)

#         # self.conv3 = nn.Conv2d(expand_size, out_size,
#         #                        kernel_size=1, stride=1, padding=0, bias=False)

#         self.bn3 = enn.InnerBatchNorm(self.out_type)

#         # self.bn3 = nn.BatchNorm2d(out_size)
    
#         if semodule_reduce > 0:
#             self.se = ESeModule(self.out_type, inner_fiber, inner_stages[-1], F, sigma, fixparams, semodule_reduce)
#         # self.se = semodule

#         self.shortcut = enn.SequentialModule()

#         # if stride != 1 or self.in_type != self.out_type:
#         if stride == 1 and self.in_type == self.out_type:
#             self.shortcut = enn.SequentialModule(
#                 conv1x1(self.in_type, self.out_type,
#                         stride=1, padding=0, bias=False, sigma=sigma, F=F, initialize=False),
#                 enn.InnerBatchNorm(self.out_type)
#             )

#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(x)))
#         out = self.relu2(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         if self.semodule_reduce > 0:
#             out = self.se(out)
#         out = out + self.shortcut(x) if self.stride == 1 and self.in_type == self.out_type else out
#         return out
    
#     def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
#         assert len(input_shape) == 4
#         assert input_shape[1] == self.in_type.size
    
#         b, c, hi, wi = input.shape
#         ho = math.floor((hi + 2 * (self.kernel_size // 2) - (self.kernel_size - 1) - 1) / self.stride + 1)
#         wo = math.floor((wi + 2 * (self.kernel_size // 2) - (self.kernel_size - 1) - 1) / self.stride + 1)

#         if self.stride == 1 and self.in_type == self.out_type:
#             return self.shortcut.evaluate_output_shape(input_shape)
#         else:
#             return b, self.out_type.size, ho, wo


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

# class Ehsigmoid(enn.ReLU):
#     def __init__(self, in_type: enn.FieldType, inplace: bool = False):
       
#         super(Ehsigmoid, self).__init__(in_type, inplace)

#     def forward(self, input: enn.GeometricTensor) -> enn.GeometricTensor:    
#         assert input.type == self.in_type, "Error! the type of the input does not match the input type of this module"
#         return enn.GeometricTensor(F.relu6(input.tensor + 3, inplace=self._inplace) / 6, self.out_type)

#     def export(self):
#         r"""
#         Export this module to a normal PyTorch :class:`torch.nn.ReLU` module and set to "eval" mode.

#         """
    
#         self.eval()
    
#         return hsigmoid()

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
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

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # nn.init.orthogonal(m.weight.data, gain=0.6)
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant_(m.bias.data, 0.01)
        except:
            pass
    return

class DP_Conv(nn.Module):
    """depthwise + pointwise"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dp_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.dp_conv(x)

class DoubleConv_1(nn.Module):
    """depthwise + pointwise"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv_1(x)

class Down_1(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv_1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_1(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv_1(x)

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

        # # version 1, 2
        # # 8 96
        # self.features = nn.Sequential(
        #     nn.Conv2d(1, 24, kernel_size=3, padding=1, bias = False),
        #     nn.BatchNorm2d(24, affine=False),
        #     nn.ReLU(),
        #     nn.Conv2d(24, 24, kernel_size=3, padding=1, bias = False),
        #     nn.BatchNorm2d(24, affine=False),
        #     nn.ReLU(),
        #     nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias = False),
        #     nn.BatchNorm2d(48, affine=False),
        #     nn.ReLU(),
        #     nn.Conv2d(48, 48, kernel_size=3, padding=1, bias = False),
        #     nn.BatchNorm2d(48, affine=False),
        #     nn.ReLU(),
        #     nn.Conv2d(48, 96, kernel_size=3, padding=1, bias = False),
        #     nn.BatchNorm2d(96, affine=False),
        #     nn.ReLU(),
        #     # nn.Dropout(0.1),
        #     nn.Conv2d(96, 96, kernel_size=8, bias = False),
        #     nn.BatchNorm2d(96, affine=False),
        # )

        # xiaowangluo 1，不变网络结构改为深度可分离
        # self.features = nn.Sequential(
        #     DoubleConv_1(1, 32),
        #     # DoubleConv_1(32, 32),
        #     Down_1(32, 64),
            
        #     # DoubleConv_1(64, 64),
        #     Down_1(64, 128),
        #     # DoubleConv_1(64, 128),
        #     # nn.Conv2d(64, 64, kernel_size=4, padding=0, groups=64),
        #     # nn.BatchNorm2d(64),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(64, 128, kernel_size=1, padding=0),
        #     # nn.BatchNorm2d(128),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=4, padding=0, bias = False),
        #     nn.BatchNorm2d(128, affine=False),
        #     nn.ReLU(),
        # )
        # self.features = nn.Sequential(
        #     DoubleConv_1(1, 32),
        #     DoubleConv_1(32, 32),
        #     Down_1(32, 64),
        #     DoubleConv_1(64, 64),
        #     Down_1(64, 128),
        #     # nn.Conv2d(128, 128, kernel_size=3, padding=0, bias = False),
        #     # nn.BatchNorm2d(128, affine=False),
        #     # nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=0, groups=128),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),

        #     nn.MaxPool2d(2)
        # )

        # factor = 2 if bilinear else 1
        # self.down4 = Down_1(64, 128 // factor)

            # DP_Conv(1, 8),
            # nn.MaxPool2d(2),
            # DP_Conv(8,16),
            # DP_Conv(16,32),
            # nn.MaxPool2d(2),
            # DP_Conv(32,64),
            # nn.Conv2d(64, 128, kernel_size=4, bias = False),
            # nn.BatchNorm2d(128, affine=False),
        
        
        # # # version2
        # # self.features = nn.Sequential(
        # #     nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
        # #     nn.BatchNorm2d(32, affine=False),
        # #     nn.ReLU(),
        # #     nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
        # #     nn.BatchNorm2d(32, affine=False),
        # #     nn.ReLU(),
        # #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
        # #     nn.BatchNorm2d(64, affine=False),
        # #     nn.ReLU(),
        # #     nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
        # #     nn.BatchNorm2d(64, affine=False),
        # #     nn.ReLU(),
        # #     nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
        # #     nn.BatchNorm2d(128, affine=False),
        # #     nn.ReLU(),
        # #     nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
        # #     nn.BatchNorm2d(128, affine=False),
        # #     nn.ReLU(),
        # #     # nn.Dropout(0.1),
        # #     nn.Conv2d(128, 128, kernel_size=8, bias = False),
        # #     nn.BatchNorm2d(128, affine=False),
        # # )
        
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
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
            nolinear=hswish(), semodule=SeModule(int(64)), stride=1),

            nn.Conv2d(64, 128, kernel_size=4, bias = False),
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

class HardNet_smaller(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_smaller, self).__init__()
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
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128, kernel_size=1, padding=0)
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

class HardNet_smaller_patch(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_smaller_patch, self).__init__()
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
            nn.Conv2d(128, 8, kernel_size=1, padding=0)     # 8x4x4
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
    
    def forward(self, input, angle=0):
        x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans =  x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x = x_features_trans.view(x_features_trans.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_Sift(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_Sift, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128
        self.features = nn.Sequential(
                            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias = False),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                                nolinear=hswish(), semodule=SeModule(int(16)), stride=1)
                        )
        self.downdim = nn.Conv2d(16, 8, kernel_size=1, padding=0)     # 8x4x4

                
        if self.train_flag:
            self.features.apply(weights_init)
            self.downdim.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input, angle=0):
        ib, _, _, _ = input.shape
        x_norm = self.input_norm(input)
        x_patch_trans = x_norm.view(ib, -1, 4, 4, 4).transpose(2, 3).reshape(ib, -1, 4, 4) # bx16x4x4
        x_bdim = self.features(x_patch_trans)      # bx16x4x4
        x_bdim_trans = x_bdim.permute(0, 2, 3, 1).contiguous().view(ib, -1, 4, 4)
        x_features = self.downdim(x_bdim_trans)     # bx8x4x4

        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(ib, -1, 4) # bx32x4
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x = x_features_trans.view(x_features_trans.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_Sift_Deform(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_Sift_Deform, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128
        self.features = nn.Sequential(
                            # nn.Conv2d(16, 16, kernel_size=3, padding=1, bias = False),
                            DeformConv2d(16, 16, kernel_size=3, padding=1, bias = False),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            Block_Deform(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                                 nolinear=hswish(), semodule=SeModule(int(16)), stride=1)
                            # Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                            #     nolinear=hswish(), semodule=SeModule(int(16)), stride=1)
                        )
        self.downdim = nn.Conv2d(16, 8, kernel_size=1, padding=0)     # 8x4x4

                
        if self.train_flag:
            self.features.apply(weights_init)
            self.downdim.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input, angle=0):
        ib, _, _, _ = input.shape
        x_norm = self.input_norm(input)
        x_patch_trans = x_norm.view(ib, -1, 4, 4, 4).transpose(2, 3).reshape(ib, -1, 4, 4) # bx16x4x4
        x_bdim = self.features(x_patch_trans)      # bx16x4x4
        x_bdim_trans = x_bdim.permute(0, 2, 3, 1).contiguous().view(ib, -1, 4, 4)
        x_features = self.downdim(x_bdim_trans)     # bx8x4x4

        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(ib, -1, 8) # bx16x8
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x = x_features_trans.view(x_features_trans.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast, self).__init__()
        self.train_flag = train_flag
       
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        b, o, _, _ = x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x = x_features_trans.view(x_features_trans.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_twice(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_twice_half(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        # count = 0
        # for m in self.features:
        #     if count > 0:
        #         out = m(out)
        #     else:
        #         out = m(input)
        #     if count == 5:
        #         print(out[0,0,:,:])
        #     count += 1
        # exit()
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_twice_half2(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half2, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_twice_half3(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half3, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_noshortcut(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_twice_half3_short(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half3_short, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_noshortcut_short(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule_short(int(8)), stride=2),
            Block_noshortcut_short(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule_short(int(16)), stride=2),
            Block_noshortcut_short(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule_short(int(16)), stride=1),
            Block_noshortcut_short(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule_short(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut_short(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule_short(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut_short(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule_short(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        wb_mask = (torch.mean(input[:,:,7:9,7:9], dim=[2,3])>0.45).squeeze(1)

        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x), wb_mask

    def model_save_quant_param(self, convw_short_list, convb_short_list, save_pth):
        save_dict = {}

        count = 0
        # modules_name = self.get_quant_modules_name()
        

        for m in self.modules():
            if type(m) in [nn.Conv2d]:
                # name = modules_name[count]

                # save_dict.update({name:{"scale":scale,"zero_point":zero_point,"dequant_scale":dequant_scale}})
                convw = torch.nn.Parameter(convw_short_list[count])
                # print(convw.size(), m.weight.size())
                convb = torch.nn.Parameter(convb_short_list[count])
                assert convw.size() == m.weight.size()
                assert convb.size() == m.bias.size()
                m.weight = convw
                m.bias = convb

                count = count + 1
        

        #保存量化后的模型
        model_state_dict = self.state_dict()
        save_dict = {
                "n_iter": 999,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": None,
                "loss": None,
            }

  
        torch.save(save_dict,save_pth)
        print("Save Short Model Success!")


class HardNet_fast_twice_half3_MO(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half3_MO, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_noshortcut(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            MobileOne(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False),
            MobileOne(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False),
            MobileOne(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            # #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_twice_half3_MO_short(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half3_MO_short, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_noshortcut_short(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule_short(int(8)), stride=2),
            Block_noshortcut_short(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule_short(int(16)), stride=2),
            MobileOne_short(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=True),
            MobileOne_short(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=True),
            MobileOne_short(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=True),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            # #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut_short(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule_short(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


    def model_save_quant_param(self, convw_short_list, convb_short_list, save_pth):
        save_dict = {}

        count = 0
        # modules_name = self.get_quant_modules_name()
        

        for m in self.modules():
            if type(m) in [nn.Conv2d]:
                # name = modules_name[count]

                # save_dict.update({name:{"scale":scale,"zero_point":zero_point,"dequant_scale":dequant_scale}})
                convw = torch.nn.Parameter(convw_short_list[count])
                # print(convw.size(), m.weight.size())
                convb = torch.nn.Parameter(convb_short_list[count])
                assert convw.size() == m.weight.size()
                assert convb.size() == m.bias.size()
                m.weight = convw
                m.bias = convb

                count = count + 1
        

        #保存量化后的模型
        model_state_dict = self.state_dict()
        save_dict = {
                "n_iter": 999,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": None,
                "loss": None,
            }

  
        torch.save(save_dict,save_pth)
        print("Save Short Model Success!")

class HardNet_fast_twice_half3_MO_new(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half3_MO_new, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_noshortcut(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            MobileOne_new(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False, nolinear=hswish()),
            MobileOne_new(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False, nolinear=hswish()),
            MobileOne_new(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False, nolinear=hswish()),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            # #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class HardNet_fast_twice_half3_MO_MOE(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half3_MO_MOE, self).__init__()
        self.train_flag = train_flag
       
        self.features_w = nn.Sequential(
            Block_noshortcut(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            MobileOne_new(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False, nolinear=hswish()),
            MobileOne_new(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False, nolinear=hswish()),
            MobileOne_new(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False, nolinear=hswish()),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            # #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 16, kernel_size=1, padding=0)
        )

        self.features_b = nn.Sequential(
            Block_noshortcut(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            MobileOne_new(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False, nolinear=hswish()),
            MobileOne_new(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False, nolinear=hswish()),
            MobileOne_new(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False, nolinear=hswish()),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            # #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 16, kernel_size=1, padding=0)
        )

        if self.train_flag:
            self.features_w.apply(weights_init)
            self.features_b.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input, angle=0):
        wb_mask = (torch.mean(input[:,:,7:9,7:9], dim=[2,3])>0.45).squeeze(1)
        input_w = input[wb_mask]
        input_b = input[~wb_mask]
        bw, bb = 0, 0
        # 白点分支
        if input_w.shape[0] != 0:
            x_features_w = self.features_w(input_w)      # bx8x4x4
            bw, o, fh, fw = x_features_w.shape
            x_features_trans_w = x_features_w.permute(0, 2, 3, 1).contiguous().view(bw, -1, o)
            if angle == 1:
                x_features_expand_w = torch.flip(x_features_trans_w, dims=[1])
                x_features_expand_w = x_features_expand_w.view(bw, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
                x_w = x_features_expand_w.view(x_features_expand_w.size(0), -1)
            else:
                x_features_expand_w = x_features_trans_w.view(bw, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
                x_w = x_features_expand_w.view(x_features_expand_w.size(0), -1)
        # 黑点分支
        if input_b.shape[0] != 0:
            x_features_b = self.features_b(input_b)      # bx8x4x4
            bb, o, fh, fw = x_features_b.shape
            x_features_trans_b = x_features_b.permute(0, 2, 3, 1).contiguous().view(bb, -1, o)
            if angle == 1:
                x_features_expand_b = torch.flip(x_features_trans_b, dims=[1])
                x_features_expand_b = x_features_expand_b.view(bb, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
                x_b = x_features_expand_b.view(x_features_expand_b.size(0), -1)
            else:
                x_features_expand_b = x_features_trans_b.view(bb, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
                x_b = x_features_expand_b.view(x_features_expand_b.size(0), -1)
        
        
        b = bw + bb # batch_size x N
        out_device = x_w.device if input_w.shape[0] != 0 else x_b.device 
        out_size = x_w.size(1) if input_w.shape[0] != 0 else x_b.size(1)  
        out_fea_sizeN, out_fea_sizeC, out_fea_sizeHW = (x_features_expand_w.size(1), x_features_expand_w.size(2), x_features_expand_w.size(3)) if input_w.shape[0] != 0 else (x_features_expand_b.size(1), x_features_expand_b.size(2), x_features_expand_b.size(3))

        x = torch.zeros(wb_mask.size(0), out_size, device=out_device)
        # x_features_expand = torch.zeros(wb_mask.size(0),  out_fea_sizeN, out_fea_sizeC, out_fea_sizeHW , device=out_device)
        if input_w.shape[0] != 0:
            x[wb_mask.nonzero().squeeze()] = x_w
            # x_features_expand[wb_mask.nonzero().squeeze()] = x_features_expand_w
        if input_b.shape[0] != 0:
            x[(~wb_mask).nonzero().squeeze()] = x_b
            # x_features_expand[(~wb_mask).nonzero().squeeze()] = x_features_expand_b

        return L2Norm()(x)

class HardNet_fast_twice_half3_MO_MOE_short(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half3_MO_MOE_short, self).__init__()
        self.train_flag = train_flag
       
        self.features_w = nn.Sequential(
            Block_noshortcut_short(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule_short(int(8)), stride=2),
            Block_noshortcut_short(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule_short(int(16)), stride=2),
            MobileOne_new_short(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=True, nolinear=hswish()),
            MobileOne_new_short(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=True, nolinear=hswish()),
            MobileOne_new_short(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=True, nolinear=hswish()),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            # #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut_short(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule_short(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 16, kernel_size=1, padding=0)
        )

        self.features_b = nn.Sequential(
            Block_noshortcut_short(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule_short(int(8)), stride=2),
            Block_noshortcut_short(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule_short(int(16)), stride=2),
            MobileOne_new_short(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=True, nolinear=hswish()),
            MobileOne_new_short(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=True, nolinear=hswish()),
            MobileOne_new_short(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=True, nolinear=hswish()),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            # #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut_short(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule_short(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 16, kernel_size=1, padding=0)
        )

        if self.train_flag:
            self.features_w.apply(weights_init)
            self.features_b.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input, angle=0):
        wb_mask = (torch.mean(input[:,:,7:9,7:9], dim=[2,3])>0.45).squeeze(1)
        input_w = input[wb_mask]
        input_b = input[~wb_mask]
        bw, bb = 0, 0
        # 白点分支
        if input_w.shape[0] != 0:
            x_features_w = self.features_w(input_w)      # bx8x4x4
            bw, o, fh, fw = x_features_w.shape
            x_features_trans_w = x_features_w.permute(0, 2, 3, 1).contiguous().view(bw, -1, o)
            if angle == 1:
                x_features_expand_w = torch.flip(x_features_trans_w, dims=[1])
                x_features_expand_w = x_features_expand_w.view(bw, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
                x_w = x_features_expand_w.view(x_features_expand_w.size(0), -1)
            else:
                x_features_expand_w = x_features_trans_w.view(bw, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
                x_w = x_features_expand_w.view(x_features_expand_w.size(0), -1)
        # 黑点分支
        if input_b.shape[0] != 0:
            x_features_b = self.features_b(input_b)      # bx8x4x4
            bb, o, fh, fw = x_features_b.shape
            x_features_trans_b = x_features_b.permute(0, 2, 3, 1).contiguous().view(bb, -1, o)
            if angle == 1:
                x_features_expand_b = torch.flip(x_features_trans_b, dims=[1])
                x_features_expand_b = x_features_expand_b.view(bb, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
                x_b = x_features_expand_b.view(x_features_expand_b.size(0), -1)
            else:
                x_features_expand_b = x_features_trans_b.view(bb, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
                x_b = x_features_expand_b.view(x_features_expand_b.size(0), -1)
        
        
        b = bw + bb # batch_size x N
        out_device = x_w.device if input_w.shape[0] != 0 else x_b.device 
        out_size = x_w.size(1) if input_w.shape[0] != 0 else x_b.size(1)  
        out_fea_sizeN, out_fea_sizeC, out_fea_sizeHW = (x_features_expand_w.size(1), x_features_expand_w.size(2), x_features_expand_w.size(3)) if input_w.shape[0] != 0 else (x_features_expand_b.size(1), x_features_expand_b.size(2), x_features_expand_b.size(3))

        x = torch.zeros(wb_mask.size(0), out_size, device=out_device)
        # x_features_expand = torch.zeros(wb_mask.size(0),  out_fea_sizeN, out_fea_sizeC, out_fea_sizeHW , device=out_device)
        if input_w.shape[0] != 0:
            x[wb_mask.nonzero().squeeze()] = x_w
            # x_features_expand[wb_mask.nonzero().squeeze()] = x_features_expand_w
        if input_b.shape[0] != 0:
            x[(~wb_mask).nonzero().squeeze()] = x_b
            # x_features_expand[(~wb_mask).nonzero().squeeze()] = x_features_expand_b

        return L2Norm()(x)

    def model_save_quant_param(self, convw_short_list, convb_short_list, save_pth):
        save_dict = {}

        count = 0
        # modules_name = self.get_quant_modules_name()
        

        for m in self.modules():
            if type(m) in [nn.Conv2d]:
                # name = modules_name[count]

                # save_dict.update({name:{"scale":scale,"zero_point":zero_point,"dequant_scale":dequant_scale}})
                convw = torch.nn.Parameter(convw_short_list[count])
                # print(convw.size(), m.weight.size())
                convb = torch.nn.Parameter(convb_short_list[count])
                assert convw.size() == m.weight.size()
                assert convb.size() == m.bias.size()
                m.weight = convw
                m.bias = convb

                count = count + 1
        

        #保存量化后的模型
        model_state_dict = self.state_dict()
        save_dict = {
                "n_iter": 999,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": None,
                "loss": None,
            }

  
        torch.save(save_dict,save_pth)
        print("Save Short Model Success!")


class HardNet_fast_twice_half3_MOA(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half3_MOA, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            MobileOne(in_planes=1, planes=8, stride=2, use_se=False, num_conv_branches=4, inference_mode=False),
            MobileOne(in_planes=8, planes=16, stride=2, use_se=False, num_conv_branches=4, inference_mode=False),
            MobileOne(in_planes=16, planes=16, stride=1, use_se=False, num_conv_branches=4, inference_mode=False),
            MobileOne(in_planes=16, planes=16, stride=1, use_se=False, num_conv_branches=4, inference_mode=False),
            MobileOne(in_planes=16, planes=16, stride=1, use_se=False, num_conv_branches=4, inference_mode=False),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            # #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            MobileOne(in_planes=16, planes=32, stride=1, use_se=True, num_conv_branches=4, inference_mode=False),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class HardNet_fast_half(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_half, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_noshortcut(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x = x_features_trans.view(x_features_trans.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_half_standard(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_half_standard, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_noshortcut(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=nn.Hardswish(inplace=True), semodule=SeModule_standard(int(8)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=nn.Hardswish(inplace=True), semodule=SeModule_standard(int(16)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=nn.Hardswish(inplace=True), semodule=SeModule_standard(int(16)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=nn.Hardswish(inplace=True), semodule=SeModule_standard(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=nn.Hardswish(inplace=True), semodule=SeModule_standard(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=nn.Hardswish(inplace=True), semodule=SeModule_standard(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4

        # b, o, _, _ = x_features.shape
        # # print('x_features.shape: ', x_features.shape
        # x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        # if angle == 1:
        #     x_features_expand = torch.flip(x_features_trans, dims=[1])
        #     x = x_features_expand.view(x_features_expand.size(0), -1)
        # else:
        #     x = x_features_trans.view(x_features_trans.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return x_features


class HardNet_fast_half_MOE(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_half_MOE, self).__init__()
        self.train_flag = train_flag
       
        self.features_w = nn.Sequential(
            Block_noshortcut(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0)
        )

        self.features_b = nn.Sequential(
            Block_noshortcut(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0)
        )

        if self.train_flag:
            self.features_w.apply(weights_init)
            self.features_b.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    

    def forward(self, input, angle=0):
        wb_mask = (torch.mean(input[:,:,15:17,3:5], dim=[2,3])>0.45).squeeze(1)
        input_w = input[wb_mask]
        input_b = input[~wb_mask] 
        bw, bb = 0, 0
        if input_w.shape[0] != 0:
            x_features_w = self.features_w(input_w) 
            bw, o, fh, fw = x_features_w.shape
            # 白点分支
            x_features_trans_w = x_features_w.permute(0, 2, 3, 1).contiguous().view(bw, -1, o)
            if angle == 1:
                x_features_trans_w = torch.flip(x_features_trans_w, dims=[1])
                x_w = x_features_trans_w.view(x_features_trans_w.size(0), -1)
            else:
                x_w = x_features_trans_w.view(x_features_trans_w.size(0), -1)

        if input_b.shape[0] != 0:
            # print(input_b.shape)
            x_features_b = self.features_b(input_b)
            bb, o, fh, fw = x_features_b.shape
            # 黑点分支
            x_features_trans_b = x_features_b.permute(0, 2, 3, 1).contiguous().view(bb, -1, o)
            if angle == 1:
                x_features_trans_b = torch.flip(x_features_trans_b, dims=[1])
                x_b = x_features_trans_b.view(x_features_trans_b.size(0), -1)
            else:
                x_b = x_features_trans_b.view(x_features_trans_b.size(0), -1)
        
        b = bw + bb # batch_size x N 
        out_device =  x_w.device if input_w.shape[0] != 0 else x_b.device 
        out_size = x_w.size(1) if input_w.shape[0] != 0 else x_b.size(1)  
        out_fea_sizeC, out_fea_sizeHW  = (x_features_trans_w.size(1), x_features_trans_w.size(2)) if input_w.shape[0] != 0 else (x_features_trans_b.size(1), x_features_trans_b.size(2))
        x = torch.zeros(wb_mask.size(0), out_size, device=out_device)
        # x_features_trans = torch.zeros(wb_mask.size(0), out_fea_sizeC, out_fea_sizeHW, device=out_device)
        if input_w.shape[0] != 0:
            x[wb_mask.nonzero().squeeze()] = x_w
            # x_features_trans[wb_mask.nonzero().squeeze()] = x_features_trans_w
        if input_b.shape[0] != 0:    
            x[(~wb_mask).nonzero().squeeze()] = x_b
            # x_features_trans[(~wb_mask).nonzero().squeeze()] = x_features_trans_b

        return L2Norm()(x)  # , x_features_trans.transpose(1, 2).contiguous().view(b, -1, fh, fw)

class HardNet_fast_half_AMOE(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_half_AMOE, self).__init__()
        self.train_flag = train_flag
       
        self.features_w = nn.Sequential(
            Block_noshortcut(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0)
        )

        self.features_b = nn.Sequential(
            Block_noshortcut(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0)
        )

        if self.train_flag:
            self.features_w.apply(weights_init)
            self.features_b.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input, angle=0, wb_mask=None):
        #  wb_mask = (torch.mean(input[:,:,15:17,3:5], dim=[2,3])>0.45).squeeze()
        input_w = input[wb_mask]
        input_b = input[~wb_mask] 
        bw, bb = 0, 0
        if input_w.shape[0] != 0:
            x_features_w = self.features_w(input_w) 
            bw, o, fh, fw = x_features_w.shape
            # 白点分支
            x_features_trans_w = x_features_w.permute(0, 2, 3, 1).contiguous().view(bw, -1, o)
            if angle == 1:
                x_features_trans_w = torch.flip(x_features_trans_w, dims=[1])
                x_w = x_features_trans_w.view(x_features_trans_w.size(0), -1)
            else:
                x_w = x_features_trans_w.view(x_features_trans_w.size(0), -1)

        if input_b.shape[0] != 0:
            x_features_b = self.features_b(input_b)
            bb, o, fh, fw = x_features_b.shape
            # 黑点分支
            x_features_trans_b = x_features_b.permute(0, 2, 3, 1).contiguous().view(bb, -1, o)
            if angle == 1:
                x_features_trans_b = torch.flip(x_features_trans_b, dims=[1])
                x_b = x_features_trans_b.view(x_features_trans_b.size(0), -1)
            else:
                x_b = x_features_trans_b.view(x_features_trans_b.size(0), -1)
        
        b = bw + bb # batch_size x N 
        out_device =  x_w.device if input_w.shape[0] != 0 else x_b.device 
        out_size = x_w.size(1) if input_w.shape[0] != 0 else x_b.size(1)  
        out_fea_sizeC, out_fea_sizeHW  = (x_features_trans_w.size(1), x_features_trans_w.size(2)) if input_w.shape[0] != 0 else (x_features_trans_b.size(1), x_features_trans_b.size(2))
        x = torch.zeros(wb_mask.size(0), out_size, device=out_device)
        # x_features_trans = torch.zeros(wb_mask.size(0), out_fea_sizeC, out_fea_sizeHW, device=out_device)
        if input_w.shape[0] != 0:
            x[wb_mask.nonzero().squeeze()] = x_w
            # x_features_trans[wb_mask.nonzero().squeeze()] = x_features_trans_w
        if input_b.shape[0] != 0:    
            x[(~wb_mask).nonzero().squeeze()] = x_b
            # x_features_trans[(~wb_mask).nonzero().squeeze()] = x_features_trans_b

        return L2Norm()(x)  # , x_features_trans.transpose(1, 2).contiguous().view(b, -1, fh, fw)


class HardNet_fast_half_short(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_half_short, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_noshortcut_short(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule_short(int(8)), stride=2),
            Block_noshortcut_short(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule_short(int(16)), stride=2),
            Block_noshortcut_short(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule_short(int(16)), stride=1),
            Block_noshortcut_short(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule_short(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut_short(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule_short(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut_short(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule_short(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x = x_features_trans.view(x_features_trans.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

    def model_save_quant_param(self, convw_short_list, convb_short_list, save_pth):
        save_dict = {}

        count = 0
        # modules_name = self.get_quant_modules_name()
        

        for m in self.modules():
            if type(m) in [nn.Conv2d]:
                # name = modules_name[count]

                # save_dict.update({name:{"scale":scale,"zero_point":zero_point,"dequant_scale":dequant_scale}})
                convw = torch.nn.Parameter(convw_short_list[count])
                # print(convw.size(), m.weight.size())
                convb = torch.nn.Parameter(convb_short_list[count])
                assert convw.size() == m.weight.size()
                assert convb.size() == m.bias.size()
                m.weight = convw
                m.bias = convb

                count = count + 1
        

        #保存量化后的模型
        model_state_dict = self.state_dict()
        save_dict = {
                "n_iter": 999,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": None,
                "loss": None,
            }

  
        torch.save(save_dict,save_pth)
        print("Save Short Model Success!")

class HardNet_fast_half_MO(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_half_MO, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_noshortcut(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            MobileOne(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False),
            MobileOne(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False),
            MobileOne(in_planes=16, planes=16, stride=1, use_se=True, num_conv_branches=4, inference_mode=False),         
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            # #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=1),

            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x = x_features_trans.view(x_features_trans.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)



class HardNet_fast_twice_half_cr(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half_cr, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_CRelu(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_CRelu(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_CRelu(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_twice_half_cr2(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half_cr2, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_CRelu(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_CRelu(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_twice_half_norm2(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half_norm2, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # 分别归一化
        x1, x2 = L2Norm()(x[:,:128]), L2Norm()(x[:,128:])
        x = torch.cat((x1, x2), dim=-1)
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return x

class HardNet_fast_twice_big(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_big, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_twice_big_MO(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_big_MO, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),

            MobileOne_shortcut(in_planes=16, planes=32, stride=1, use_se=True, num_conv_branches=4, inference_mode=False, nolinear=hswish()),
            MobileOne_shortcut(in_planes=32, planes=64, stride=1, use_se=True, num_conv_branches=4, inference_mode=False, nolinear=hswish()),

            # Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
            #          nolinear=hswish(), semodule=SeModule(int(64)), stride=1),

            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            MobileOne_shortcut(in_planes=64, planes=64, stride=1, use_se=True, num_conv_branches=4, inference_mode=False, nolinear=hswish()),
            # Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
            #          nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class HardNet_fast_twice_big_norm2(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_big_norm2, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # 分别归一化
        x1, x2 = L2Norm()(x[:,:128]), L2Norm()(x[:,128:])
        x = torch.cat((x1, x2), dim=-1)
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return x

class HardNet_fast_twice_half3_norm2(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_half3_norm2, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_noshortcut(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_noshortcut(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # 分别归一化
        x1, x2 = L2Norm()(x[:,:128]), L2Norm()(x[:,128:])
        x = torch.cat((x1, x2), dim=-1)
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return x



class HardNet_fast_big(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_big, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x = x_features_trans.view(x_features_trans.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_big_ap(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_big_ap, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x = x_features_trans.view(x_features_trans.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_big_mp(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_big_mp, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape)
        # exit()
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x = x_features_trans.view(x_features_trans.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class HardNet_fast_twice_big_short(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_big_short, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_short(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule_short(int(8)), stride=2),
            Block_short(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule_short(int(16)), stride=2),
            Block_short(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule_short(int(32)), stride=1),
            Block_short(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule_short(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_short(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule_short(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_short(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule_short(int(128)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

    def model_save_quant_param(self, convw_short_list, convb_short_list, save_pth):
        save_dict = {}

        count = 0
        # modules_name = self.get_quant_modules_name()
        

        for m in self.modules():
            if type(m) in [nn.Conv2d]:
                # name = modules_name[count]

                # save_dict.update({name:{"scale":scale,"zero_point":zero_point,"dequant_scale":dequant_scale}})
                convw = torch.nn.Parameter(convw_short_list[count])
                convb = torch.nn.Parameter(convb_short_list[count])
                assert convw.size() == m.weight.size()
                assert convb.size() == m.bias.size()
                m.weight = convw
                m.bias = convb

                count = count + 1
        

        #保存量化后的模型
        model_state_dict = self.state_dict()
        save_dict = {
                "n_iter": 999,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": None,
                "loss": None,
            }

  
        torch.save(save_dict,save_pth)
        print("Save Short Model Success!")



class HardNet_fast_twice_big2(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_big2, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=2),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(64), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class HardNet_fast_twice_short(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_twice_short, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_short(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule_short(int(8)), stride=2),
            Block_short(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule_short(int(16)), stride=2),
            Block_short(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule_short(int(32)), stride=1),
            Block_short(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule_short(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_short(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule_short(int(32)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_short(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule_short(int(64)), stride=1),
            # Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

    def model_save_quant_param(self, convw_short_list, convb_short_list, save_pth):
        save_dict = {}

        count = 0
        # modules_name = self.get_quant_modules_name()
        

        for m in self.modules():
            if type(m) in [nn.Conv2d]:
                # name = modules_name[count]

                # save_dict.update({name:{"scale":scale,"zero_point":zero_point,"dequant_scale":dequant_scale}})
                convw = torch.nn.Parameter(convw_short_list[count])
                convb = torch.nn.Parameter(convb_short_list[count])

                assert convw.size() == m.weight.size()
                assert convb.size() == m.bias.size()
                m.weight = convw
                m.bias = convb

                count = count + 1
        

        #保存量化后的模型
        model_state_dict = self.state_dict()
        save_dict = {
                "n_iter": 999,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": None,
                "loss": None,
            }

  
        torch.save(save_dict,save_pth)
        print("Save Short Model Success!")


class HardNet_fast_Pconv(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_Pconv, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(2), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
            #          nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            Block_Pconv(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(input)      # bx8x4x4
        # x_features = self.features(self.input_norm(input))      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x = x_features_trans.view(x_features_trans.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class HardNet_fast_deform(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_deform, self).__init__()
        self.train_flag = train_flag
       
        self.features = nn.Sequential(
            Block_Deform(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_Deform(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block_Deform(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Deform(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Deform(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Deform(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(self.input_norm(input))      # bx8x34x10
        b, o, dh, dw = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), dh, dw, o)
        else:
            x = x_features_trans.view(x_features_trans.size(0), dh, dw, o)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return x.permute(0, 3, 1, 2)
        # return L2Norm()(x)

class HardNet_fast_deform_last(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_deform_last, self).__init__()
        self.train_flag = train_flag
       
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
            Block_Deform(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
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
    

    def forward(self, input, angle=0):
        x_features = self.features(self.input_norm(input))      # bx8x34x10
        b, o, dh, dw = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), dh, dw, o)
        else:
            x = x_features_trans.view(x_features_trans.size(0), dh, dw, o)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return x.permute(0, 3, 1, 2)
        # return L2Norm()(x)


# class WideBasic(enn.EquivariantModule):   
#     def __init__(self,
#                  in_fiber: enn.FieldType,
#                  inner_fiber: enn.FieldType,
#                  dropout_rate, 
#                  stride=1,
#                  out_fiber: enn.FieldType = None,
#                  F: float = 1.,
#                  sigma: float = 0.45,
#                  ):
#         super(WideBasic, self).__init__()
        
#         if out_fiber is None:
#             out_fiber = in_fiber
        
#         self.in_type = in_fiber
#         inner_class = inner_fiber
#         self.out_type = out_fiber
        
#         if isinstance(in_fiber.gspace, gspaces.FlipRot2dOnR2):
#             rotations = in_fiber.gspace.fibergroup.rotation_order
#         elif isinstance(in_fiber.gspace, gspaces.Rot2dOnR2):
#             rotations = in_fiber.gspace.fibergroup.order()
#         else:
#             rotations = 0
        
#         if rotations in [0, 2, 4]:
#             conv = conv3x3
#         else:
#             conv = conv5x5
        
#         self.bn1 = enn.InnerBatchNorm(self.in_type)
#         self.relu1 = enn.ReLU(self.in_type, inplace=True)
#         self.conv1 = conv(self.in_type, inner_class, sigma=sigma, F=F, initialize=False)
        
#         self.bn2 = enn.InnerBatchNorm(inner_class)
#         self.relu2 = enn.ReLU(inner_class, inplace=True)
        
#         self.dropout = enn.PointwiseDropout(inner_class, p=dropout_rate)
        
#         self.conv2 = conv(inner_class, self.out_type, stride=stride, sigma=sigma, F=F, initialize=False)
        
#         self.shortcut = None
#         # if stride == 1 and in_size == out_size:
#         if stride != 1 or self.in_type != self.out_type:
#             self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride, bias=False, sigma=sigma, F=F, initialize=False)
#             # if rotations in [0, 2, 4]:
#             #     self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride, bias=False, sigma=sigma, F=F, initialize=False)
#             # else:
#             #     self.shortcut = conv3x3(self.in_type, self.out_type, stride=stride, bias=False, sigma=sigma, F=F, initialize=False)
    
#     def forward(self, x):
#         x_n = self.relu1(self.bn1(x))
#         out = self.relu2(self.bn2(self.conv1(x_n)))
#         out = self.dropout(out)
#         out = self.conv2(out)
        
#         if self.shortcut is not None:
#             out += self.shortcut(x_n)
#         else:
#             out += x
        
#         return out
    
#     def evaluate_output_shape(self, input_shape: Tuple):
#         assert len(input_shape) == 4
#         assert input_shape[1] == self.in_type.size
#         if self.shortcut is not None:
#             return self.shortcut.evaluate_output_shape(input_shape)
#         else:
#             return input_shape

class HardNet_fast_Ecnn(nn.Module):
    """HardNet model definition
    """
    def __init__(self, dropout_rate=0.5, train_flag=False,
                 N: int = 4,
                 r: int = 0,
                 f: bool = False,
                 main_fiber: str = "regular",
                 inner_fiber: str = "regular",
                 F: float = 1.,                     # 计算attr.freq和attr.radius满足线性关系的斜率 k = PI * F
                 sigma: float = None,
                 deltaorth: bool = False,
                 fixparams: bool = False,
                 conv2triv: bool = False,
                 ):
        super(HardNet_fast_Ecnn, self).__init__()

        self.train_flag = train_flag
        self._layer = 0
        self._N = N
        self._f = f
        self._r = r
        self._F = F
        self._sigma = sigma
        self._fixparams = fixparams
        self.conv2triv = conv2triv

        if self._f:
            self.gspace = gspaces.FlipRot2dOnR2(N)
        else:
            self.gspace = gspaces.Rot2dOnR2(N)          # C8
        
        # if self._r == 0:
        #     id = (0, 1) if self._f else 1
        #     self.gspace, _, _ = self.gspace.restrict(id)
        
        # 1 channel
        r1 = enn.FieldType(self.gspace, [self.gspace.trivial_repr]) ## input 1 channels (gray scale image)  trivial_repr = 'irrep_0'
        
        # # 2 channels
        # r1 = enn.FieldType(self.gspace, [self.gspace.trivial_repr, self.gspace.trivial_repr]) ## input 1 channels (gray scale image)  trivial_repr = 'irrep_0'
        
        # nStages = [4, 8, 16, 16, 16, 32, 64]
        nStages = [8, 16, 32, 32, 32, 64]
        self.in_type = [r1] + [FIBERS[main_fiber](self.gspace, p, fixparams=self._fixparams) for p in nStages[:-1]]     # [0, 2, 5, 11, 11, 11, 22]
        self.out_type = [FIBERS[main_fiber](self.gspace, p, fixparams=self._fixparams) for p in nStages[:-1]]    # [1, 2, 5, 5, 5, 11]
        
        if self.conv2triv:
            out_fiber = "trivial"
        else:
            out_fiber = main_fiber

        self.out_type += [FIBERS[out_fiber](self.gspace, nStages[-1], fixparams=self._fixparams)]   # + [8]
        
        self.features = enn.SequentialModule(
            EBlock(kernel_size=3, in_fiber= self.in_type[0], inner_fiber=inner_fiber, inner_stages=[nStages[0]]*2, 
                   out_fiber=self.out_type[0], stride=2, F=F, sigma=sigma, fixparams=fixparams),
            EBlock(kernel_size=3, in_fiber= self.in_type[1], inner_fiber=inner_fiber, inner_stages=[nStages[1]]*2, 
                   out_fiber=self.out_type[1], stride=2, F=F, sigma=sigma, fixparams=fixparams),
            # EConstantPad2d(self.out_type[1], [3, 3, 3, 3], 0),
            EConstantPad2d(self.out_type[1], [1, 1, 2, 2], 0),
            EBlock(kernel_size=3, in_fiber= self.in_type[2], inner_fiber=inner_fiber, inner_stages=[nStages[2]]*2, 
                   out_fiber=self.out_type[2], stride=1, F=F, sigma=sigma, fixparams=fixparams),
            EBlock(kernel_size=3, in_fiber= self.in_type[3], inner_fiber=inner_fiber, inner_stages=[nStages[3]]*2, 
                   out_fiber=self.out_type[3], stride=1, F=F, sigma=sigma, fixparams=fixparams),
            EBlock(kernel_size=3, in_fiber= self.in_type[4], inner_fiber=inner_fiber, inner_stages=[nStages[4]]*2, 
                   out_fiber=self.out_type[4], stride=1, F=F, sigma=sigma, fixparams=fixparams),
            # EBlock(kernel_size=3, in_fiber= self.in_type[4], inner_fiber=inner_fiber, inner_stages=[nStages[4]]*2, 
            #        out_fiber=self.out_type[4], stride=1, F=F, sigma=sigma, fixparams=fixparams),
            EBlock(kernel_size=3, in_fiber= self.in_type[5], inner_fiber=inner_fiber, inner_stages=[nStages[5]]*2, 
                   out_fiber=self.out_type[5], stride=1, F=F, sigma=sigma, fixparams=fixparams),
            # EBlock_SA(kernel_size=3, in_fiber= self.in_type[5], inner_fiber=inner_fiber, inner_stages=[nStages[5]]*2, 
            #       out_fiber=self.out_type[5], stride=1, F=F, sigma=sigma, fixparams=fixparams, win_size=4),
            # EBlock_HiLo_SA(kernel_size=3, in_fiber= self.in_type[5], inner_fiber=inner_fiber, inner_stages=[nStages[5]]*2, 
            #        out_fiber=self.out_type[5], stride=1, F=F, sigma=sigma, fixparams=fixparams),
            EBlock(kernel_size=3, in_fiber= self.out_type[5], inner_fiber=inner_fiber, inner_stages=[nStages[5]]*2, 
                   out_fiber=self.out_type[5], stride=1, F=F, sigma=sigma, fixparams=fixparams)
            # conv1x1(self.in_type[-1], self.out_type[-1], padding=0, F=F, sigma=sigma, initialize=False)
        )
        self.downdim = nn.Conv2d(16, 8, kernel_size=1, padding=0)
        
        # if self._r > 0:
        #     id = (0, 4) if self._f else 4
        #     self.restrict1 = self._restrict_layer(id)
        # else:
        #     self.restrict1 = lambda x: x

        # if self._r > 1:
        #     id = (0, 1) if self._f else 1
        #     self.restrict2 = self._restrict_layer(id)
        # else:
        #     self.restrict2 = lambda x: x

        if self.conv2triv:
            self.relu = enn.ReLU(self.out_type[-1], inplace=True)
        else:
            self.mp = enn.GroupPooling(self.out_type[-1])
            self.relu = enn.ReLU(self.mp.out_type, inplace=True)

        # self.downdim = nn.Conv2d(11, 8, kernel_size=1, padding=0)    
        # self.linear = nn.Linear(self.relu.out_type.size, num_classes)
        
        if self.train_flag:
            for _, module in self.named_modules():
                if isinstance(module, enn.R2Conv):
                    if deltaorth:
                        init.deltaorthonormal_init(module.weights.data, module.basisexpansion)
                    else:
                        init.generalized_he_init(module.weights.data, module.basisexpansion)
                elif isinstance(module, nn.Conv2d):
                    nn.init.orthogonal_(module.weight.data, gain=0.6)
                    try:
                        nn.init.constant_(module.bias.data, 0.01)
                    except:
                        pass
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
                elif isinstance(module, nn.Linear):
                    module.bias.data.zero_()
        
        print("MODEL TOPOLOGY:")
        # for i, (name, mod) in enumerate(self.named_modules()):
        #     print(f"\t{i} - {name}")
        # for i, (name, mod) in enumerate(self.named_modules()):
        #     params = sum([p.numel() for p in mod.parameters() if p.requires_grad])
        #     if isinstance(mod, nn.EquivariantModule) and isinstance(mod.in_type, nn.FieldType) and isinstance(mod.out_type,
        #                                                                                                 nn.FieldType):
        #         print(f"\t{i: <3} - {name: <70} | {params: <8} | {mod.in_type.size: <4}- {mod.out_type.size: <4}")
        #     else:
        #         print(f"\t{i: <3} - {name: <70} | {params: <8} |")
        tot_param = sum([p.numel() for p in self.parameters() if p.requires_grad])
        print("Total number of parameters:", tot_param)
                
        # if self.train_flag:
        #     self.features.apply(weights_init)
        return

    def _restrict_layer(self, subgroup_id):
        layers = list()
        layers.append(enn.RestrictionModule(self._in_type, subgroup_id))
        layers.append(enn.DisentangleModule(layers[-1].out_type))
        self._in_type = layers[-1].out_type
        self.gspace = self._in_type.gspace
        
        restrict_layer = enn.SequentialModule(*layers)
        return restrict_layer
    

    def _wide_layer(self, block, planes: int, num_blocks: int, dropout_rate: float, stride: int,
                    main_fiber: str = "regular",
                    inner_fiber: str = "regular",
                    out_fiber: str = None,
                    ):
        
        self._layer += 1
        print("start building", self._layer)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        main_type = FIBERS[main_fiber](self.gspace, planes, fixparams=self._fixparams)
        inner_class = FIBERS[inner_fiber](self.gspace, planes, fixparams=self._fixparams)
        if out_fiber is None:
            out_fiber = main_fiber
        out_type = FIBERS[out_fiber](self.gspace, planes, fixparams=self._fixparams)
        
        for b, stride in enumerate(strides):
            if b == num_blocks - 1:
                out_f = out_type
            else:
                out_f = main_type
            layers.append(
                block(self._in_type, inner_class, dropout_rate, stride, out_fiber=out_f, sigma=self._sigma, F=self._F))
            self._in_type = out_f
        print("built", self._layer)
        return enn.SequentialModule(*layers)


    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    

    def forward(self, input, angle=0):
        # input_n = self.input_norm(input)
        g_input = enn.GeometricTensor(input, self.in_type[0])
        # print(input[0, 0, 0, :])
        out = self.features(g_input)     # bx11x34x10
        # print(out.tensor.shape)
        # count = 0
        # for m in self.features._modules.values():
        #     if count == 0:
        #         out = m(g_input)
        #         # out2 = TF.rotate(out1.tensor, -45, InterpolationMode.BILINEAR)
        #         # print(out1.shape)
        #         # print(out1.tensor[0, :8, 32, 13], out1.tensor[0, :8, 44, 15])
        #     else:
        #         out = m(out)
        #     if count == 6:
        #         print(out.tensor[0, 0, 0, :])
        #         exit()
        #     count += 1

        # g1 = self.layer1(g_input)      # bx8x34x10
        # # print(g1.tensor.shape)
        # g2 = self.layer2(self.restrict1(g1))
        # g3 = self.layer3(self.restrict2(g2))
        # # print(g3.tensor.shape)
        # out = self.bn1(g3)
        if not self.conv2triv:
            out = self.mp(out)
        # print(out.tensor.shape)
        out = self.relu(out)
        # print(out.tensor[0, 0, 3, :])
        # exit()
        # print(out.tensor.shape)
        out = out.tensor

        x_features = self.downdim(out)
        b, o, dh, dw = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), dh, dw, o)
        else:
            x = x_features_trans.view(x_features_trans.size(0), dh, dw, o)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return x.permute(0, 3, 1, 2)
        # return L2Norm()(x)

class HardNet_fast_Ecnn_twice(nn.Module):
    """HardNet model definition
    """
    def __init__(self, dropout_rate=0.5, train_flag=False,
                 N: int = 4,
                 r: int = 0,
                 f: bool = False,
                 main_fiber: str = "regular",
                 inner_fiber: str = "regular",
                 F: float = 1.,                     # 计算attr.freq和attr.radius满足线性关系的斜率 k = PI * F
                 sigma: float = None,
                 deltaorth: bool = False,
                 fixparams: bool = False,
                 conv2triv: bool = False,
                 ):
        super(HardNet_fast_Ecnn_twice, self).__init__()

        self.train_flag = train_flag
        self._layer = 0
        self._N = N
        self._f = f
        self._r = r
        self._F = F
        self._sigma = sigma
        self._fixparams = fixparams
        self.conv2triv = conv2triv

        if self._f:
            self.gspace = gspaces.FlipRot2dOnR2(N)
        else:
            self.gspace = gspaces.Rot2dOnR2(N)          # C8
        
        # if self._r == 0:
        #     id = (0, 1) if self._f else 1
        #     self.gspace, _, _ = self.gspace.restrict(id)
        
        r1 = enn.FieldType(self.gspace, [self.gspace.trivial_repr]) ## input 1 channels (gray scale image)  trivial_repr = 'irrep_0'
        
        # nStages = [4, 8, 16, 16, 16, 32]
        nStages = [8, 16, 32, 32, 32, 64]
        self.in_type = [r1] + [FIBERS[main_fiber](self.gspace, p, fixparams=self._fixparams) for p in nStages[:-1]]     # [0, 2, 5, 11, 11, 11, 22] -> [0, 1, 2, 4, 4, 4, 8]
        self.out_type = [FIBERS[main_fiber](self.gspace, p, fixparams=self._fixparams) for p in nStages[:-1]]    # [1, 2, 5, 5, 5, 11]
        
        if self.conv2triv:
            out_fiber = "trivial"
        else:
            out_fiber = main_fiber

        self.out_type += [FIBERS[out_fiber](self.gspace, nStages[-1], fixparams=self._fixparams)]   # + [8]
        
        self.features = enn.SequentialModule(
            EBlock(kernel_size=3, in_fiber= self.in_type[0], inner_fiber=inner_fiber, inner_stages=[nStages[0]]*2, 
                   out_fiber=self.out_type[0], stride=2, F=F, sigma=sigma, fixparams=fixparams),
            # EConstantPad2d(self.out_type[1], [1, 1, 0, 0], 0)
            EBlock(kernel_size=3, in_fiber= self.in_type[1], inner_fiber=inner_fiber, inner_stages=[nStages[1]]*2, 
                   out_fiber=self.out_type[1], stride=2, F=F, sigma=sigma, fixparams=fixparams),
            EConstantPad2d(self.out_type[1], [1, 1, 2, 2], 0),
            # EConstantPad2d(self.out_type[1], [3, 3, 3, 3], 0),
            # EConstantPad2d(self.out_type[1], [2, 2, 2, 2], 0),
            EBlock(kernel_size=3, in_fiber= self.in_type[2], inner_fiber=inner_fiber, inner_stages=[nStages[2]]*2, 
                   out_fiber=self.out_type[2], stride=1, F=F, sigma=sigma, fixparams=fixparams),
            # EConstantPad2d(self.out_type[2], [2, 2, 2, 2], 0),
            EBlock(kernel_size=3, in_fiber= self.in_type[3], inner_fiber=inner_fiber, inner_stages=[nStages[3]]*2, 
                   out_fiber=self.out_type[3], stride=1, F=F, sigma=sigma, fixparams=fixparams),
            EBlock(kernel_size=3, in_fiber= self.in_type[4], inner_fiber=inner_fiber, inner_stages=[nStages[4]]*2, 
                   out_fiber=self.out_type[4], stride=1, F=F, sigma=sigma, fixparams=fixparams),
            # EBlock(kernel_size=3, in_fiber= self.in_type[4], inner_fiber=inner_fiber, inner_stages=[nStages[4]]*2, 
            #       out_fiber=self.out_type[4], stride=1, F=F, sigma=sigma, fixparams=fixparams),

            EBlock(kernel_size=3, in_fiber= self.in_type[5], inner_fiber=inner_fiber, inner_stages=[nStages[5]]*2, 
                   out_fiber=self.out_type[5], stride=1, F=F, sigma=sigma, fixparams=fixparams),

            # EBlock_SA(kernel_size=3, in_fiber= self.in_type[5], inner_fiber=inner_fiber, inner_stages=[nStages[5]]*2, 
            #        out_fiber=self.out_type[5], stride=1, F=F, sigma=sigma, fixparams=fixparams, win_size=4),

            # EBlock_HiLo_SA(kernel_size=3, in_fiber= self.in_type[5], inner_fiber=inner_fiber, inner_stages=[nStages[5]]*2, 
            #        out_fiber=self.out_type[5], stride=1, F=F, sigma=sigma, fixparams=fixparams),

            EBlock(kernel_size=3, in_fiber= self.out_type[5], inner_fiber=inner_fiber, inner_stages=[nStages[5]]*2, 
                   out_fiber=self.out_type[5], stride=1, F=F, sigma=sigma, fixparams=fixparams)
            # conv1x1(self.in_type[-1], self.out_type[-1], padding=0, F=F, sigma=sigma, initialize=False)
        )
        self.downdim = nn.Conv2d(16, 16, kernel_size=1, padding=0)
        
        # if self._r > 0:
        #     id = (0, 4) if self._f else 4
        #     self.restrict1 = self._restrict_layer(id)
        # else:
        #     self.restrict1 = lambda x: x

        # if self._r > 1:
        #     id = (0, 1) if self._f else 1
        #     self.restrict2 = self._restrict_layer(id)
        # else:
        #     self.restrict2 = lambda x: x

        if self.conv2triv:
            self.relu = enn.ReLU(self.out_type[-1], inplace=True)
        else:
            self.mp = enn.GroupPooling(self.out_type[-1])
            self.relu = enn.ReLU(self.mp.out_type, inplace=True)

        # self.downdim = nn.Conv2d(11, 8, kernel_size=1, padding=0)    
        # self.linear = nn.Linear(self.relu.out_type.size, num_classes)
        
        if self.train_flag:
            for _, module in self.named_modules():
                if isinstance(module, enn.R2Conv):
                    if deltaorth:
                        init.deltaorthonormal_init(module.weights.data, module.basisexpansion)
                    else:
                        init.generalized_he_init(module.weights.data, module.basisexpansion)
                elif isinstance(module, nn.Conv2d):
                    nn.init.orthogonal_(module.weight.data, gain=0.6)
                    try:
                        nn.init.constant_(module.bias.data, 0.01)
                    except:
                        pass
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
                elif isinstance(module, nn.Linear):
                    module.bias.data.zero_()
        
        print("MODEL TOPOLOGY:")
        # for i, (name, mod) in enumerate(self.named_modules()):
        #     print(f"\t{i} - {name}")
        # for i, (name, mod) in enumerate(self.named_modules()):
        #     params = sum([p.numel() for p in mod.parameters() if p.requires_grad])
        #     if isinstance(mod, nn.EquivariantModule) and isinstance(mod.in_type, nn.FieldType) and isinstance(mod.out_type,
        #                                                                                                 nn.FieldType):
        #         print(f"\t{i: <3} - {name: <70} | {params: <8} | {mod.in_type.size: <4}- {mod.out_type.size: <4}")
        #     else:
        #         print(f"\t{i: <3} - {name: <70} | {params: <8} |")
        tot_param = sum([p.numel() for p in self.parameters() if p.requires_grad])
        print("Total number of parameters:", tot_param)
                
        # if self.train_flag:
        #     self.features.apply(weights_init)
        return

    def _restrict_layer(self, subgroup_id):
        layers = list()
        layers.append(enn.RestrictionModule(self._in_type, subgroup_id))
        layers.append(enn.DisentangleModule(layers[-1].out_type))
        self._in_type = layers[-1].out_type
        self.gspace = self._in_type.gspace
        
        restrict_layer = enn.SequentialModule(*layers)
        return restrict_layer
    

    def _wide_layer(self, block, planes: int, num_blocks: int, dropout_rate: float, stride: int,
                    main_fiber: str = "regular",
                    inner_fiber: str = "regular",
                    out_fiber: str = None,
                    ):
        
        self._layer += 1
        print("start building", self._layer)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        main_type = FIBERS[main_fiber](self.gspace, planes, fixparams=self._fixparams)
        inner_class = FIBERS[inner_fiber](self.gspace, planes, fixparams=self._fixparams)
        if out_fiber is None:
            out_fiber = main_fiber
        out_type = FIBERS[out_fiber](self.gspace, planes, fixparams=self._fixparams)
        
        for b, stride in enumerate(strides):
            if b == num_blocks - 1:
                out_f = out_type
            else:
                out_f = main_type
            layers.append(
                block(self._in_type, inner_class, dropout_rate, stride, out_fiber=out_f, sigma=self._sigma, F=self._F))
            self._in_type = out_f
        print("built", self._layer)
        return enn.SequentialModule(*layers)


    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    

    def forward(self, input, angle=0):
        # input_n = self.input_norm(input)
        g_input = enn.GeometricTensor(input, self.in_type[0])

        out = self.features(g_input)     # bx11x34x10
        # print(out.tensor.shape)

        # g1 = self.layer1(g_input)      # bx8x34x10
        # # print(g1.tensor.shape)
        # g2 = self.layer2(self.restrict1(g1))
        # g3 = self.layer3(self.restrict2(g2))
        # # print(g3.tensor.shape)
        # out = self.bn1(g3)
        if not self.conv2triv:
            out = self.mp(out)
        # print(out.tensor.shape)
        out = self.relu(out)
        # print(out.tensor.shape)
        out = out.tensor
        # print(out.shape)
        x_features = self.downdim(out)

        # print(x_features.shape)

        b, o, dh, dw = x_features.shape
        # print('x_features.shape: ', x_features.shape)
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), dh, dw, o)
        else:
            x = x_features_trans.view(x_features_trans.size(0), dh, dw, o)
    
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return x.permute(0, 3, 1, 2)
        # return L2Norm()(x)

class HardNet_fast_Ecnn_third(nn.Module):
    """HardNet model definition
    """
    def __init__(self, dropout_rate=0.5, train_flag=False,
                 N: int = 4,
                 r: int = 0,
                 f: bool = False,
                 main_fiber: str = "regular",
                 inner_fiber: str = "regular",
                 F: float = 1.,                     # 计算attr.freq和attr.radius满足线性关系的斜率 k = PI * F
                 sigma: float = None,
                 deltaorth: bool = False,
                 fixparams: bool = False,
                 conv2triv: bool = False,
                 ):
        super(HardNet_fast_Ecnn_third, self).__init__()

        self.train_flag = train_flag
        self._layer = 0
        self._N = N
        self._f = f
        self._r = r
        self._F = F
        self._sigma = sigma
        self._fixparams = fixparams
        self.conv2triv = conv2triv

        if self._f:
            self.gspace = gspaces.FlipRot2dOnR2(N)
        else:
            self.gspace = gspaces.Rot2dOnR2(N)          # C8
        
        # if self._r == 0:
        #     id = (0, 1) if self._f else 1
        #     self.gspace, _, _ = self.gspace.restrict(id)
        
        # 1 channel input
        r1 = enn.FieldType(self.gspace, [self.gspace.trivial_repr]) ## input 1 channels (gray scale image)  trivial_repr = 'irrep_0'
       
        # # 2 channels input
        # r1 = enn.FieldType(self.gspace, [self.gspace.trivial_repr, self.gspace.trivial_repr])
        
        # nStages = [4, 8, 16, 16, 16, 32]
        nStages = [8, 16, 32, 32, 32, 64]
        self.in_type = [r1] + [FIBERS[main_fiber](self.gspace, p, fixparams=self._fixparams) for p in nStages[:-1]]     # [0, 2, 5, 11, 11, 11, 22] -> [0, 1, 2, 4, 4, 4, 8]
        self.out_type = [FIBERS[main_fiber](self.gspace, p, fixparams=self._fixparams) for p in nStages[:-1]]    # [1, 2, 5, 5, 5, 11]
        
        if self.conv2triv:
            out_fiber = "trivial"
        else:
            out_fiber = main_fiber

        self.out_type += [FIBERS[out_fiber](self.gspace, nStages[-1], fixparams=self._fixparams)]   # + [8]
        
        self.features = enn.SequentialModule(
            EBlock(kernel_size=3, in_fiber= self.in_type[0], inner_fiber=inner_fiber, inner_stages=[nStages[0]]*2, 
                   out_fiber=self.out_type[0], stride=2, F=F, sigma=sigma, fixparams=fixparams),
            # EConstantPad2d(self.out_type[1], [1, 1, 0, 0], 0)
            EBlock(kernel_size=3, in_fiber= self.in_type[1], inner_fiber=inner_fiber, inner_stages=[nStages[1]]*2, 
                   out_fiber=self.out_type[1], stride=2, F=F, sigma=sigma, fixparams=fixparams),
            EConstantPad2d(self.out_type[1], [1, 1, 2, 2], 0),
            # EConstantPad2d(self.out_type[1], [3, 3, 3, 3], 0),
            # EConstantPad2d(self.out_type[1], [2, 2, 2, 2], 0),
            EBlock(kernel_size=3, in_fiber= self.in_type[2], inner_fiber=inner_fiber, inner_stages=[nStages[2]]*2, 
                   out_fiber=self.out_type[2], stride=1, F=F, sigma=sigma, fixparams=fixparams),
            # EConstantPad2d(self.out_type[2], [2, 2, 2, 2], 0),
            EBlock(kernel_size=3, in_fiber= self.in_type[3], inner_fiber=inner_fiber, inner_stages=[nStages[3]]*2,          # 32->32
                   out_fiber=self.out_type[3], stride=1, F=F, sigma=sigma, fixparams=fixparams)
        )
        self.features1 = EBlock(kernel_size=3, in_fiber= self.in_type[4], inner_fiber=inner_fiber, inner_stages=[nStages[4]]*2,          # 32->32
                   out_fiber=self.out_type[4], stride=1, F=F, sigma=sigma, fixparams=fixparams)
            # EBlock(kernel_size=3, in_fiber= self.in_type[4], inner_fiber=inner_fiber, inner_stages=[nStages[4]]*2, 
            #       out_fiber=self.out_type[4], stride=1, F=F, sigma=sigma, fixparams=fixparams),
        self.features2 = EBlock(kernel_size=3, in_fiber= self.in_type[5], inner_fiber=inner_fiber, inner_stages=[nStages[5]]*2,          # 32->64
                   out_fiber=self.out_type[5], stride=1, F=F, sigma=sigma, fixparams=fixparams)

            # EBlock_SA(kernel_size=3, in_fiber= self.in_type[5], inner_fiber=inner_fiber, inner_stages=[nStages[5]]*2, 
            #        out_fiber=self.out_type[5], stride=1, F=F, sigma=sigma, fixparams=fixparams, win_size=4),

            # EBlock_HiLo_SA(kernel_size=3, in_fiber= self.in_type[5], inner_fiber=inner_fiber, inner_stages=[nStages[5]]*2, 
            #        out_fiber=self.out_type[5], stride=1, F=F, sigma=sigma, fixparams=fixparams),
        self.features3 = EBlock(kernel_size=3, in_fiber= self.out_type[5], inner_fiber=inner_fiber, inner_stages=[nStages[5]]*2,         # 64->64
                   out_fiber=self.out_type[5], stride=1, F=F, sigma=sigma, fixparams=fixparams)
            # conv1x1(self.in_type[-1], self.out_type[-1], padding=0, F=F, sigma=sigma, initialize=False)
        
        self.downdim = nn.Conv2d(40, 8, kernel_size=1, padding=0)
        
        # if self._r > 0:
        #     id = (0, 4) if self._f else 4
        #     self.restrict1 = self._restrict_layer(id)
        # else:
        #     self.restrict1 = lambda x: x

        # if self._r > 1:
        #     id = (0, 1) if self._f else 1
        #     self.restrict2 = self._restrict_layer(id)
        # else:
        #     self.restrict2 = lambda x: x

        if self.conv2triv:
            self.relu = enn.ReLU(self.out_type[-1], inplace=True)
        else:
            self.mp_re1 = enn.SequentialModule(
                enn.GroupPooling(self.out_type[4]),
                enn.ReLU(enn.GroupPooling(self.out_type[-3]).out_type, inplace=True)
                )
            self.mp_re2 = enn.SequentialModule(
                enn.GroupPooling(self.out_type[5]),
                enn.ReLU(enn.GroupPooling(self.out_type[5]).out_type, inplace=True)
                )
            self.mp_re3 = enn.SequentialModule(
                enn.GroupPooling(self.out_type[5]),
                enn.ReLU(enn.GroupPooling(self.out_type[5]).out_type, inplace=True)
                )
            

        # self.downdim = nn.Conv2d(11, 8, kernel_size=1, padding=0)    
        # self.linear = nn.Linear(self.relu.out_type.size, num_classes)
        
        if self.train_flag:
            for _, module in self.named_modules():
                if isinstance(module, enn.R2Conv):
                    if deltaorth:
                        init.deltaorthonormal_init(module.weights.data, module.basisexpansion)
                    else:
                        init.generalized_he_init(module.weights.data, module.basisexpansion)
                elif isinstance(module, nn.Conv2d):
                    nn.init.orthogonal_(module.weight.data, gain=0.6)
                    try:
                        nn.init.constant_(module.bias.data, 0.01)
                    except:
                        pass
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
                elif isinstance(module, nn.Linear):
                    module.bias.data.zero_()
        
        print("MODEL TOPOLOGY:")
        # for i, (name, mod) in enumerate(self.named_modules()):
        #     print(f"\t{i} - {name}")
        # for i, (name, mod) in enumerate(self.named_modules()):
        #     params = sum([p.numel() for p in mod.parameters() if p.requires_grad])
        #     if isinstance(mod, nn.EquivariantModule) and isinstance(mod.in_type, nn.FieldType) and isinstance(mod.out_type,
        #                                                                                                 nn.FieldType):
        #         print(f"\t{i: <3} - {name: <70} | {params: <8} | {mod.in_type.size: <4}- {mod.out_type.size: <4}")
        #     else:
        #         print(f"\t{i: <3} - {name: <70} | {params: <8} |")
        tot_param = sum([p.numel() for p in self.parameters() if p.requires_grad])
        print("Total number of parameters:", tot_param)
                
        # if self.train_flag:
        #     self.features.apply(weights_init)
        return

    def _restrict_layer(self, subgroup_id):
        layers = list()
        layers.append(enn.RestrictionModule(self._in_type, subgroup_id))
        layers.append(enn.DisentangleModule(layers[-1].out_type))
        self._in_type = layers[-1].out_type
        self.gspace = self._in_type.gspace
        
        restrict_layer = enn.SequentialModule(*layers)
        return restrict_layer
    

    def _wide_layer(self, block, planes: int, num_blocks: int, dropout_rate: float, stride: int,
                    main_fiber: str = "regular",
                    inner_fiber: str = "regular",
                    out_fiber: str = None,
                    ):
        
        self._layer += 1
        print("start building", self._layer)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        main_type = FIBERS[main_fiber](self.gspace, planes, fixparams=self._fixparams)
        inner_class = FIBERS[inner_fiber](self.gspace, planes, fixparams=self._fixparams)
        if out_fiber is None:
            out_fiber = main_fiber
        out_type = FIBERS[out_fiber](self.gspace, planes, fixparams=self._fixparams)
        
        for b, stride in enumerate(strides):
            if b == num_blocks - 1:
                out_f = out_type
            else:
                out_f = main_type
            layers.append(
                block(self._in_type, inner_class, dropout_rate, stride, out_fiber=out_f, sigma=self._sigma, F=self._F))
            self._in_type = out_f
        print("built", self._layer)
        return enn.SequentialModule(*layers)


    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    

    def forward(self, input, angle=0):
        # input_n = self.input_norm(input)
        g_input = enn.GeometricTensor(input, self.in_type[0])

        out = self.features(g_input)     # bx11x34x10
        out1 = self.features1(out)
        out2 = self.features2(out1)
        out3 = self.features3(out2)

        if not self.conv2triv:
            out = torch.cat((self.mp_re1(out1).tensor, self.mp_re2(out2).tensor, self.mp_re3(out3).tensor), dim=1)
        # print(out.tensor.shape)
        # out = self.relu(out)
        # print(out.tensor.shape)
        # out = out.tensor
        # print(out.shape)
        x_features = self.downdim(out)

        # print(x_features.shape)

        b, o, dh, dw = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), dh, dw, o)
        else:
            x = x_features_trans.view(x_features_trans.size(0), dh, dw, o)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return x.permute(0, 3, 1, 2)
        # return L2Norm()(x)


class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.act_num = act_num
        self.deploy = deploy
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num*2 + 1, act_num*2 + 1))
        if deploy:
            self.bias = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None
            self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        nn.init.trunc_normal_(self.weight, std=0.02)
        # weight_init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        if self.deploy:
            return F.conv2d(
                super(activation, self).forward(x), 
                self.weight, self.bias, padding=self.act_num, groups=self.dim)
        else:
            return self.bn(F.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=self.act_num, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class BlockVanilla(nn.Module):
    def __init__(self, dim, dim_out, semodule, act_num=3, stride=2, deploy=False, ada_pool=None):
        super(BlockVanilla, self).__init__()
        self.act_learn = 1
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6)
            )

        if not ada_pool:
            self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        else:
            self.pool = nn.Identity() if stride == 1 else nn.AdaptiveMaxPool2d((ada_pool, ada_pool))

        self.act = activation(dim_out, act_num, deploy=self.deploy)

        self.se = semodule
 
    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            
            # We use leakyrelu to implement the deep training technique.
            x = F.leaky_relu(x,self.act_learn)
            
            x = self.conv2(x)

        x = self.pool(x)
        x = self.act(x)
        # se
        x = self.se(x)
        return x

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        self.conv1[0].weight.data = kernel
        self.conv1[0].bias.data = bias
        # kernel, bias = self.conv2[0].weight.data, self.conv2[0].bias.data
        kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight.data = torch.matmul(kernel.transpose(1,3), self.conv1[0].weight.data.squeeze(3).squeeze(2)).transpose(1,3)
        self.conv.bias.data = bias + (self.conv1[0].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True
 

class VanillaNet(nn.Module):
    def __init__(self, in_chans=1, num_classes=16, dims=[8, 8, 16, 32, 64, 64, 128], 
                 act_num=1, strides=[1,2,1,1,1,1], deploy=False, ada_pool=None, **kwargs):
        super().__init__()
        self.deploy = deploy
        if self.deploy:
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.stem1 = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0], eps=1e-6),
            )
            self.stem2 = nn.Sequential(
                nn.Conv2d(dims[0], dims[0], kernel_size=1, stride=1),
                nn.BatchNorm2d(dims[0], eps=1e-6),
                nn.ReLU(inplace=True)
            )

        self.act_learn = 1

        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            if not ada_pool:
                stage = BlockVanilla(dim=dims[i], dim_out=dims[i+1], semodule=SeModule(dims[i+1]), act_num=act_num, stride=strides[i], deploy=deploy)
            else:
                stage = BlockVanilla(dim=dims[i], dim_out=dims[i+1], semodule=SeModule(dims[i+1]), act_num=act_num, stride=strides[i], deploy=deploy, ada_pool=ada_pool[i])

            self.stages.append(stage)
        self.depth = len(strides)

        if self.deploy:
            self.cls = nn.Sequential(
                nn.Conv2d(dims[-1], num_classes, 1),
            )
        else:
            self.cls1 = nn.Sequential(
                nn.Conv2d(dims[-1], num_classes, 1),
                nn.BatchNorm2d(num_classes, eps=1e-6),
            )
            self.cls2 = nn.Sequential(
                nn.Conv2d(num_classes, num_classes, 1)
            )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            try:
                nn.init.constant_(m.bias, 0)
            except:
                pass

    def change_act(self, m):
        for i in range(self.depth):
            self.stages[i].act_learn = m
        self.act_learn = m

    def forward(self, x, angle=0):
        if self.deploy:
            x = self.stem(x)
        else:
            x = self.stem1(x)
            x = torch.nn.functional.leaky_relu(x,self.act_learn)
            x = self.stem2(x)

        for i in range(self.depth):
            x = self.stages[i](x)

        if self.deploy:
            x_features = self.cls(x)
        else:
            x = self.cls1(x)
            x = F.leaky_relu(x,self.act_learn)
            x_features = self.cls2(x)
        # return x.view(x.size(0),-1)

        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x_features_expand = x_features_expand.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x_features_expand = x_features_trans.view(b, 16, 2, o//2).permute(0, 2, 1, 3).contiguous()
            x = x_features_expand.view(x_features_expand.size(0), -1)

        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)



    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        self.stem2[2].switch_to_deploy()
        kernel, bias = self._fuse_bn_tensor(self.stem1[0], self.stem1[1])
        self.stem1[0].weight.data = kernel
        self.stem1[0].bias.data = bias
        kernel, bias = self._fuse_bn_tensor(self.stem2[0], self.stem2[1])
        self.stem1[0].weight.data = torch.einsum('oi,icjk->ocjk', kernel.squeeze(3).squeeze(2), self.stem1[0].weight.data)
        self.stem1[0].bias.data = bias + (self.stem1[0].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.stem = torch.nn.Sequential(*[self.stem1[0], self.stem2[2]])
        self.__delattr__('stem1')
        self.__delattr__('stem2')

        for i in range(self.depth):
            self.stages[i].switch_to_deploy()

        kernel, bias = self._fuse_bn_tensor(self.cls1[2], self.cls1[3])
        self.cls1[2].weight.data = kernel
        self.cls1[2].bias.data = bias
        kernel, bias = self.cls2[0].weight.data, self.cls2[0].bias.data
        self.cls1[2].weight.data = torch.matmul(kernel.transpose(1,3), self.cls1[2].weight.data.squeeze(3).squeeze(2)).transpose(1,3)
        self.cls1[2].bias.data = bias + (self.cls1[2].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.cls = torch.nn.Sequential(*self.cls1[0:3])
        self.__delattr__('cls1')
        self.__delattr__('cls2')
        self.deploy = True

# @register_model
def HardNet_fast_twice_big_vanila(train_flag=False):
    model = VanillaNet(dims=[8, 8, 16, 32, 64, 64, 128], strides=[1,2,1,1,1,1])
    return model

def HardNet_fast_twice_half3_vanila(train_flag=False):
    model = VanillaNet(dims=[8, 8, 16, 16, 16, 16, 32], strides=[1,2,1,1,1,1])
    return model

def HardNet_fast_twice_mid_vanila(train_flag=False):
    model = VanillaNet(dims=[12, 12, 20, 20, 20, 20, 32], strides=[1,2,1,1,1,1])
    return model