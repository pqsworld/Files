from sympy import re
import torch
import math
import torch.nn as nn
from typing import Tuple
import e2cnn.nn as enn
from e2cnn import gspaces
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Pad 
from torchvision.transforms import Resize 
from torchvision.transforms import ToTensor

from .enn_layers import FIELD_TYPE, convnxn, ennAdaptiveAvgPool

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

class Rehswish(nn.Module):
    def __init__(self, in_type, inplace=True):
        super(Rehswish, self).__init__()
        # self.in_type = FIELD_TYPE['regular'](gspace, in_size)
        self.in_type = in_type
        self.activate = enn.ReLU(in_type, inplace=inplace)
    def forward(self, x):
        relu6 = torch.clamp(self.activate(x).tensor, 0, 6)
        out = x.tensor * (relu6 + 3) / 6
        # out = x * enn.Relu(x + 3, inplace=True) / 6
        return enn.GeometricTensor(out, self.in_type)

class Rehsigmoid(nn.Module):
    def __init__(self, in_type, inplace=True):
        super(Rehsigmoid, self).__init__()
        self.in_type = in_type
        self.activate = enn.ReLU(in_type, inplace=inplace)
    def forward(self, x):
        relu6 = torch.clamp(self.activate(x).tensor, 0, 6)
        out = (relu6 + 3) / 6
        # out = F.relu6(x + 3, inplace=True) / 6
        return enn.GeometricTensor(out, self.in_type)

# def Rehsigmoid(in_type):
#     activate = enn.ReLU(in_type, inplace=True)
#     relu6 = torch.clamp(activate(x), 0, 6)
#     out = (relu6 + 3) / 6

class ReSeModule(nn.Module):
    def __init__(self, in_size, reduction=4, gspace=None):
        super(ReSeModule, self).__init__()
        self.in_field_type = FIELD_TYPE['regular'](gspace, in_size)
        self.reductiong_field_type = FIELD_TYPE['regular'](gspace, in_size // reduction)

        self.se = enn.SequentialModule(
            ennAdaptiveAvgPool(gspace, in_size, 1),
            enn.R2Conv(self.in_field_type, self.reductiong_field_type, kernel_size=1,
                      stride=1, padding=0, bias=False),
            enn.InnerBatchNorm(self.reductiong_field_type),
            enn.ReLU(self.reductiong_field_type, inplace=True),
            enn.R2Conv(self.reductiong_field_type, self.in_field_type, kernel_size=1,
                      stride=1, padding=0, bias=False),
            enn.InnerBatchNorm(self.in_field_type),
            enn.ReLU(self.in_field_type, inplace=True)
            # self.activate()
            # Rehsigmoid(self.in_field_type)
        )
        # self.activate = Rehsigmoid(self.in_field_type)
        # self.activate = enn.ReLU(self.in_field_type, inplace=True)

    def forward(self, x):
        '''GeometricTensor 不支持向量积'''
        return x * self.se(x)

class WideBasic(enn.EquivariantModule):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, stride, gspace, fixparams=False, in_type='regular', out_type='regular'):
        super(WideBasic, self).__init__()
        self.stride = stride

        if in_type == 'trivial':
            self.in_type =  FIELD_TYPE[in_type](gspace, in_size)
        else:
            self.in_type =  FIELD_TYPE[in_type](gspace, in_size, fixparams)
        self.expand_type =  FIELD_TYPE[out_type](gspace, expand_size, fixparams)
        self.out_type =  FIELD_TYPE[out_type](gspace, out_size, fixparams)

        self.conv1 = enn.R2Conv(self.in_type, self.expand_type,
                               kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn1 = enn.InnerBatchNorm(self.expand_type)
        self.nolinear1 = enn.ReLU(self.expand_type, inplace=True)

        self.conv2 = enn.R2Conv(self.expand_type, self.out_type,
                               kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn2 = enn.InnerBatchNorm(self.out_type)

        self.se = None

        self.shortcut = None
        if stride == 1 and in_size != out_size:
            self.shortcut = enn.SequentialModule(
                enn.R2Conv(self.in_type, self.out_type, kernel_size=1,
                          stride=stride, padding=0, bias=False),
                enn.InnerBatchNorm(self.out_type),
            )
        # if stride == 1 and in_size == out_size:
        #     self.shortcut = enn.SequentialModule(
        #         enn.R2Conv(self.in_field_type, self.out_field_type, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         enn.InnerBatchNorm(self.out_field_type),
        #     )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        # out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn2(self.conv2(out))
        if self.se != None:
            out = self.se(out)
        # out = out + self.shortcut(x) if self.stride == 1 else out
        if self.shortcut is not None:
            out = out + self.shortcut(x) 
        return out
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape

class ReBlock(enn.EquivariantModule):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, ReNolinear, Resemodule, stride, gspace, fixparams=False, in_type='regular', out_type='regular'):
        super(ReBlock, self).__init__()
        self.stride = stride

        if in_type == 'trivial':
            self.in_type =  FIELD_TYPE[in_type](gspace, in_size)
        else:
            self.in_type =  FIELD_TYPE[in_type](gspace, in_size, fixparams)
        self.expand_type =  FIELD_TYPE[out_type](gspace, expand_size, fixparams)
        self.out_type =  FIELD_TYPE[out_type](gspace, out_size, fixparams)

        self.conv1 = enn.R2Conv(self.in_type, self.expand_type,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = enn.InnerBatchNorm(self.expand_type)
        self.nolinear1 = ReNolinear(self.expand_type, inplace=True)
        self.conv2 = enn.R2Conv(self.expand_type, self.expand_type, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=self.expand_type.size // gspace.fibergroup.order(), bias=False)
        self.bn2 = enn.InnerBatchNorm(self.expand_type)
        self.nolinear2 = ReNolinear(self.expand_type, inplace=True)
        self.conv3 = enn.R2Conv(self.expand_type, self.out_type,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = enn.InnerBatchNorm(self.out_type)
        self.se = ReSeModule(out_size, gspace=gspace) if Resemodule is not None else None
        self.shortcut = enn.SequentialModule()
        # if stride == 1 and in_size != out_size:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_size, out_size, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )
        if stride == 1 and in_size == out_size:
            self.shortcut = enn.SequentialModule(
                enn.R2Conv(self.in_type, self.out_type, kernel_size=1,
                          stride=1, padding=0, bias=False),
                enn.InnerBatchNorm(self.out_type),
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

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape

class ReBlock_v2(enn.EquivariantModule):
    '''expand + depthwise + pointwise'''
    ''' stride=2时进行shortcut '''

    def __init__(self, kernel_size, in_size, expand_size, out_size, ReNolinear, Resemodule, stride, gspace, fixparams=False, in_type='regular', out_type='regular'):
        super(ReBlock_v2, self).__init__()
        self.stride = stride

        if in_type == 'trivial':
            self.in_type =  FIELD_TYPE[in_type](gspace, in_size)
        else:
            self.in_type =  FIELD_TYPE[in_type](gspace, in_size, fixparams)
        self.expand_type =  FIELD_TYPE[out_type](gspace, expand_size, fixparams)
        self.out_type =  FIELD_TYPE[out_type](gspace, out_size, fixparams)

        self.conv1 = enn.R2Conv(self.in_type, self.expand_type,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = enn.InnerBatchNorm(self.expand_type)
        self.nolinear1 = ReNolinear(self.expand_type, inplace=True)
        self.conv2 = enn.R2Conv(self.expand_type, self.expand_type, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=self.expand_type.size // gspace.fibergroup.order(), bias=False)
        self.bn2 = enn.InnerBatchNorm(self.expand_type)
        self.nolinear2 = ReNolinear(self.expand_type, inplace=True)
        self.conv3 = enn.R2Conv(self.expand_type, self.out_type,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = enn.InnerBatchNorm(self.out_type)
        self.se = ReSeModule(out_size, gspace=gspace) if Resemodule is not None else None
        self.shortcut = None
        # if stride == 1 and in_size != out_size:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_size, out_size, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )
        if stride != 1 and in_size != out_size:
            self.shortcut = enn.SequentialModule(
                enn.R2Conv(self.in_type, self.out_type, kernel_size=1,
                          stride=stride, padding=0, bias=False),
                enn.InnerBatchNorm(self.out_type),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        # out = out + self.shortcut(x) if self.stride == 1 else out
        out = out + self.shortcut(x) if self.shortcut != None else out
        return out

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape

class RePad2d(enn.EquivariantModule):
    def __init__(self, in_type=None, padding: Tuple=None, value: float=0):
        super(RePad2d, self).__init__()
        self.in_type    = in_type
        self.out_type   = in_type
        self.padding    = padding
        self.value      = value

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        assert x.type == self.out_type, "RePadding error! The x.type and out_type do not match."
        out_pad = F.pad(x.tensor, pad=self.padding, mode='constant', value=self.value)
        return enn.GeometricTensor(out_pad, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, _, hi, wi = input.shape
        ho = hi + self.padding[2] + self.padding[3]
        wo = wi + self.padding[0] + self.padding[1]

        return b, self.out_type.size, ho, wo

class ReHardNet_fast_featuremap_depth(nn.Module):
    """Rotation-equivariant HardNet model definition
    """
    def __init__(self, train_flag=False, orientation=4):
        super(ReHardNet_fast_featuremap_depth, self).__init__()
        self.train_flag = train_flag
        self.orientation = orientation
        # self.fixparams = fixparams
        self.gspace = gspaces.Rot2dOnR2(orientation)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 1)
        # self.out1_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * (64 // orientation))
        # self.gpool_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * (64 // orientation))

        self.layer1 = ReBlock(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8), ReNolinear=enn.ReLU,
                            Resemodule=None, stride=2, gspace=self.gspace, fixparams=False, in_type='trivial')
        self.layer2 = ReBlock(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16), ReNolinear=enn.ReLU,
                            Resemodule=None, stride=2, gspace=self.gspace, fixparams=False)
        self.layer_pad = RePad2d(self.layer2.out_type, padding=(1, 1, 2, 2), value=0.)    # 37x14
        self.layer3 = ReBlock(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32), ReNolinear=enn.ReLU,
                            Resemodule=None, stride=1, gspace=self.gspace, fixparams=False)
        self.layer4 = ReBlock(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32), ReNolinear=enn.ReLU,
                            Resemodule=None, stride=1, gspace=self.gspace, fixparams=False)
        self.layer5 = ReBlock(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32), ReNolinear=enn.ReLU,
                            Resemodule=None, stride=1, gspace=self.gspace, fixparams=False)
        self.layer6 = ReBlock(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64), ReNolinear=enn.ReLU,
                            Resemodule=None, stride=1, gspace=self.gspace, fixparams=False)
        self.layer7 = ReBlock(kernel_size=3, in_size=int(64), expand_size=int(64), out_size=int(64), ReNolinear=enn.ReLU,
                            Resemodule=None, stride=1, gspace=self.gspace, fixparams=False)

        self.gpool = enn.GroupPooling(self.layer6.out_type)
        self.active = enn.ReLU(self.gpool.out_type, inplace=True)

        self.downconv = nn.Conv2d(self.active.out_type.size, 8, kernel_size=1, padding=0)
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
        # input_norm = self.input_norm(input)
        input_norm = input
        input_geometrictensor = enn.GeometricTensor(input_norm, self.input_type)

        x1 = self.layer1(input_geometrictensor)
        x2 = self.layer2(x1)
        x2_pad = self.layer_pad(x2)
        x3 = self.layer3(x2_pad)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)

        x_out = self.gpool(x7)
        x_out = self.active(x_out)

        x_out = x_out.tensor
        x_features = self.downconv(x_out)
        
        bs, bins, h, w = x_features.shape

        if flip_flag:
            x_features_flip = x_features.permute(0, 2, 3, 1).contiguous().view(bs, -1, bins)
            x_features_flip = torch.flip(x_features_flip, dims=[1])
            x = x_features_flip.view(x_features_flip.size(0), -1)
        else:
            x = x_features.permute(0, 2, 3, 1).contiguous().view(x_features.size(0), -1)
        # print(x.shape)
        # return L2Norm()(x)
        return x.view(bs, h, w, bins).permute(0, 3, 1, 2)

    def forward_test(self, input, flip_flag=False):
        input_norm = self.input_norm(input)
        input_geometrictensor = enn.GeometricTensor(input_norm, self.input_type)

        x_features1 = self.layer1(input_geometrictensor)
        x_features2 = self.layer2(x_features1)
        x_features3 = self.layer3(x_features2)
        x_features4 = self.layer4(x_features3)
        x_features5 = self.layer5(x_features4)
        x_features6 = self.layer6(x_features5)

        x_out = self.gpool(x_features6)
        x_features = x_out.tensor
        
        bs, bins, h, w = x_features.shape
        x = x_features.permute(0, 2, 3, 1).contiguous().view(x_features.size(0), -1)
        out = x.view(bs, h, w, bins).permute(0, 3, 1, 2)
        out_part = out
        # out_part = out[0, :8, 32, 8]
        # out_part = out[0, :8, 34, 10]
        # out_part = out[0, :8, 17, 5]

        return out_part

class ReGhostModule(enn.EquivariantModule):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, gspace=None, init_type='regular', new_type='regular'):
        super(ReGhostModule, self).__init__()
        self.relu = relu
        self.gspace = gspace
        self.inp_c = int(inp.size / gspace.fibergroup.order())
        self.oup_c = int(oup.size / gspace.fibergroup.order())

        init_channels = math.ceil(self.oup_c / ratio)
        new_channels = init_channels*(ratio-1)
        self.init_type = FIELD_TYPE[init_type](gspace, init_channels * gspace.fibergroup.order(), fixparams=False)
        self.new_type = FIELD_TYPE[new_type](gspace, new_channels * gspace.fibergroup.order(), fixparams=False)
        
        self.primary_conv = enn.R2Conv(inp, self.init_type, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.primary_bn = enn.InnerBatchNorm(self.init_type)
        self.primary_act = enn.ReLU(self.init_type, inplace=True) if relu else enn.SequentialModule()

        self.cheap_conv = enn.R2Conv(self.init_type, self.new_type, kernel_size=dw_size, stride=1, padding=dw_size//2, groups=init_channels, bias=False)
        self.cheap_bn = enn.InnerBatchNorm(self.new_type)
        self.cheap_act = enn.ReLU(self.new_type, inplace=True) if relu else enn.SequentialModule()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x1 = self.primary_bn(x1)
        if self.relu:
            x1 = self.primary_act(x1)

        x2 = self.cheap_conv(x1)
        x2 = self.cheap_bn(x2)
        if self.relu:
            x2 = self.cheap_act(x2)

        out = enn.tensor_directsum(list([x1, x2]))
        # out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup_c,:,:]

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class ReGhostBottleneck(enn.EquivariantModule):
    def __init__(self, kernel_size, in_size, expand_size, out_size, ReNolinear, Resemodule, stride, gspace, fixparams=False, in_type='regular', out_type='regular'):
        super(ReGhostBottleneck, self).__init__()
        self.stride = stride

        if in_type == 'trivial':
            self.input_type =  FIELD_TYPE[in_type](gspace, in_size)
        else:
            self.input_type =  FIELD_TYPE[in_type](gspace, in_size, fixparams)
        self.expand_type =  FIELD_TYPE[out_type](gspace, expand_size, fixparams)
        self.output_type =  FIELD_TYPE[out_type](gspace, out_size, fixparams)

        # Point-wise expansion
        self.ghost1 = ReGhostModule(self.input_type, self.expand_type, relu=True, gspace=gspace)

         # Depth-wise convolution
        # if self.stride > 1:
        self.conv_dw = enn.R2Conv(self.expand_type, self.expand_type, kernel_size=kernel_size, stride=stride,
                            padding=(kernel_size-1)//2, groups=self.expand_type.size // gspace.fibergroup.order(), bias=False)
        self.bn_dw = enn.InnerBatchNorm(self.expand_type)

        # Squeeze-and-excitation
        if Resemodule:
            self.se = ReSeModule(out_size, gspace=gspace)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = ReGhostModule(self.expand_type, self.output_type, relu=False, gspace=gspace)

        self.shortcut = enn.SequentialModule()
        # if stride == 1 and in_size != out_size:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_size, out_size, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )
        if stride == 1 and in_size == out_size:
            self.shortcut = enn.SequentialModule(
                enn.R2Conv(self.input_type, self.output_type, kernel_size=1,
                          stride=1, padding=0, bias=False),
                enn.InnerBatchNorm(self.output_type),
            )

    def forward(self, x):
        out = self.ghost1(x)
        out = self.bn_dw(self.conv_dw(out))
        
        if self.se != None:
            out = self.se(out)

        out = self.ghost2(out)
        # out = out + self.shortcut(x) if self.stride == 1 else out
        out = out + self.shortcut(x) if (self.stride == 1 and out.shape[1] == x.shape[1]) else out
        return out

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.input_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape


class ReHardNet_fast_featuremap_ghost(nn.Module):
    """Rotation-equivariant HardNet model definition
    """
    def __init__(self, train_flag=False, orientation=4):
        super(ReHardNet_fast_featuremap_ghost, self).__init__()
        self.train_flag = train_flag
        self.orientation = orientation
        # self.fixparams = fixparams
        self.gspace = gspaces.Rot2dOnR2(orientation)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 1)
        # self.out1_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * (64 // orientation))
        # self.gpool_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * (64 // orientation))

        self.layer1 = ReBlock(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8), ReNolinear=enn.ReLU,
                            Resemodule=None, stride=2, gspace=self.gspace, fixparams=True, in_type='trivial')
        self.layer2 = ReBlock(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16), ReNolinear=enn.ReLU,
                            Resemodule=None, stride=2, gspace=self.gspace, fixparams=True)
        self.layer_pad = RePad2d(self.layer2.out_type, padding=(1, 1, 2, 2), value=0.)    # 37x14
        self.layer3 = ReBlock(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32), ReNolinear=enn.ReLU,
                            Resemodule=None, stride=1, gspace=self.gspace, fixparams=True)
        self.layer4 = ReBlock(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32), ReNolinear=enn.ReLU,
                            Resemodule=None, stride=1, gspace=self.gspace, fixparams=True)
        self.layer5 = ReBlock(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32), ReNolinear=enn.ReLU,
                            Resemodule=None, stride=1, gspace=self.gspace, fixparams=True)
        self.layer6 = ReGhostBottleneck(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64), ReNolinear=enn.ReLU,
                            Resemodule=None, stride=1, gspace=self.gspace, fixparams=True)
        
        self.gpool = enn.GroupPooling(self.layer6.output_type)
        self.active = enn.ReLU(self.gpool.out_type, inplace=True)

        self.downconv = nn.Conv2d(self.active.out_type.size, 8, kernel_size=1, padding=0)
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
        # input_norm = self.input_norm(input)   # 数据增强后不建议正则化
        input_norm = input
        input_geometrictensor = enn.GeometricTensor(input_norm, self.input_type)

        x_features1 = self.layer1(input_geometrictensor)
        x_features2 = self.layer2(x_features1)
        x_ft_pad    = self.layer_pad(x_features2)
        x_features3 = self.layer3(x_ft_pad)
        x_features4 = self.layer4(x_features3)
        x_features5 = self.layer5(x_features4)
        x_features6 = self.layer6(x_features5)

        x_out = self.gpool(x_features6)
        x_out = self.active(x_out)

        x_out = x_out.tensor
        x_features = self.downconv(x_out)
        
        bs, bins, h, w = x_features.shape

        if flip_flag:
            x_features_flip = x_features.permute(0, 2, 3, 1).contiguous().view(bs, -1, bins)
            x_features_flip = torch.flip(x_features_flip, dims=[1])
            x = x_features_flip.view(x_features_flip.size(0), -1)
        else:
            x = x_features.permute(0, 2, 3, 1).contiguous().view(x_features.size(0), -1)
        # print(x.shape)
        # return L2Norm()(x)
        return x.view(bs, h, w, bins).permute(0, 3, 1, 2)


def weights_init(m):
    if isinstance(m, enn.R2Conv):
        # nn.init.orthogonal(m.weight.data, gain=0.6)
        enn.init.deltaorthonormal_init(m.weight.data, m.basisexpansion)
        # try:
        #     enn.init.constant(m.bias.data, 0.01)
        # except:
        #     pass
    return

def test_model(model: torch.nn.Module, x: Image, device='cpu'):
    # evaluate the `model` on 8 rotated versions of the input image `x`     
    # model.eval()
    # wrmup = model(x.to(device))
    # del wrmup
    # x = torch.tensor(np.array(x)).to(device).reshape(1, 1, 29, 29)
    totensor = ToTensor()
    # pad = Pad((0, 0, 2, 2), fill=0)
    # x = pad(x)
    print()
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(8):
            x_transformed = totensor(x.rotate(r*45., Image.BILINEAR)).reshape(1, 1, 145, 53)
            x_transformed = x_transformed.to(device)
            y = model(x_transformed)
            y = y.to('cpu').numpy().squeeze()
            angle = r * 45

            m = cv2.getRotationMatrix2D((41 / 2, 137 / 2), angle, 1)
            M = np.concatenate((m, [[0, 0, 1.]]), axis=0)
            pts = M @ np.array([17., 65., 1.]).T
            pts_ds = pts / 4

            y = y[:8, round(pts_ds[1]), round(pts_ds[0])]
            print("{:5d} : {}".format(angle, y))
    print('##########################################################################################')     
    print()      
if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    GPU_ids = 2  
    device = torch.device('cuda:' + str(GPU_ids))
    
    # net = ReHardNet_fast_featuremap_depth(train_flag=False).to(device)
    net = ReHardNet_fast_featuremap_ghost(train_flag=False).to(device)
    # from torchstat import stat
    net.eval()
    # x = torch.rand(16,1,16,16)
    # net(x)
    # from torchsummary import summary
    # summary(x, input_size=(1, 136, 32))
    print('# net parameters:', sum(param.numel() for param in net.parameters()))
    print('# net parameters memory:', sum(param.numel()*4/1024 for param in net.parameters()), 'kb')

    import numpy as np
    import random
    import cv2
    # image = Image.fromarray(np.array(np.random.randn(137, 41)))
    image = Image.fromarray(np.array(np.random.randn(145, 53)))
    test_model(net, image, device=device)
