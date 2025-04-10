from typing import Tuple

import torch
import torch.nn.functional as F

import math

import sys
sys.path.append("/data/yey/master/e2cnn-master/e2cnn-master")

import e2cnn.nn as enn
from e2cnn.nn import init
from e2cnn import gspaces

from argparse import ArgumentParser
import torch

from e2cnn import gspaces
from e2cnn import nn

import numpy as np

from PIL import Image
import torchvision

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # nn.init.orthogonal(m.weight.data, gain=0.6)
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return

class EConstantPad2d(enn.EquivariantModule):
    def __init__(self, in_type, padding: Tuple = None, value: float = None):
        super(EConstantPad2d, self).__init__()
        
        self.in_type = in_type
        self.out_type = in_type
        self.padding = padding
        self.value = value

    def forward(self, input: enn.GeometricTensor) -> enn.GeometricTensor:
        assert input.type == self.in_type, "Error! the type of the input does not match the input type of this module"
        return enn.GeometricTensor(F.pad(input.tensor, self.padding, 'constant', self.value), self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, _, hi, wi = input.shape
        ho = hi + self.padding[2] + self.padding[3]
        wo = wi + self.padding[0] + self.padding[1]

        return b, self.out_type.size, ho, wo

class Block_E2_Rot8(enn.EquivariantModule):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, stride, r2_act):
        super(Block_E2_Rot8, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)
        
        assert in_size % 4 in (0,1)
        assert expand_size % 4 == 0
        assert out_size % 4 == 0

        if in_size % 4 == 0:
            _in_size = int(in_size // 4)
        _expand_size = int(expand_size // 4)
        _out_size = int(out_size // 4)

        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = r2_act
        # the input image is a scalar field, corresponding to the trivial representation
        if in_size % 4 == 1:
            in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        else:
            in_type = nn.FieldType(self.r2_act, _in_size*[self.r2_act.regular_repr])
        
        self.in_type = in_type

        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = nn.FieldType(self.r2_act, _expand_size*[self.r2_act.regular_repr])

        self.block1_conv = enn.R2Conv(in_type, out_type, kernel_size=1, padding=0, bias=False)
        self.block1_bn = enn.InnerBatchNorm(out_type)
        self.block1_relu = enn.ReLU(out_type, inplace=True)

        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1_relu.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, _expand_size*[self.r2_act.regular_repr])
        if stride == 1:
            self.block2_conv = enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False, groups=_expand_size)
            self.block2_bn = enn.InnerBatchNorm(out_type)
            self.block2_relu = enn.ReLU(out_type, inplace=True)
        else:
            self.block2_conv = enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False, stride=2, groups=_expand_size)
            # nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2),
            self.block2_bn = enn.InnerBatchNorm(out_type)
            self.block2_relu = enn.ReLU(out_type, inplace=True)



        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2_relu.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, _out_size*[self.r2_act.regular_repr])

        self.block3_conv = enn.R2Conv(in_type, out_type, kernel_size=1, padding=0, bias=False)
        self.block3_bn = enn.InnerBatchNorm(out_type)
        self.block3_relu = enn.ReLU(out_type, inplace=True)
        
        self.out_type = out_type

        self.shortcut = enn.SequentialModule(enn.IdentityModule(self.out_type))

        if stride == 1 and in_size != out_size:
            self.shortcut = enn.SequentialModule(
                enn.R2Conv(self.in_type, self.out_type, kernel_size=1,stride=1, padding=0, bias=False),
                enn.InnerBatchNorm(self.out_type)
            )

    def forward(self, x):
        out = self.block1_relu(self.block1_bn(self.block1_conv(x)))
        out = self.block2_relu(self.block2_bn(self.block2_conv(out)))
        out = self.block3_relu(self.block3_bn(self.block3_conv(out)))
        # out = self.block3_bn(self.block3_conv(out))
        # out = out + self.shortcut(x) if self.shortcut_flag else out
        out = out + self.shortcut(x) if self.stride == 1 else out

        return out
    
    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input.shape
        ho = math.floor((hi + 2 * (self.kernel_size // 2) - (self.kernel_size - 1) - 1) / self.stride + 1)
        wo = math.floor((wi + 2 * (self.kernel_size // 2) - (self.kernel_size - 1) - 1) / self.stride + 1)

        if self.shortcut_flag:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return b, self.out_type.size, ho, wo

class HardNet_dense_E2(torch.nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_dense_E2, self).__init__()
        self.train_flag = train_flag
        self.gauss = torch.nn.Parameter(torch.ones(3), requires_grad=False)

        self.r2_act = gspaces.Rot2dOnR2(N=4)
        self.in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.out_type = nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])

        self.features = nn.SequentialModule(  
            Block_E2_Rot8(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),stride=2, r2_act=self.r2_act),
            Block_E2_Rot8(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),stride=2, r2_act=self.r2_act),
            EConstantPad2d(nn.FieldType(self.r2_act, 4*[self.r2_act.regular_repr]),(1,1,2,2),0),
            Block_E2_Rot8(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),stride=1, r2_act=self.r2_act),
            Block_E2_Rot8(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),stride=1, r2_act=self.r2_act),
            Block_E2_Rot8(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),stride=1, r2_act=self.r2_act),
            Block_E2_Rot8(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),stride=1, r2_act=self.r2_act),
            Block_E2_Rot8(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),stride=1, r2_act=self.r2_act),
            enn.GroupPooling(self.out_type)
        )
        self.post = torch.nn.Conv2d(16, 8, kernel_size=1, padding=0)
        
        # self.features = Block_E2_Rot8(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),stride=2, r2_act=self.r2_act)
        # if self.train_flag:
        #     self.features.apply(weights_init)
          
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        assert input.size(2) in [144,128]
        assert input.size(3) == 52

        x = nn.GeometricTensor(input, self.in_type)

        x_features = self.features(x)
        x = self.post(x_features.tensor)
        return x

# class HardNet_dense_E2(torch.nn.Module):
#     """HardNet model definition
#     """
#     def __init__(self, train_flag=False):
#         super(HardNet_dense_E2, self).__init__()
#         self.train_flag = train_flag

#         self.r2_act = gspaces.Rot2dOnR2(N=16)
#         self.in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
#         self.out_type = nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])

#         self.pad = torch.nn.ZeroPad2d((8,8,8,8))
#         self.features = nn.SequentialModule(  
#             Block_E2_Rot8(kernel_size=3, in_size=int(1), expand_size=int(16), out_size=int(16),stride=2, r2_act=self.r2_act),
#             Block_E2_Rot8(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),stride=1, r2_act=self.r2_act),
#             Block_E2_Rot8(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),stride=1, r2_act=self.r2_act),
#             Block_E2_Rot8(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),stride=1, r2_act=self.r2_act),
#             Block_E2_Rot8(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),stride=1, r2_act=self.r2_act),
#             Block_E2_Rot8(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),stride=1, r2_act=self.r2_act),
#             # enn.GroupPooling(self.out_type)
#         )
#         self.post = torch.nn.Conv2d(16, 16, kernel_size=1, padding=0)

#         # self.features = Block_E2_Rot8(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),stride=2, r2_act=self.r2_act)
#         # if self.train_flag:
#         #     self.features.apply(weights_init)
          
#         return
    
#     def input_norm(self,x):
#         flat = x.view(x.size(0), -1)
#         mp = torch.mean(flat, dim=1)
#         sp = torch.std(flat, dim=1) + 1e-7
#         return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
#     def forward(self, input):
#         assert input.size(2) == 136
#         assert input.size(3) == 40

#         input = torchvision.transforms.functional.resize(input, [137,41])
#         x = self.input_norm(input)
#         x = nn.GeometricTensor(self.pad(x), self.in_type)

#         x_features = self.features(x).tensor
#         # x = self.post(x_features.tensor)
#         return x_features



def test_model(model: torch.nn.Module):
    # evaluate the `model` on 8 rotated versions of the input image `x`
    model.eval()
    
    wrmup = (torch.randn(1, 1, 136, 40)*10).to(device)
    wrmup_90 = wrmup.permute(0,1,3,2).flip(dims=[-2]) #逆时针旋转

    
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(16):
            x_transformed = torchvision.transforms.functional.rotate(wrmup,r*22.5)
            # x_transformed = torchvision.transforms.functional.resize(x_transformed, [137,41])
            x_transformed = x_transformed.to(device)

            # x_transformed_90 = torchvision.transforms.functional.rotate(wrmup_90,r*45)
            # x_transformed_90 = torchvision.transforms.functional.resize(x_transformed_90, [41,137])
            # x_transformed_90 = x_transformed_90.to(device)

            y = model(x_transformed)
            # y_90 = model(x_transformed_90)
    
            y = y[0, ...].detach().unsqueeze(1)
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
    model = HardNet_dense_E2(train_flag=True).to(device)
    tot_param = sum([p.numel() for p in model.parameters() if p.requires_grad])

    # evaluate the model
    test_model(model)