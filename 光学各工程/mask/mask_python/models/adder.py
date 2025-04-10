'''
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of BSD 3-Clause License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3-Clause License for more details.
'''
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import math

def adder2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    W_col = W.view(n_filters, -1)
    
    out = adder.apply(W_col,X_col)
    
    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()
    
    return out

class adder(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col,X_col)
        output = -(W_col.unsqueeze(2)-X_col.unsqueeze(0)).abs().sum(1)
        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5
        grad_X_col = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)
        
        return grad_W_col, grad_X_col
    
class adder2d(nn.Module):

    def __init__(self,input_channel,output_channel,kernel_size, stride=1, padding=0, bias = False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = torch.nn.Parameter(nn.init.normal_(torch.randn(output_channel,input_channel,kernel_size,kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x,self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        return output


from .MobileNet import hswish, hsigmoid
class AdderSeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(AdderSeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            adder2d(in_size, in_size // reduction, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            adder2d(in_size // reduction, in_size, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class AdderBlock(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(AdderBlock, self).__init__()
        self.stride = stride

        self.conv1 = adder2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = adder2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                adder2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MNV3_bufen_adder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_adder, self).__init__()

        model = [  # nn.ReflectionPad2d(3),
            adder2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]
        model += [  # nn.ReflectionPad2d(3),
            adder2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]

        model += [adder2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(8),
                  nn.ReLU(True)]
        model += [adder2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(16),
                  nn.ReLU(True)]
        model += [adder2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(32),
                  nn.ReLU(True)]
        model += [adder2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(40),
                  nn.ReLU(True)]
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                # CBABlock(kernel_size=3, in_size=int(ngf * mult), expand_size=int(ngf * mult * 1), out_size=int(ngf * mult),
                #       nolinear=hswish(), cbamodule=CBAModule(int(ngf * mult)), stride=1)]
                AdderBlock(kernel_size=3, in_size=int(40), expand_size=int(40), out_size=int(40),
                     nolinear=hswish(), semodule=AdderSeModule(int(40)), stride=1)]

        model += [nn.ConvTranspose2d(40, int(32),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=False),
                  norm_layer(int(32)),
                  nn.ReLU(True)]
        model += [nn.ConvTranspose2d(32, int(16),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=False),
                  norm_layer(int(16)),
                  nn.ReLU(True)]
        model += [nn.ConvTranspose2d(16, int(8),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=False),
                  norm_layer(int(8)),
                  nn.ReLU(True)]
        model += [nn.ConvTranspose2d(8, int(4),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=False),
                  norm_layer(int(4)),
                  nn.ReLU(True)]

        model += [  # nn.ReflectionPad2d(3),
            adder2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]
        model += [adder2d(ngf, output_nc, kernel_size=3, padding=1),
                  norm_layer(output_nc),
                  ]
        model += [hsigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):

        return self.model(input)
