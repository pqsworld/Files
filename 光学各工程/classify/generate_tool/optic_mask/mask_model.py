import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision
import numpy as np


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModuleMask(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModuleMask, self).__init__()
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


class BlockMask(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(BlockMask, self).__init__()
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
        if stride == 1 and in_size != out_size:
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
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MNV3_bufen_new(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new, self).__init__()

        model = [  # nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]
        model += [  # nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]

        model += [nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(8),
                  nn.ReLU(True)]
        model += [nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(16),
                  nn.ReLU(True)]
        model += [nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(32),
                  nn.ReLU(True)]
        model += [nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(40),
                  nn.ReLU(True)]
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                # CBABlock(kernel_size=3, in_size=int(ngf * mult), expand_size=int(ngf * mult * 1), out_size=int(ngf * mult),
                #       nolinear=hswish(), cbamodule=CBAModule(int(ngf * mult)), stride=1)]
                BlockMask(kernel_size=3, in_size=int(40), expand_size=int(40), out_size=int(40),
                     nolinear=hswish(), semodule=SeModuleMask(int(40)), stride=1)]

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
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
     
        model += [hsigmoid()]

        self.model = nn.Sequential(*model)
    def forward(self, input):
        return self.model(input)


