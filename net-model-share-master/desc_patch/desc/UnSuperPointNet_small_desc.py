"""latest version of SuperpointNet. Use it!

"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from ..models.unet_parts import *
import numpy as np

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
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

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

# from models.SubpixelNet import SubpixelNet
class UnSuperPointNet_small_desc(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, subpixel_channel=1):
        super(UnSuperPointNet_small_desc, self).__init__()
       
        self.block_descriptor = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
       

        self.convPb_descriptor = torch.nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)        
        # self.reshape_descriptor = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)   
        # self.bnDb_descriptor = nn.BatchNorm2d(128)
        self.output = None



    def forward(self, x4):
        desc = None
        # Descriptor Head.
        cPa_descriptor = self.block_descriptor(x4)
        desc = self.convPb_descriptor(cPa_descriptor)
        # desc = self.bnDb_descriptor(desc)
        # desc = self.reshape_descriptor(desc)

        desc = torch.nn.functional.normalize(input=desc, p=2, dim=1, eps=1e-12)  # -1 - 1
       
        # dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        output = desc

        return output


if __name__ == '__main__':
    model = UnSuperPointNet_small()
    total = 0
    for name, parameters in model.named_parameters():
        total += parameters.nelement()
        print(name, ":", parameters.size())
    print("Number of parameter: %.5fM" % (total / 1e6))





