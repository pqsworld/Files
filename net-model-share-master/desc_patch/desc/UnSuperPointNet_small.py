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
class UnSuperPointNet_small(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, subpixel_channel=1):
        super(UnSuperPointNet_small, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 1
        model = [  # nn.ReflectionPad2d(3),
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True)]
        model += [  # nn.ReflectionPad2d(3),
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True)]

        model += [nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(8),
                  nn.ReLU(True)]
        model += [nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(16),
                  nn.ReLU(True)]
        model += [nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(32),
                  nn.ReLU(True)]
        for i in range(6):  # add ResNet blocks
            model += [
                Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1)]

        self.model = nn.Sequential(*model)

        self.nonlinear = hsigmoid()

        # Detector Head Score.
        self.block_score = Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1)
        self.convPb_score = torch.nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
       
        # Detector Head Position.
        self.block_position = Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1)
        self.convPb_position = torch.nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)
        self.bnPb_position = nn.BatchNorm2d(2)

        self.output = None



    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu
        x4 = self.model(x)  # encoder output

        # Score Head.
        cPa_score = self.block_score(x4)
        # cPa_score = self.bnPa_score(self.convPa_score(x4))
        # semi_score = self.nonlinear(self.bnPb_score(self.convPb_score(cPa_score)))
        semi_score = self.nonlinear(self.convPb_score(cPa_score))

        # Position Head.
        cPa_position = self.block_position(x4)
        # semi_position = self.nonlinear(self.convPb_position(cPa_position))
        semi_position = self.nonlinear(self.bnPb_position(self.convPb_position(cPa_position)))
# 
        semi = torch.cat((semi_score, semi_position), dim=1)

        desc = None
        # Descriptor Head.
       
       
        output = {'semi': semi, 'desc': desc, 'encoder': x4}
        self.output = output

        return output


if __name__ == '__main__':
    model = UnSuperPointNet_small()
    total = 0
    for name, parameters in model.named_parameters():
        total += parameters.nelement()
        print(name, ":", parameters.size())
    print("Number of parameter: %.5fM" % (total / 1e6))





