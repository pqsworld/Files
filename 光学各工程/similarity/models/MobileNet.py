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


class ChannelAttention(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1,
                      stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1,
                      stride=1, padding=0),
        )

        self.nonlinear = hsigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.nonlinear(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3,
                      stride=1, padding=1),
        )
        self.nonlinear = hsigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.nonlinear(x)

class CBAModule(nn.Module):  # CNN Block Attention Module
    def __init__(self, in_size, reduction=4):
        super(CBAModule, self).__init__()
        self.ca = ChannelAttention(in_size)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class SABlock(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, cbamodule, stride):
        super(SABlock, self).__init__()
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
        self.se = cbamodule
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

class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.nolinear = nn.ReLU(True)

    def forward(self, x):
        out = self.depthwise_conv(x)
        #out = self.nolinear(out)
        out = self.pointwise_conv(out)
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
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        if self.nolinear1 != None:
            out = self.nolinear1(self.bn1(self.conv1(x)))
            out = self.nolinear2(self.bn2(self.conv2(out)))
        else:
            out = self.bn1(self.conv1(x))
            out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out
class Block_nc(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_nc, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False,padding_mode = 'reflect')
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
        if self.nolinear1 != None:
            out = self.nolinear1(self.bn1(self.conv1(x)))
            out = self.nolinear2(self.bn2(self.conv2(out)))
        else:
            out = self.bn1(self.conv1(x))
            out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 3 else out
        return out
class Block_4(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_4, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False, padding_mode = 'reflect')
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
        if self.nolinear1 != None:
            out = self.nolinear1(self.bn1(self.conv1(x)))
            out = self.nolinear2(self.bn2(self.conv2(out)))
        else:
            out = self.bn1(self.conv1(x))
            out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out
class Block_up(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride,padding=None,output_padding=1):
        super(Block_up, self).__init__()
        self.stride = stride
        padding_p = 1
        
        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        #nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1),  # featrues_upconv1
        self.conv2 = nn.ConvTranspose2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=padding_p, groups=expand_size, bias=False,output_padding=output_padding)
        #self.conv2 = nn.ConvTranspose2d(expand_size, expand_size, kernel_size=kernel_size,
                               #stride=stride, padding=padding_p, output_padding=padding_p, groups=expand_size, bias=False)
        #self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               #stride=stride, padding=padding_p, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        #self.shortcut = nn.Sequential()
        #if stride == 1 and in_size != out_size:
            #self.shortcut = nn.Sequential(
                #nn.Conv2d(in_size, out_size, kernel_size=1,
                          #stride=1, padding=0, bias=False),
                #nn.BatchNorm2d(out_size),
            #)

    def forward(self, x):  
        if self.nolinear1 != None:
            out = self.nolinear1(self.bn1(self.conv1(x)))
            out = self.nolinear2(self.bn2(self.conv2(out)))
        else:
            out = self.bn1(self.conv1(x))
            out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        #out = out + self.shortcut(x) if self.stride == 3 else out
        return out

class APBlock(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, downsample):
        super(APBlock, self).__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        if downsample:
            self.nolinear2 = nn.Sequential(
                nolinear,
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()
        # if downsample:
        #     self.shortcut = nn.Sequential(
        #         nn.AvgPool2d(kernel_size=2, stride=2),
        #         nn.Conv2d(in_size, out_size, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )
        # else:
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
        out = out + self.shortcut(x) if self.downsample != 1 else out
        # out = out
        return out

class MessagePass(nn.Module):
    def __init__(self, output_dim,
                 axis):
        super(MessagePass, self).__init__()

        self.output_dim = output_dim
        self.axis = axis

        assert self.axis in [1, 2]

        if self.axis == 1:
            self.conv = nn.Conv2d(output_dim, output_dim,
                                  kernel_size=(1, 9), stride=1, padding=(0, 4), bias=False)
            self.bn = nn.BatchNorm2d(output_dim)
            self.nonlinear = nn.ReLU(True)
        if self.axis == 2:
            self.conv = nn.Conv2d(output_dim, output_dim,
                                  kernel_size=(9, 1), stride=1, padding=(4, 0), bias=False)
            self.bn = nn.BatchNorm2d(output_dim)
            self.nonlinear = nn.ReLU(True)

    def forward(self, inputs):
        h, w = int(inputs.shape[2]), int(inputs.shape[3])

        if self.axis == 1:
            n = h
        if self.axis == 2:
            n = w

        feature_slice_old = []
        feature_slice_new = []

        for i in range(n):
            if self.axis == 1:
                cur_slice = torch.unsqueeze(inputs[:, :, i, :], 2)
            else:
                cur_slice = torch.unsqueeze(inputs[:, :, :, i], 3)
            feature_slice_old.append(cur_slice)

            if i == 0:
                feature_slice_new.append(cur_slice)
            else:
                tmp = self.nonlinear(self.bn(self.conv(feature_slice_new[i - 1])))
                tmp = tmp + feature_slice_old[i]
                feature_slice_new.append(tmp)

        feature_slice_old = feature_slice_new

        for i in reversed(range(n)):
            if i == (n - 1):
                pass
            else:
                tmp = self.nonlinear(self.bn(self.conv(feature_slice_new[i + 1])))
                tmp = tmp + feature_slice_old[i]
                feature_slice_new[i] = tmp


        output = torch.stack(feature_slice_new, dim=self.axis + 1)
        output = torch.squeeze(output, dim=self.axis + 2)

        return output

class TransposeConv(nn.Module):
    def __init__(self, in_size, out_size):
        super(TransposeConv, self).__init__()
        self.transposeconv = nn.ConvTranspose2d(in_channels=in_size, out_channels=out_size,
                                                kernel_size=3, stride=2,
                                                padding=1, output_padding=1,
                                                bias=False)
        self.norm = nn.BatchNorm2d(out_size)
        self.nonlinear = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        out = self.nonlinear(self.norm(self.transposeconv(x)))
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation_function = hsigmoid()

    def forward(self, x):
        return self.activation_function(self.conv(x))


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class MNV3_test_large(nn.Module):
    def __init__(self, numclasses):
        super(MNV3_test_large, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            SeModule(8),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            SeModule(16),

            # nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
            #           stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
        )
        self.bneck = nn.Sequential(
            Block(kernel_size=3, in_size=16, expand_size=64, out_size=16,
                  nolinear=hswish(), semodule=SeModule(16), stride=1),
            Block(kernel_size=3, in_size=16, expand_size=64, out_size=16,
                  nolinear=hswish(), semodule=SeModule(16), stride=1),
            Block(kernel_size=3, in_size=16, expand_size=64, out_size=16,
                  nolinear=hswish(), semodule=SeModule(16), stride=1),

        )
        self.up = nn.Sequential(
            # TransposeConv(in_size=32, out_size=16),
            TransposeConv(in_size=16, out_size=8),
            SeModule(8),
            TransposeConv(in_size=8, out_size=4),
            SeModule(4),

            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, True)
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        )
        self.activate_function = hsigmoid()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.down(x)
        x = self.bneck(x)
        x = self.up(x)
        x = self.outconv(x)
        x = self.activate_function(x)
        return x


class MNV3_GAN(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_GAN, self).__init__()

        model = [  # nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]
        model += [  # nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]
        # model += [SeModule(ngf)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            # model += [SeModule(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                # CBABlock(kernel_size=3, in_size=int(ngf * mult), expand_size=int(ngf * mult * 1), out_size=int(ngf * mult),
                #       nolinear=hswish(), cbamodule=CBAModule(int(ngf * mult)), stride=1)]
                Block(kernel_size=3, in_size=int(ngf * mult), expand_size=int(ngf * mult * 1), out_size=int(ngf * mult),
                     nolinear=hswish(), semodule=SeModule(int(ngf * mult)), stride=1)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=False),
                      norm_layer(int(ngf * mult / 2)),
                      nn.LeakyReLU(0.1, True)]
            # model += [SeModule(int(ngf * mult / 2))]

        model += [  # nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.LeakyReLU(0.1, True)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
        model += [hsigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class MNV3_bufen(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen, self).__init__()

        model = [  # nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]
        model += [  # nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]
        # model += [SeModule(ngf)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            # model += [SeModule(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                # CBABlock(kernel_size=3, in_size=int(ngf * mult), expand_size=int(ngf * mult * 1), out_size=int(ngf * mult),
                #       nolinear=hswish(), cbamodule=CBAModule(int(ngf * mult)), stride=1)]
                Block(kernel_size=3, in_size=int(ngf * mult), expand_size=int(ngf * mult * 1), out_size=int(ngf * mult),
                     nolinear=hswish(), semodule=SeModule(int(ngf * mult)), stride=1)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=False),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            # model += [SeModule(int(ngf * mult / 2))]

        model += [  # nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
        model += [hsigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        # print(self.model[:21](input))
        # print(self.model(input))
        return self.model(input)

class MNV3_bufen_new(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
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
                Block(kernel_size=3, in_size=int(40), expand_size=int(40), out_size=int(40),
                     nolinear=hswish(), semodule=SeModule(int(40)), stride=1)]

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

class MNV3_bufen_new3(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new3,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv5=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(True),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        )
        self.classifier=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            #nn.Dropout2d(0.5),
            nn.Conv2d(32, 1, kernel_size=1,stride=1, padding=0, bias=True),
            hsigmoid(),
        )
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        self.featrues_upconv5=nn.Sequential(
            nn.ConvTranspose2d(32, int(16),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(16)),
            nn.ReLU(True),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(8)),
            nn.ReLU(True),
        )
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        #print("out4:")
        #print(out4)
        out = self.featrues_deconv5(out4)
        #print("out5:")
        #print(out)
        out = self.featrues_deconv7(out)
        #print("out6666:")
        #print(out)
        classify = self.classifier(out)
        classify = classify.view(classify.size(0), -1)
        out = self.featrues_upconv5(out)
        #print("out6:")
        #print(out)
        out = torch.cat([out4,out],1)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out,classify
class MNV3_bufen_new4(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new4,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(16),
            nn.ReLU(True),
        )
        #self.featrues_deconv4_2=nn.Sequential(
        #    nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
        #)
        self.featrues_deconv5=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(32),
            nn.ReLU(True),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            Block_4(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        )
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        self.featrues_upconv5=nn.Sequential(
            nn.ConvTranspose2d(32, int(16),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(16)),
            nn.ReLU(True),
        )
        #self.featrues_upconv5_2=nn.Sequential(
        #    nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=False)
        #    #nn.ConvTranspose2d(16, int(16),
        #    #                   kernel_size=3, stride=(2,1),
        #    #                   padding=1, output_padding=(1,0),
        #    #                   bias=False),
        #    #norm_layer(int(16)),
        #    #nn.ReLU(True),
        #)
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(8)),
            nn.ReLU(True),
        )
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1, padding_mode = 'reflect'),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        #out = self.featrues_deconv4_2(out4)
        #print("out4:")
        #print(out4)
        out = self.featrues_deconv5(out4)
        #print("out5:")
        #print(out)
        out = self.featrues_deconv7(out)
        #print("out6666:")
        #print(out)
        out = self.featrues_upconv5(out)
        #out = self.featrues_upconv5_2(out)
        #print("out6:")
        #print(out)
        out = torch.cat([out4,out],1)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class MNV3_bufen_new4_ooo(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new4_ooo,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        '''
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv5=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(True),
        )
        '''
        #self.featrues_deconv3=nn.Sequential(
            #Block(kernel_size=3, in_size=int(4), expand_size=int(8), out_size=int(8),
                #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        self.featrues_deconv4=nn.Sequential(
            Block(kernel_size=3, in_size=int(8), expand_size=int(24), out_size=int(16),
                nolinear=hswish(), semodule=None, stride=2),
        )
        self.featrues_deconv5=nn.Sequential(
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=None, stride=2),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        )
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        '''
        self.featrues_upconv5=nn.Sequential(
            nn.ConvTranspose2d(32, int(16),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=2),
            norm_layer(int(16)),
            nn.ReLU(True),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=2),
            norm_layer(int(8)),
            nn.ReLU(True),
        )
        '''
        self.featrues_upconv5 = nn.Sequential(
            Block_up(kernel_size=3, in_size=32, expand_size=32, out_size=16,
                  nolinear=hswish(), semodule=None, stride=2),
        )
        self.featrues_upconv4 = nn.Sequential(
            Block_up(kernel_size=3, in_size=32, expand_size=32, out_size=8,
                  nolinear=hswish(), semodule=None, stride=2),
        )
        
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        #print("out4:")
        #print(out4)
        out = self.featrues_deconv5(out4)
        #print("out5:")
        #print(out)
        out = self.featrues_deconv7(out)
        out = self.featrues_upconv5(out)
        #print("out6:")
        #print(out)
        out = torch.cat([out4,out],1)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class MNV3_bufen_new4_o(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new4_o,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        '''
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv5=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(True),
        )
        '''
        #self.featrues_deconv3=nn.Sequential(
            #Block(kernel_size=3, in_size=int(4), expand_size=int(8), out_size=int(8),
                #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        self.featrues_deconv4=nn.Sequential(
            Block(kernel_size=3, in_size=int(8), expand_size=int(24), out_size=int(16),
                nolinear=hswish(), semodule=None, stride=2),
        )
        self.featrues_deconv5=nn.Sequential(
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=None, stride=2),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        )
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        self.featrues_upconv5=nn.Sequential(
            nn.ConvTranspose2d(32, int(16),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=2),
            norm_layer(int(16)),
            nn.ReLU(True),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=2),
            norm_layer(int(8)),
            nn.ReLU(True),
        )
        '''
        self.featrues_upconv5 = nn.Sequential(
            Block_up(kernel_size=3, in_size=32, expand_size=32, out_size=16,
                  nolinear=hswish(), semodule=None, stride=2),
        )
        self.featrues_upconv4 = nn.Sequential(
            Block_up(kernel_size=3, in_size=32, expand_size=32, out_size=8,
                  nolinear=hswish(), semodule=None, stride=2),
        )
        '''
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        #print("out4:")
        #print(out4)
        out = self.featrues_deconv5(out4)
        #print("out5:")
        #print(out)
        out = self.featrues_deconv7(out)
        out = self.featrues_upconv5(out)
        #print("out6:")
        #print(out)
        out = torch.cat([out4,out],1)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class MNV3_bufen_new5_214(nn.Module): #214保持182 297 5.338
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_214,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv4_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            #nn.Conv2d(16, 16, kernel_size=3, stride=(2,1), padding=1, bias=False),
            #norm_layer(16),
            #nn.ReLU(True),
        )
        #self.featrues_deconv5=nn.Sequential(
        #    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
        #    norm_layer(32),
        #    nn.ReLU(True),
        #)
        #self.featrues_deconv3=nn.Sequential(
            #Block(kernel_size=3, in_size=int(4), expand_size=int(8), out_size=int(8),
                #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        #self.featrues_deconv4=nn.Sequential(
        #    Block(kernel_size=3, in_size=int(8), expand_size=int(24), out_size=int(16),
        #        nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
        #)
        #self.featrues_deconv4_1=nn.Sequential(
            #Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        self.featrues_deconv5=nn.Sequential(
            Block_4(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=None, stride=2),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        #self.featrues_deconv7=nn.Sequential(
        #    #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
        #    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False,groups=32, padding_mode = 'reflect'),
        #    norm_layer(32),
        #    hswish(),
        #    nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
        #    nn.BatchNorm2d(32),
        #    #hswish(),
        #    SeModule(int(32))
        #    #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
        #        #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        #)
        #self.featrues_deconv8=nn.Sequential(
        #    #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
        #    nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False,groups=24),
        #    norm_layer(24),
        #    hswish(),
        #    nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
        #    nn.BatchNorm2d(24),
        #    hswish(),
        #    #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
        #        #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        #)
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        #self.featrues_upconv5=nn.Sequential(#4.534
        #    nn.ConvTranspose2d(32, int(16),
        #                       kernel_size=3, stride=2,
        #                       padding=1, output_padding=1,
        #                       bias=False,groups=1),
        #    norm_layer(int(16)),
        #    nn.ReLU(True),
        #    #nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #    #norm_layer(16),
        #    #nn.ReLU(True),
        #)
        #self.featrues_upconv5 = nn.Sequential(
        #    nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,groups=32),
        #    norm_layer(32),
        #    hswish(),
        #    nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #    nn.BatchNorm2d(16),
        #    hswish(),
        #)
        #self.featrues_upconv5 = nn.Sequential(#这个比上面两个好
        #    Block_up(kernel_size=3, in_size=32, expand_size=32, out_size=16,#32 40
        #          nolinear=hswish(), semodule=None, stride=2),
        #    nn.ReLU(True),
        #)
        self.featrues_upconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,groups=4),
            norm_layer(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.featrues_upconv5_2=nn.Sequential(
            nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=False)
            #nn.ConvTranspose2d(16, int(16),
            #                   kernel_size=3, stride=(2,1),
            #                   padding=1, output_padding=(1,0),
            #                   bias=False),
            #norm_layer(int(16)),
            #nn.ReLU(True),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=1),
            norm_layer(int(8)),
            nn.ReLU(True),
            #nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False),
            #norm_layer(8),
        )
        #self.featrues_upconv5 = nn.Sequential(
        #    Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=16,#32 40
        #          nolinear=hswish(), semodule=None, stride=2),
        #)

        #self.featrues_upconv4_1 = nn.Sequential(
            #Block(kernel_size=3, in_size=16, expand_size=16, out_size=16,
                  #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        #self.featrues_upconv4_1 = nn.Sequential(
            #SeModule(int(32)),
        #)
        #self.featrues_upconv4 = nn.Sequential(
            #Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=8,#32,40   16,32
                  #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1, padding_mode = 'reflect'),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        out4_2 = self.featrues_deconv4_2(out4)
        #print("out4:")
        #print(out4)
        out5 = self.featrues_deconv5(out4_2)
        #print("out5:")
        #print(out)
        #out7 = self.featrues_deconv7(out5)
        #out = out5+out7
        #out7 = self.featrues_deconv7(out)
        #out = out+out7
        #out = out7 + out5
        out = self.featrues_upconv5(out5)
        out = self.featrues_upconv5_2(out)
        #print("out6:")
        #print(out)
        #out = self.featrues_upconv4_1(out)
        out = torch.cat([out4,out],1)
        #out = self.featrues_upconv4_1(out)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class MNV3_bufen_new5_ddd(nn.Module): #199保持182 261 4.282
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_ddd,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        #self.featrues_deconv2 = nn.Sequential(
        #    nn.Conv2d(ngf, ngf, kernel_size=3,stride=(2,1), padding=1, bias=False),
        #    norm_layer(ngf),
        #    nn.ReLU(True),
        #)
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3,stride=1, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv4_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            #nn.Conv2d(16, 16, kernel_size=3, stride=(2,1), padding=1, bias=False),
            #norm_layer(16),
            #nn.ReLU(True),
        )
        self.featrues_deconv5=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(True),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        )
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        self.featrues_upconv5=nn.Sequential(
            nn.ConvTranspose2d(32, int(16),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(16)),
            nn.ReLU(True),
        )
        self.featrues_upconv5_2=nn.Sequential(
            nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=False)
            #nn.ConvTranspose2d(16, int(16),
            #                   kernel_size=3, stride=(2,1),
            #                   padding=1, output_padding=(1,0),
            #                   bias=False),
            #norm_layer(int(16)),
            #nn.ReLU(True),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(8)),
            nn.ReLU(True),
        )
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        #self.featrues_upconv2=nn.Sequential(
        #    nn.ConvTranspose2d(ngf, ngf, kernel_size=3,stride=(2,1), padding=1, bias=False, output_padding=(1,0),),
        #    norm_layer(ngf),
        #    nn.ReLU(True),
        #)
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out.shape)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)

        #print("out4:")
        #print(out4_2.shape)
        out = self.featrues_deconv5(out4)
        out4_2 = self.featrues_deconv4_2(out)
        #print("out5:")
        #print(out)
        out = self.featrues_deconv7(out4_2)
        out = self.featrues_upconv5_2(out)
        out = self.featrues_upconv5(out)
        #print(out.shape)

        #print("out6:")
        #print(out.shape)
        out = torch.cat([out4,out],1)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class MNV3_bufen_new5_199(nn.Module): #199保持182 261 4.282
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_199,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        #self.featrues_deconv2 = nn.Sequential(
        #    nn.Conv2d(ngf, ngf, kernel_size=3,stride=(2,1), padding=1, bias=False),
        #    norm_layer(ngf),
        #    nn.ReLU(True),
        #)
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3,stride=1, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv4_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            #nn.Conv2d(16, 16, kernel_size=3, stride=(2,1), padding=1, bias=False),
            #norm_layer(16),
            #nn.ReLU(True),
        )
        self.featrues_deconv5=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(True),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        )
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        self.featrues_upconv5=nn.Sequential(
            nn.ConvTranspose2d(32, int(16),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(16)),
            nn.ReLU(True),
        )
        self.featrues_upconv5_2=nn.Sequential(
            nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=False)
            #nn.ConvTranspose2d(16, int(16),
            #                   kernel_size=3, stride=(2,1),
            #                   padding=1, output_padding=(1,0),
            #                   bias=False),
            #norm_layer(int(16)),
            #nn.ReLU(True),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(8)),
            nn.ReLU(True),
        )
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        #self.featrues_upconv2=nn.Sequential(
        #    nn.ConvTranspose2d(ngf, ngf, kernel_size=3,stride=(2,1), padding=1, bias=False, output_padding=(1,0),),
        #    norm_layer(ngf),
        #    nn.ReLU(True),
        #)
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out.shape)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        out4_2 = self.featrues_deconv4_2(out4)
        #print("out4:")
        #print(out4_2.shape)
        out = self.featrues_deconv5(out4_2)
        #print("out5:")
        #print(out)
        out = self.featrues_deconv7(out)
        out = self.featrues_upconv5(out)
        #print(out.shape)
        out = self.featrues_upconv5_2(out)
        #print("out6:")
        #print(out.shape)
        out = torch.cat([out4,out],1)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out

class MNV3_bufen_new5_197(nn.Module): #197保持182 282 4.901
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_197,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True),
        )
        #self.featrues_deconv5=nn.Sequential(
        #    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
        #    norm_layer(32),
        #    nn.ReLU(True),
        #)
        #self.featrues_deconv3=nn.Sequential(
            #Block(kernel_size=3, in_size=int(4), expand_size=int(8), out_size=int(8),
                #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        #self.featrues_deconv4=nn.Sequential(
        #    Block(kernel_size=3, in_size=int(8), expand_size=int(24), out_size=int(16),
        #        nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
        #)
        #self.featrues_deconv4_1=nn.Sequential(
            #Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        self.featrues_deconv5=nn.Sequential(
            Block(kernel_size=3, in_size=int(16), expand_size=int(40), out_size=int(32),
                nolinear=hswish(), semodule=SeModule(int(32)), stride=2),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False,groups=32),
            norm_layer(32),
            hswish(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
            nn.BatchNorm2d(32),
            hswish(),
            #Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #    nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        )
        #self.featrues_deconv8=nn.Sequential(
        #    #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
        #    nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False,groups=24),
        #    norm_layer(24),
        #    hswish(),
        #    nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
        #    nn.BatchNorm2d(24),
        #    hswish(),
        #    #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
        #        #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        #)
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        #self.featrues_upconv5=nn.Sequential(#4.534
        #    nn.ConvTranspose2d(32, int(16),
        #                       kernel_size=3, stride=2,
        #                       padding=1, output_padding=1,
        #                       bias=False,groups=1),
        #    norm_layer(int(16)),
        #    nn.ReLU(True),
        #    #nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #    #norm_layer(16),
        #    #nn.ReLU(True),
        #)
        self.featrues_upconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,groups=2),
            norm_layer(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        #self.featrues_upconv5 = nn.Sequential(#这个比上面两个好
        #    Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=16,#32 40
        #          nolinear=hswish(), semodule=None, stride=2),
        #    nn.ReLU(True),
        #)
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=1),
            norm_layer(int(8)),
            nn.ReLU(True),
            #nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False),
            #norm_layer(8),
        )
        #self.featrues_upconv5 = nn.Sequential(
        #    Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=16,#32 40
        #          nolinear=hswish(), semodule=None, stride=2),
        #)

        #self.featrues_upconv4_1 = nn.Sequential(
            #Block(kernel_size=3, in_size=16, expand_size=16, out_size=16,
                  #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        #self.featrues_upconv4_1 = nn.Sequential(
            #SeModule(int(32)),
        #)
        #self.featrues_upconv4 = nn.Sequential(
            #Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=8,#32,40   16,32
                  #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        #out = self.featrues_deconv4_1(out4)
        #print("out4:")
        #print(out4)
        out5 = self.featrues_deconv5(out4)
        #print("out5:")
        #print(out)
        out7 = self.featrues_deconv7(out5)
        out = out5+out7
        #out7 = self.featrues_deconv7(out)
        #out = out+out7
        #out = out7 + out5
        out = self.featrues_upconv5(out)
        #print("out6:")
        #print(out)
        #out = self.featrues_upconv4_1(out)
        out = torch.cat([out4,out],1)
        #out = self.featrues_upconv4_1(out)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class MNV3_bufen_new5(nn.Module): #195保持182 257 4.469
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv4_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            #nn.Conv2d(16, 16, kernel_size=3, stride=(2,1), padding=1, bias=False),
            #norm_layer(16),
            #nn.ReLU(True),
        )
        #self.featrues_deconv5=nn.Sequential(
        #    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
        #    norm_layer(32),
        #    nn.ReLU(True),
        #)
        #self.featrues_deconv3=nn.Sequential(
            #Block(kernel_size=3, in_size=int(4), expand_size=int(8), out_size=int(8),
                #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        #self.featrues_deconv4=nn.Sequential(
        #    Block(kernel_size=3, in_size=int(8), expand_size=int(24), out_size=int(16),
        #        nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
        #)
        #self.featrues_deconv4_1=nn.Sequential(
            #Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        self.featrues_deconv5=nn.Sequential(
            Block_4(kernel_size=3, in_size=int(16), expand_size=int(40), out_size=int(32),
                nolinear=hswish(), semodule=None, stride=2),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False,groups=32, padding_mode = 'reflect'),
            norm_layer(32),
            hswish(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
            nn.BatchNorm2d(32),
            #hswish(),
            SeModule(int(32)),
            #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
                #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        )
        self.classifier=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            #nn.Dropout2d(0.5),
            nn.Conv2d(32, 1, kernel_size=1,stride=1, padding=0, bias=True),
            hsigmoid(),
        )
        #self.featrues_deconv8=nn.Sequential(
        #    #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
        #    nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False,groups=24),
        #    norm_layer(24),
        #    hswish(),
        #    nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
        #    nn.BatchNorm2d(24),
        #    hswish(),
        #    #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
        #        #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        #)
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        #self.featrues_upconv5=nn.Sequential(#4.534
        #    nn.ConvTranspose2d(32, int(16),
        #                       kernel_size=3, stride=2,
        #                       padding=1, output_padding=1,
        #                       bias=False,groups=1),
        #    norm_layer(int(16)),
        #    nn.ReLU(True),
        #    #nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #    #norm_layer(16),
        #    #nn.ReLU(True),
        #)
        '''
        self.featrues_upconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,groups=4),
            norm_layer(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        #self.featrues_upconv5 = nn.Sequential(#这个比上面两个好
        #    Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=16,#32 40
        #          nolinear=hswish(), semodule=None, stride=2),
        #    nn.ReLU(True),
        #)
        self.featrues_upconv5_2=nn.Sequential(
            nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=False)
            #nn.ConvTranspose2d(16, int(16),
            #                   kernel_size=3, stride=(2,1),
            #                   padding=1, output_padding=(1,0),
            #                   bias=False),
            #norm_layer(int(16)),
            #nn.ReLU(True),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=1),
            norm_layer(int(8)),
            nn.ReLU(True),
            #nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False),
            #norm_layer(8),
        )
        #self.featrues_upconv5 = nn.Sequential(
        #    Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=16,#32 40
        #          nolinear=hswish(), semodule=None, stride=2),
        #)

        #self.featrues_upconv4_1 = nn.Sequential(
            #Block(kernel_size=3, in_size=16, expand_size=16, out_size=16,
                  #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        #self.featrues_upconv4_1 = nn.Sequential(
            #SeModule(int(32)),
        #)
        #self.featrues_upconv4 = nn.Sequential(
            #Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=8,#32,40   16,32
                  #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1, padding_mode = 'reflect'),
            hsigmoid(),
        )
        '''
    def forward(self, input):
        #print("input:")
        # print(input.shape)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        out = self.featrues_deconv4_2(out4)
        #print("out4:")
        #print(out.shape)
        out5 = self.featrues_deconv5(out)
        #print("out5:")
        #print(out)
        out7 = self.featrues_deconv7(out5)
        out = out5+out7
        classify = self.classifier(out)
        # print(classify.shape)
        classify = classify.view(classify.size(0),1)
        # print(classify.shape)
        '''
        #out7 = self.featrues_deconv7(out)
        #out = out+out7
        #out = out7 + out5
        out = self.featrues_upconv5(out)
        out = self.featrues_upconv5_2(out)
        #print("out6:")
        #print(out)
        #out = self.featrues_upconv4_1(out)
        out = torch.cat([out4,out],1)
        #out = self.featrues_upconv4_1(out)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        '''
        classify = classify  #*355
        # print(classify)
        # classify = (classify+1)*127.5
        return classify

class MNV3_bufen_new5_out2(nn.Module): #195保持182 257 4.469
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_out2,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv4_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            #nn.Conv2d(16, 16, kernel_size=3, stride=(2,1), padding=1, bias=False),
            #norm_layer(16),
            #nn.ReLU(True),
        )

        self.featrues_deconv5=nn.Sequential(
            Block_4(kernel_size=3, in_size=int(16), expand_size=int(40), out_size=int(32),
                nolinear=hswish(), semodule=None, stride=2),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False,groups=32, padding_mode = 'reflect'),
            norm_layer(32),
            hswish(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
            nn.BatchNorm2d(32),
            #hswish(),
            SeModule(int(32)),
            #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
                #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        )
        self.classifier_pool=nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        )
        self.classifier=nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            #nn.Dropout2d(0.5),
            nn.Conv2d(32, 2, kernel_size=1,stride=1, padding=0, bias=True),
        )
        self.regression=nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1,stride=1, padding=0, bias=True),
            hsigmoid(),
        )

    def forward(self, input):
        #print("input:")
        # print(input.shape)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        out = self.featrues_deconv4_2(out4)
        #print("out4:")
        #print(out.shape)
        out5 = self.featrues_deconv5(out)
        #print("out5:")
        #print(out)
        out7 = self.featrues_deconv7(out5)
        out = out5+out7
        out = self.classifier_pool(out)

        #simi_value 回归
        reg = self.regression(out)
        reg = reg.view(reg.size(0), 1)
        
        # 分类头
        classify = self.classifier(out)
        # print(classify.shape)
        classify = classify.view(classify.size(0),-1)
        # print(classify.shape)

        return reg,classify


class MNV3_bufen_new5_bin_reg(nn.Module): #195保持182 257 4.469
    # 简单分类
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_bin_reg,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv4_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            #nn.Conv2d(16, 16, kernel_size=3, stride=(2,1), padding=1, bias=False),
            #norm_layer(16),
            #nn.ReLU(True),
        )
        
        self.featrues_deconv5=nn.Sequential(
            Block_4(kernel_size=3, in_size=int(16), expand_size=int(40), out_size=int(32),
                nolinear=hswish(), semodule=None, stride=2),
        )
        
        self.featrues_deconv7=nn.Sequential(
            #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False,groups=32, padding_mode = 'reflect'),
            norm_layer(32),
            hswish(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
            nn.BatchNorm2d(32),
            #hswish(),
            SeModule(int(32)),
            #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
                #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        )
       
        # self.classifier=nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(32, 1, kernel_size=1,stride=1, padding=0, bias=True),
        #     hsigmoid(),
        #     # nn.Linear(32,1)
        # )
        
        # 加label,最后一层用回归还是分类呢？
        self.feat1=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier=nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=1,stride=1, padding=0, bias=True),
            # hsigmoid(),
        )
        self.regresssion=nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1,stride=1, padding=0, bias=True),
            # hsigmoid(),
        )
        # 第一次模型
        # self.linear = nn.Sequential(nn.Linear(32,2))
        

        
    def forward(self, input):
        out = self.featrues_deconv1(input)  #bchw
        out = self.featrues_deconv2(out)
        out = self.featrues_deconv3(out)
        out4 = self.featrues_deconv4(out)
        out = self.featrues_deconv4_2(out4)
        out5 = self.featrues_deconv5(out)
        # print(out5.shape)
        out7 = self.featrues_deconv7(out5)
        out = out5+out7
        out = self.feat1(out)
        out_cls =self.classifier(out)
        out_cls = out_cls.view(out_cls.size(0), -1)
        out_reg = self.regresssion(out)
        out_reg = out_reg.view(out_reg.size(0), -1)
        # print(out_cls.shape,out_reg.shape)
        # # # 回归的方式
        # # classify = self.classifier(out)
        # # classify = classify.view(classify.size(0), 2)
        # # 分类
        # classify = self.classifier(out)
        # # classify = classify.view(-1,32)
        # # classify= self.linear(classify)

        # classify = classify.view(classify.size(0), 1)
        # print(classify.shape)
        return out_reg,out_cls
class MNV3_bufen_new5_bin_reg_pool2(nn.Module): #195保持182 257 4.469
    # 简单分类
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_bin_reg_pool2,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv4_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            #nn.Conv2d(16, 16, kernel_size=3, stride=(2,1), padding=1, bias=False),
            #norm_layer(16),
            #nn.ReLU(True),
        )
        
        self.featrues_deconv5=nn.Sequential(
            Block_4(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(24),
                nolinear=hswish(), semodule=None, stride=2),
        )
        
        self.featrues_deconv7=nn.Sequential(
            #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False,groups=24, padding_mode = 'reflect'),
            norm_layer(24),
            hswish(),
            nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
            nn.BatchNorm2d(24),
            #hswish(),
            SeModule(int(24)),
            #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
                #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        )
       
        # self.classifier=nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(32, 1, kernel_size=1,stride=1, padding=0, bias=True),
        #     hsigmoid(),
        #     # nn.Linear(32,1)
        # )
        
        # 加label,最后一层用回归还是分类呢？
        self.feat1=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier=nn.Sequential(
            nn.Conv2d(24, 2, kernel_size=1,stride=1, padding=0, bias=True),
            # hsigmoid(),
        )
        self.regresssion=nn.Sequential(
            nn.Conv2d(24, 1, kernel_size=1,stride=1, padding=0, bias=True),
            # hsigmoid(),
        )
        # 第一次模型
        # self.linear = nn.Sequential(nn.Linear(32,2))
        

        
    def forward(self, input):
        out = self.featrues_deconv1(input)
        out = self.featrues_deconv2(out)
        out = self.featrues_deconv3(out)
        out4 = self.featrues_deconv4(out)
        out = self.featrues_deconv4_2(out4)
        out5 = self.featrues_deconv5(out)
        # print(out5.shape)
        out7 = self.featrues_deconv7(out5)
        out = out5+out7
        out = self.feat1(out)
        out_cls =self.classifier(out)
        out_cls = out_cls.view(out_cls.size(0), -1)
        out_reg = self.regresssion(out)
        out_reg = out_reg.view(out_reg.size(0), -1)
        # print(out_cls.shape,out_reg.shape)
        # # # 回归的方式
        # # classify = self.classifier(out)
        # # classify = classify.view(classify.size(0), 2)
        # # 分类
        # classify = self.classifier(out)
        # # classify = classify.view(-1,32)
        # # classify= self.linear(classify)

        # classify = classify.view(classify.size(0), 1)
        # print(classify.shape)
        return out_reg,out_cls

class MNV3_bufen_new5_194(nn.Module): #194保持182 298 4.622
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_194,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True),
        )
        #self.featrues_deconv5=nn.Sequential(
        #    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
        #    norm_layer(32),
        #    nn.ReLU(True),
        #)
        #self.featrues_deconv3=nn.Sequential(
            #Block(kernel_size=3, in_size=int(4), expand_size=int(8), out_size=int(8),
                #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        #self.featrues_deconv4=nn.Sequential(
        #    Block(kernel_size=3, in_size=int(8), expand_size=int(24), out_size=int(16),
        #        nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
        #)
        #self.featrues_deconv4_1=nn.Sequential(
            #Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        self.featrues_deconv5=nn.Sequential(
            Block(kernel_size=3, in_size=int(16), expand_size=int(40), out_size=int(24),
                nolinear=hswish(), semodule=None, stride=2),
            nn.ReLU(True),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            DepthwiseSeparableConvolution(24, 24, kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False,groups=32),
            #norm_layer(32),
            #hswish(),
            #nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
            #nn.BatchNorm2d(32),
            #hswish(),
            #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
                #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
            nn.ReLU(True),
        )
        self.featrues_deconv8=nn.Sequential(
            DepthwiseSeparableConvolution(24, 24, kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False,groups=24),
            #norm_layer(24),
            #hswish(),
            #nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
            #nn.BatchNorm2d(24),
            #hswish(),
            #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
                #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
            nn.ReLU(True),
        )
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        #self.featrues_upconv5=nn.Sequential(#4.534
        #    nn.ConvTranspose2d(32, int(16),
        #                       kernel_size=3, stride=2,
        #                       padding=1, output_padding=1,
        #                       bias=False,groups=1),
        #    norm_layer(int(16)),
        #    nn.ReLU(True),
        #    #nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #    #norm_layer(16),
        #    #nn.ReLU(True),
        #)
        #self.featrues_upconv5 = nn.Sequential(
        #    nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,groups=32),
        #    norm_layer(32),
        #    hswish(),
        #    nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #    nn.BatchNorm2d(16),
        #    hswish(),
        #)
        self.featrues_upconv5 = nn.Sequential(#这个比上面两个好
            Block_up(kernel_size=3, in_size=24, expand_size=40, out_size=16,#32 40
                  nolinear=hswish(), semodule=None, stride=2),
            nn.ReLU(True),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=1),
            norm_layer(int(8)),
            nn.ReLU(True),
            #nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False),
            #norm_layer(8),
        )
        #self.featrues_upconv5 = nn.Sequential(
        #    Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=16,#32 40
        #          nolinear=hswish(), semodule=None, stride=2),
        #)

        #self.featrues_upconv4_1 = nn.Sequential(
            #Block(kernel_size=3, in_size=16, expand_size=16, out_size=16,
                  #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        #self.featrues_upconv4_1 = nn.Sequential(
            #SeModule(int(32)),
        #)
        #self.featrues_upconv4 = nn.Sequential(
            #Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=8,#32,40   16,32
                  #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        #out = self.featrues_deconv4_1(out4)
        #print("out4:")
        #print(out4)
        out5 = self.featrues_deconv5(out4)
        #print("out5:")
        #print(out)
        out7 = self.featrues_deconv7(out5)
        out = out5+out7
        out8 = self.featrues_deconv8(out)
        out = out+out8
        #out = out7 + out5
        out = self.featrues_upconv5(out)
        #print("out6:")
        #print(out)
        #out = self.featrues_upconv4_1(out)
        out = torch.cat([out4,out],1)
        #out = self.featrues_upconv4_1(out)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class MNV3_bufen_new5_192(nn.Module): #192保持182 297 4.427
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_192,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv4_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            #nn.Conv2d(16, 16, kernel_size=3, stride=(2,1), padding=1, bias=False),
            #norm_layer(16),
            #nn.ReLU(True),
        )
        #self.featrues_deconv5=nn.Sequential(
        #    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
        #    norm_layer(32),
        #    nn.ReLU(True),
        #)
        #self.featrues_deconv3=nn.Sequential(
            #Block(kernel_size=3, in_size=int(4), expand_size=int(8), out_size=int(8),
                #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        #self.featrues_deconv4=nn.Sequential(
        #    Block(kernel_size=3, in_size=int(8), expand_size=int(24), out_size=int(16),
        #        nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
        #)
        #self.featrues_deconv4_1=nn.Sequential(
            #Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        self.featrues_deconv5=nn.Sequential(
            Block_4(kernel_size=3, in_size=int(16), expand_size=int(40), out_size=int(32),
                nolinear=hswish(), semodule=None, stride=2),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False,groups=32, padding_mode = 'reflect'),
            norm_layer(32),
            hswish(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
            nn.BatchNorm2d(32),
            #hswish(),
            SeModule(int(32))
            #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
                #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        )
        #self.featrues_deconv8=nn.Sequential(
        #    #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
        #    nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False,groups=24),
        #    norm_layer(24),
        #    hswish(),
        #    nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
        #    nn.BatchNorm2d(24),
        #    hswish(),
        #    #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
        #        #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        #)
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        #self.featrues_upconv5=nn.Sequential(#4.534
        #    nn.ConvTranspose2d(32, int(16),
        #                       kernel_size=3, stride=2,
        #                       padding=1, output_padding=1,
        #                       bias=False,groups=1),
        #    norm_layer(int(16)),
        #    nn.ReLU(True),
        #    #nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #    #norm_layer(16),
        #    #nn.ReLU(True),
        #)
        #self.featrues_upconv5 = nn.Sequential(
        #    nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,groups=32),
        #    norm_layer(32),
        #    hswish(),
        #    nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #    nn.BatchNorm2d(16),
        #    hswish(),
        #)
        self.featrues_upconv5 = nn.Sequential(#这个比上面两个好
            Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=16,#32 40
                  nolinear=hswish(), semodule=None, stride=2),
            nn.ReLU(True),
        )
        self.featrues_upconv5_2=nn.Sequential(
            nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=False)
            #nn.ConvTranspose2d(16, int(16),
            #                   kernel_size=3, stride=(2,1),
            #                   padding=1, output_padding=(1,0),
            #                   bias=False),
            #norm_layer(int(16)),
            #nn.ReLU(True),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=1),
            norm_layer(int(8)),
            nn.ReLU(True),
            #nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False),
            #norm_layer(8),
        )
        #self.featrues_upconv5 = nn.Sequential(
        #    Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=16,#32 40
        #          nolinear=hswish(), semodule=None, stride=2),
        #)

        #self.featrues_upconv4_1 = nn.Sequential(
            #Block(kernel_size=3, in_size=16, expand_size=16, out_size=16,
                  #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        #self.featrues_upconv4_1 = nn.Sequential(
            #SeModule(int(32)),
        #)
        #self.featrues_upconv4 = nn.Sequential(
            #Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=8,#32,40   16,32
                  #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1, padding_mode = 'reflect'),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        out4_2 = self.featrues_deconv4_2(out4)
        #print("out4:")
        #print(out4)
        out5 = self.featrues_deconv5(out4_2)
        #print("out5:")
        #print(out)
        out7 = self.featrues_deconv7(out5)
        out = out5+out7
        #out7 = self.featrues_deconv7(out)
        #out = out+out7
        #out = out7 + out5
        out = self.featrues_upconv5(out)
        out = self.featrues_upconv5_2(out)
        #print("out6:")
        #print(out)
        #out = self.featrues_upconv4_1(out)
        out = torch.cat([out4,out],1)
        #out = self.featrues_upconv4_1(out)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class MNV3_bufen_new5_190(nn.Module): #190保持182 298 4.627
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_190,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.LeakyReLU(0.2),
        )
        #self.featrues_deconv5=nn.Sequential(
        #    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
        #    norm_layer(32),
        #    nn.ReLU(True),
        #)
        #self.featrues_deconv3=nn.Sequential(
            #Block(kernel_size=3, in_size=int(4), expand_size=int(8), out_size=int(8),
                #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        #self.featrues_deconv4=nn.Sequential(
        #    Block(kernel_size=3, in_size=int(8), expand_size=int(24), out_size=int(16),
        #        nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
        #)
        #self.featrues_deconv4_1=nn.Sequential(
            #Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        self.featrues_deconv5=nn.Sequential(
            Block(kernel_size=3, in_size=int(16), expand_size=int(40), out_size=int(32),
                nolinear=hswish(), semodule=SeModule(int(32)), stride=2),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False,groups=32),
            norm_layer(32),
            hswish(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
            nn.BatchNorm2d(32),
            hswish(),
            #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
                #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        )
        #self.featrues_deconv8=nn.Sequential(
        #    #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
        #    nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False,groups=24),
        #    norm_layer(24),
        #    hswish(),
        #    nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
        #    nn.BatchNorm2d(24),
        #    hswish(),
        #    #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
        #        #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        #)
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        #self.featrues_upconv5=nn.Sequential(
        #    nn.ConvTranspose2d(32, int(16),
        #                       kernel_size=3, stride=2,
        #                       padding=1, output_padding=1,
        #                       bias=False,groups=1),
        #    norm_layer(int(16)),
        #    nn.ReLU(True),
        #    #nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #    #norm_layer(16),
        #    #nn.ReLU(True),
        #)
        self.featrues_upconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,groups=32),
            norm_layer(32),
            hswish(),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            hswish(),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=1),
            norm_layer(int(8)),
            nn.ReLU(True),
            #nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False),
            #norm_layer(8),
        )
        #self.featrues_upconv5 = nn.Sequential(
        #    Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=16,#32 40
        #          nolinear=hswish(), semodule=None, stride=2),
        #)

        #self.featrues_upconv4_1 = nn.Sequential(
            #Block(kernel_size=3, in_size=16, expand_size=16, out_size=16,
                  #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        #self.featrues_upconv4_1 = nn.Sequential(
            #SeModule(int(32)),
        #)
        #self.featrues_upconv4 = nn.Sequential(
            #Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=8,#32,40   16,32
                  #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        #out = self.featrues_deconv4_1(out4)
        #print("out4:")
        #print(out4)
        out5 = self.featrues_deconv5(out4)
        #print("out5:")
        #print(out)
        out7 = self.featrues_deconv7(out5)
        out = out5+out7
        #out7 = self.featrues_deconv7(out)
        #out = out+out7
        #out = out7 + out5
        out = self.featrues_upconv5(out)
        #print("out6:")
        #print(out)
        #out = self.featrues_upconv4_1(out)
        out = torch.cat([out4,out],1)
        #out = self.featrues_upconv4_1(out)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class MNV3_bufen_new5_x(nn.Module): #188保持182 269 4.715
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_x,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        #self.featrues_deconv4=nn.Sequential(
            #nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(16),
            #nn.LeakyReLU(0.2),
        #)
        #self.featrues_deconv5=nn.Sequential(
            #nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(32),
            #nn.ReLU(True),
        #)
        #self.featrues_deconv3=nn.Sequential(
            #Block(kernel_size=3, in_size=int(4), expand_size=int(8), out_size=int(8),
                #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        self.featrues_deconv4=nn.Sequential(
            Block(kernel_size=3, in_size=int(8), expand_size=int(24), out_size=int(16),
                nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
        )
        #self.featrues_deconv4_1=nn.Sequential(
            #Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        self.featrues_deconv5=nn.Sequential(
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=SeModule(int(32)), stride=2),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False,groups=32),
            norm_layer(32),
            hswish(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
            nn.BatchNorm2d(32),
            hswish(),
            #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
                #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        )
        #self.featrues_deconv8=nn.Sequential(
        #    #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
        #    nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False,groups=24),
        #    norm_layer(24),
        #    hswish(),
        #    nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
        #    nn.BatchNorm2d(24),
        #    hswish(),
        #    #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
        #        #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        #)
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        #self.featrues_upconv5=nn.Sequential(
        #    nn.ConvTranspose2d(32, int(16),
        #                       kernel_size=3, stride=2,
        #                       padding=1, output_padding=1,
        #                       bias=False,groups=1),
        #    norm_layer(int(16)),
        #    nn.ReLU(True),
        #    #nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #    #norm_layer(16),
        #    #nn.ReLU(True),
        #)
        self.featrues_upconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,groups=32),
            norm_layer(32),
            hswish(),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            hswish(),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=1),
            norm_layer(int(8)),
            nn.ReLU(True),
            #nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False),
            #norm_layer(8),
        )
        #self.featrues_upconv5 = nn.Sequential(
        #    Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=16,#32 40
        #          nolinear=hswish(), semodule=None, stride=2),
        #)

        #self.featrues_upconv4_1 = nn.Sequential(
            #Block(kernel_size=3, in_size=16, expand_size=16, out_size=16,
                  #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        #self.featrues_upconv4_1 = nn.Sequential(
            #SeModule(int(32)),
        #)
        #self.featrues_upconv4 = nn.Sequential(
            #Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=8,#32,40   16,32
                  #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        #out = self.featrues_deconv4_1(out4)
        #print("out4:")
        #print(out4)
        out5 = self.featrues_deconv5(out4)
        #print("out5:")
        #print(out)
        out7 = self.featrues_deconv7(out5)
        out = out5+out7
        #out7 = self.featrues_deconv7(out)
        #out = out+out7
        #out = out7 + out5
        out = self.featrues_upconv5(out)
        #print("out6:")
        #print(out)
        #out = self.featrues_upconv4_1(out)
        out = torch.cat([out4,out],1)
        #out = self.featrues_upconv4_1(out)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class MNV3_bufen_new5_p(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_p,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        #self.featrues_deconv4=nn.Sequential(
            #nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(16),
            #nn.LeakyReLU(0.2),
        #)
        #self.featrues_deconv5=nn.Sequential(
            #nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(32),
            #nn.ReLU(True),
        #)
        #self.featrues_deconv3=nn.Sequential(
            #Block(kernel_size=3, in_size=int(4), expand_size=int(8), out_size=int(8),
                #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        self.featrues_deconv4=nn.Sequential(
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                nolinear=hswish(), semodule=None, stride=2),
        )
        #self.featrues_deconv4_1=nn.Sequential(
            #Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        self.featrues_deconv5=nn.Sequential(
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=None, stride=2),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False,groups=32),
            norm_layer(32),
            hswish(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
            nn.BatchNorm2d(32),
            hswish(),
            #Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                #nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        )
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        self.featrues_upconv5=nn.Sequential(
            nn.ConvTranspose2d(32, int(16),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=1),
            norm_layer(int(16)),
            nn.ReLU(True),
            #nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
            #norm_layer(16),
            #nn.ReLU(True),
        )
        
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=1),
            norm_layer(int(8)),
            nn.ReLU(True),
            #nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False),
            #norm_layer(8),
        )
        #self.featrues_upconv5 = nn.Sequential(
            #Block_up(kernel_size=3, in_size=24, expand_size=32, out_size=16,
                  #nolinear=hswish(), semodule=None, stride=2),
        #)
        #self.featrues_upconv4_1 = nn.Sequential(
            #Block(kernel_size=3, in_size=16, expand_size=16, out_size=16,
                  #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        #self.featrues_upconv4_1 = nn.Sequential(
            #SeModule(int(32)),
        #)
        #self.featrues_upconv4 = nn.Sequential(
            #Block_up(kernel_size=3, in_size=32, expand_size=32, out_size=8,
                  #nolinear=hswish(), semodule=None, stride=2),
        #)
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        #out = self.featrues_deconv4_1(out4)
        #print("out4:")
        #print(out4)
        out5 = self.featrues_deconv5(out4)
        #print("out5:")
        #print(out)
        out7 = self.featrues_deconv7(out5)
        out = out7 + out5
        out = self.featrues_upconv5(out7)
        #print("out6:")
        #print(out)
        #out = self.featrues_upconv4_1(out)
        out = torch.cat([out4,out],1)
        #out = self.featrues_upconv4_1(out)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class MNV3_bufen_new5_k(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        super(MNV3_bufen_new5_k, self).__init__()
        #self.resblocks = res_blocks

        dkersize = 3
        de_nc = 4
        self.featrues_deconv1 = nn.Sequential(  # 64*64
            # nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, de_nc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(de_nc),
        )
        
        self.featrues_deconv2_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(de_nc, 4, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(4),
        )
        self.featrues_deconv2_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 4, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(4),
        )
        
        self.featrues_deconv3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 12, kernel_size=dkersize, stride=2, padding=1),  # 16*16                        
            nn.BatchNorm2d(12),
        )
        self.featrues_deconv4_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            # nn.Conv2d(12, 12, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            #DepthwiseSeparableConvolution(12, 12, kernel_size=3, stride=1, padding=1),
            Block(kernel_size=3, in_size=int(12), expand_size=int(12), out_size=int(12),
                nolinear=hswish(), semodule=None, stride=1),
            #nn.BatchNorm2d(12),
        )
        
        
        self.featrues_deconv4_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            # nn.Conv2d(12, 12, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            #DepthwiseSeparableConvolution(12, 12, kernel_size=3, stride=1, padding=1),
            Block(kernel_size=3, in_size=int(12), expand_size=int(12), out_size=int(12),
                nolinear=hswish(), semodule=None, stride=1),
            #nn.BatchNorm2d(12),
        )
        
        
        self.featrues_resnet1 = nn.Sequential(  # 8*8
            #DepthwiseSeparableConvolution(24, 24, kernel_size=3, stride=1, padding=1),
            Block(kernel_size=3, in_size=int(24), expand_size=int(24), out_size=int(24),
                nolinear=hswish(), semodule=None, stride=1),
            #nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            #DepthwiseSeparableConvolution(24, 24, kernel_size=3, stride=1, padding=1),
            Block(kernel_size=3, in_size=int(24), expand_size=int(24), out_size=int(24),
                nolinear=hswish(), semodule=None, stride=1),
            #nn.BatchNorm2d(24),
            # nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(48),
        )        
        self.featrues_upconv3 = nn.Sequential(           
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1,groups = 4),  #g8
            nn.Conv2d(24, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,groups = 4),  #g1,
            # nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16)
        )
        self.featrues_upconv2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            # Block(kernel_size=dkersize, in_size=16, expand_size=8, out_size=8,
            #       nolinear=hswish(), semodule=None, stride=1),
            nn.BatchNorm2d(8),
        )
        self.featrues_upconv1_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        self.featrues_upconv1_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        
        self.featrues_upconv1 = nn.Sequential(
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1,groups=4),  # 32*32  featrues_upconv3
            nn.Conv2d(8, 4, kernel_size=1, stride=1, padding=0),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        self.featrues_upconv0 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),  # 32*32  featrues_upconv3
            # nn.BatchNorm2d(1),
            nn.Tanh()
        )      
    def forward(self, input):
        input1 = self.featrues_deconv1(input)   # 4 72 26
        # print(input1)
        input01 = self.featrues_deconv2_1(input1)
        input02 = self.featrues_deconv2_2(input01)
        input = torch.cat([input01, input02], 1)
        input = self.featrues_deconv3(input)
        # print(input)
        input11 = self.featrues_deconv4_1(input)
        # print(input11)
        input12 = self.featrues_deconv4_2(input11)
        # print(input12)
        input = torch.cat([input11, input12], 1)
        out = input + self.featrues_resnet1(input)
        # print(out)
        out = self.featrues_upconv3(out)
        # print(out)
        out = self.featrues_upconv2(out)
        # print(out)
        out_u1 = self.featrues_upconv1_1(out)
        out_u2 = self.featrues_upconv1_2(out_u1)
        out = torch.cat([out_u1, out_u2], 1)  
        out = self.featrues_upconv1(out)   # 4 72 26
        # print(out)  
        # out = out*input1
        out = self.featrues_upconv0(out)
        # print(out)
        return out




class MNV3_bufen_new5_oo(nn.Module):#0.95 不行
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_oo,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        #self.featrues_deconv4=nn.Sequential(
            #nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(16),
            #nn.ReLU(True),
        #)
        #self.featrues_deconv5=nn.Sequential(
            #nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(32),
            #nn.ReLU(True),
        #)
        #self.featrues_deconv3=nn.Sequential(
            #Block(kernel_size=3, in_size=int(4), expand_size=int(8), out_size=int(8),
                #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        self.featrues_deconv4=nn.Sequential(
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                nolinear=hswish(), semodule=None, stride=2),
        )
        self.featrues_deconv4_1=nn.Sequential(
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        )
        self.featrues_deconv5=nn.Sequential(
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=None, stride=2),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        #self.featrues_deconv7=nn.Sequential(
            #Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                #nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #)
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        #self.featrues_upconv5=nn.Sequential(
            #nn.ConvTranspose2d(32, int(16),
                               #kernel_size=3, stride=2,
                               #padding=1, output_padding=1,
                               #bias=False,groups=4),
            #norm_layer(int(16)),
            #nn.ReLU(True),
        #)
        #self.featrues_upconv4=nn.Sequential(
            #nn.ConvTranspose2d(32, int(8),
                               #kernel_size=3, stride=2,
                               #padding=1, output_padding=1,
                               #bias=False,groups=2),
            #norm_layer(int(8)),
            #nn.ReLU(True),
        #)
        self.featrues_upconv5 = nn.Sequential(
            Block_up(kernel_size=3, in_size=32, expand_size=32, out_size=16,
                  nolinear=hswish(), semodule=None, stride=2),
        )
        self.featrues_upconv4_1 = nn.Sequential(
            Block(kernel_size=3, in_size=16, expand_size=16, out_size=16,
                  nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        )
        self.featrues_upconv4 = nn.Sequential(
            Block_up(kernel_size=3, in_size=32, expand_size=32, out_size=8,
                  nolinear=hswish(), semodule=None, stride=2),
        )
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        out = self.featrues_deconv4_1(out4)
        #print("out4:")
        #print(out4)
        out = self.featrues_deconv5(out)
        #print("out5:")
        #print(out)
        #out = self.featrues_deconv7(out)
        out = self.featrues_upconv5(out)
        #print("out6:")
        #print(out)
        out = self.featrues_upconv4_1(out)
        out = torch.cat([out4,out],1)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class MNV3_bufen_new6_o(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new6_o,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv5=nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(32),
            nn.ReLU(True),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        # self.featrues_deconv7=nn.Sequential(
        #     Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
        #         nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        # )
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        self.featrues_upconv5=nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False,groups=4),
            norm_layer(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, int(16),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(16)),
            nn.ReLU(True),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(8)),
            nn.ReLU(True),
        )
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1, padding_mode = 'reflect'),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input.size())
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        #print("out4:")
        #print(out4)
        out = self.featrues_deconv5(out4)
        #print("out5:")
        #print(out)
        #out = self.featrues_deconv7(out)
        out = self.featrues_upconv5(out)
        #print("out6:")
        #print(out)
        #print(out4.size(),out.size())
        out = torch.cat([out4,out],1)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(self.featrues_upconv1[0](out))
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class Umobilenet_004(nn.Module):#28057
    def __init__(self,intput_nc = 1, res_blocks=1):
        super(Umobilenet_004, self).__init__()
        self.featrues_deconv0 = nn.Sequential(
            nn.Conv2d(in_channels=intput_nc, out_channels=8, kernel_size=3,
                      stride=2, padding=1, bias=False,padding_mode = 'reflect'),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.2),
            #nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #self.featrues_deconv1 = nn.Sequential(
            #Block(kernel_size=3, in_size=8, expand_size=8, out_size=8,
                  #nolinear=hswish(), semodule=None, stride=1),
        #)
        self.featrues_deconv2 = nn.Sequential(
            Block_nc(kernel_size=3, in_size=8, expand_size=16, out_size=16,
                  nolinear=hswish(), semodule=None, stride=2),
        )
        self.featrues_deconv3 = nn.Sequential(
            Block_nc(kernel_size=3, in_size=16, expand_size=32, out_size=16,
                  nolinear=hswish(), semodule=None, stride=1),
        )
        self.featrues_deconv4 = nn.Sequential(
            Block_nc(kernel_size=3, in_size=16, expand_size=32, out_size=24,
                  nolinear=hswish(), semodule=None, stride=2),
        )


        self.featrues_upconv4 = nn.Sequential(
            Block_up(kernel_size=3, in_size=24, expand_size=32, out_size=16,
                  nolinear=hswish(), semodule=None, stride=2),
        )
        #self.featrues_upconv3 = nn.Sequential(
            #Block_up(kernel_size=4, in_size=16, expand_size=32, out_size=16,
                  #nolinear=hswish(), semodule=None, stride=1),
        #)
        self.featrues_upconv3 = nn.Sequential(
            Block_nc(kernel_size=3, in_size=32, expand_size=32, out_size=16,
                  nolinear=hswish(), semodule=None, stride=1),
        )
        self.featrues_upconv2 = nn.Sequential(
            Block_up(kernel_size=3, in_size=16, expand_size=16, out_size=8,
                  nolinear=hswish(), semodule=None, stride=2),
        )
        #self.featrues_upconv1 = nn.Sequential(
            #Block(kernel_size=3, in_size=8, expand_size=8, out_size=8,
                  #nolinear=hswish(), semodule=None, stride=1),
        #)
        self.featrues_upconv0 = nn.Sequential(
            #nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1,output_padding=1),
            hsigmoid(),
        )
    def forward(self, input):
        out = self.featrues_deconv0(input)
        #out0 = out
        #out = self.featrues_deconv1(out)
        #out1 = out
        out = self.featrues_deconv2(out)
        #out2 = out
        out = self.featrues_deconv3(out)
        out3 = out
        out = self.featrues_deconv4(out)
        #out4 = out
        #out = self.featrues_upconv5(out)
        #out = torch.cat([out, out3], 1)
        
        out = self.featrues_upconv4(out)
        #out4 = out
        #out = out + out3
        #out[:,:,:37,:10] = out[:,:,:37,:10]/2+out3[:,:,:37,:10]/2
        out = torch.cat([out3, out], 1)
        out = self.featrues_upconv3(out)
        out = self.featrues_upconv2(out)
        #out = self.featrues_upconv1(out)
        out = self.featrues_upconv0(out)
        return out
class MNV3_bufen_new3_2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new3_2,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv5=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(True),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        )
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        self.featrues_upconv5=nn.Sequential(
            nn.ConvTranspose2d(32, int(16),
                               kernel_size=3, stride=2,
                               padding=1,
                               bias=False),
            norm_layer(int(16)),
            nn.ReLU(True),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1,
                               bias=False),
            norm_layer(int(8)),
            nn.ReLU(True),
        )
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        #print("out4:")
        #print(out4)
        out = self.featrues_deconv5(out4)
        #print("out5:")
        #print(out)
        out = self.featrues_deconv7(out)
        out = self.featrues_upconv5(out)
        #print("out6:")
        #print(out)
        out = torch.cat([out4,out],1)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
class MNV3_bufen_new4_old(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new4_old,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        #self.featrues_deconv2 = nn.Sequential(
            #nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            #norm_layer(ngf),
            #nn.ReLU(True),
        #)
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv5=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(True),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        )
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        self.featrues_upconv5=nn.Sequential(
            nn.ConvTranspose2d(32, int(16),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(16)),
            nn.ReLU(True),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(8)),
            nn.ReLU(True),
        )
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(16, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        #self.featrues_upconv2=nn.Sequential(
            #nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            #norm_layer(ngf),
            #nn.ReLU(True),
        #)
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        out = self.featrues_deconv1(input)
        #out = self.featrues_deconv2(out)
        out = self.featrues_deconv3(out)
        out3=out
        out4 = self.featrues_deconv4(out)
        out = self.featrues_deconv5(out4)
        out = self.featrues_deconv7(out)
        out = self.featrues_upconv5(out)
        out = torch.cat([out4,out],1)
        out = self.featrues_upconv4(out)
        out = torch.cat([out3,out],1)
        out = self.featrues_upconv3(out)
        #out = self.featrues_upconv2(out)
        out = self.featrues_upconv1(out)
        return out
class MNV3_bufen_new5_o(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new5_o,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        #self.featrues_deconv2 = nn.Sequential(
        #    nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
        #    norm_layer(ngf),
        #    nn.ReLU(True),
        #)
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv5=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(True),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        )
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        self.featrues_upconv5=nn.Sequential(
            nn.ConvTranspose2d(32, int(16),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(16)),
            nn.ReLU(True),
        )
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(32, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(8)),
            nn.ReLU(True),
        )
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        #self.featrues_upconv2=nn.Sequential(
        #    nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
        #    norm_layer(ngf),
        #    nn.ReLU(True),
        #)
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        #out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        #print("out4:")
        #print(out4)
        out = self.featrues_deconv5(out4)
        #print("out5:")
        #print(out)
        out = self.featrues_deconv7(out)
        out = self.featrues_upconv5(out)
        #print("out6:")
        #print(out)
        out = torch.cat([out4,out],1)
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        #out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out
#L2_0026_6193_DK4_partialPress_12mul_merge_DK4_0175_fake_B
'''
class MNV3_bufen_new4(nn.Module):
    def __init__(self, intput_nc, res_blocks=1):
        super(MNV3_bufen_new4, self).__init__()

        self.resblocks = res_blocks

        dkersize = 3
        de_nc = 4
        self.featrues_deconv1 = nn.Sequential(  # 64*64
            # nn.ReflectionPad2d(1),
            nn.Conv2d(intput_nc, de_nc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(de_nc),
        )

        self.featrues_deconv2_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(de_nc, 4, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(4),
        )
        self.featrues_deconv2_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 4, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(4),
        )
       
        self.featrues_deconv3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=dkersize, stride=2, padding=1),  # 16*16                        
            nn.BatchNorm2d(16),
        )
        

        self.featrues_deconv4_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            # nn.Conv2d(16, 16, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            DepthwiseSeparableConvolution(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
        )
        
        
        self.featrues_deconv4_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            # nn.Conv2d(16, 16, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            DepthwiseSeparableConvolution(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
        )
       

        self.featrues_resnet1 = nn.Sequential(  # 8*8
            DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            # nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(48),
        )        
        
        self.featrues_upconv3 = nn.Sequential(             
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,groups=4),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16)
        )
        
        self.featrues_upconv2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(8),
        )
        
        self.featrues_upconv1_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        
        self.featrues_upconv1_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        
        self.featrues_upconv1 = nn.Sequential(
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=4),  # 32*32  featrues_upconv3
            nn.Conv2d(8, 4, kernel_size=1, stride=1, padding=0),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        
        self.featrues_upconv0 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),  # 32*32  featrues_upconv3
            # nn.BatchNorm2d(1),
            nn.Tanh()
        )      
        
    def forward(self, input):        
        input1 = self.featrues_deconv1(input)   # 4 72 26
        input01 = self.featrues_deconv2_1(input1)
        input02 = self.featrues_deconv2_2(input01)
        input = torch.cat([input01, input02], 1)

        input = self.featrues_deconv3(input)

        # print(input)
        input11 = self.featrues_deconv4_1(input)
        # print(input11)
        input12 = self.featrues_deconv4_2(input11)
        # print(input12)
        input = torch.cat([input11, input12], 1)
        # print(input)
        out = input + self.featrues_resnet1(input)
        # print(out)
        out = self.featrues_upconv3(out)
        # print(out)
        # print(out.shape)
        out = self.featrues_upconv2(out)
        # print(out)
        out_u1 = self.featrues_upconv1_1(out)
        out_u2 = self.featrues_upconv1_2(out_u1)
        
        out = torch.cat([out_u1, out_u2], 1)  
        
        out = self.featrues_upconv1(out)   # 4 72 26
        # print(out)  

        # out = out*input1
        out = self.featrues_upconv0(out)
        # print(out)
        
        return out
'''
class MNV3_bufen_Message(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_Message, self).__init__()

        model = [  # nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]
        model += [  # nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]
        # model += [SeModule(ngf)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            # model += [SeModule(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                # CBABlock(kernel_size=3, in_size=int(ngf * mult), expand_size=int(ngf * mult * 1), out_size=int(ngf * mult),
                #       nolinear=hswish(), cbamodule=CBAModule(int(ngf * mult)), stride=1)]
                Block(kernel_size=3, in_size=int(ngf * mult), expand_size=int(ngf * mult * 1), out_size=int(ngf * mult),
                      nolinear=hswish(), semodule=SeModule(int(ngf * mult)), stride=1)]
        model += [MessagePass(int(ngf * mult), 1)]
        model += [MessagePass(int(ngf * mult), 2)]
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=False),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            # model += [SeModule(int(ngf * mult / 2))]

        model += [  # nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
        model += [hsigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        # print(self.model[:21](input))
        # print(self.model(input))
        return self.model(input)
class MNV3_bufen_new6(nn.Module): #195保持182 257 4.469
    def __init__(self, input_nc, output_nc, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_new6,self).__init__()
        self.featrues_deconv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_deconv3=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(8),
            nn.ReLU(True),
        )
        self.featrues_deconv4=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(16),
            nn.ReLU(True),
        )
        self.featrues_deconv4_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            #nn.Conv2d(16, 16, kernel_size=3, stride=(2,1), padding=1, bias=False),
            #norm_layer(16),
            #nn.ReLU(True),
        )
        #self.featrues_deconv5=nn.Sequential(
        #    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
        #    norm_layer(32),
        #    nn.ReLU(True),
        #)
        #self.featrues_deconv3=nn.Sequential(
            #Block(kernel_size=3, in_size=int(4), expand_size=int(8), out_size=int(8),
                #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        #self.featrues_deconv4=nn.Sequential(
        #    Block(kernel_size=3, in_size=int(8), expand_size=int(24), out_size=int(16),
        #        nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
        #)
        #self.featrues_deconv4_1=nn.Sequential(
            #Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        self.featrues_deconv5=nn.Sequential(
            Block_4(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(24),
                nolinear=hswish(), semodule=None, stride=2),
        )
        #self.featrues_deconv6=nn.Sequential(
            #nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            #norm_layer(40),
            #nn.ReLU(True)]
        #)
        self.featrues_deconv7=nn.Sequential(
            #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False,groups=24, padding_mode = 'reflect'),
            norm_layer(24),
            hswish(),
            nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
            nn.BatchNorm2d(24),
            #hswish(),
            SeModule(int(24)),
            #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
                #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        )
        self.classifier=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            #nn.Dropout2d(0.5),
            nn.Conv2d(24, 1, kernel_size=1,stride=1, padding=0, bias=True),
            hsigmoid(),
        )
        #self.featrues_deconv8=nn.Sequential(
        #    #DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
        #    nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False,groups=24),
        #    norm_layer(24),
        #    hswish(),
        #    nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, bias=False,groups=1),
        #    nn.BatchNorm2d(24),
        #    hswish(),
        #    #Block(kernel_size=3, in_size=int(24), expand_size=int(32), out_size=int(24),
        #        #nolinear=hswish(), semodule=SeModule(int(24)), stride=1),
        #)
        #self.featrues_upconv6=nn.Sequential(
        #    nn.ConvTranspose2d(40, int(32),
        #                             kernel_size=3, stride=2,
        #                             padding=1, output_padding=1,
        #                             bias=False),
        #          norm_layer(int(32)),
        #          nn.ReLU(True),
        #)
        #self.featrues_upconv5=nn.Sequential(#4.534
        #    nn.ConvTranspose2d(32, int(16),
        #                       kernel_size=3, stride=2,
        #                       padding=1, output_padding=1,
        #                       bias=False,groups=1),
        #    norm_layer(int(16)),
        #    nn.ReLU(True),
        #    #nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #    #norm_layer(16),
        #    #nn.ReLU(True),
        #)
        self.featrues_upconv5 = nn.Sequential(
            nn.ConvTranspose2d(24, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,groups=4),
            norm_layer(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        #self.featrues_upconv5 = nn.Sequential(#这个比上面两个好
        #    Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=16,#32 40
        #          nolinear=hswish(), semodule=None, stride=2),
        #    nn.ReLU(True),
        #)
        self.featrues_upconv5_2=nn.Sequential(
            nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=False)
            #nn.ConvTranspose2d(16, int(16),
            #                   kernel_size=3, stride=(2,1),
            #                   padding=1, output_padding=(1,0),
            #                   bias=False),
            #norm_layer(int(16)),
            #nn.ReLU(True),
        )
        #self.featrues_upconv4_1=nn.Sequential(
        #    nn.Conv2d(32, int(16),
        #                       kernel_size=1, stride=1,
        #                       padding=0,
        #                       bias=False,groups=2),
        #    norm_layer(int(16)),
        #    nn.ReLU(True),
        #    #nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False),
        #    #norm_layer(8),
        #)
        self.featrues_upconv4=nn.Sequential(
            nn.ConvTranspose2d(16, int(8),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False,groups=1),
            norm_layer(int(8)),
            nn.ReLU(True),
            #nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False),
            #norm_layer(8),
        )
        #self.featrues_upconv5 = nn.Sequential(
        #    Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=16,#32 40
        #          nolinear=hswish(), semodule=None, stride=2),
        #)

        #self.featrues_upconv4_1 = nn.Sequential(
            #Block(kernel_size=3, in_size=16, expand_size=16, out_size=16,
                  #nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
        #)
        #self.featrues_upconv4_1 = nn.Sequential(
            #SeModule(int(32)),
        #)
        #self.featrues_upconv4 = nn.Sequential(
            #Block_up(kernel_size=3, in_size=32, expand_size=40, out_size=8,#32,40   16,32
                  #nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
        #)
        self.featrues_upconv3=nn.Sequential(
            nn.ConvTranspose2d(8, int(4),
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=False),
            norm_layer(int(4)),
            nn.ReLU(True),
        )
        self.featrues_upconv2=nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.featrues_upconv1=nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1, padding_mode = 'reflect'),
            hsigmoid(),
        )
        
    def forward(self, input):
        #print("input:")
        #print(input)
        out = self.featrues_deconv1(input)
        #print("out1:")
        #print(out)
        out = self.featrues_deconv2(out)
        #print("out2:")
        #print(out)
        out = self.featrues_deconv3(out)
        #print("out3:")
        #print(out)
        out4 = self.featrues_deconv4(out)
        out = self.featrues_deconv4_2(out4)
        #print("out4:")
        #print(out4)
        out5 = self.featrues_deconv5(out)
        #print("out5:")
        #print(out)
        out7 = self.featrues_deconv7(out5)
        out = out5+out7
        classify = self.classifier(out)
        classify = classify.view(classify.size(0), -1)
        #out7 = self.featrues_deconv7(out)
        #out = out+out7
        #out = out7 + out5
        out = self.featrues_upconv5(out)
        out = self.featrues_upconv5_2(out)
        #print("out6:")
        #print(out)
        #out = torch.cat([out4,out],1)
        #out = self.featrues_upconv4_1(out)

        out = out4+out
        out = self.featrues_upconv4(out)
        #print("out7:")
        #print(out)
        out = self.featrues_upconv3(out)
        #print("out8:")
        #print(out)
        out = self.featrues_upconv2(out)
        #print("out9:")
        #print(out)
        out = self.featrues_upconv1(out)
        #print("out10:")
        #print(out)
        return out,classify
if __name__ == '__main__':
    '''
    pthpath = r'test_acc_99.6159936658749.pth'
    net = MNV3_small(2)

    net.load_state_dict(torch.load(pthpath).state_dict())
    net.to('cpu')
    net.eval()

    for i in net.modules():#childerens
        if not isinstance(i, nn.Sequential) and not isinstance(i, Block):
            # print(i)
            pass
    

    a = torch.ones((1, 1, 98, 98))
    c = net(a)
    print(c)
    '''

    '''print the number of parameters '''
    net = MNV3_bufen_new(1, 1, 4, n_blocks=9)
    total = 0
    for name, parameters in net.named_parameters():
        total += parameters.nelement()
        print(name, ":", parameters.size())
    print("Number of parameter: %.2fM" % (total / 1e6))

    '''
    import netron
    import torch.onnx

    onnx_path = "onnx_model_name.onnx"
    d = torch.rand([1, 1,98,98])
    model = net
    torch.onnx.export(model, torch.rand([1, 1,98,98]),onnx_path)

    netron.start(onnx_path)
    '''
