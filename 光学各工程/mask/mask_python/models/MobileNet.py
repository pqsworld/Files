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


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


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
        output = output.permute(0, 3, 1, 2)

        return output

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
        # print("input: ", input)
        # conv = nn.utils.fusion.fuse_conv_bn_eval(self.model[0], self.model[1])
        # print(conv.weight)
        # print(self.model[:30])
        # print(self.model[:30](input))
        # print("out",self.model(input))
        return self.model(input)

class MNV3_bufen_test(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_test, self).__init__()

        model = [nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(8),
                  nn.ReLU(True)]
        model += [Block(kernel_size=3, in_size=8, expand_size=16, out_size=16,
                     nolinear=hswish(), semodule=SeModule(16), stride=2)]
        model += [Block(kernel_size=3, in_size=16, expand_size=32, out_size=32,
                     nolinear=hswish(), semodule=SeModule(32), stride=2)]
        model += [Block(kernel_size=3, in_size=32, expand_size=40, out_size=40,
                     nolinear=hswish(), semodule=SeModule(40), stride=2)]
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
        
        model += [Block(kernel_size=3, in_size=4, expand_size=8, out_size=4,
                     nolinear=hswish(), semodule=SeModule(4), stride=1)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
     
        model += [hsigmoid()]

        self.model = nn.Sequential(*model)
    def forward(self, input):
        return self.model(input)


class MNV3_bufen_GNN(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_GNN, self).__init__()

        model = [  # nn.ReflectionPad2d(3),
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True)]
        model += [  # nn.ReflectionPad2d(3),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True)]

        model += [nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(32),
                  nn.ReLU(True)]
        model += [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(64),
                  nn.ReLU(True)]
        model += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(128),
                  nn.ReLU(True)]
        model += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(128),
                  nn.ReLU(True)]
        for i in range(5):  # add ResNet blocks
            model += [
                Block_SA(kernel_size=3, in_size=int(128), expand_size=int(128), out_size=int(128),
                     nolinear=hswish(), semodule=SeModule(int(128)), stride=1)]

        model += [nn.ConvTranspose2d(128, int(128),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=False),
                  norm_layer(int(128)),
                  nn.ReLU(True)]
        model += [nn.ConvTranspose2d(128, int(64),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=False),
                  norm_layer(int(64)),
                  nn.ReLU(True)]
        model += [nn.ConvTranspose2d(64, int(32),
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
        
        model += [  # nn.ReflectionPad2d(3),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True)]
        model += [nn.Conv2d(16, output_nc, kernel_size=3, padding=1)]
     
        model += [hsigmoid()]

        self.model = nn.Sequential(*model)
    def forward(self, input):
        return self.model(input)

class MNV3_bufen_subpixel(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MNV3_bufen_subpixel, self).__init__()

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
        # model += [nn.ConvTranspose2d(32, int(16),
        #                              kernel_size=3, stride=2,
        #                              padding=1, output_padding=1,
        #                              bias=False),
        #           norm_layer(int(16)),
        #           nn.ReLU(True)]
        # model += [nn.ConvTranspose2d(16, int(8),
        #                              kernel_size=3, stride=2,
        #                              padding=1, output_padding=1,
        #                              bias=False),
        #           norm_layer(int(8)),
        #           nn.ReLU(True)]
        # model += [nn.ConvTranspose2d(8, int(4),
        #                              kernel_size=3, stride=2,
        #                              padding=1, output_padding=1,
        #                              bias=False),
        #           norm_layer(int(4)),
        #           nn.ReLU(True)]
        #
        # model += [  # nn.ReflectionPad2d(3),
        #     nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False),
        #     norm_layer(ngf),
        #     nn.ReLU(True)]
        # model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
        model += [nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                  norm_layer(64),
                  nn.ReLU(True)]
        model += [nn.Conv2d(64, 64, kernel_size=1, padding=0)]
        model += [hsigmoid()]

        self.model = nn.Sequential(*model)
        self.d2s = DepthToSpace(8)
    def forward(self, input):

        return self.d2s(self.model(input))

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
