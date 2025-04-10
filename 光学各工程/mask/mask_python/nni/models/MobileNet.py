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
                               stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
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
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
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
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(MNV3_GAN, self).__init__()

        model = [# nn.ReflectionPad2d(3),
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
        for i in range(n_blocks):       # add ResNet blocks

            model += [Block(kernel_size=3, in_size=int(ngf * mult), expand_size=int(ngf * mult * 1), out_size=int(ngf * mult),
                        nolinear=hswish(), semodule=SeModule(int(ngf*mult)), stride=1)]

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
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(MNV3_bufen, self).__init__()

        model = [# nn.ReflectionPad2d(3),
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
        for i in range(n_blocks):       # add ResNet blocks

            model += [Block(kernel_size=3, in_size=int(ngf * mult), expand_size=int(ngf * mult * 1), out_size=int(ngf * mult),
                        nolinear=hswish(), semodule=SeModule(int(ngf*mult)), stride=1)]

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
    net = UNet(1,2)
    total = 0
    for name,parameters in  net.named_parameters():
        total += parameters.nelement()
        print(name,":",parameters.size())
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