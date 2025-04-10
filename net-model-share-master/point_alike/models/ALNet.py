from stringprep import c22_specials
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet
from typing import Optional, Callable
#from utils.d2s import DepthToSpace, SpaceToDepth

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
        # print(list(self.conv1.parameters()))
        out = self.nolinear2(self.conv2(out))
        out = self.conv3(out)
        if self.se != None:
            out = self.se(out)
        # out = out + self.shortcut(x) if self.stride == 1 else out
        out = out + self.shortcut(x) if (self.stride == 1 and x.shape[1] == out.shape[1]) else out
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x

class ConvBlock_New(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x

class ConvBlock_Single(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        # self.conv2 = resnet.conv3x3(out_channels, out_channels)
        # self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        # x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


class ConvBlock_New_Short(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=True)
        # resnet.conv3x3(in_channels, out_channels)
        # self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=True)
        # resnet.conv3x3(out_channels, out_channels)
        # self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.conv1(x))  # B x in_channels x H x W
        x = self.gate(self.conv2(x))  # B x out_channels x H x W
        return x

# copied from torchvision\models\resnet.py#27->BasicBlock
class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


class ALNet(nn.Module):
    def __init__(self, c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128, dim: int = 128,
                 single_head: bool = True,
                 ):
        super().__init__()

        self.gate = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.block1 = ConvBlock(1, c1, self.gate, nn.BatchNorm2d)

        self.block2 = ResBlock(inplanes=c1, planes=c2, stride=1,
                               downsample=nn.Conv2d(c1, c2, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        self.block3 = ResBlock(inplanes=c2, planes=c3, stride=1,
                               downsample=nn.Conv2d(c2, c3, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        self.block4 = ResBlock(inplanes=c3, planes=c4, stride=1,
                               downsample=nn.Conv2d(c3, c4, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)

        # ================================== feature aggregation
        self.conv1 = resnet.conv1x1(c1, dim // 4)
        self.conv2 = resnet.conv1x1(c2, dim // 4)
        self.conv3 = resnet.conv1x1(c3, dim // 4)
        self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        # ================================== detector and descriptor head
        self.single_head = single_head
        if not self.single_head:
            self.convhead1 = resnet.conv1x1(dim, dim)
        self.convhead2 = resnet.conv1x1(dim, dim + 1)

    def forward(self, image):
        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool2(x2)
        x3 = self.block3(x3)  # B x c3 x H/4 x W/4
        x4 = self.pool2(x3)
        x4 = self.block4(x4)  # B x dim x H/8 x W/8

        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//4 x W//4
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//8 x W//8
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        x3_up = self.upsample4(x3)  # B x dim//4 x H x W
        x4_up = self.upsample8(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)

        # ================================== detector and descriptor head
        if not self.single_head:
            x1234 = self.gate(self.convhead1(x1234))
        x = self.convhead2(x1234)  # B x dim+1 x H x W

        descriptor_map = x[:, :-1, :, :]
        scores_map = torch.sigmoid(x[:, -1, :, :]).unsqueeze(1)

        return scores_map, descriptor_map


class ALNet_New(nn.Module):
    def __init__(self, c1: int = 8, c2: int = 16, c3: int = 32, c4: int = 64, dim: int = 32,
                 single_head: bool = True,
                 ):
        super().__init__()

        self.aligned_corner = True

        self.gate = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.block1 = ConvBlock_New(1, c1, self.gate, nn.BatchNorm2d)

        self.block2 = Block_noshortcut(kernel_size=3, in_size=int(c1), expand_size=int(c2), out_size=int(c2),
                     nolinear=hswish(), semodule=SeModule(int(c2)), stride=1)
        # ResBlock(inplanes=c1, planes=c2, stride=1,
        #                        downsample=nn.Conv2d(c1, c2, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)
        self.block3 = Block_noshortcut(kernel_size=3, in_size=int(c2), expand_size=int(c3), out_size=int(c3),
                     nolinear=hswish(), semodule=SeModule(int(c3)), stride=1)
        # ResBlock(inplanes=c2, planes=c3, stride=1,
        #                        downsample=nn.Conv2d(c2, c3, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)
        self.block4 = Block_noshortcut(kernel_size=3, in_size=int(c3), expand_size=int(c4), out_size=int(c4),
                     nolinear=hswish(), semodule=SeModule(int(c4)), stride=1)
        # ResBlock(inplanes=c3, planes=c4, stride=1,
        #                        downsample=nn.Conv2d(c3, c4, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)

        # ================================== feature aggregation
        self.conv1 = nn.Conv2d(c1, dim // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(c2, dim // 4, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(c3, dim // 4, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(c4, dim // 4, kernel_size=1, padding=0)
        # self.conv1 = resnet.conv1x1(c1, dim // 4)
        # self.conv2 = resnet.conv1x1(c2, dim // 4)
        # self.conv3 = resnet.conv1x1(c3, dim // 4)
        # self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.aligned_corner)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=self.aligned_corner)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=self.aligned_corner)
        # self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        # ================================== detector and descriptor head
        self.single_head = single_head
        if not self.single_head:
            self.convhead1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
            # self.convhead1 = resnet.conv1x1(dim, dim)
        # self.convhead2 = nn.Conv2d(dim, dim + 1, kernel_size=1, padding=0)
        self.convhead2 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)

    def forward(self, image):
        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool2(x2)
        x3 = self.block3(x3)  # B x c3 x H/4 x W/4
        x4 = self.pool2(x3)
        x4 = self.block4(x4)  # B x dim x H/8 x W/8

        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//4 x W//4
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//8 x W//8

        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        # print(x2_up[0, 0, 0, :])
        # exit()
        x3_up = self.upsample4(x3)  # B x dim//4 x H x W
        
        x4_up = self.upsample8(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)


        # ================================== detector and descriptor head
        if not self.single_head:
            x1234 = self.gate(self.convhead1(x1234))
        # print(x1234.shape, self.single_head)
        x = self.convhead2(x1234)  # B x dim+1 x H x W

        # descriptor_map = x[:, :-1, :, :]
        descriptor_map = None
        scores_map = torch.sigmoid(x[:, -1, :, :]).unsqueeze(1)

        # from PIL import Image
        # imgs = Image.fromarray((scores_map[0, 0]).float().squeeze().cpu().numpy()*255).convert("L")
        # imgs.save("/hdd/file-input/linwc/Descriptor/code/Test_sys/logs/output/pnt/0711a_239800_thr0.2_nms2_XArot_modify_t/all.bmp")
        # exit()
        return scores_map, descriptor_map

class ALNet_Standard(nn.Module):
    def __init__(self, c1: int = 8, c2: int = 16, c3: int = 32, c4: int = 64, dim: int = 32,
                 single_head: bool = True,
                 ):
        super().__init__()

        self.gate = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.block1 = ConvBlock_New(1, c1, self.gate, nn.BatchNorm2d)

        self.block2 = Block_noshortcut(kernel_size=3, in_size=int(c1), expand_size=int(c2), out_size=int(c2),
                     nolinear=nn.Hardswish(inplace=True), semodule=SeModule_standard(int(c2)), stride=1)
        # ResBlock(inplanes=c1, planes=c2, stride=1,
        #                        downsample=nn.Conv2d(c1, c2, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)
        self.block3 = Block_noshortcut(kernel_size=3, in_size=int(c2), expand_size=int(c3), out_size=int(c3),
                     nolinear=nn.Hardswish(inplace=True), semodule=SeModule_standard(int(c3)), stride=1)
        # ResBlock(inplanes=c2, planes=c3, stride=1,
        #                        downsample=nn.Conv2d(c2, c3, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)
        self.block4 = Block_noshortcut(kernel_size=3, in_size=int(c3), expand_size=int(c4), out_size=int(c4),
                     nolinear=nn.Hardswish(inplace=True), semodule=SeModule_standard(int(c4)), stride=1)
        # ResBlock(inplanes=c3, planes=c4, stride=1,
        #                        downsample=nn.Conv2d(c3, c4, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)

        # ================================== feature aggregation
        self.conv1 = nn.Conv2d(c1, dim // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(c2, dim // 4, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(c3, dim // 4, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(c4, dim // 4, kernel_size=1, padding=0)
        # self.conv1 = resnet.conv1x1(c1, dim // 4)
        # self.conv2 = resnet.conv1x1(c2, dim // 4)
        # self.conv3 = resnet.conv1x1(c3, dim // 4)
        # self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        # self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        # ================================== detector and descriptor head
        self.single_head = single_head
        if not self.single_head:
            self.convhead1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
            # self.convhead1 = resnet.conv1x1(dim, dim)
        # self.convhead2 = nn.Conv2d(dim, dim + 1, kernel_size=1, padding=0)
        self.convhead2 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)

    def forward(self, image):
        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool2(x2)
        x3 = self.block3(x3)  # B x c3 x H/4 x W/4
        x4 = self.pool2(x3)
        x4 = self.block4(x4)  # B x dim x H/8 x W/8

        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//4 x W//4
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//8 x W//8
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        x3_up = self.upsample4(x3)  # B x dim//4 x H x W
        x4_up = self.upsample8(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)

        # ================================== detector and descriptor head
        if not self.single_head:
            x1234 = self.gate(self.convhead1(x1234))
        x = self.convhead2(x1234)  # B x dim+1 x H x W

        # descriptor_map = x[:, :-1, :, :]
        descriptor_map = None
        scores_map = torch.sigmoid(x[:, -1, :, :]).unsqueeze(1)

        return scores_map, descriptor_map


class ALNet_Angle(nn.Module):
    def __init__(self, c1: int = 8, c2: int = 16, c3: int = 32, c4: int = 64, dim: int = 32,
                 single_head: bool = True,
                 ):
        super().__init__()

        self.aligned_corner = True

        self.gate = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.block1 = ConvBlock_New(1, c1, self.gate, nn.BatchNorm2d)

        self.block2 = Block_noshortcut(kernel_size=3, in_size=int(c1), expand_size=int(c2), out_size=int(c2),
                     nolinear=hswish(), semodule=SeModule(int(c2)), stride=1)
        # ResBlock(inplanes=c1, planes=c2, stride=1,
        #                        downsample=nn.Conv2d(c1, c2, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)
        self.block3 = Block_noshortcut(kernel_size=3, in_size=int(c2), expand_size=int(c3), out_size=int(c3),
                     nolinear=hswish(), semodule=SeModule(int(c3)), stride=1)
        # ResBlock(inplanes=c2, planes=c3, stride=1,
        #                        downsample=nn.Conv2d(c2, c3, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)
        self.block4 = Block_noshortcut(kernel_size=3, in_size=int(c3), expand_size=int(c4), out_size=int(c4),
                     nolinear=hswish(), semodule=SeModule(int(c4)), stride=1)
        # ResBlock(inplanes=c3, planes=c4, stride=1,
        #                        downsample=nn.Conv2d(c3, c4, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)

        # ================================== feature aggregation
        self.conv1 = nn.Conv2d(c1, dim // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(c2, dim // 4, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(c3, dim // 4, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(c4, dim // 4, kernel_size=1, padding=0)
        # self.conv1 = resnet.conv1x1(c1, dim // 4)
        # self.conv2 = resnet.conv1x1(c2, dim // 4)
        # self.conv3 = resnet.conv1x1(c3, dim // 4)
        # self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.aligned_corner)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=self.aligned_corner)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=self.aligned_corner)
        # self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        # ================================== detector and descriptor head
        self.single_head = single_head
        if not self.single_head:
            self.convhead1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
            # self.convhead1 = resnet.conv1x1(dim, dim)
        # self.convhead2 = nn.Conv2d(dim, dim + 1, kernel_size=1, padding=0)
        self.convhead2 = nn.Conv2d(dim, 3, kernel_size=1, padding=0)

    def forward(self, image):
        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool2(x2)
        x3 = self.block3(x3)  # B x c3 x H/4 x W/4
        x4 = self.pool2(x3)
        x4 = self.block4(x4)  # B x dim x H/8 x W/8

        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//4 x W//4
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//8 x W//8
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        x3_up = self.upsample4(x3)  # B x dim//4 x H x W
        x4_up = self.upsample8(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)

        # ================================== detector and descriptor head
        if not self.single_head:
            x1234 = self.gate(self.convhead1(x1234))
        x = self.convhead2(x1234)  # B x dim+1 x H x W

        # descriptor_map = x[:, :-1, :, :]
        descriptor_map = None
        scores_map = torch.sigmoid(x)

        return scores_map, descriptor_map

class ALNet_Angle_Short(nn.Module):
    def __init__(self, 
                 c1: int = 8, c2: int = 16, c3: int = 32, c4: int = 64, dim: int = 32,
                 # c1: int = 4, c2: int = 8, c3: int = 16, c4: int = 32, dim: int = 16,
                 # c1: int = 16, c2: int = 32, c3: int = 64, c4: int = 128, dim: int = 64,
                 single_head: bool = True,
                 ):
        super().__init__()

        self.aligned_corner = True

        self.gate = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.block1 = ConvBlock_New_Short(1, c1, self.gate, nn.BatchNorm2d)

        self.block2 = Block_noshortcut_short(kernel_size=3, in_size=int(c1), expand_size=int(c2), out_size=int(c2),
                     nolinear=hswish(), semodule=SeModule_short(int(c2)), stride=1)
        # ResBlock(inplanes=c1, planes=c2, stride=1,
        #                        downsample=nn.Conv2d(c1, c2, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)
        self.block3 = Block_noshortcut_short(kernel_size=3, in_size=int(c2), expand_size=int(c3), out_size=int(c3),
                     nolinear=hswish(), semodule=SeModule_short(int(c3)), stride=1)
        # ResBlock(inplanes=c2, planes=c3, stride=1,
        #                        downsample=nn.Conv2d(c2, c3, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)
        self.block4 = Block_noshortcut_short(kernel_size=3, in_size=int(c3), expand_size=int(c4), out_size=int(c4),
                     nolinear=hswish(), semodule=SeModule_short(int(c4)), stride=1)
        # ResBlock(inplanes=c3, planes=c4, stride=1,
        #                        downsample=nn.Conv2d(c3, c4, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)

        # ================================== feature aggregation
        self.conv1 = nn.Conv2d(c1, dim // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(c2, dim // 4, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(c3, dim // 4, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(c4, dim // 4, kernel_size=1, padding=0)
        # self.conv1 = resnet.conv1x1(c1, dim // 4)
        # self.conv2 = resnet.conv1x1(c2, dim // 4)
        # self.conv3 = resnet.conv1x1(c3, dim // 4)
        # self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.aligned_corner)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=self.aligned_corner)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=self.aligned_corner)
        # self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        # ================================== detector and descriptor head
        self.single_head = single_head
        if not self.single_head:
            self.convhead1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
            # self.convhead1 = resnet.conv1x1(dim, dim)
        # self.convhead2 = nn.Conv2d(dim, dim + 1, kernel_size=1, padding=0)
        self.convhead2 = nn.Conv2d(dim, 3, kernel_size=1, padding=0)

    def forward(self, image):
        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2

        x3 = self.pool2(x2)

        x3 = self.block3(x3)  # B x c3 x H/4 x W/4

        x4 = self.pool2(x3)
        # print(x4[0,0,1,:])
        # exit()
        x4 = self.block4(x4)  # B x dim x H/8 x W/8

        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2

        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//4 x W//4
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//8 x W//8
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W

        x3_up = self.upsample4(x3)  # B x dim//4 x H x W

        x4_up = self.upsample8(x4)  # B x dim//4 x H x W

        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)

        # ================================== detector and descriptor head
        if not self.single_head:
            x1234 = self.gate(self.convhead1(x1234))

        x = self.convhead2(x1234)  # B x dim+1 x H x W

        # descriptor_map = x[:, :-1, :, :]
        descriptor_map = None
        scores_map = torch.sigmoid(x)

        return scores_map, descriptor_map

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


class ALNet_Angle_New(nn.Module):
    def __init__(self, c1: int = 8, c2: int = 8, c3: int = 16, c4: int = 32, dim: int = 16,
                 single_head: bool = True,
                 ):
        super().__init__()

        self.aligned_corner = True

        self.gate = nn.ReLU(inplace=True)

        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.block1 = ConvBlock_Single(1, c1, self.gate, nn.BatchNorm2d)

        self.block2 = Block_noshortcut(kernel_size=3, in_size=int(c1), expand_size=int(c2), out_size=int(c2),
                     nolinear=hswish(), semodule=SeModule(int(c2)), stride=2)
        # ResBlock(inplanes=c1, planes=c2, stride=1,
        #                        downsample=nn.Conv2d(c1, c2, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)
        self.block3 = Block_noshortcut(kernel_size=3, in_size=int(c2), expand_size=int(c3), out_size=int(c3),
                     nolinear=hswish(), semodule=SeModule(int(c3)), stride=2)
        # ResBlock(inplanes=c2, planes=c3, stride=1,
        #                        downsample=nn.Conv2d(c2, c3, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)
        self.block4 = Block_noshortcut(kernel_size=3, in_size=int(c3), expand_size=int(c4), out_size=int(c4),
                     nolinear=hswish(), semodule=SeModule(int(c4)), stride=2)
        # ResBlock(inplanes=c3, planes=c4, stride=1,
        #                        downsample=nn.Conv2d(c3, c4, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)

        # ================================== feature aggregation
        self.conv1 = nn.Conv2d(c1, dim // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(c2, dim // 4, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(c3, dim // 4, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(c4, dim // 4, kernel_size=1, padding=0)
        # self.conv1 = resnet.conv1x1(c1, dim // 4)
        # self.conv2 = resnet.conv1x1(c2, dim // 4)
        # self.conv3 = resnet.conv1x1(c3, dim // 4)
        # self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.aligned_corner)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=self.aligned_corner)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=self.aligned_corner)
        # self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        # ================================== detector and descriptor head
        self.single_head = single_head
        if not self.single_head:
            self.convhead1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
            # self.convhead1 = resnet.conv1x1(dim, dim)
        # self.convhead2 = nn.Conv2d(dim, dim + 1, kernel_size=1, padding=0)
        self.convhead2 = nn.Conv2d(dim, 3, kernel_size=1, padding=0)

    def forward(self, image):
        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        # x2 = self.pool2(x1)
        x2 = self.block2(x1)  # B x c2 x H/2 x W/2
        # x3 = self.pool2(x2)
        x3 = self.block3(x2)  # B x c3 x H/4 x W/4
        # x4 = self.pool2(x3)
        x4 = self.block4(x3)  # B x dim x H/8 x W/8

        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//4 x W//4
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//8 x W//8
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        x3_up = self.upsample4(x3)  # B x dim//4 x H x W
        x4_up = self.upsample8(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)

        # ================================== detector and descriptor head
        if not self.single_head:
            x1234 = self.gate(self.convhead1(x1234))
        x = self.convhead2(x1234)  # B x dim+1 x H x W

        # descriptor_map = x[:, :-1, :, :]
        descriptor_map = None
        scores_map = torch.sigmoid(x)

        return scores_map, descriptor_map

#class ALNet_nodesc(nn.Module):
#    def __init__(self, input_nc=1, output_nc=65, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=2,
#                 padding_type='reflect'):
#        assert (n_blocks >= 0)
#        super(ALNet_nodesc, self).__init__()
#
#        model = [  # nn.ReflectionPad2d(3),
#            nn.Conv2d(input_nc, 4, kernel_size=3, padding=1, bias=False),
#            norm_layer(4),
#            nn.ReLU(True)]
#        model += [  # nn.ReflectionPad2d(3),
#            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
#            norm_layer(4),
#            nn.ReLU(True)]
#
#        model += [nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
#                  norm_layer(8),
#                  nn.ReLU(True)]
#        model += [nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
#                  norm_layer(16),
#                  nn.ReLU(True)]
#        model += [nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
#                  norm_layer(32),
#                  nn.ReLU(True)]
#        for i in range(n_blocks):  # add ResNet blocks
#            model += [
#                Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
#                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1)]
#
#        self.model = nn.Sequential(*model)
#
#        # Detector Head.
#        model_1 = [  # nn.ReflectionPad2d(3),
#            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
#            norm_layer(64),
#            nn.ReLU(True)]
#        model_1 += [nn.Conv2d(64, output_nc, kernel_size=1, padding=0)]
#        # model += [hsigmoid()]
#        self.model_detector = nn.Sequential(*model_1)
#        self.space2depth = DepthToSpace(8)
#
#    def forward(self, image):
#        x = self.model(image)
#        # Detector Head.
#        scores = self.model_detector(x)       # B x 64 x H/8 x W/4
#        # print(scores.shape)
#        scores = self.space2depth(scores)     # B x 1 x H x W
#        descriptor_map = None
#        scores_map = torch.sigmoid(scores)
#
#        return scores_map, descriptor_map





