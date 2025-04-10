import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet
from typing import Optional, Callable

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
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)    # [64,16,16,64,1]
        spl = t_1.split(self.block_size, 3)     # 在第3维分割成每块包含block_size   len(spl) : 8     spl[0].shape : [64,16,16,8,1]
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]    # stack is a list, stack[0].shape : [64,16,128,1]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)     # [64,128,128,1]
        output = output.permute(0, 3, 1, 2)     # [64,1,128,128]

        return output

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size        # 8
        self.block_size_sq = block_size*block_size

    def forward(self, input):   # 将labels标签(128*128大小的尺寸)转换为特征图的尺寸(16*16*64)
        output = input.permute(0, 2, 3, 1)      # 将tensor的维度换位; [64,1,128,128] => [64,128,128,1]
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)  # split_size=block_size,dim=2，沿第2轴进行拆分,每个划分大小为block_size
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]     # len(stack) : 16  , stack[0].shape : [64,16,64]
        output = torch.stack(stack, 1)          # [64,16,16,64]
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)     # [64,64,16,16]
        return output

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

        self.aligned_corner = True # False

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
        
        scores_map = torch.sigmoid(x)
        # scores_map = torch.sigmoid(x[:, -1, :, :]).unsqueeze(1)

        # from PIL import Image
        # imgs = Image.fromarray((scores_map[0, 0]).float().squeeze().cpu().numpy()*255).convert("L")
        # imgs.save("/hdd/file-input/linwc/Descriptor/code/Test_sys/logs/output/pnt/0711a_239800_thr0.2_nms2_XArot_modify_t/all.bmp")
        # exit()
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

class ALNet_Angle_Standard(nn.Module):
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
        # print(x4_up[:, 0, 1, :])
        # exit()
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


class ALNet_nodesc(nn.Module):
    def __init__(self, input_nc=1, output_nc=65, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=2,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ALNet_nodesc, self).__init__()

        model = [  # nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 4, kernel_size=3, padding=1, bias=False),
            norm_layer(4),
            nn.ReLU(True)]
        model += [  # nn.ReflectionPad2d(3),
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
            norm_layer(4),
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
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1)]

        self.model = nn.Sequential(*model)

        # Detector Head.
        model_1 = [  # nn.ReflectionPad2d(3),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(True)]
        model_1 += [nn.Conv2d(64, output_nc, kernel_size=1, padding=0)]
        # model += [hsigmoid()]
        self.model_detector = nn.Sequential(*model_1)
        self.space2depth = DepthToSpace(8)

    def forward(self, image):
        x = self.model(image)
        # Detector Head.
        scores = self.model_detector(x)       # B x 64 x H/8 x W/4

        scores = self.space2depth(scores)     # B x 1 x H x W
        descriptor_map = None
        scores_map = torch.sigmoid(scores)

        return scores_map, descriptor_map


class ALNet_Angle_Deep(nn.Module):
    def __init__(self, c1: int = 4, c2: int = 8, c3: int = 16, c4: int = 32, dim: int = 16,
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
        # if not self.single_head:
        self.convhead1 = nn.Conv2d(dim, 3, kernel_size=1, padding=0)
            # self.convhead1 = resnet.conv1x1(dim, dim)
        # self.convhead2 = nn.Conv2d(dim, dim + 1, kernel_size=1, padding=0)
        self.convhead2 = nn.Conv2d(3, 3, kernel_size=1, padding=0)

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
        # if not self.single_head:
        #     x1234 = self.gate(self.convhead1(x1234))
        x1234 = self.gate(self.convhead1(x1234))
        x = self.convhead2(x1234)  # B x dim+1 x H x W

        # descriptor_map = x[:, :-1, :, :]
        descriptor_map = None
        scores_map = torch.sigmoid(x)

        return scores_map, descriptor_map


class ALNet_Angle_Deep_Short(nn.Module):
    def __init__(self, c1: int = 4, c2: int = 8, c3: int = 16, c4: int = 32, dim: int = 16,
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
        # if not self.single_head:
        self.convhead1 = nn.Conv2d(dim, 3, kernel_size=1, padding=0)
            # self.convhead1 = resnet.conv1x1(dim, dim)
        # self.convhead2 = nn.Conv2d(dim, dim + 1, kernel_size=1, padding=0)
        self.convhead2 = nn.Conv2d(3, 3, kernel_size=1, padding=0)

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
        # if not self.single_head:
        #     x1234 = self.gate(self.convhead1(x1234))
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


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, groups=1, modulation=False):
    # def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, groups=1, modulation=False, offset=True):
        """
        Args:
            moduleation(bool, optional): If True, Modulated Defromable Convolution(Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.zero_padding = nn.ZeroPad2d(padding)
        # self.offset = offset
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias, groups=groups)
        
        # self.p_conv偏置层，学习公式（2）中的偏移量。
        # 2*kernel_size*kernel_size：代表了卷积核中所有元素的偏移坐标，因为同时存在x和y的偏移，故要乘以2。
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        # if not offset:
        #     self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        #     nn.init.constant_(self.p_conv.weight, 0)
        # # register_backward_hook是为了方便查看这几层学出来的结果，对网络结构无影响。
        # self.p_conv.register_backward_hook(self._set_lr)
        
        self.modulation = modulation
        if modulation:
            # self.m_conv权重学习层，是后来提出的第二个版本的卷积也就是公式（3）描述的卷积。
            # kernel_size*kernel_size：代表了卷积核中每个元素的权重。
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            # register_backward_hook是为了方便查看这几层学出来的结果，对网络结构无影响。
            # self.m_conv.register_backward_hook(self._set_lr)
            
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
    
    # 生成卷积核的邻域坐标
    def _get_p_n(self, N, dtype):
        """
        torch.meshgrid():Creates grids of coordinates specified by the 1D inputs in attr:tensors.
        功能是生成网格，可以用于生成坐标。
        函数输入两个数据类型相同的一维张量，两个输出张量的行数为第一个输入张量的元素个数，
        列数为第二个输入张量的元素个数，当两个输入张量数据类型不同或维度不是一维时会报错。
        
        其中第一个输出张量填充第一个输入张量中的元素，各行元素相同；
        第二个输出张量填充第二个输入张量中的元素各列元素相同。
        """
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        
        # p_n ===>offsets_x(kernel_size*kernel_size,) concat offsets_y(kernel_size*kernel_size,) 
        #     ===> (2*kernel_size*kernel_size,)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        # （1， 2*kernel_size*kernel_size, 1, 1）
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n
    
    # 获取卷积核在feature map上所有对应的中心坐标，也就是p0
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # (b, 2*kernel_size, h, w)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0
    
    # 将获取的相对坐标信息与中心坐标相加就获得了卷积核的所有坐标。
    # 再加上之前学习得到的offset后，就是加上了偏移量后的坐标信息。
    # 即对应论文中公式(2)中的(p0+pn+Δpn)
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        # p_n ===> (1, 2*kernel_size*kernel_size, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # p_0 ===> (1, 2*kernel_size*kernel_size, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # (1, 2*kernel_size*kernel_size, h, w)
        p = p_0.to(offset.device) + p_n.to(offset.device) + offset
        return p
    
    def _get_x_q(self, x, q, N):
        # b, h, w, 2*kerel_size*kernel_size
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # x ===> (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        # 因为x是与h轴方向平行，y是与w轴方向平行。故将2D卷积核投影到1D上，位移公式如下：
        # 各个卷积核中心坐标及邻域坐标的索引 offsets_x * w + offsets_y
        # (b, h, w, kernel_size*kernel_size)
        index = q[..., :N] * padded_w + q[..., N:]
        # (b, c, h, w, kernel_size*kernel_size) ===> (b, c, h*w*kernel_size*kernel_size)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        # (b, c, h*w)
        # x_offset[0][0][0] = x[0][0][index[0][0][0]]
        # index[i][j][k]的值应该是一一对应着输入x的(h*w)的坐标，且在之前将index[i][j][k]的值clamp在[0, h]及[0, w]范围里。
        # (b, c, h, w, kernel_size*kernel_size)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset
    
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        # (b, c, h, w, kernel_size*kernel_size)
        b, c, h, w, N = x_offset.size()
        # (b, c, h, w*kernel_size)
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        # (b, c, h*kernel_size, w*kernel_size)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
        
        return x_offset
    
    def forward(self, x):
    # def forward(self, x, extern_offset):
        # (b, c, h, w) ===> (b, 2*kernel_size*kernel_size, h, w)
        # if self.offset:
        #     offset = extern_offset
        # else:
        #     offset = self.p_conv(x)
        offset = self.p_conv(x)
        if self.modulation:
            # (b, c, h, w) ===> (b, kernel_size*kernel_size, h, w)
            m = torch.sigmoid(self.m_conv(x))
        
        dtype = offset.data.type()
        ks = self.kernel_size
        # kernel_size*kernel_size
        N = offset.size(1) // 2
        
        if self.padding:
            x = self.zero_padding(x)
        # (b, 2*kernel_size*kernel_size, h, w)
        p = self._get_p(offset, dtype)
        # (b, h, w, 2*kernel_size*kernel_size)
        p = p.contiguous().permute(0, 2, 3, 1)
        # 将p从tensor的前向计算中取出来，并向下取整得到左上角坐标q_lt。
        q_lt = p.detach().floor()
        # 将p向上再取整，得到右下角坐标q_rb。
        q_rb = q_lt + 1
        
        # 学习的偏移量是float类型，需要用双线性插值的方法去推算相应的值。
        # 同时防止偏移量太大，超出feature map，故需要torch.clamp来约束。
        # Clamps all elements in input into the range [ min, max ].
        # torch.clamp(a, min=-0.5, max=0.5)
        
        # p左上角x方向的偏移量不超过h,y方向的偏移量不超过w。
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # p右下角x方向的偏移量不超过h,y方向的偏移量不超过w。
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # p左上角的x方向的偏移量和右下角y方向的偏移量组合起来，得到p左下角的值。
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        # p右下角的x方向的偏移量和左上角y方向的偏移量组合起来，得到p右上角的值。
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        
        # clip p。
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
        
        # 双线性插值公式里的四个系数。即bilinear kernel。
        # 作者代码为了保持整齐，每行的变量计算形式一样，所以计算需要做一点对应变量的对应变化。
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        
        # 计算双线性插值的四个坐标对应的像素值。
        # (b, c, h, w, kernel_size*kernel_size)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        
        # 双线性插值的最后计算
        # (b, c, h, w, kernel_size*kernel_size)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                    g_rb.unsqueeze(dim=1) * x_q_rb + \
                    g_lb.unsqueeze(dim=1) * x_q_lb + \
                    g_rt.unsqueeze(dim=1) * x_q_rt
        
        # modulation
        if self.modulation:
            # (b, kernel_size*kernel_size, h, w) ===> (b, h, w, kernel_size*kernel_size)
            m = m.contiguous().permute(0, 2, 3, 1)
            # (b, h, w, kernel_size*kernel_size) ===>  (b, 1, h, w, kernel_size*kernel_size)
            m = m.unsqueeze(dim=1)
            # (b, c, h, w, kernel_size*kernel_size)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        # x_offset: (b, c, h, w, kernel_size*kernel_size)
        # x_offset: (b, c, h*kernel_size, w*kernel_size)
        x_offset = self._reshape_x_offset(x_offset, ks)
        # out: (b, c, h, w)
        out = self.conv(x_offset)

        return out
        # return out, offset / self.stride


class Block_Deform_noshortcut(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_Deform_noshortcut, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        # self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
        #                        stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)

        self.conv2 = DeformConv2d(expand_size, expand_size, kernel_size=kernel_size,
                        stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False, modulation=True)

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


class ALNet_Angle_Deform(nn.Module):
    def __init__(self, c1: int = 4, c2: int = 8, c3: int = 16, c4: int = 32, dim: int = 16,
                 single_head: bool = True,
                 ):
        super().__init__()

        self.aligned_corner = True

        self.gate = nn.SELU(inplace=True) # nn.ReLU(inplace=True) # 

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.block1 = ConvBlock_New(1, c1, self.gate, nn.BatchNorm2d)

        self.block2 = Block_noshortcut(kernel_size=3, in_size=int(c1), expand_size=int(c2), out_size=int(c2),
                     nolinear=hswish(), semodule=SeModule(int(c2)), stride=1)
        # ResBlock(inplanes=c1, planes=c2, stride=1,
        #                        downsample=nn.Conv2d(c1, c2, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)
        self.block3 = Block_Deform_noshortcut(kernel_size=3, in_size=int(c2), expand_size=int(c3), out_size=int(c3),
                     nolinear=hswish(), semodule=SeModule(int(c3)), stride=1)
        # ResBlock(inplanes=c2, planes=c3, stride=1,
        #                        downsample=nn.Conv2d(c2, c3, 1),
        #                        gate=self.gate,
        #                        norm_layer=nn.BatchNorm2d)
        self.block4 = Block_Deform_noshortcut(kernel_size=3, in_size=int(c3), expand_size=int(c4), out_size=int(c4),
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
