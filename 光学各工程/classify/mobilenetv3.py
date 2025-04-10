import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision
import numpy as np
from collections import deque
from torch.autograd import Function
class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

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
        x1 = torch.cat([avgout, maxout], dim=1)
        x1 = self.conv(x1)
        return x*self.nonlinear(x1)

class CBAModule(nn.Module):  # CNN Block Attention Module
    def __init__(self, in_size, reduction=4):
        super(CBAModule, self).__init__()
        self.ca = ChannelAttention(in_size)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride,padding=None):
        super(Block, self).__init__()
        self.stride = stride
        padding_p = kernel_size//2 if padding == None else padding

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=padding_p, groups=expand_size, bias=False)
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
class Product_Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size,  nolinear):
        super(Product_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_size, in_size,
                               kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_size)
        self.nolinear1 = nolinear



        self.conv2 = nn.Conv2d(in_size, in_size, kernel_size=kernel_size,
                               stride=1, padding=3, bias=False, dilation = 3)
        self.bn2 = nn.BatchNorm2d(in_size)
        self.nolinear2 = nolinear



        #self.conv3 = nn.Conv2d(in_size, in_size,
                               #kernel_size=kernel_size, stride=1, padding=5, bias=False, dilation = 5)
        #self.bn3 = nn.BatchNorm2d(in_size)
        #self.nolinear3 = nolinear


        self.conv4 = nn.Conv2d(in_size*2, in_size,
                               kernel_size=kernel_size, stride=1, padding=1, bias=False, dilation = 1)
        self.bn4 = nn.BatchNorm2d(in_size)
        self.nolinear4 = nolinear

        self.sig = hsigmoid()

    def forward(self, x):  
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out1 = self.nolinear2(self.bn2(self.conv2(out)))
        #out2 = self.nolinear3(self.bn3(self.conv3(out1)))
        #out3 = torch.cat((out,out1,out2),1)
        out3 = torch.cat((out,out1),1)
        out4 = self.nolinear4(self.bn4(self.conv4(out3)))
        out4 = self.sig(out4)
        out4 = x * out4

        
        return out4
class Block_p(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride,padding=None):
        super(Block_p, self).__init__()
        self.stride = stride
        padding_p = kernel_size//2 if padding == None else padding

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=padding_p, groups=expand_size, bias=False, padding_mode='replicate')
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
        out = out + self.shortcut(x) if self.stride == 3 else out
        return out
class Block_cc(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride,padding=None):
        super(Block_cc, self).__init__()
        self.stride = stride
        padding_p = kernel_size//2 if padding == None else padding

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=padding_p, groups=expand_size, bias=False)
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



class MNV3_small_6_3_6_10(nn.Module):#20562    也可
    def __init__(self, numclasses):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.bneck = nn.Sequential(
            Block(kernel_size=3, in_size=8, expand_size=8, out_size=8,
                  nolinear=hswish(), semodule=None, stride=1),
            Block(kernel_size=3, in_size=8, expand_size=32, out_size=16,
                  nolinear=hswish(), semodule=None, stride=2),

            # nn.AvgPool2d(kernel_size=2, stride=2),
            Block(kernel_size=3, in_size=16, expand_size=64, out_size=16,
                  nolinear=hswish(), semodule=None, stride=1),
            Block(kernel_size=3, in_size=16, expand_size=64, out_size=24,
                  nolinear=hswish(), semodule=None, stride=2),

            # nn.AvgPool2d(kernel_size=2, stride=2),
            Block(kernel_size=3, in_size=24, expand_size=96, out_size=24,
                  nolinear=hswish(), semodule=None, stride=1),
            Block(kernel_size=3, in_size=24, expand_size=96, out_size=32,
                  nolinear=hswish(), semodule=None, stride=2),

            # nn.AvgPool2d(kernel_size=2, stride=2),
            #Block(kernel_size=3, in_size=32, expand_size=128, out_size=40,
                  #nolinear=hswish(), semodule=None, stride=2),
            #Block(kernel_size=3, in_size=40, expand_size=160, out_size=40,
                  #nolinear=hswish(), semodule=None, stride=1),


            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(),
        )

        self.classifier = nn.Conv2d(in_channels=32, out_channels=numclasses, kernel_size=1,
                      stride=1, padding=0, bias=True)

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
        x = self.layer1(x)       
        x = self.bneck(x)
        features = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return features,x
class MNV3_small_6_3_6_10_c2(nn.Module):#20562    也可
    def __init__(self, numclasses):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.bneck = nn.Sequential(
            #Block(kernel_size=3, in_size=8, expand_size=8, out_size=8,
                  #nolinear=hswish(), semodule=None, stride=1),
            Block(kernel_size=3, in_size=8, expand_size=32, out_size=16,
                  nolinear=hswish(), semodule=None, stride=2),

            # nn.AvgPool2d(kernel_size=2, stride=2),
            Block(kernel_size=3, in_size=16, expand_size=32, out_size=16,
                  nolinear=hswish(), semodule=None, stride=1),
            Block(kernel_size=3, in_size=16, expand_size=64, out_size=24,
                  nolinear=hswish(), semodule=None, stride=2),

            # nn.AvgPool2d(kernel_size=2, stride=2),
            Block(kernel_size=3, in_size=24, expand_size=48, out_size=24,
                  nolinear=hswish(), semodule=None, stride=1),
            #Block(kernel_size=3, in_size=24, expand_size=96, out_size=32,
                  #nolinear=hswish(), semodule=None, stride=2),

            # nn.AvgPool2d(kernel_size=2, stride=2),
            #Block(kernel_size=3, in_size=32, expand_size=128, out_size=40,
                  #nolinear=hswish(), semodule=None, stride=2),
            #Block(kernel_size=3, in_size=40, expand_size=160, out_size=40,
                  #nolinear=hswish(), semodule=None, stride=1),


            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(),
        )

        self.classifier = nn.Conv2d(in_channels=24, out_channels=numclasses, kernel_size=1,
                      stride=1, padding=0, bias=True)

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
        x = self.layer1(x)       
        x = self.bneck(x)
        features = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return features,x
class MNV3_small_6_3_6_10_c3(nn.Module):#20562    也可
    def __init__(self, numclasses):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #self.P_block = Product_Block(3,8,nn.ReLU())
        self.bneck = nn.Sequential(
            #Block(kernel_size=3, in_size=8, expand_size=8, out_size=8,
                  #nolinear=hswish(), semodule=None, stride=1),
            Block(kernel_size=3, in_size=8, expand_size=32, out_size=16,
                  nolinear=hswish(), semodule=None, stride=2),

            # nn.AvgPool2d(kernel_size=2, stride=2),
            Block(kernel_size=3, in_size=16, expand_size=64, out_size=16,
                  nolinear=hswish(), semodule=None, stride=1),
            Block(kernel_size=3, in_size=16, expand_size=64, out_size=24,
                  nolinear=hswish(), semodule=None, stride=2),

            # nn.AvgPool2d(kernel_size=2, stride=2),
            Block(kernel_size=3, in_size=24, expand_size=96, out_size=24,
                  nolinear=hswish(), semodule=None, stride=1),
            Block(kernel_size=3, in_size=24, expand_size=96, out_size=32,
                  nolinear=hswish(), semodule=None, stride=2),

            # nn.AvgPool2d(kernel_size=2, stride=2),
            #Block(kernel_size=3, in_size=32, expand_size=128, out_size=40,
                  #nolinear=hswish(), semodule=None, stride=2),
            #Block(kernel_size=3, in_size=40, expand_size=160, out_size=40,
                  #nolinear=hswish(), semodule=None, stride=1),


            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(),
        )

        self.classifier = nn.Conv2d(in_channels=32, out_channels=numclasses, kernel_size=1,
                      stride=1, padding=0, bias=True)

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
        x = self.layer1(x)  
        #x = self.P_block(x)     
        features = self.bneck(x)
        #features = x.view(x.size(0), -1)
        x = self.classifier(features)
        x = x.view(x.size(0), -1)
        return features,x



def MNV30811_SMALL():
    #return MNV3_large2(2)
    #return MNV3_small_cmopare_b(2)
    #return MNV3_small_6_3_6_10(2)
    #return MNV3_small_6_3_6_10_c2(2)
    return MNV3_small_6_3_6_10_c3(2)
    #return MNV3_bufen_new5(2)
    #return MNV3_small_6_3_6_12(2)
