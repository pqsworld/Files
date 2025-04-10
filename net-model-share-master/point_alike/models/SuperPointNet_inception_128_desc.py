"""latest version of SuperpointNet. Use it!

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
import numpy as np

class Inception(nn.Module):
    def __init__(self,in_c,c1,c2,c3,c4):#1,
        super(Inception,self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_c,c1,kernel_size=1),
            nn.ReLU()
        )  
        self.p2 = nn.Sequential(
            nn.Conv2d(in_c,c2[0],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(in_c, c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5,padding=2),
            nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_c,c4,kernel_size=1),
            nn.ReLU()
        )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        y = torch.cat((p1,p2,p3,p4),dim=1)
        return y

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

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):#kernel_size:3
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)#depthwise
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
class SuperPointNet_inception_128_desc(nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, input_nc=1, output_nc=65, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=2,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(SuperPointNet_inception_128_desc, self).__init__()
        self.tran = True#上采样转置卷积

        model = [ 
            nn.Conv2d(input_nc, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16),
            nn.ReLU(True)]
        
        model +=  [Inception(16,4,(4,12),(4,8),8),#32  #多尺度尝试
                    nn.MaxPool2d(kernel_size=3,stride=2,padding=0), #1_downsample
                    ]
                    
        model += [nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),#2_downsample
                    norm_layer(32),
                    nn.ReLU(True)]
                    
        model += [nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),#3_downsample
                    norm_layer(32),
                    nn.ReLU(True)]	
                    
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),#3*3 kernel stride = 1的感受野
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

        # Descriptor Head.
        model_2 = [
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(True)]
        model_2 += [
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(True)]
        #model_2 += [nn.Conv2d(64, 128, kernel_size=1, padding=0)]#128维数据，并不进行下采样
        #------------------#
        if self.tran:
            model_2 += [
                nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False),
                norm_layer(128),
                nn.ReLU(True)]
            model_2 += [
                nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1,output_padding=1,bias=False),
                norm_layer(128),
                nn.ReLU(True)]
            model_2 += [
                nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1,output_padding=1,bias=False)]
        else:
            model_2 += [nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)]
            model_2 += [
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                norm_layer(128),
                nn.ReLU(True)]
            model_2 += [nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)]
            model_2 += [
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                norm_layer(128),
                nn.ReLU(True)]
            model_2 += [nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)]
            model_2 += [
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)]
            
        self.model_descriptor = nn.Sequential(*model_2)

    def forward(self, input):
        x = self.model(input)
        # Detector Head.
        semi = self.model_detector(x)
        # Descriptor Head.
        desc = self.model_descriptor(x)
        # print(desc.shape)
        #dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        #desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        output = {'semi': semi, 'desc': desc}

        self.output = output
        return output

    def process_output(self, sp_processer):
        """
        input:
          N: number of points
        return: -- type: tensorFloat
          pts: tensor [batch, N, 2] (no grad)  (x, y)
          pts_offset: tensor [batch, N, 2] (grad) (x, y)
          pts_desc: tensor [batch, N, 256] (grad)
        """
        from utils.utils import flattenDetection
        # from models.model_utils import pred_soft_argmax, sample_desc_from_points
        output = self.output
        semi = output['semi']
        desc = output['desc']
        # flatten
        heatmap = flattenDetection(semi) # [batch_size, 1, H, W] decoder，将深度图转为heatmap
        # nms
        heatmap_nms_batch = sp_processer.heatmap_to_nms(heatmap, tensor=True)
        # extract offsets
        outs = sp_processer.pred_soft_argmax(heatmap_nms_batch, heatmap)
        residual = outs['pred']
        # extract points
        outs = sp_processer.batch_extract_features(desc, heatmap_nms_batch, residual)

        # output.update({'heatmap': heatmap, 'heatmap_nms': heatmap_nms, 'descriptors': descriptors})
        output.update(outs)
        self.output = output
        return output

if __name__ == '__main__':
    net = SuperPointNet_inception_128_desc()
    net.eval()
    x = torch.rand(1,1,136,32)
    net(x)
        



