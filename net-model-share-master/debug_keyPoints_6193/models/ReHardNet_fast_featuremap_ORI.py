import torch
import math
import torch.nn as nn
import e2cnn.nn as enn
from e2cnn import gspaces
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Pad 
from torchvision.transforms import Resize 
from torchvision.transforms import ToTensor

from .enn_layers import FIELD_TYPE, convnxn, ennAdaptiveAvgPool

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x

class Rehswish(nn.Module):
    def __init__(self, in_type, inplace=True):
        super(Rehswish, self).__init__()
        # self.in_type = FIELD_TYPE['regular'](gspace, in_size)
        self.in_type = in_type
        self.activate = enn.ReLU(in_type, inplace=inplace)
    def forward(self, x):
        relu6 = torch.clamp(self.activate(x).tensor, 0, 6)
        out = x.tensor * (relu6 + 3) / 6
        # out = x * enn.Relu(x + 3, inplace=True) / 6
        return enn.GeometricTensor(out, self.in_type)

class Rehsigmoid(nn.Module):
    def __init__(self, in_type, inplace=True):
        super(Rehsigmoid, self).__init__()
        self.in_type = in_type
        self.activate = enn.ReLU(in_type, inplace=inplace)
    def forward(self, x):
        relu6 = torch.clamp(self.activate(x).tensor, 0, 6)
        out = (relu6 + 3) / 6
        # out = F.relu6(x + 3, inplace=True) / 6
        return enn.GeometricTensor(out, self.in_type)

# def Rehsigmoid(in_type):
#     activate = enn.ReLU(in_type, inplace=True)
#     relu6 = torch.clamp(activate(x), 0, 6)
#     out = (relu6 + 3) / 6

class ReSeModule(nn.Module):
    def __init__(self, in_size, reduction=4, gspace=None):
        super(ReSeModule, self).__init__()
        self.in_field_type = FIELD_TYPE['regular'](gspace, in_size)
        self.reductiong_field_type = FIELD_TYPE['regular'](gspace, in_size // reduction)

        self.se = enn.SequentialModule(
            ennAdaptiveAvgPool(gspace, in_size, 1),
            enn.R2Conv(self.in_field_type, self.reductiong_field_type, kernel_size=1,
                      stride=1, padding=0, bias=False),
            enn.InnerBatchNorm(self.reductiong_field_type),
            enn.ReLU(self.reductiong_field_type, inplace=True),
            enn.R2Conv(self.reductiong_field_type, self.in_field_type, kernel_size=1,
                      stride=1, padding=0, bias=False),
            enn.InnerBatchNorm(self.in_field_type),
            enn.ReLU(self.in_field_type, inplace=True)
            # self.activate()
            # Rehsigmoid(self.in_field_type)
        )
        # self.activate = Rehsigmoid(self.in_field_type)
        # self.activate = enn.ReLU(self.in_field_type, inplace=True)

    def forward(self, x):
        '''GeometricTensor 不支持向量积'''
        return x * self.se(x)

class ReBlock(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, ReNolinear, Resemodule, stride, gspace, in_type='regular', out_type='regular'):
        super(ReBlock, self).__init__()
        self.stride = stride

        self.in_field_type =  FIELD_TYPE[in_type](gspace, in_size)
        self.expand_field_type =  FIELD_TYPE[out_type](gspace, expand_size)
        self.out_field_type =  FIELD_TYPE[out_type](gspace, out_size)

        self.conv1 = enn.R2Conv(self.in_field_type, self.expand_field_type,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = enn.InnerBatchNorm(self.expand_field_type)
        self.nolinear1 = ReNolinear(self.expand_field_type, inplace=True)
        self.conv2 = enn.R2Conv(self.expand_field_type, self.expand_field_type, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size // gspace.fibergroup.order(), bias=False)
        self.bn2 = enn.InnerBatchNorm(self.expand_field_type)
        self.nolinear2 = ReNolinear(self.expand_field_type, inplace=True)
        self.conv3 = enn.R2Conv(self.expand_field_type, self.out_field_type,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = enn.InnerBatchNorm(self.out_field_type)
        self.se = ReSeModule(out_size, gspace=gspace) if Resemodule is not None else None
        self.shortcut = enn.SequentialModule()
        # if stride == 1 and in_size != out_size:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_size, out_size, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )
        if stride == 1 and in_size == out_size:
            self.shortcut = enn.SequentialModule(
                enn.R2Conv(self.in_field_type, self.out_field_type, kernel_size=1,
                          stride=1, padding=0, bias=False),
                enn.InnerBatchNorm(self.out_field_type),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        # out = out + self.shortcut(x) if self.stride == 1 else out
        out = out + self.shortcut(x) if (self.stride == 1 and out.shape[1] == x.shape[1]) else out
        return out

class ReHardNet_fast_featuremap_ORI(nn.Module):
    """Rotation-equivariant HardNet model definition
    """
    def __init__(self, train_flag=False, orientation=8):
        super(ReHardNet_fast_featuremap_ORI, self).__init__()
        self.train_flag = train_flag
        self.orientation = orientation
        # self.fixparams = fixparams
        self.gspace = gspaces.Rot2dOnR2(orientation)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 1)
        self.out1_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * (64 // orientation))
        self.gpool_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * (64 // orientation))

        self.features = nn.Sequential(
            ReBlock(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     ReNolinear=enn.ReLU, Resemodule=None, stride=2, gspace=self.gspace, in_type='trivial'),
            ReBlock(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     ReNolinear=enn.ReLU, Resemodule=None, stride=2, gspace=self.gspace),
            ReBlock(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     ReNolinear=enn.ReLU, Resemodule=None, stride=1, gspace=self.gspace),
            ReBlock(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     ReNolinear=enn.ReLU, Resemodule=None, stride=1, gspace=self.gspace),
            ReBlock(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     ReNolinear=enn.ReLU, Resemodule=None, stride=1, gspace=self.gspace),
            ReBlock(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     ReNolinear=enn.ReLU, Resemodule=None, stride=1, gspace=self.gspace),
            # enn.R2Conv(self.out1_type, self.out2_type, kernel_size=1, padding=0)
        )
        # self.block1 = ReBlock(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
        #              ReNolinear=Rehswish, Resemodule=ReSeModule, stride=2, gspace=self.gspace, in_type='trivial')
        # self.block2 = ReBlock(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
        #              ReNolinear=Rehswish, Resemodule=ReSeModule, stride=2, gspace=self.gspace)
        # self.block3 = ReBlock(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
        #              ReNolinear=Rehswish, Resemodule=ReSeModule, stride=1, gspace=self.gspace)
        # self.block4 = ReBlock(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
        #              ReNolinear=Rehswish, Resemodule=ReSeModule, stride=1, gspace=self.gspace)
        # self.block5 = ReBlock(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
        #              ReNolinear=Rehswish, Resemodule=ReSeModule, stride=1, gspace=self.gspace)
        # self.block6 = ReBlock(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
        #              ReNolinear=Rehswish, Resemodule=ReSeModule, stride=1, gspace=self.gspace)
        # self.Conv = enn.R2Conv(self.out1_type, self.out2_type, kernel_size=1, padding=0)
        self.gpool = enn.GroupPooling(self.gpool_type)
        if self.train_flag:
            self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    # def forward(self, input):
    #     x_features = self.features(self.input_norm(input))
    #     # print('x_features.shape: ', x_features.shape)
    #     x = x_features.view(x_features.size(0), -1)
       
    #     # x_features = self.input_norm(input)
    #     # # print('x_features.shape: ', x_features.shape)
    #     # x = x_features.view(x_features.size(0), -1)
    #     return L2Norm()(x)
    def forward(self, input, flip_flag=False):
        input_norm = self.input_norm(input)
        input_geometrictensor = enn.GeometricTensor(input_norm, self.input_type)
        x_features = self.features(input_geometrictensor)
        x_features = self.gpool(x_features)
        x_features = x_features.tensor
        bs, bins, h, w = x_features.shape

        if flip_flag:
            x_features_flip = x_features.permute(0, 2, 3, 1).contiguous().view(bs, -1, bins)
            x_features_flip = torch.flip(x_features_flip, dims=[1])
            x = x_features_flip.view(x_features_flip.size(0), -1)
        else:
            x = x_features.permute(0, 2, 3, 1).contiguous().view(x_features.size(0), -1)
        # print(x.shape)
        # return L2Norm()(x)
        return x.view(bs, h, w, bins).permute(0, 3, 1, 2)

def weights_init(m):
    if isinstance(m, enn.R2Conv):
        # nn.init.orthogonal(m.weight.data, gain=0.6)
        enn.init.deltaorthonormal_init(m.weight.data, m.basisexpansion)
        # try:
        #     enn.init.constant(m.bias.data, 0.01)
        # except:
        #     pass
    return

def test_model(model: torch.nn.Module, x: Image, device='cpu'):
    # evaluate the `model` on 8 rotated versions of the input image `x`     
    # model.eval()
    # wrmup = model(x.to(device))
    # del wrmup
    # x = torch.tensor(np.array(x)).to(device).reshape(1, 1, 29, 29)
    totensor = ToTensor()
    pad = Pad((0, 0, 1, 1), fill=0)
    x = pad(x)
    print()
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(8):
            x_transformed = totensor(x.rotate(r*45., Image.BILINEAR)).reshape(1, 1, 30, 30)
            x_transformed = x_transformed.to(device)
            y = model(x_transformed)
            y = y.to('cpu').numpy().squeeze()
            angle = r * 45
            print("{:5d} : {}".format(angle, y))
    print('##########################################################################################')     
    print()      
if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    GPU_ids = 6     
    device = torch.device('cuda:' + str(GPU_ids))
    
    net = ReHardNet_fast_featuremap(train_flag=False).to(device)
    # from torchstat import stat
    net.eval()
    # x = torch.rand(16,1,16,16)
    # net(x)
    # from torchsummary import summary
    # summary(x, input_size=(1, 136, 32))
    print('# net parameters:', sum(param.numel() for param in net.parameters()))
    print('# net parameters memory:', sum(param.numel()*4/1024 for param in net.parameters()), 'kb')

    import numpy as np
    image = Image.fromarray(np.array(torch.randn(29, 29)))
    test_model(net, image, device=device)
