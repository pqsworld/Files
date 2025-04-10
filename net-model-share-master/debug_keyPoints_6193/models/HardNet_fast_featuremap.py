import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # if stride == 1 and in_size != out_size:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_size, out_size, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )
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
        # out = out + self.shortcut(x) if self.stride == 1 else out
        out = out + self.shortcut(x) if (self.stride == 1 and out.shape[1] == x.shape[1]) else out
        return out

class HardNet_fast_featuremap(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_featuremap, self).__init__()
        self.train_flag = train_flag

        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            nn.Conv2d(64, 8, kernel_size=1, padding=0)
        )
                
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
        x_features = self.features(self.input_norm(input))
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
    if isinstance(m, nn.Conv2d):
        # nn.init.orthogonal(m.weight.data, gain=0.6)
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return

if __name__ == '__main__':
    net = HardNet_fast(train_flag=True)
    # from torchstat import stat
    net.eval()
    # x = torch.rand(16,1,16,16)
    # net(x)
    # from torchsummary import summary
    # summary(x, input_size=(1, 136, 32))
    print('# net parameters:', sum(param.numel() for param in net.parameters()))
    print('# net parameters memory:', sum(param.numel()*4/1024 for param in net.parameters()), 'kb')
