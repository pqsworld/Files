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

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class HardNet_flip(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet_flip, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 8, kernel_size=1, padding=0)
        )
        self.features.apply(weights_init)
        self.hsigmoid = hsigmoid()
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input, flip_flag=False):
        x_features = self.features(self.input_norm(input))

        if flip_flag:
            bs, bins, _, _ = x_features.shape
            x_features_flip = x_features.permute(0, 2, 3, 1).contiguous().view(bs, -1, bins)
            x_features_flip = torch.flip(x_features_flip, dims=[1])
            x = x_features_flip.view(x_features_flip.size(0), -1)
        else:
            x = x_features.permute(0, 2, 3, 1).contiguous().view(x_features.size(0), -1)
        return L2Norm()(x)

    # def forward(self, input):
    #     x_features = self.features(self.input_norm(input))
    #     x = x_features.view(x_features.size(0), -1)
    #     return self.hsigmoid(x)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # nn.init.orthogonal(m.weight.data, gain=0.6)
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return

