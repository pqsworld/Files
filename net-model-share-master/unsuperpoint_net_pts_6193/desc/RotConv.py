import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
import torch
from copy import deepcopy
from torch.autograd import Variable
from torch._six import container_abcs
from itertools import repeat
import torchvision
# from ..utils.utils import inv_warp_patch_batch, inv_warp_patch, inv_warp_patch_batch_rec

def warp_points(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    # expand points len to (x, y, 1)
    homographies = homographies.double()
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    # homographies = homographies.unsqueeze(0) if len(homographies.shape) == 2 else homographies
    batch_size = homographies.shape[0]
    points = torch.cat((points.double(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)
    homographies = homographies.view(batch_size*3,3)
    # warped_points = homographies*points
    # points = points.double()
    homographies = homographies.to(points.device)
    warped_points = homographies@points.transpose(0,1)
    # warped_points = np.tensordot(homographies, points.transpose(), axes=([2], [0]))
    # normalize the points
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0,:,:] if no_batches else warped_points

def get_rotation_matrix(theta):
    batchsize = len(theta)
    theta_r = theta*3.14159265/180
    rotate_maxtrix = torch.zeros((batchsize, 3,3))
    rotate_maxtrix[:,0,0] = torch.cos(theta_r)
    rotate_maxtrix[:,0,1] = torch.sin(theta_r)
    rotate_maxtrix[:,0,2] = 0
    rotate_maxtrix[:,1,0] = -torch.sin(theta_r)
    rotate_maxtrix[:,1,1] = torch.cos(theta_r)
    rotate_maxtrix[:,1,2] = 0
    rotate_maxtrix[:,2,0] = 0
    rotate_maxtrix[:,2,1] = 0
    rotate_maxtrix[:,2,2] = 1

    return rotate_maxtrix

# from utils.utils import inv_warp_image_batch
def inv_warp_patch_batch(img, points_batch, theta_batch, patch_size=16, sample_size = 16, mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param points:
        batch of points
        tensor [batch_size, N, 2]
    :param theta:
        batch of orientation [-90 +90]
        tensor [batch_size, N]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, patch_size, patch_size]
    '''
    batch_size, points_num = points_batch.size(0),points_batch.size(1)
    points = points_batch.view(-1,2)
    theta = theta_batch.view(-1)

    mat_homo_inv = get_rotation_matrix(theta)
    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)
    device = img.device
    _, channel, H, W = img.shape
    Batch = len(points)
  
    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, patch_size), torch.linspace(-1, 1, patch_size)), dim=2)  # 产生两个网格
    if sample_size == 1:
        coor_cells = torch.zeros_like(coor_cells)
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv.double(), device)
    src_pixel_coords = src_pixel_coords.view([Batch, patch_size, patch_size, 2])
    src_pixel_coords = src_pixel_coords.float() * (sample_size // 2) + points.unsqueeze(1).unsqueeze(1).repeat(1,patch_size,patch_size,1)


    src_pixel_coords_ofs = torch.floor(src_pixel_coords)
    src_pixel_coords_ofs_Q11 = src_pixel_coords_ofs.view([Batch, -1, 2])

    batch_image_coords_correct = torch.linspace(0, (batch_size-1)*H*W, batch_size).long().to(device)

    src_pixel_coords_ofs_Q11 = (src_pixel_coords_ofs_Q11[:,:,0] + src_pixel_coords_ofs_Q11[:,:,1]*W).long()
    src_pixel_coords_ofs_Q21 = src_pixel_coords_ofs_Q11 + 1
    src_pixel_coords_ofs_Q12 = src_pixel_coords_ofs_Q11 + W
    src_pixel_coords_ofs_Q22 = src_pixel_coords_ofs_Q11 + W + 1

    warp_weight = (src_pixel_coords - src_pixel_coords_ofs).view([Batch, -1, 2])

    alpha = warp_weight[:,:,0]
    beta = warp_weight[:,:,1]
    src_Q11 = img.take(src_pixel_coords_ofs_Q11.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q21 = img.take(src_pixel_coords_ofs_Q21.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q12 = img.take(src_pixel_coords_ofs_Q12.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q22 = img.take(src_pixel_coords_ofs_Q22.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)

    warped_img = src_Q11*(1 - alpha)*(1 - beta) + src_Q21*alpha*(1 - beta) + \
        src_Q12*(1 - alpha)*beta + src_Q22*alpha*beta
    warped_img = warped_img.view([Batch, patch_size,patch_size])
    return warped_img


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class RotConv(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        angles: int = 4
    ):
        kernel_size_ = _pair(kernel_size+2)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(nn.Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)
        self.angles = angles

    def forward(self, input):
        outputs = []
        angles = torch.linspace(0,180,(self.angles + 1))[:-1]
        for angle in angles:
            _weight = torchvision.transforms.functional.rotate(self.weight,-angle.item())[:,:,1:-1,1:-1]
            x_features = self._conv_forward(input, _weight, self.bias)
            outputs.append((x_features).unsqueeze(-1))

        # Get the maximum direction (Orientation Pooling)
        strength, max_ind = torch.max(torch.cat(outputs, -1), -1)

        return strength


class RotConv_C(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        kernel_size_ = _pair(kernel_size+2)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(nn.Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, input, theta):
        assert self.out_channels == self.groups

        kernel_size = self.kernel_size[0]-2
        stride = self.stride[0]

        batch_size, fdim, imgH, imgW = input.size()
        H_o, W_o = imgH // stride, imgW // stride

        unfold_Input = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=stride)
        input = unfold_Input(input)
        input = input.view(batch_size, fdim, 9, H_o, W_o).permute(0,1,3,4,2)

        #对输入进行变换
        unfold_theta = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=stride)
        theta = unfold_theta(theta)
        theta = theta.view(batch_size, 1, 9, H_o, W_o)[:,:,4] * 360

        _weight = self.weight.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        _weight = _weight.view(-1, self.kernel_size[0], self.kernel_size[1]).unsqueeze(1)
        orientation_theta_batch = theta.squeeze().unsqueeze(1).repeat(1,self.out_channels,1,1).view(-1,H_o*W_o)
        keypoints_batch_correct = torch.zeros(batch_size*fdim, H_o*W_o, 2).to(input.device)
        keypoints_batch_correct[:,:] = 2
        patch = inv_warp_patch_batch(_weight, keypoints_batch_correct, orientation_theta_batch, patch_size=3, sample_size=3)
        outs = patch.unsqueeze(1)
        _weight_unfold = outs.view(batch_size, fdim, H_o, W_o, -1)

        #最终计算
        output = torch.sum(_weight_unfold * input,dim=-1)
        
        return output



class RotConv_sym(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(nn.Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, input):
        assert self.kernel_size == (3,3)
        # '''SYM'''
        # _weight = torch.ones_like(self.weight)
        # _weight[:,:,0,0] = self.weight[:,:,0,0]
        # _weight[:,:,2,0] = self.weight[:,:,0,0]
        # _weight[:,:,0,2] = self.weight[:,:,0,0]
        # _weight[:,:,2,2] = self.weight[:,:,0,0]
        # _weight[:,:,0,1] = self.weight[:,:,0,1]
        # _weight[:,:,1,0] = self.weight[:,:,0,1]
        # _weight[:,:,1,2] = self.weight[:,:,0,1]
        # _weight[:,:,2,1] = self.weight[:,:,0,1]
        # _weight[:,:,1,1] = self.weight[:,:,1,1]

        # x_features = self._conv_forward(input, _weight, self.bias)

        # '''CIRCLE'''
        # _weight = torch.ones_like(self.weight)
        # _a = self.weight[:,:,0,0]
        # _b = self.weight[:,:,0,1]
        # _sq_2 = 1.414213562

        # _weight[:,:,0,0] = 0.5*_b
        # _weight[:,:,2,0] = 0.5*_b
        # _weight[:,:,0,2] = 0.5*_b
        # _weight[:,:,2,2] = 0.5*_b
        # _weight[:,:,0,1] = _sq_2*_b
        # _weight[:,:,1,0] = _sq_2*_b
        # _weight[:,:,1,2] = _sq_2*_b
        # _weight[:,:,2,1] = _sq_2*_b
        # _weight[:,:,1,1] = _a + (6 - 4*_sq_2)*_b


        '''CIRCLE Integral'''
        # _weight = torch.ones_like(self.weight)
        # _a = self.weight[:,:,0,0]
        # _b = self.weight[:,:,0,1]
        # _c = self.weight[:,:,0,2]
        # _d = self.weight[:,:,1,0]
        # _e = self.weight[:,:,1,1]
        # _f = self.weight[:,:,1,2]
        # _g = self.weight[:,:,2,0]
        # _h = self.weight[:,:,2,1]
        # _i = self.weight[:,:,2,2]

        # rb = 1/8
        # rc = 2/8
        # rd = 3/8
        # re = 4/8
        # rf = 5/8
        # rg = 6/8
        # rh = 7/8
        # ri = 8/8
    
        # _pi = 3.14159265357

        # _t = 0.5*(rb*rb*_b + rc*rc*_c + rd*rd*_d + re*re*_e + rf*rf*_f + rg*rg*_g + rh*rh*_h + ri*ri*_i)

        # _m = (2*rb-rb*rb)*_b + (2*rc-rc*rc)*_c + (2*rd-rd*rd)*_d + (2*re-re*re)*_e + \
        # (2*rf-rf*rf)*_f + (2*rg-rg*rg)*_g + (2*rh-rh*rh)*_h + (2*ri-ri*ri)*_i

        # _n = _a + (-8*rb+2*rb*rb)*_b + (-8*rc+2*rc*rc)*_c + (-8*rd+2*rd*rd)*_d + (-8*re+2*re*re)*_e + \
        # (-8*rf+2*rf*rf)*_f + (-8*rg+2*rg*rg)*_g + (-8*rh+2*rh*rh)*_h + (-8*ri+2*ri*ri)*_i + \
        # (_b + _c + _d + _e + _f + _g + _h + _i)*2*_pi
        
        # _weight[:,:,0,0] = _t
        # _weight[:,:,2,0] = _t
        # _weight[:,:,0,2] = _t
        # _weight[:,:,2,2] = _t
        # _weight[:,:,0,1] = _m
        # _weight[:,:,1,0] = _m
        # _weight[:,:,1,2] = _m
        # _weight[:,:,2,1] = _m
        # _weight[:,:,1,1] = _n

        
        #单圆弧
        _weight = torch.ones_like(self.weight)
        _a = self.weight[:,:,0,0]
        _b = self.weight[:,:,0,1]
    
        _pi = 3.14159265357

        _t = 0.5*_b

        _m = _b

        _n = _a + (2*_pi-6)*_b 
        
        _weight[:,:,0,0] = _t
        _weight[:,:,2,0] = _t
        _weight[:,:,0,2] = _t
        _weight[:,:,2,2] = _t
        _weight[:,:,0,1] = _m
        _weight[:,:,1,0] = _m
        _weight[:,:,1,2] = _m
        _weight[:,:,2,1] = _m
        _weight[:,:,1,1] = _n

        x_features = self._conv_forward(input, _weight, self.bias)


        return x_features


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
        # if stride == 1 and in_size != out_size:
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
        out = out + self.shortcut(x) if self.shortcut_flag else out
        return out


class Block_Rot(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_Rot, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = RotConv(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False, angles=4)
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

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.nolinear2(self.bn2(out))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.shortcut_flag else out
        return out

class Block_Rot_C(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_Rot_C, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = RotConv_C(expand_size, expand_size, kernel_size=kernel_size,
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

    def forward(self, x, theta):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.conv2(out, theta)
        out = self.nolinear2(self.bn2(out))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.shortcut_flag else out

        if self.stride == 2:
            ds2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
            theta = ds2(theta)
        return out, theta


class Block_Rot_sym_grop(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_Rot_sym_grop, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = RotConv_sym(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size // 4, bias=False)
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

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.nolinear2(self.bn2(out))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.shortcut_flag else out

        return out


class Block_Rot_sym(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_Rot_sym, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = RotConv_sym(expand_size, expand_size, kernel_size=kernel_size,
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

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.nolinear2(self.bn2(out))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.shortcut_flag else out

        return out

class Block_Rot_bilinear(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_Rot_bilinear, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.ds2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.conv2 = RotConv_sym(expand_size, expand_size, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2, groups=expand_size, bias=False)
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

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        if self.stride == 2:
            out = self.ds2(out)

        out = self.conv2(out)
        out = self.nolinear2(self.bn2(out))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.shortcut_flag else out

        return out

class Block_Rot_circle2(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_Rot_circle2, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = RotConv_sym(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)
        self.conv2_double = RotConv_sym(expand_size, expand_size, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.bn2_double = nn.BatchNorm2d(expand_size)
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

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.bn2_double(self.conv2_double(out))
        out = self.nolinear2(out)
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.shortcut_flag else out

        return out

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

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            RotConv_sym(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck_sym(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=4):
        super(GhostBottleneck_sym, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = RotConv_sym(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SeModule(mid_chs, reduction=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                RotConv_sym(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # nn.init.orthogonal(m.weight.data, gain=0.6)
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return

class ThetaSequential(nn.Sequential):
    def __init__(self, *args):
        super(ThetaSequential, self).__init__(*args)
    def forward(self, input, theta):
        for module in self:
            input, theta = module(input, theta)
        return input, theta

class HardNet_dense_rot(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_dense_rot, self).__init__()
        self.train_flag = train_flag

        #Orientation Head
        self.theta = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            nn.ZeroPad2d((3,3,3,3)),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
            hsigmoid()
        )

        self.pre = ThetaSequential(
            Block_Rot_C(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_Rot_C(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2)
        )

        self.pad1 = nn.ZeroPad2d((3,3,3,3))

        self.features = ThetaSequential(         
            Block_Rot_C(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_C(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_C(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_C(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_C(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_C(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_C(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1)
        )

        self.post = nn.Conv2d(64, 8, kernel_size=1, padding=0)


        if self.train_flag:
            self.pre.apply(weights_init)
            self.features.apply(weights_init)
            self.post.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        us4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        ori = self.theta(input)
        ori_us4 = us4(ori[:,:,3:-3,3:-3])
        x_features, _ = self.pre(self.input_norm(input), ori_us4)
        x_features = self.pad1(x_features)
        x_features, _ = self.features(x_features, ori)
        x_features = self.post(x_features)
        x = x_features
        return x



class HardNet_dense_sym(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_dense_sym, self).__init__()
        self.train_flag = train_flag
        self.gauss = torch.nn.Parameter(torch.ones(3), requires_grad=False)

        self.features = nn.Sequential(
            # nn.ZeroPad2d((6,6,4,4)),
            Block_Rot_sym(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_Rot_sym(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            nn.ZeroPad2d((1,1,2,2)),               
            Block_Rot_sym(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            nn.Conv2d(64, 8, kernel_size=1, padding=0)
        )

        # self.features = nn.Sequential(
        #     nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True), 
        #     nn.Conv2d(1, 16, kernel_size=3, padding=1, bias = False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2,stride=2),
        #     nn.ZeroPad2d((1,1,2,2)),               
        #     Block_Rot_sym(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #     Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #     Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #     Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #     Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #     Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #     Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
        #              nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
        #     nn.Conv2d(64, 8, kernel_size=1, padding=0)
        # )


        if self.train_flag:
            self.features.apply(weights_init)
          
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        assert input.size(2) == 144
        assert input.size(3) == 52

        # input = torchvision.transforms.functional.resize(input, [137,41])

        x_features = self.features(input)
        x = x_features
        return x

    def forward_patch(self, patch):
        gauss_kernel_0 = torch.hstack([self.gauss[0], self.gauss[1], self.gauss[1], self.gauss[0]])
        gauss_kernel_1 = torch.hstack([self.gauss[1], self.gauss[2], self.gauss[2], self.gauss[1]])
        gauss_kernel_2 = torch.hstack([self.gauss[1], self.gauss[2], self.gauss[2], self.gauss[1]])
        gauss_kernel_3 = torch.hstack([self.gauss[0], self.gauss[1], self.gauss[1], self.gauss[0]])
        gauss_kernel = torch.vstack([gauss_kernel_0, gauss_kernel_1, gauss_kernel_2, gauss_kernel_3]).to(patch.device)

        return patch*gauss_kernel


class HardNet_dense_sym_grop(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_dense_sym_grop, self).__init__()
        self.train_flag = train_flag
        self.gauss = torch.nn.Parameter(torch.ones(3), requires_grad=True)

        self.features = nn.Sequential(
            # nn.ZeroPad2d((6,6,4,4)),
            Block_Rot_sym_grop(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_Rot_sym_grop(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            nn.ZeroPad2d((1,1,2,2)),               
            Block_Rot_sym_grop(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym_grop(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym_grop(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym_grop(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
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
    
    def forward(self, input):
        assert input.size(2) == 144
        assert input.size(3) == 52

        # input = torchvision.transforms.functional.resize(input, [137,41])

        x_features = self.features(input)
        x = x_features
        return x

    def forward_patch(self, patch):
        gauss_kernel_0 = torch.hstack([self.gauss[0], self.gauss[1], self.gauss[1], self.gauss[0]])
        gauss_kernel_1 = torch.hstack([self.gauss[1], self.gauss[2], self.gauss[2], self.gauss[1]])
        gauss_kernel_2 = torch.hstack([self.gauss[1], self.gauss[2], self.gauss[2], self.gauss[1]])
        gauss_kernel_3 = torch.hstack([self.gauss[0], self.gauss[1], self.gauss[1], self.gauss[0]])
        gauss_kernel = torch.vstack([gauss_kernel_0, gauss_kernel_1, gauss_kernel_2, gauss_kernel_3]).to(patch.device)

        return patch*gauss_kernel


class HardNet_dense_sym_unfold(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_dense_sym_unfold, self).__init__()
        self.train_flag = train_flag
        self.gauss = torch.nn.Parameter(torch.ones(3), requires_grad=True)

        self.features = nn.Sequential(
            Block_Rot_sym(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.ZeroPad2d((1,1,2,2)),               
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
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
    
    def forward(self, input):
        assert input.size(2) == 144
        assert input.size(3) == 52

        # input = torchvision.transforms.functional.resize(input, [137,41])

        unfold = nn.Unfold(kernel_size=4, dilation=1, padding=0, stride=4)
        input_unfold = unfold(input).view(input.size(0),16,36,13)

        x_features = self.features(input_unfold)
        x = x_features
        return x

    def forward_patch(self, patch):
        gauss_kernel_0 = torch.hstack([self.gauss[0], self.gauss[1], self.gauss[1], self.gauss[0]])
        gauss_kernel_1 = torch.hstack([self.gauss[1], self.gauss[2], self.gauss[2], self.gauss[1]])
        gauss_kernel_2 = torch.hstack([self.gauss[1], self.gauss[2], self.gauss[2], self.gauss[1]])
        gauss_kernel_3 = torch.hstack([self.gauss[0], self.gauss[1], self.gauss[1], self.gauss[0]])
        gauss_kernel = torch.vstack([gauss_kernel_0, gauss_kernel_1, gauss_kernel_2, gauss_kernel_3]).to(patch.device)

        return patch*gauss_kernel



class HardNet_dense_sym_ghost(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_dense_sym_ghost, self).__init__()
        self.train_flag = train_flag
        self.gauss = torch.nn.Parameter(torch.ones(3), requires_grad=True)

        self.features = nn.Sequential(  
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            GhostBottleneck_sym(in_chs=8, mid_chs=16, out_chs=16, stride=2),
            GhostBottleneck_sym(in_chs=16, mid_chs=32, out_chs=32, stride=2),
            nn.ZeroPad2d((3,3,3,3)),               
            GhostBottleneck_sym(in_chs=32, mid_chs=32, out_chs=32, stride=1),
            GhostBottleneck_sym(in_chs=32, mid_chs=32, out_chs=32, stride=1),
            GhostBottleneck_sym(in_chs=32, mid_chs=32, out_chs=32, stride=1),
            GhostBottleneck_sym(in_chs=32, mid_chs=32, out_chs=32, stride=1),
            GhostBottleneck_sym(in_chs=32, mid_chs=32, out_chs=32, stride=1),
            GhostBottleneck_sym(in_chs=32, mid_chs=64, out_chs=64, stride=1),
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
    
    def forward(self, input):
        assert input.size(2) == 136
        assert input.size(3) == 40

        input = torchvision.transforms.functional.resize(input, [137,41])

        x_features = self.features(input)
        x = x_features
        return x

    def forward_patch(self, patch):
        gauss_kernel_0 = torch.hstack([self.gauss[0], self.gauss[1], self.gauss[1], self.gauss[0]])
        gauss_kernel_1 = torch.hstack([self.gauss[1], self.gauss[2], self.gauss[2], self.gauss[1]])
        gauss_kernel_2 = torch.hstack([self.gauss[1], self.gauss[2], self.gauss[2], self.gauss[1]])
        gauss_kernel_3 = torch.hstack([self.gauss[0], self.gauss[1], self.gauss[1], self.gauss[0]])
        gauss_kernel = torch.vstack([gauss_kernel_0, gauss_kernel_1, gauss_kernel_2, gauss_kernel_3]).to(patch.device)

        return patch*gauss_kernel



class HardNet_dense_ds3_us2(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_dense_ds3_us2, self).__init__()
        self.train_flag = train_flag

        self.features = nn.Sequential(  
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True), 
            Block_Rot_sym(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_Rot_sym(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            nn.ZeroPad2d((3,3,3,3)),               
            Block_Rot_sym(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            nn.Conv2d(64, 8, kernel_size=1, padding=0)
        )

        self.post = nn.Sequential(  
            # nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True), 
            Block_Rot_sym(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            nn.Conv2d(16, 8, kernel_size=1, padding=0)
        )



        if self.train_flag:
            self.features.apply(weights_init)
            self.post.apply(weights_init)
          
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):

        x_features = self.features(self.input_norm(input))
        x = x_features
        return x
    
    def _forward(self, input):
        x_features = self.post(input)
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class HardNet_dense_sym_test(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_dense_sym_test, self).__init__()
        self.train_flag = train_flag
        self.gauss = torch.nn.Parameter(torch.ones(3), requires_grad=True)

        self.features = nn.Sequential(  
            # nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True), 
            # nn.ZeroPad2d((8,8,8,8)),
            Block_Rot_sym(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            # Block_Rot_sym(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            # nn.ZeroPad2d((3,3,3,3)),               
            Block_Rot_sym(kernel_size=3, in_size=int(8), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            Block_Rot_sym(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
                     nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # nn.Conv2d(64, 8, kernel_size=1, padding=0)
        )


        if self.train_flag:
            self.features.apply(weights_init)
          
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # assert input.size(2) == 136
        # assert input.size(3) == 40

        # input = torchvision.transforms.functional.resize(input, [137,41])

        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        x = x_features
        return x

    def forward_patch(self, patch):
        gauss_kernel_0 = torch.hstack([self.gauss[0], self.gauss[1], self.gauss[1], self.gauss[0]])
        gauss_kernel_1 = torch.hstack([self.gauss[1], self.gauss[2], self.gauss[2], self.gauss[1]])
        gauss_kernel_2 = torch.hstack([self.gauss[1], self.gauss[2], self.gauss[2], self.gauss[1]])
        gauss_kernel_3 = torch.hstack([self.gauss[0], self.gauss[1], self.gauss[1], self.gauss[0]])
        gauss_kernel = torch.vstack([gauss_kernel_0, gauss_kernel_1, gauss_kernel_2, gauss_kernel_3]).to(patch.device)

        return patch*gauss_kernel

def test_model(model: torch.nn.Module):
    # evaluate the `model` on 8 rotated versions of the input image `x`
    model.eval()
    
    wrmup = (torch.randn(1, 1, 29, 29)).to(device)
    wrmup = torchvision.transforms.functional.resize(wrmup, [87,87])
    wrmup_90 = wrmup.permute(0,1,3,2).flip(dims=[-2]) #逆时针旋转

    
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(8):
            x_transformed = torchvision.transforms.functional.rotate(wrmup,r*45)
            x_transformed = torchvision.transforms.functional.resize(x_transformed, [29,29])
            x_transformed = x_transformed.to(device)

            x_transformed_90 = torchvision.transforms.functional.rotate(wrmup_90,r*45)
            x_transformed_90 = torchvision.transforms.functional.resize(x_transformed_90, [29,29])
            x_transformed_90 = x_transformed_90.to(device)

            y = model(x_transformed)
            y_90 = model(x_transformed_90)
    
            y = y[0, ...].detach().unsqueeze(0)
            # y = torchvision.transforms.functional.rotate(y,-r*45)
            y = y[:,:,7,7]
            y = y.view(-1)[:8].cpu().numpy()

            # y_90 = y_90[0, ...].detach().unsqueeze(0)
            # # y_90 = torchvision.transforms.functional.rotate(y_90,-r*45)
            # y_90 = y_90.permute(0,1,3,2).flip(dims=[-1]) #逆时针旋转
            # # y_90 = y_90[:,:,0,0]
            # y_90 = y_90.view(-1)[:8].cpu().numpy()
            
            angle = r * 45
            print("{:5d} : {}".format(angle, y))
            # print("{:5d}_90 : {}".format(angle, y_90))
    print('##########################################################################################')


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HardNet_dense_sym_test(train_flag=True).to(device)

    test_model(model)

# if __name__ == '__main__':
#     '''print the number of parameters '''
#     net = HardNet_dense_ds3_us2()
#     total = 0
#     for name, parameters in net.named_parameters():
#         total += parameters.nelement()
#         print(name, ":", parameters.size())
#     print("Number of parameter: %.4fM" % (total / 1e6))

#     input_tensor = torch.randn(32, 1, 136, 40)
#     import torchvision

#     input_tensor_90 = torchvision.transforms.functional.rotate(input_tensor,90)
#     y = net(input_tensor)
#     y_90 = net(input_tensor_90)
#     y_90 = torchvision.transforms.functional.rotate(y_90.view(-1,8,4,4),180).view(y_90.size(0),-1)

