import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

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

def nshear2H(rotate_matrix,nshear):

    shear_matrix = torch.eye(3,device=rotate_matrix.device).unsqueeze(0).repeat(rotate_matrix.size(0),1,1)
    shear_matrix[:,0,1] = nshear
    rotate_matrix = shear_matrix@rotate_matrix
    return rotate_matrix

# from utils.utils import inv_warp_image_batch
def inv_warp_patch_batch(img, points_batch, theta_batch, patch_size=16, sample_size = 16, mode='bilinear',nshear=0):
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
    mat_homo_inv = nshear2H(mat_homo_inv, nshear)
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
    src_pixel_coords = src_pixel_coords.float() * (sample_size / 2) + points.unsqueeze(1).unsqueeze(1).repeat(1,patch_size,patch_size,1)


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



class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

# class SeModule(nn.Module):
#     def __init__(self, in_size, reduction=4):
#         super(SeModule, self).__init__()
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_size, in_size // reduction, kernel_size=1,
#                       stride=1, padding=0, bias=True),
#             # nn.BatchNorm2d(in_size // reduction),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_size // reduction, in_size, kernel_size=1,
#                       stride=1, padding=0, bias=True),
#             # nn.BatchNorm2d(in_size),
#             hsigmoid()
#         )

#     def forward(self, x):
#         return x * self.se(x)
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
                      stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
class Block_short(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block_short, self).__init__()
        self.stride = stride
        self.shortcut_flag = (self.stride == 1 and in_size == out_size)

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=True)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=True)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=True)
        self.se = semodule
        self.shortcut = nn.Sequential()
        # if stride == 1 and in_size != out_size:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_size, out_size, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )

    def forward(self, x):
        out = self.nolinear1(self.conv1(x))
        out = self.nolinear2(self.conv2(out))
        out = self.conv3(out)
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

class HardNet_tiny_short(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_tiny_short, self).__init__()
        self.train_flag = train_flag


        self.features = nn.Sequential(
            Block_short(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block_short(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block_short(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block_short(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block_short(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block_short(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            nn.Conv2d(32, 16, kernel_size=1, padding=0)
        )

        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):

        # from PIL import Image
        # img_save = Image.fromarray(input[50,0].squeeze().cpu().numpy()*255).convert("L")
        # img_save.save("/hdd/file-input/yey/work/desc/test/out/rotate/RE/hh0.bmp")
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # patch_inpaint = self.decoder(x_features)
        # x_features = self.nonlinear_projection(x_features)
        # for idx in range(8):
        #     img_save = Image.fromarray(x_features[50,idx].squeeze().float().cpu().numpy()*255).convert("L")
        #     img_save.save("/hdd/file-input/yey/work/desc/test/out/rotate/RE/desc_{:d}.bmp".format(idx))
        # exit()

        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
    
        factor_dim = x.size(1) // 128
        x = x.view(-1,factor_dim*8,4,4).permute(0,2,3,1).reshape(x.size(0),-1)
        # 转二进制
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        # x = x[:,:128]
        return L2Norm()(x)

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

class HardNet_fast_s(nn.Module):
    """HardNet model definition
    """
    def __init__(self, train_flag=False):
        super(HardNet_fast_s, self).__init__()
        self.train_flag = train_flag
       
        # version 1, 2
        # 16 128
        self.features = nn.Sequential(
            Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
                     nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(16), out_size=int(16),
                     nolinear=hswish(), semodule=SeModule(int(16)), stride=1),
            Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # nn.Conv2d(32, 16, kernel_size=1, padding=0)
            nn.Conv2d(32, 16, kernel_size=1, padding=0, bias=False)
        )
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        # x_features = self.features(self.input_norm(input))
        x_features = self.features(input)
        # print('x_features.shape: ', x_features.shape)
        x = x_features.view(x_features.size(0), -1)
       
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class Test():
    def __init__(self, DO_CUDA=True):
        # models_path = "/hdd/file-input/zhangsn/light_simi/descnet/93061_short.pth.tar"
        models_path = "/hdd/file-input/zhangsn/light_simi/descnet/superPointNet_90000_desc.pth.tar"
        if DO_CUDA:
            # self.descriptor_net = HardNet_tiny_short().cuda()
            self.descriptor_net = HardNet_fast_s().cuda()
            self.descriptor_net.load_state_dict(torch.load(models_path,map_location='cuda')['model_state_dict'])
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('Extracting on GPU')
        else:
            print('Extracting on CPU')
            # self.descriptor_net = HardNet_tiny_short().cpu()
            self.descriptor_net = HardNet_fast_s().cpu()
            self.descriptor_net.load_state_dict(torch.load(models_path,map_location='cpu')['model_state_dict'])
            self.device="cpu"
        # self.w_expand=8
        # self.h_expand=3
        self.w_expand=0
        self.h_expand=0
        
        
    def get_sift_orientation_batch(self, img, keypoints, patch_size=19, bin_size=10):
        '''
        img:tensor
        '''
        patch_size=19
        # print("img.device: ", img.device)
        w_gauss = torch.tensor([0.0,0.0,0.0,1.0,1.0,2.0,3.0,5.0,6.0,6.0,6.0,5.0,3.0,2.0,1.0,1.0,0.0,0.0,0.0,
        0.0,0.0,1.0,1.0,3.0,5.0,8.0,10.0,12.0,13.0,12.0,10.0,8.0,5.0,3.0,1.0,1.0,0.0,0.0,
        0.0,1.0,2.0,3.0,6.0,11.0,16.0,22.0,26.0,28.0,26.0,22.0,16.0,11.0,6.0,3.0,2.0,1.0,0.0,
        1.0,1.0,3.0,7.0,13.0,22.0,34.0,45.0,55.0,58.0,55.0,45.0,34.0,22.0,13.0,7.0,3.0,1.0,1.0,
        1.0,3.0,6.0,13.0,24.0,42.0,65.0,88.0,106.0,112.0,106.0,88.0,65.0,42.0,24.0,13.0,6.0,3.0,1.0,
        2.0,5.0,11.0,22.0,42.0,73.0,112.0,151.0,182.0,193.0,182.0,151.0,112.0,73.0,42.0,22.0,11.0,5.0,2.0,
        3.0,8.0,16.0,34.0,65.0,112.0,171.0,232.0,279.0,296.0,279.0,232.0,171.0,112.0,65.0,34.0,16.0,8.0,3.0,
        5.0,10.0,22.0,45.0,88.0,151.0,232.0,314.0,378.0,401.0,378.0,314.0,232.0,151.0,88.0,45.0,22.0,10.0,5.0,
        6.0,12.0,26.0,55.0,106.0,182.0,279.0,378.0,456.0,483.0,456.0,378.0,279.0,182.0,106.0,55.0,26.0,12.0,6.0,
        6.0,13.0,28.0,58.0,112.0,193.0,296.0,401.0,483.0,512.0,483.0,401.0,296.0,193.0,112.0,58.0,28.0,13.0,6.0,
        6.0,12.0,26.0,55.0,106.0,182.0,279.0,378.0,456.0,483.0,456.0,378.0,279.0,182.0,106.0,55.0,26.0,12.0,6.0,
        5.0,10.0,22.0,45.0,88.0,151.0,232.0,314.0,378.0,401.0,378.0,314.0,232.0,151.0,88.0,45.0,22.0,10.0,5.0,
        3.0,8.0,16.0,34.0,65.0,112.0,171.0,232.0,279.0,296.0,279.0,232.0,171.0,112.0,65.0,34.0,16.0,8.0,3.0,
        2.0,5.0,11.0,22.0,42.0,73.0,112.0,151.0,182.0,193.0,182.0,151.0,112.0,73.0,42.0,22.0,11.0,5.0,2.0,
        1.0,3.0,6.0,13.0,24.0,42.0,65.0,88.0,106.0,112.0,106.0,88.0,65.0,42.0,24.0,13.0,6.0,3.0,1.0,
        1.0,1.0,3.0,7.0,13.0,22.0,34.0,45.0,55.0,58.0,55.0,45.0,34.0,22.0,13.0,7.0,3.0,1.0,1.0,
        0.0,1.0,2.0,3.0,6.0,11.0,16.0,22.0,26.0,28.0,26.0,22.0,16.0,11.0,6.0,3.0,2.0,1.0,0.0,
        0.0,0.0,1.0,1.0,3.0,5.0,8.0,10.0,12.0,13.0,12.0,10.0,8.0,5.0,3.0,1.0,1.0,0.0,0.0,
        0.0,0.0,0.0,1.0,1.0,2.0,3.0,5.0,6.0,6.0,6.0,5.0,3.0,2.0,1.0,1.0,0.0,0.0,0.0],device=img.device)

        ori_max = 180
        bins = ori_max // bin_size
        batch, c, h, w = img.shape
        offset = patch_size // 2
        device = img.device

        Gx=torch.zeros((batch, c, h+offset*2, w+offset*2), dtype=img.dtype, device=img.device)
        Gy=torch.zeros((batch, c, h+offset*2, w+offset*2), dtype=img.dtype, device=img.device)
        # Gm=torch.zeros((batch, c, h+patch_size, w+patch_size), dtype=img.dtype, device=img.device)
        # Gm[:,:,patch_size:h+patch_size,patch_size:w+patch_size] = 1

        Gx0=torch.zeros_like(img)
        Gx2=torch.zeros_like(img)
        Gy0=torch.zeros_like(img)
        Gy2=torch.zeros_like(img)

        Gx0[:,:,:,1:-1] = img[:,:,:,:-2]*255
        Gx2[:,:,:,1:-1] = img[:,:,:,2:]*255
        Gx[:,:,offset:-offset,offset:-offset] = (Gx0 - Gx2)

        Gy0[:,:,1:-1,:] = img[:,:,:-2,:]*255
        Gy2[:,:,1:-1,:] = img[:,:,2:,:]*255
        Gy[:,:,offset:-offset,offset:-offset] = (Gy2 - Gy0)

        coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, patch_size), torch.linspace(-1, 1, patch_size)), dim=2)  # 产生两个网格
        coor_cells = coor_cells.transpose(0, 1)
        coor_cells = coor_cells.to(self.device)
        coor_cells = coor_cells.contiguous()
        
        keypoints_num = keypoints.size(1)
        keypoints_correct = torch.round(keypoints.clone())
        keypoints_correct += offset
        
        src_pixel_coords = coor_cells.unsqueeze(0).repeat(batch, keypoints_num,1,1,1)
        src_pixel_coords = src_pixel_coords.float() * (patch_size // 2) + keypoints_correct.unsqueeze(2).unsqueeze(2).repeat(1,1,patch_size,patch_size,1)
        
        src_pixel_coords = src_pixel_coords.view([batch, keypoints_num, -1, 2])
        batch_image_coords_correct = torch.linspace(0, (batch-1)*(h+patch_size-1)*(w+patch_size-1), batch).long().to(device)
        src_pixel_coords_index = (src_pixel_coords[:,:,:,0] + src_pixel_coords[:,:,:,1]*(w+patch_size-1)).long()
        src_pixel_coords_index  = src_pixel_coords_index + batch_image_coords_correct[:,None,None]

        eps = 1e-12
        
        #生成幅值图和角度图
        Grad_Amp = ((torch.sqrt(Gx**2 + Gy**2)) * 256)

        #边界反射
        Grad_Amp[:,:,9] = Grad_Amp[:,:,10]
        Grad_Amp[:,:,-10] = Grad_Amp[:,:,-11]
        Grad_Amp[:,:,:,9] = Grad_Amp[:,:,:,10]
        Grad_Amp[:,:,:,-10] = Grad_Amp[:,:,:,-11]

        degree_value = Gy / (Gx + eps)
        Grad_ori = torch.atan(degree_value)
        Grad_ori = Grad_ori*180 / math.pi #180/(3.1415926)
        a_mask = (Gx >= 0)
        b_mask = (Gy >= 0)
        apbp_mask = a_mask * b_mask
        apbn_mask = a_mask * (~b_mask)
        anbp_mask = (~a_mask) * b_mask
        anbn_mask = (~a_mask) * (~b_mask)
        Grad_ori[apbp_mask] = Grad_ori[apbp_mask]
        Grad_ori[apbn_mask] = Grad_ori[apbn_mask] + 360
        Grad_ori[anbp_mask] = Grad_ori[anbp_mask] + 180
        Grad_ori[anbn_mask] = Grad_ori[anbn_mask] + 180


        #边界反射
        Grad_ori[:,:,9] = Grad_ori[:,:,10]
        Grad_ori[:,:,-10] = Grad_ori[:,:,-11]
        Grad_ori[:,:,:,9] = Grad_ori[:,:,:,10]
        Grad_ori[:,:,:,-10] = Grad_ori[:,:,:,-11]
        
        Grad_ori = Grad_ori % ori_max

        angle = Grad_ori.take(src_pixel_coords_index)

        #高斯加权
        w_gauss /= 512
        Amp = Grad_Amp.take(src_pixel_coords_index)
        Amp = Amp*w_gauss[None,None,:]
        angle_d = ((angle // bin_size)).long() % bins
        angle_d_onehot = F.one_hot(angle_d,num_classes=bins)
        hist = torch.sum(Amp.unsqueeze(-1)*angle_d_onehot,dim=-2) #[0,pi)

        #平滑
        h_t=torch.zeros((batch, keypoints_num, hist.size(-1)+4), dtype=hist.dtype, device=hist.device)
        h_t[:,:,2:-2] = hist
        h_t[:,:,-2:] = hist[:,:,:2]
        h_t[:,:,:2] = hist[:,:,-2:]

        h_p2=h_t[:,:,4:]
        h_n2=h_t[:,:,:-4]
        h_p1=h_t[:,:,3:-1]
        h_n1=h_t[:,:,1:-3]

        Hist = (h_p2 + h_n2 + 4*(h_p1 + h_n1) + 6*hist) / 16
        Hist = Hist.long()
        
        #获取主方向i
        H_p_i = torch.max(Hist,dim=-1).indices
        H_t=torch.zeros((batch, keypoints_num, Hist.size(-1)+2), dtype=Hist.dtype, device=Hist.device)
        H_t[:,:,1:-1] = Hist
        H_t[:,:,-1:] = Hist[:,:,:1]
        H_t[:,:,:1] = Hist[:,:,-1:]

        H_p1=H_t[:,:,2:]
        H_n1=H_t[:,:,:-2]

        H_i_offset = (H_n1 - H_p1) / (2*(H_n1 + H_p1 - 2*Hist) + eps)
        H_p_i_onehot = F.one_hot(H_p_i,num_classes=bins)
        H_p_offset = torch.sum(H_i_offset*H_p_i_onehot,dim=-1)
        H_p = (H_p_i + H_p_offset + 0.5) * bin_size
        H_p = H_p % 180 - 90


        return H_p
                                                # warp_A, pts_nms_A_batch, patch_size=16, sample_size=22, correct=True, sift=0, theta=180, trans_theta=0

    def forward_patches_correct_batch_expand(self, img_batch, keypoints_batch, patch_size=16, sample_size=16,correct=True,sift=False,theta=0,trans_theta=0,train_flag=False):
        # 根据关键点获得patch，并输入网络
        # 返回元组 B kpts_num desc_dim
        # theta == 0 时，默认是A图输入，180时为B图输入
        assert sample_size <= 32, "padding more!"
        # # 输入非扩边图时
        # pad_img = torch.nn.ZeroPad2d((8,8,3,3))
        # img_batch = pad_img(img_batch)
        # assert img_batch.size(2) == 128
        # assert img_batch.size(3) == 52
        results = None
        Batchsize, kpt_num  = keypoints_batch.size(0), keypoints_batch.size(1)
        img_H, img_W = img_batch.size(2), img_batch.size(3)
        
        img_batch = img_batch.to(self.device)

        patch_padding = 50
        add_offset = patch_padding//2
        add_offset_x = patch_padding//2

        img_batch_padding = torch.zeros((Batchsize, 1, img_H + patch_padding, img_W + patch_padding),device=self.device)
        img_batch_padding[:, :,add_offset:(img_H+add_offset),add_offset_x:(img_W+add_offset_x)] = img_batch
        #生成mask
        mask_batch_padding = torch.zeros((Batchsize, 1, img_H + patch_padding, img_W + patch_padding),device=self.device)
        mask_batch_padding[:, :,add_offset:(img_H+add_offset),add_offset_x:(img_W+add_offset_x)] = 1
        


        keypoints_batch_correct = keypoints_batch.clone()
        # keypoints_batch_correct[:,:,0] += (add_offset_x + self.w_expand)
        # keypoints_batch_correct[:,:,1] += (add_offset + self.h_expand)
        keypoints_batch_correct[:,:,0] += (add_offset_x + self.w_expand)
        keypoints_batch_correct[:,:,1] += (add_offset + self.h_expand)

        
        # keypoints_batch_correct[:,:,0] += (add_offset_x + 6)
        # keypoints_batch_correct[:,:,1] += (add_offset + 3)

        return_theta = None
        if correct:
            keypoints_batch_ori = keypoints_batch.clone()
            keypoints_batch_ori[:,:,0] += self.w_expand
            keypoints_batch_ori[:,:,1] += self.h_expand
            orientation_theta_batch = self.get_sift_orientation_batch(img_batch, keypoints_batch_ori)

            # #theta == 0,即A图时进行theta校准
            # if theta == 0:
            #     correct_theta, AT_theta = self.trans_ori_correct(orientation_theta_batch, trans_theta)
            #     orientation_theta_batch += correct_theta
            #     return_theta = AT_theta
            # else:
            #     return_theta = orientation_theta_batch

            # if sift:
            #     # orientation_theta_batch = self.get_orientation_batch(img_batch, keypoints_batch_ori, 16) + correct_theta + theta
            #     orientation_theta_batch += theta

        else:
            orientation_theta_batch = torch.zeros(keypoints_batch_correct.size(0),keypoints_batch_correct.size(1))
        
        # #nshear
        # nshear_enable = random.choice([0,0,0,0,1])
        # nshear = random.uniform(-0.174,0.174)
        # nshear = nshear_enable*(theta == 0)*nshear
        
        nshear = 0
        patch = inv_warp_patch_batch(img_batch_padding, keypoints_batch_correct, orientation_theta_batch, patch_size=patch_size, sample_size=sample_size, nshear=nshear)

        # if theta == 0:
        #     '''patch Mask'''
        #     patch_mask = inv_warp_patch_batch(mask_batch_padding, keypoints_batch_correct, orientation_theta_batch, patch_size=patch_size, sample_size=sample_size)
        #     #逆转mask面积：即mask大的乘以小mask，反之亦然
        #     patch_mask_idx = torch.sort(torch.mean(patch_mask,dim=[1,2])).indices
        #     patch_mask_t = patch_mask.clone()
        #     patch_mask_t[patch_mask_idx] = patch_mask[torch.flip(patch_mask_idx,dims=[0])]
        #     patch = patch * patch_mask_t

        results = patch.unsqueeze(1)

        patch_mask = inv_warp_patch_batch(mask_batch_padding, keypoints_batch_correct, orientation_theta_batch, patch_size=patch_size, sample_size=sample_size)

        # #双通道
        # patch = inv_warp_patch_batch(img_batch_padding, keypoints_batch_correct, orientation_theta_batch, patch_size=patch_size, sample_size=sample_size)
        # patch_32 = inv_warp_patch_batch(img_batch_padding, keypoints_batch_correct, orientation_theta_batch, patch_size=patch_size, sample_size=32)
        
        # results = torch.cat([patch.unsqueeze(1),patch_32.unsqueeze(1)],dim=1)
        # patch_4 = inv_warp_patch_batch(img_batch_padding, keypoints_batch_correct, orientation_theta_batch, patch_size=32, sample_size=sample_size)
        # patch_4 = patch_4.unsqueeze(1)
        # patch0 = patch_4[:,:,:16,:16]
        # patch1 = patch_4[:,:,:16,16:]
        # patch2 = patch_4[:,:,16:,:16]
        # patch3 = patch_4[:,:,16:,16:]
        # results = torch.cat([patch0,patch1,patch2,patch3],dim=1)
        
        results_batch = Variable(results)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)

        outs = outs.view(Batchsize, kpt_num, -1) #[B*N dim] -> [B dim N]

        patch = patch.view(Batchsize, kpt_num, patch_size, patch_size)
        patch_mask = patch_mask.view(Batchsize, kpt_num, patch_size, patch_size)
        return outs, patch, patch_mask, return_theta,orientation_theta_batch

# descAs_undetached, patchAs, patchAs_mask, _ = self.forward_patches_correct_batch_expand(imgA, pnts_A, patch_size=16, sample_size=22, correct=True, sift=0, theta=180, trans_theta=0)