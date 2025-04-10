import logging
import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
import torch.nn.functional as F
from torchvision.transforms import ToTensor, InterpolationMode
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import math
from Model_component import draw_keypoints_pair

from models.ALNet import ALNet, ALNet_New, ALNet_Angle, ALNet_Angle_Short, ALNet_Angle_New, ALNet_Angle_Deep_Short, ALNet_Angle_Deform, ALNet_Angle_Deep
from models.modules import DKD
from models.hardnet_model import HardNet_fast_twice_half3_short, HardNet_fast_half_standard, HardNet_fast_twice_half3_MO_MOE_short, HardNet_fast_half_AMOE, HardNet_fast_twice_half3_MO_MOE, HardNet_fast_half_MOE, HardNet_fast_twice_big_MO, HardNet_fast_twice_mid_vanila, HardNet_fast_half_MO, HardNet_small, HardNet, HardNet_smaller, HardNet_smaller_patch, HardNet_Sift, HardNet_fast, HardNet_fast_half, HardNet_fast_half_short, HardNet_fast_twice, HardNet_fast_twice_half, HardNet_fast_twice_half2, HardNet_fast_twice_half3, HardNet_fast_twice_half3_norm2, HardNet_fast_twice_half3_MO, HardNet_fast_twice_half3_MO_short, HardNet_fast_twice_half3_MOA, HardNet_fast_big, HardNet_fast_big_ap, HardNet_fast_big_mp, HardNet_fast_twice_big_short, HardNet_fast_twice_half_cr, HardNet_fast_twice_half_cr2, HardNet_fast_twice_half_norm2, HardNet_fast_twice_big, HardNet_fast_twice_big_norm2, HardNet_fast_twice_big2, HardNet_fast_twice_short, HardNet_fast_Ecnn_third, HardNet_fast_Pconv, HardNet_Sift_Deform, HardNet_fast_deform, HardNet_fast_deform_last, HardNet_fast_Ecnn, HardNet_fast_Ecnn_twice, HardNet_fast_twice_half3_vanila, HardNet_fast_twice_half3_MO_new, L2Norm
from models.hardnet_model_quant import HardNet_fast_quant
from models.PTQ.ptq_config import Config
import time
from models.Superglue_small import Superglue_small
from models.Superglue_small_alldesc import Superglue_small_alldesc
from models.Superglue_small_relposition import Superglue_small_relposition
from models.Lightglue_small_relposition import Lightglue_small_relposition
from models.Lightglue_small_relposition_downdim import Lightglue_small_relposition_downdim
from models.Lightglue_big_relposition_downdim import Lightglue_big_relposition_downdim
from models.Lightglue_small_relposition_downdim_softlabel import Lightglue_small_relposition_downdim_softlabel
from models.Lightglue_small_relposition_downdim_alldesc import Lightglue_small_relposition_downdim_alldesc
from models.Lightglue_small_relposition_downdim_innersim import Lightglue_small_relposition_downdim_innersim
from models.Lightglue_big_relposition_downdim_innersim import Lightglue_big_relposition_downdim_innersim
from models.Lightglue_small_relposition_downdim_innersim_recurrent import Lightglue_small_relposition_downdim_innersim_recurrent

configs = {
    'alike-t': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-t.pth')},
    'alike-s': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 96, 'dim': 96, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-s.pth')},
    'alike-n': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dim': 128, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-n.pth')},
    'alike-l': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 128, 'dim': 128, 'single_head': False, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-l.pth')},
}

class FeatureLoss(nn.Module):

    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 # name,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0, # 0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,
                 ):
        super(FeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))

        self.reset_parameters()


    def forward(self,
                preds_S,
                preds_T,
                # gt_bboxes,
                # img_metas
                ):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'

        if self.align is not None:
            preds_S = self.align(preds_S)
        
        N,C,H,W = preds_S.shape

        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)

        Mask_fg = torch.ones_like(S_attention_t)
        Mask_bg = torch.zeros_like(S_attention_t)

        # wmin,wmax,hmin,hmax = [],[],[],[]
        # for i in range(N):
        #     new_boxxes = torch.ones_like(gt_bboxes[i])
        #     new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
        #     new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

        #     wmin.append(torch.floor(new_boxxes[:, 0]).int())
        #     wmax.append(torch.ceil(new_boxxes[:, 2]).int())
        #     hmin.append(torch.floor(new_boxxes[:, 1]).int())
        #     hmax.append(torch.ceil(new_boxxes[:, 3]).int())

        #     area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))

        #     for j in range(len(gt_bboxes[i])):
        #         Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
        #                 torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])

        #     Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
        #     if torch.sum(Mask_bg[i]):
        #         Mask_bg[i] /= torch.sum(Mask_bg[i])

        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg, 
                           C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        rela_loss = self.get_rela_loss(preds_S, preds_T)


        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
               + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
            
        return loss


    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W= preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
        C_attention = C * F.softmax(channel_map/temp, dim=1)

        return S_attention, C_attention


    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')
        
        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t= torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t)/len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t)/len(Mask_bg)

        return fg_loss, bg_loss


    def get_mask_loss(self, C_s, C_t, S_s, S_t):

        mask_loss = torch.sum(torch.abs((C_s-C_t)))/len(C_s) + torch.sum(torch.abs((S_s-S_t)))/len(S_s)

        return mask_loss
     
    
    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context


    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t)/len(out_s)
        
        return rela_loss


    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            for m_i in m:
                self.weight_init(m_i)
            self.weight_init(m[-1], 1)
            # constant_init(m[-1], val=0)
        else:
            self.weight_init(m, 1)
            # constant_init(m, val=0)

    def weight_init(self, m, mode=0):
        # mode
        # 0: 'kaiming_normal'
        # 1: 'constant'
        if isinstance(m, nn.Conv2d):
            if mode == 0:
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.constant_(m.weight.data, 0.)
            try:
                nn.init.constant_(m.bias.data, 0.)
            except:
                pass
        return 
    
    def reset_parameters(self):
        self.weight_init(self.conv_mask_s)
        self.weight_init(self.conv_mask_t)
        # kaiming_init(self.conv_mask_s, mode='fan_in')
        # kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)

# MGD
class MGDLoss(nn.Module):

    """PyTorch version of `Masked Generative Distillation`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.65
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 # name,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.45, # 0.65,
                 ):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        # self.name = name
    
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

        self.reset_parameters()

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)
    
        loss = self.get_dis_loss(preds_S, preds_T)*self.alpha_mgd
            
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat>1-self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss

    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            for m_i in m:
                self.weight_init(m_i)
            self.weight_init(m[-1], 1)
            # constant_init(m[-1], val=0)
        else:
            self.weight_init(m, 1)
            # constant_init(m, val=0)

    def weight_init(self, m, mode=0):
        # mode
        # 0: 'kaiming_normal'
        # 1: 'constant'
        if isinstance(m, nn.Conv2d):
            if mode == 0:
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.constant_(m.weight.data, 0.)
            try:
                nn.init.constant_(m.bias.data, 0.)
            except:
                pass
        return 
    
    def reset_parameters(self):
        self.last_zero_init(self.generation)
        # self.last_zero_init(self.channel_add_conv_t)


def draw_orientation(input_img, input_pts, color=(255, 0, 0), radius=3, s=3,):
    img = np.array(input_img['img']) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    # print(input_pts['pts'], input_pts['angles'])
    for c, angle in zip(np.stack(input_pts['pts']), np.stack(input_pts['angles'])):
        if c.size == 1:
            break
        c_new = np.zeros_like(c)
        # print(np.cos(angle*0.017453292))
        c_new[0] = c[0] + 5*np.cos(angle*0.017453292)
        c_new[1] = c[1] - 5*np.sin(angle*0.017453292)
        # print(c_new, c, np.sin(angle*0.017453292))
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)    # 图像坐标系，先列后行
        # cv2.circle(img, tuple((s * c_new[:2]).astype(int)), radius, color, thickness=-1)
        cv2.line(img, tuple((s * c[:2]).astype(int)),tuple((s * c_new[:2]).astype(int)), (0, 0, 255), 1, shift=0)
    
    return img

def inv_warp_image_batch_cv2(img, mat_homo_inv, device='cpu', mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    '''
    # compute inverse warped points
    if len(img.shape) == 2:
        img = img.view(1, 1, img.shape[0], img.shape[1])
    if len(img.shape) == 3:
        img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)

    Batch, channel, H, W = img.shape

    warped_img = cv2.warpPerspective(img.squeeze().numpy(), mat_homo_inv.squeeze().numpy(), (W, H))
    warped_img = torch.tensor(warped_img, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

    # warped_img = cv2.warpAffine(img.squeeze().numpy(), mat_homo_inv[0, :2, :].squeeze().numpy(), (W, H))
    # warped_img = torch.tensor(warped_img, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    return warped_img

def inv_warp_image(img, mat_homo_inv, device='cpu', mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [H, W]
    '''
    warped_img = inv_warp_image_batch_cv2(img, mat_homo_inv, device, mode)
    return warped_img.squeeze()

def batch_inv_warp_image(img_batch, mat_homo_inv, device='cpu', mode='billinear'):
    img_batch_shape = img_batch.shape
    for idx in range(img_batch_shape[0]):
        if idx == 0:
            warped_img = inv_warp_image(img_batch[idx,:,:,:].squeeze().squeeze().detach().cpu(), mat_homo_inv[idx].squeeze().cpu(), device, mode).unsqueeze(0).unsqueeze(0)
        else:
            mid_warped_img = inv_warp_image(img_batch[idx,:,:,:].squeeze().squeeze().detach().cpu(), mat_homo_inv[idx].squeeze().cpu(), device, mode).unsqueeze(0).unsqueeze(0)
            warped_img = torch.cat([warped_img, mid_warped_img], dim=0)
    return warped_img

def warp_points_batch(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    # expand points len to (x, y, 1)
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
def inv_warp_patch_batch(img, points, theta, patch_size=16, sample_size = 16):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    '''
    mat_homo_inv = get_rotation_matrix(theta)

    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)
    device = img.device
    B, _, H, W = img.shape
    Batch = len(points)
    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, patch_size), torch.linspace(-1, 1, patch_size)), dim=2)  # 产生两个网格
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous().double()
    # print(mat_homo_inv.shape, coor_cells.shape)
    src_pixel_coords = warp_points_batch(coor_cells.view([-1, 2]), mat_homo_inv.double(), device) 
    # print(src_pixel_coords.shape)
    src_pixel_coords = src_pixel_coords.view([Batch, patch_size, patch_size, 2])
    src_pixel_coords = src_pixel_coords.float() * (sample_size / 2) + points.unsqueeze(1).unsqueeze(1).repeat(1,patch_size,patch_size,1)

    src_pixel_coords_ofs = torch.floor(src_pixel_coords)
    src_pixel_coords_ofs_Q11 = src_pixel_coords_ofs.view([Batch, -1, 2])        # (BxM)x(patch_size,patch_size)x2
    batch_offset = H * W * torch.linspace(0, B-1, steps=B, device=device).repeat(Batch//B, 1).transpose(0, 1).unsqueeze(0).repeat(1, 1, patch_size*patch_size).view(Batch, -1).long()
    # print(batch_offset)
    src_pixel_coords_ofs_Q11 = (src_pixel_coords_ofs_Q11[:,:,0] + src_pixel_coords_ofs_Q11[:,:,1]*W).long()
    src_pixel_coords_ofs_Q11 += batch_offset
    src_pixel_coords_ofs_Q21 = src_pixel_coords_ofs_Q11 + 1
    src_pixel_coords_ofs_Q12 = src_pixel_coords_ofs_Q11 + W
    src_pixel_coords_ofs_Q22 = src_pixel_coords_ofs_Q11 + W + 1

    warp_weight = (src_pixel_coords - src_pixel_coords_ofs).view([Batch, -1, 2])

    alpha = warp_weight[:,:,0]
    beta = warp_weight[:,:,1]
    # print(img.shape, src_pixel_coords_ofs_Q11.shape)
    src_Q11 = img.take(src_pixel_coords_ofs_Q11).view(-1, patch_size*patch_size)
    src_Q21 = img.take(src_pixel_coords_ofs_Q21).view(-1, patch_size*patch_size)
    src_Q12 = img.take(src_pixel_coords_ofs_Q12).view(-1, patch_size*patch_size)
    src_Q22 = img.take(src_pixel_coords_ofs_Q22).view(-1, patch_size*patch_size)
    # print(src_Q11.shape)
    warped_img = src_Q11*(1 - alpha)*(1 - beta) + src_Q21*alpha*(1 - beta) + \
        src_Q12*(1 - alpha)*beta + src_Q22*alpha*beta
    warped_img = warped_img.view([Batch, patch_size,patch_size])
    return warped_img

# from utils.utils import inv_warp_image_batch
def inv_warp_patch_batch_rec(img, points, theta, patch_size=(32,8), sample_factor = 1, mode='bilinear'):
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
    batch_size, _, H, W = img.shape
    # points = points_batch.view(-1,2)
    # theta = theta_batch.view(-1)

    mat_homo_inv = get_rotation_matrix(theta)
    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)
    device = img.device
    _, channel, H, W = img.shape
    Batch = len(points)
    points_num = Batch // batch_size

    patch_y = patch_size[0]*sample_factor / 2
    patch_x = patch_size[1]*sample_factor / 2

    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-patch_x, patch_x, patch_size[1]), torch.linspace(-patch_y, patch_y, patch_size[0])), dim=2)  # 产生两个网格
    coor_cells = coor_cells.transpose(1,0)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points_batch(coor_cells.view([-1, 2]).double(), mat_homo_inv.double(), device)
    src_pixel_coords = src_pixel_coords.view([Batch, patch_size[0], patch_size[1], 2])
    src_pixel_coords = src_pixel_coords.float() + points.unsqueeze(1).unsqueeze(1).repeat(1,patch_size[0],patch_size[1],1)


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
    
    # src_Q11 = img.take(src_pixel_coords_ofs_Q11).view(-1, patch_size*patch_size)
    # src_Q21 = img.take(src_pixel_coords_ofs_Q21).view(-1, patch_size*patch_size)
    # src_Q12 = img.take(src_pixel_coords_ofs_Q12).view(-1, patch_size*patch_size)
    # src_Q22 = img.take(src_pixel_coords_ofs_Q22).view(-1, patch_size*patch_size)
    src_Q11 = img.take(src_pixel_coords_ofs_Q11.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q21 = img.take(src_pixel_coords_ofs_Q21.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q12 = img.take(src_pixel_coords_ofs_Q12.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q22 = img.take(src_pixel_coords_ofs_Q22.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)

    warped_img = src_Q11*(1 - alpha)*(1 - beta) + src_Q21*alpha*(1 - beta) + \
        src_Q12*(1 - alpha)*beta + src_Q22*alpha*beta
    warped_img = warped_img.view([Batch, patch_size[0],patch_size[1]])
    return warped_img

# ALNet_New ALNet_Angle ALNet_Angle_Short ALNet_Angle_New ALNet_Angle_Deep ALNet_Angle_Deep_Short ALNet_Angle_Deform
class ALikeWithHardPntMatch(ALNet_Angle):
    def __init__(self,
                 # ================================== feature encoder
                 c1: int = 8, c2: int = 16, c3: int = 32, c4: int = 64, dim: int = 32,
                 # c1: int = 4, c2: int = 8, c3: int = 16, c4: int = 32, dim: int = 16,
                 # c1: int = 16, c2: int = 32, c3: int = 64, c4: int = 128, dim: int = 64,
                 # c1: int = 8, c2: int = 8, c3: int = 16, c4: int = 32, dim: int = 16,
                 single_head: bool = True, # False,
                 # ================================== detect parameterss
                 radius: int = 2,
                 top_k: int = -1, scores_th: float = 0.2,      # default: 1 / ((2*raduis+1) * (2*raduis+1))
                 n_limit: int = 400,
                 device: str = 'cpu',
                 model_path: str = '',
                 phase: str = 'test'
                 ):
        super().__init__(c1, c2, c3, c4, dim, single_head)
        self.radius = radius            # nms radius
        self.top_k = top_k
        self.n_limit = n_limit
        self.scores_th = scores_th
        self.dkd = DKD(radius=self.radius, top_k=self.top_k,
                       scores_th=self.scores_th, n_limit=self.n_limit)
        

        self.match_net = Lightglue_small_relposition_downdim_innersim() 
        # Superglue_small() Superglue_small_alldesc() Superglue_small_relposition() 
        # Lightglue_small_relposition() Lightglue_small_relposition_downdim() Lightglue_big_relposition_downdim() Lightglue_small_relposition_downdim_softlabel() Lightglue_small_relposition_downdim_alldesc() Lightglue_small_relposition_downdim_innersim()
        # Lightglue_big_relposition_downdim_innersim() Lightglue_small_relposition_downdim_innersim_recurrent()

        # self.descriptor_net = HardNet(train_flag=(phase == 'train'))
        # self.descriptor_net = nn.DataParallel(HardNet_smaller(train_flag=(phase == 'train')), device_ids=[3]) # HardNet_small(train_flag=(phase == 'train'))  # HardNet(train_flag=(phase == 'train'))   
        # self.descriptor_net = nn.DataParallel(HardNet_smaller_patch(train_flag=(phase == 'train')), device_ids=[3])
        # self.descriptor_net = nn.DataParallel(HardNet_Sift(train_flag=(phase == 'train')), device_ids=[3])
        # self.descriptor_net = nn.DataParallel(HardNet_fast(train_flag=(phase == 'train')), device_ids=[5])
        # self.descriptor_net = nn.DataParallel(HardNet_Sift_Deform(train_flag=(phase == 'train')), device_ids=[5])
        # self.descriptor_net = nn.DataParallel(HardNet_fast_deform(train_flag=(phase == 'train')), device_ids=[5])
        # self.descriptor_net = nn.DataParallel(HardNet_fast_deform_last(train_flag=(phase == 'train')), device_ids=[5])
        
        # self.descriptor_net = HardNet_fast_Ecnn(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_Ecnn_twice(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_Ecnn_third(train_flag=(phase == 'train'))

        # self.descriptor_net = HardNet_fast(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half2(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half3(train_flag=(phase == 'train'))
        self.descriptor_net = HardNet_fast_twice_half3_short(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half3_norm2(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half_norm2(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half3_MO(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half3_MO_new(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half3_MO_short(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half3_vanila(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_mid_vanila(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half3_MOA(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half3_MO_MOE(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half3_MO_MOE_short(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_big(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_big_MO(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_big_norm2(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_big_short(train_flag=(phase == 'train'))
        
        # self.descriptor_net = HardNet_fast_big(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_big_ap(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_big_mp(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_half(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_half_standard(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_half_MOE(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_half_AMOE(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_half_MO(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_half_short(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_big2(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half_cr(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half_cr2(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_short(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_Pconv(train_flag=(phase == 'train'))

        # self.descriptor_assist_net = HardNet_fast_twice_big(train_flag=(phase == 'train'))
        # self.descriptor_tea_net = HardNet_fast_twice_big(train_flag=(phase== 'train'))
        # self.descriptor_tea_net = HardNet_fast_twice_big_MO(train_flag=(phase== 'train'))
        # self.descriptor_tea_net = HardNet_fast_big(train_flag=(phase == 'train'))

        # cfg = Config(lis=False)
        # self.descriptor_net = HardNet_fast_quant(cfg=cfg, train_flag=(phase == 'train'))

        self.only1 = True
        self.is_rec = False
        self.is_square_ncal = False
        self.is_expand_rec = False
        self.Guass_flag = False
        self.has_amp = False
        self.desc_patch = 4 # 4 # 16

        self.sample_desc_patch = 22 # 22 # 32 # 4 # 6 # 4 # 32

        self.desc_patch_expand = 4 # 6 # 6 # 48
        self.orient_patch = 17
        self.patch_unfold = nn.Unfold(kernel_size=self.desc_patch, padding=self.desc_patch // 2)
        self.patch_unfold_expand = nn.Unfold(kernel_size=self.desc_patch_expand, padding=self.desc_patch_expand // 2)
        self.patch_unfold_orient = nn.Unfold(kernel_size=self.orient_patch, padding=self.orient_patch // 2)
        self.device = device
        self.phase = phase

        # 高斯距离衰减核
        norm_x = torch.linspace(-1, 1, 16)
        norm_grid = torch.stack(torch.meshgrid([norm_x, norm_x])).view(2, -1).t()[:, [1, 0]]
        self.gauss_kernel = torch.exp(-torch.sum(norm_grid*norm_grid, dim=-1) / 2).view(16, 16)

        # dis_sigma1 = nn.Parameter(torch.ones(1, 8))
        # dis_sigma2 = nn.Parameter(torch.ones(1, 8))
        # dis_sigma3 = nn.Parameter(torch.ones(1, 8))
        # dis_sigma_one = torch.cat((dis_sigma1, dis_sigma2.repeat(2, 1), dis_sigma1), dim=0)
        # dis_sigma_two = torch.cat((dis_sigma2, dis_sigma3.repeat(2, 1), dis_sigma2), dim=0)
        # dis_sigma_weight = torch.cat((dis_sigma_one, dis_sigma_two, dis_sigma_two, dis_sigma_one), dim=0).view(1, -1)
        # self.register_buffer('dis_sigma_weight', dis_sigma_weight)

        # # FGD loss
        # fea_dim = 16 # 16
        # self.fgd_loss = FeatureLoss(
        #     fea_dim, 
        #     fea_dim, 
        #     temp=0.5,       # 斜率超参，论文里表明与性能相关性不强
        #     alpha_fgd=0.001,
        #     beta_fgd=0, # 0.0005,
        #     gamma_fgd=0.001,
        #     lambda_fgd=0.000005)

        # # MGD loss
        # fea_dim = 8 # 16
        # self.mgd_loss = MGDLoss(
        #     fea_dim,
        #     fea_dim,
        #     # name,
        #     alpha_mgd=0.0002,
        #     lambda_mgd=0.45 # 0.65
        #     )

        if model_path != '':
            state_dict = torch.load(model_path, self.device)
            self.load_state_dict(state_dict)
            self.to(self.device)
            self.eval()
            logging.info(f'Loaded model parameters from {model_path}')
            logging.info(
                f"Number of model parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e3}KB")

    def extract_dense_map(self, image, ret_dict=False):
        # ====================================================
        # check image size, should be integer multiples of 2^5
        # if it is not a integer multiples of 2^5, padding zeros
        device = image.device
        b, c, h, w = image.shape
        # h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
        # w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w

        # # right bottom padding zero
        # if h_ != h:
        #     h_padding = torch.zeros(b, c, h_ - h, w, device=device)       
        #     image = torch.cat([image, h_padding], dim=2)
        # if w_ != w:
        #     w_padding = torch.zeros(b, c, h_, w_ - w, device=device)
        #     image = torch.cat([image, w_padding], dim=3)
        # ====================================================

        scores_map, descriptor_map = super().forward(image)

        # # ====================================================
        # if h_ != h or w_ != w:
        #     descriptor_map = descriptor_map[:, :, :h, :w]
        #     scores_map = scores_map[:, :, :h, :w]  # Bx1xHxW
        # # ====================================================

        if descriptor_map is not None:
            # BxCxHxW
            descriptor_map = F.normalize(descriptor_map, p=2, dim=1)        # 沿着channel维L2归一化描述子
        else:
            descriptor_map = None

        if ret_dict:
            return {'descriptor_map': descriptor_map, 'scores_map': scores_map, }
        else:
            return descriptor_map, scores_map

    def generate_homograhy_by_angle(self, H, W, angles):
        scale = 1
        M = [cv2.getRotationMatrix2D((W / 2, H / 2), i, scale) for i in angles]
        # center = np.mean(pts2, axis=0, keepdims=True)
        homo = [np.concatenate((m, [[0, 0, 1.]]), axis=0) for m in M]

        # valid = np.arange(n_angles)
        # idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        # homo = M[idx]

        return homo

    def get_orientation(self, img, keypoints):
        img = np.array(img)
        img = img.astype(np.float)
        # start = timer()
        Gx=np.zeros_like(img)
        Gy=np.zeros_like(img)
        h, w = img.shape
        for i in range(1,h-1):
            Gy[i,:] = img[i-1,:] - img[i+1,:] 
        Gy[0, :] = 2 * (img[0, :] - img[1, :])
        Gy[-1, :] = 2 * (img[-2, :] - img[-1, :])

        for j in range(1,w-1):
            Gx[:,j] = img[:,j+1] - img[:,j-1]
        Gx[:, 0] = 2 * (img[:,1] - img[:,0])
        Gx[:, -1] = 2 * (img[:,-1] - img[:,-2])

        Gxx = Gx*Gx
        Gyy = Gy*Gy
        Gxy = Gx*Gy
        Gxx_unfold = self.patch_unfold_orient(torch.tensor(Gxx).unsqueeze(0).unsqueeze(0).to(keypoints.device)).view(1, -1, h, w)
        Gyy_unfold = self.patch_unfold_orient(torch.tensor(Gyy).unsqueeze(0).unsqueeze(0).to(keypoints.device)).view(1, -1, h, w)
        Gxy_unfold = self.patch_unfold_orient(torch.tensor(Gxy).unsqueeze(0).unsqueeze(0).to(keypoints.device)).view(1, -1, h, w)
        Gxx_unfold_sum = torch.sum(Gxx_unfold, dim=1)
        Gyy_unfold_sum = torch.sum(Gyy_unfold, dim=1)       
        Gxy_unfold_sum = torch.sum(Gxy_unfold, dim=1)

        eps = 1e-12
        degree_value_all = 2 * Gxy_unfold_sum / (Gxx_unfold_sum - Gyy_unfold_sum + eps)
        angle_all = torch.atan(degree_value_all)    
        angle_all = angle_all*57.29578049 #180/(3.1415926)

        cond1 = torch.logical_and(Gxx_unfold_sum - Gyy_unfold_sum < 0, Gxy_unfold_sum >= 0)
        cond2 = torch.logical_and(Gxx_unfold_sum - Gyy_unfold_sum < 0, Gxy_unfold_sum < 0)
        angle_all[cond1] = (angle_all[cond1] + 180) / 2
        angle_all[cond2] = (angle_all[cond2] - 180) / 2
        angle_all[Gxx_unfold_sum - Gyy_unfold_sum >= 0] = angle_all[Gxx_unfold_sum - Gyy_unfold_sum >= 0] / 2
        angle_all += 90
        angle_all[angle_all > 90] = angle_all[angle_all > 90] - 180
        angle = F.grid_sample(angle_all.float().unsqueeze(0),
                            keypoints.view(1, 1, -1, 2),
                            mode='bilinear', align_corners=True).squeeze(2).squeeze(0)        # 1 x M

        return angle

    def get_orientation_batch(self, img_batch, keypoints=None, device='cpu'):
        b, _, h, w = img_batch.shape
        img_batch_left = img_batch[:, :, :, :-2]
        img_batch_right = img_batch[:, :, :, 2:]
        img_batch_top = img_batch[:, :, :-2, :]
        img_batch_bottom = img_batch[:, :, 2:, :]
        Gx = torch.zeros_like(img_batch, device=device)
        Gy = torch.zeros_like(img_batch, device=device)
        Gx[:, :, :, 1:-1] = img_batch_right - img_batch_left
        Gy[:, :, 1:-1, :] = img_batch_top - img_batch_bottom       # w-2 
        Gx[:, :, :, 0] = 2 * (img_batch[:, :, :, 1] - img_batch[:, :, :, 0])
        Gx[:, :, :, -1] = 2 * (img_batch[:, :, :,-1] - img_batch[:, :, :,-2])
        Gy[:, :, 0, :] = 2 * (img_batch[:, :, 0, :] - img_batch[:, :, 1, :])
        Gy[:, :, -1, :] = 2 * (img_batch[:, :, -2, :] - img_batch[:, :, -1, :])

        Gxx = Gx*Gx
        Gyy = Gy*Gy
        Gxy = Gx*Gy
        Gxx_unfold = self.patch_unfold_orient(Gxx).view(b, -1, h, w)
        Gyy_unfold = self.patch_unfold_orient(Gyy).view(b, -1, h, w)
        Gxy_unfold = self.patch_unfold_orient(Gxy).view(b, -1, h, w)
        Gxx_unfold_sum = torch.sum(Gxx_unfold, dim=1)
        Gyy_unfold_sum = torch.sum(Gyy_unfold, dim=1)       
        Gxy_unfold_sum = torch.sum(Gxy_unfold, dim=1)
        
        eps = 1e-12
        degree_value_all = 2 * Gxy_unfold_sum / (Gxx_unfold_sum - Gyy_unfold_sum + eps)
        angle_all = torch.atan(degree_value_all)    
        angle_all = angle_all*57.29578049 #180/(3.1415926)
        cond1 = torch.logical_and(Gxx_unfold_sum - Gyy_unfold_sum < 0, Gxy_unfold_sum >= 0)
        cond2 = torch.logical_and(Gxx_unfold_sum - Gyy_unfold_sum < 0, Gxy_unfold_sum < 0)
        angle_all[cond1] = (angle_all[cond1] + 180) / 2
        angle_all[cond2] = (angle_all[cond2] - 180) / 2
        angle_all[Gxx_unfold_sum - Gyy_unfold_sum >= 0] = angle_all[Gxx_unfold_sum - Gyy_unfold_sum >= 0] / 2
        angle_all += 90
        angle_all[angle_all > 90] = angle_all[angle_all > 90] - 180

        if keypoints is None:
            return angle_all.view(b, 1, -1)
        else:
            angle = F.grid_sample(angle_all.float().unsqueeze(0),
                                keypoints.view(1, 1, -1, 2),
                                mode='bilinear', align_corners=True).squeeze(2).squeeze(0)        # 1 x M

        return angle

    def get_sift_orientation_batch(self, img, patch_size=19, bin_size=10):
        '''
        img:tensor bx1xhxw
        keypoints: bxnx2 [h,w]

        '''
        patch_size=19

        batch, c, h, w = img.shape
        x = torch.linspace(0, w-1, w)     # 128 x 52 ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img.device)
        keypoints = mesh_points.repeat(batch, 1, 1).to(img.device)     # b x N x 2

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
        coor_cells = coor_cells.to(img.device)
        coor_cells = coor_cells.contiguous()
        
        keypoints_num = keypoints.size(1)
        keypoints_correct = torch.round(keypoints.clone()).to(img.device)
        keypoints_correct += offset
     
        src_pixel_coords = coor_cells.unsqueeze(0).repeat(batch, keypoints_num,1,1,1)
        src_pixel_coords = src_pixel_coords.float() * (patch_size // 2) + keypoints_correct.unsqueeze(2).unsqueeze(2).repeat(1,1,patch_size,patch_size,1)
      
        src_pixel_coords = src_pixel_coords.view([batch, keypoints_num, -1, 2])
        batch_image_coords_correct = torch.linspace(0, (batch-1)*(h+patch_size-1)*(w+patch_size-1), batch).long().to(device)
        src_pixel_coords_index = (src_pixel_coords[:,:,:,0] + src_pixel_coords[:,:,:,1]*(w+patch_size-1)).long()
        src_pixel_coords_index  = src_pixel_coords_index + batch_image_coords_correct[:,None,None]

        eps = 1e-12
        
        #生成幅值图和角度图
        # Grad_Amp = ((torch.sqrt(Gx**2 + Gy**2)) * 256)
        # 精度问题
        Grad_Amp = ((torch.sqrt(Gx**2 + Gy**2)) * 256).long()

        #边界反射
        Grad_Amp[:,:,9] = Grad_Amp[:,:,10]
        Grad_Amp[:,:,-10] = Grad_Amp[:,:,-11]
        Grad_Amp[:,:,:,9] = Grad_Amp[:,:,:,10]
        Grad_Amp[:,:,:,-10] = Grad_Amp[:,:,:,-11]

        degree_value = Gy / (Gx + eps)
        # Grad_ori = torch.atan(degree_value)
        Grad_ori = (torch.atan(degree_value) * 4096 + 0.5).long() / 4096
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


        return H_p.view(batch, -1)

    def get_grid_coords(self, kernel_size: int):
        
        origin = kernel_size / 2 - 0.5
        
        points = []
        
        for y in range(kernel_size):
            for x in range(kernel_size):
                p = (x - origin, -y + origin)
                points.append(p)
        
        points = np.array(points).T
        dis_weight = np.sqrt((points **2).sum(0))   # 根号（kernel_size // 2)**2 * 2
        return dis_weight

    def get_sift_amp_grad_batch(self, img, patch_size=19):
        '''
        img:tensor bx1xhxw
        keypoints: bxnx2 [h,w]

        '''
        patch_size=19

        batch, c, h, w = img.shape
        x = torch.linspace(0, w-1, w)     # 128 x 52 ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img.device)
        keypoints = mesh_points.repeat(batch, 1, 1).to(img.device)     # b x N x 2

        ori_max = 180
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
        coor_cells = coor_cells.to(img.device)
        coor_cells = coor_cells.contiguous()
        
        keypoints_num = keypoints.size(1)
        keypoints_correct = torch.round(keypoints.clone()).to(img.device)
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
        
        return Grad_Amp[:,:,offset:-offset,offset:-offset] / torch.max(torch.max(Grad_Amp[:,:,offset:-offset,offset:-offset], dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1)


    def cut_patch(self, img_batch, points, patch_size=16, train_flag=False):
        b, c, h, w = img_batch.shape
        descriptors = []
        results = None 
        # Padding Zero
        pad_size = (patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2)
        img_pad_batch = F.pad(img_batch, pad_size, "constant", 0)
        for batch_idx in range(b):
            keypoints = points[batch_idx]
            keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]]).to(keypoints.device)
            img = img_pad_batch[batch_idx]
            
            for point in keypoints:
                x = int(point[0] + 0.5)
                y = int(point[1] + 0.5)
                # crop_x = 0 if x-patch_size/2<0 else x-patch_size//2
                # crop_y = 0 if y-patch_size/2<0 else y-patch_size//2
                crop_x = x
                crop_y = y
                # print(x, y ,crop_x, crop_y)
                patch = img[:,crop_y:crop_y+patch_size,crop_x:crop_x+patch_size]

                data = patch.unsqueeze(0)
                # print(data.shape)
                if results is None:
                    results = data
                else:
                    results = torch.cat([results, data],dim=0)
        
        # compute output for patch a
        results_batch = Variable(results)   
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
        # outs = torch.chunk(outs, b, dim=0)
    
        return outs    

    def cut_patch_unfold(self, img_batch, points, train_flag=False, pt_norm_flag=True):
        b, _, h, w = img_batch.shape
        results = None 
        # Padding Zero
        img_unfold_batch = self.patch_unfold(img_batch).view(b, -1, h+1, w+1)[:, :, :-1, :-1]     # Bx(patch_size x patch_size)x(HxW) 
        for batch_idx in range(b):
            keypoints = points[batch_idx]
            if pt_norm_flag:
                keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]]).to(keypoints.device)
            img_patches = img_unfold_batch[batch_idx].transpose(0, 2)     # w x h x (patch_size x patch_size)
            keypoints = (keypoints + 0.5).long()
            data = img_patches[keypoints[:, 0], keypoints[:, 1]].view(-1, self.desc_patch, self.desc_patch).unsqueeze(1)    # M x (patch_size x patch_size)
            if results is None:
                results = data
            else:
                results = torch.cat([results, data],dim=0)
        
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
        # outs = torch.chunk(outs, b, dim=0)
    
        return outs    

    def cut_patch_unfold_interpolation(self, img_batch, points, train_flag=False, pt_norm_flag=True):
        b, _, h, w = img_batch.shape
        results = None 
        # Padding Zero
        # homography = sample_homography_cv(self.desc_patch, self.desc_patch, max_angle=180, n_angles=360)
        # homography = torch.tensor(homography, dtype=torch.float32)
        img_unfold_batch = self.patch_unfold(img_batch).view(b, -1, h+1, w+1)   # Bx(patch_size x patch_size) x H x W
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            if not pt_norm_flag:
                keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_offset = keypoints * keypoints.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
            img_patches = F.grid_sample(img_unfold_batch[batch_idx].unsqueeze(0),
                                    keypoints_offset.view(1, 1, -1, 2),
                                    mode='bilinear', align_corners=True).squeeze(2).squeeze(0)
            # no rotate
            data = img_patches.transpose(0, 1).view(-1, self.desc_patch, self.desc_patch).unsqueeze(1)  # M x 1 x patch_size x patch_size

            # for count in range(data.shape[0]):
            #     cv2.imwrite('demo/demo_patch_old' + str(count) + '.bmp', 255*data[count, 0, :, :].detach().cpu().numpy())
            # data = img_patches[keypoints[:, 0], keypoints[:, 1]].view(-1, self.desc_patch, self.desc_patch).unsqueeze(1)    # M x 1 x patch_size x patch_size
            if results is None:
                results = data
            else:
                results = torch.cat([results, data],dim=0)

            # data = img_patches[keypoints[:, 0], keypoints[:, 1]].view(-1, self.desc_patch, self.desc_patch).unsqueeze(1)
            # # print(data.shape, homography.shape)
            # data_rotate = batch_inv_warp_image(data.cpu(), homography.repeat(data.shape[0], 1, 1), mode="bilinear").to(data.device)
            # # print(data_rotate.shape)
            # if results is None:
            #     results = data_rotate
            # else:
            #     results = torch.cat([results, data_rotate],dim=0)

        # compute output for patch a
        results_batch = Variable(results)   
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
        # outs = torch.chunk(outs, b, dim=0)
    
        return outs.squeeze(-1)   

    def cut_patch_unfold_interpolation_aligned(self, img_batch, points, train_flag=False, pt_norm_flag=True):
        b, _, h, w = img_batch.shape
        results = None 
        # Padding Zero
        # homography = sample_homography_cv(self.desc_patch, self.desc_patch, max_angle=180, n_angles=360)
        # homography = torch.tensor(homography, dtype=torch.float32)
        img_unfold_batch_expand = self.patch_unfold_expand(img_batch).view(b, -1, h+1, w+1)     # Bx(expand_patch_size x expand_patch_size) x (H + 1) x (W + 1)
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            if not pt_norm_flag:
                keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_offset = keypoints * keypoints.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
            img_patches = F.grid_sample(img_unfold_batch_expand[batch_idx].unsqueeze(0),
                                    keypoints_offset.view(1, 1, -1, 2),
                                    mode='bilinear', align_corners=True).squeeze(2).squeeze(0)      # (expand_patch_size x expand_patch_size) x M
            orientation_patch = self.get_orientation(img_batch[batch_idx].squeeze().cpu().numpy(), keypoints)
            homography_expand_patch = self.generate_homograhy_by_angle(self.desc_patch_expand, self.desc_patch_expand, -orientation_patch.squeeze(0).detach().cpu().numpy())
            homography_expand_patch = torch.tensor(homography_expand_patch, dtype=torch.float32)        # Mx3x3
            data_expand = img_patches.transpose(0, 1).view(-1, self.desc_patch_expand, self.desc_patch_expand).unsqueeze(1) # M x 1 x expand_patch_size x expand_patch_size
            data_expand_rotate = batch_inv_warp_image(data_expand.cpu(), homography_expand_patch, mode="bilinear").to(data_expand.device)
            
            # 24x24 -> 16x16
            data = data_expand_rotate[:, :, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2].contiguous()

            # 画点和方向场
            # pred, img_pair = {}, {}
            # pred.update({
            #     "pts": ((keypoints + 1) / 2 * keypoints.new_tensor([w - 1, h - 1]).to(keypoints.device)).detach().cpu().numpy(), 
            #     "angles": orientation_patch.squeeze(0).detach().cpu().numpy(),
            #     })
            # img_pair.update({
            #     "img": img_batch[batch_idx].cpu().numpy().squeeze(),
            #     })
            # img_pts = draw_orientation(img_pair, pred, radius=1, s=1)
            # cv2.imwrite('demo/demo_patch_new' + str(count) + '.bmp', img_pts)

            # # 画校准后的patch
            # for count in range(data.shape[0]):
            #     cv2.imwrite('demo/demo_patch_new' + str(count) + '.bmp', 255*data[count, 0, :, :].detach().cpu().numpy())
            #     if count == 100:
            #         exit()
            if results is None:
                results = data
            else:
                results = torch.cat([results, data],dim=0)

            # data = img_patches[keypoints[:, 0], keypoints[:, 1]].view(-1, self.desc_patch, self.desc_patch).unsqueeze(1)
            # # print(data.shape, homography.shape)
            # data_rotate = batch_inv_warp_image(data.cpu(), homography.repeat(data.shape[0], 1, 1), mode="bilinear").to(data.device)
            # # print(data_rotate.shape)
            # if results is None:
            #     results = data_rotate
            # else:
            #     results = torch.cat([results, data_rotate],dim=0)

        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
        # outs = torch.chunk(outs, b, dim=0)
    
        return outs.squeeze(-1)   

    def cut_patch_unfold_patch_map_interpolation_aligned_batch(self, img_batch, points, train_flag=False, pt_norm_flag=True, fixed_angle=-200, sift_ori=None):
        b, _, h, w = img_batch.shape
        pad_size = (self.desc_patch_expand // 2, self.desc_patch_expand // 2, self.desc_patch_expand // 2, self.desc_patch_expand // 2)
        img_pad_batch = F.pad(img_batch, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 200
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)

                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0)  
        if not pt_norm_flag:
            keypoints_expand = keypoints_expand / keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
        if fixed_angle >=-180 and fixed_angle < 180:
            orient_batch = torch.ones(b, 1, h*w) * fixed_angle
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        ori_expand = torch.zeros(max_num, device=img_batch.device)
        if sift_ori is not None:
            orient_batch_kp = torch.cat((sift_ori.to(img_batch.device), ori_expand[expand_mask_all.view(-1)==0]), dim=-1)                                 # nx1
            # print(orient_batch_kp, orient_batch_kp.shape)
        else:
            orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                            keypoints_expand.view(b, 1, -1, 2),
                            mode='bilinear', align_corners=True)        # B X 1 X 1 X (HXW)
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device) + self.desc_patch_expand // 2 
        # print(keypoints_expand[0, :] - self.desc_patch_expand // 2, orient_batch_kp.view(-1)[0]) 
        # print(keypoints_expand[149, :] - self.desc_patch_expand // 2, orient_batch_kp.view(-1)[149]) 
        data_kp = inv_warp_patch_batch(img_pad_batch, keypoints_expand, orient_batch_kp.view(-1), self.desc_patch, self.sample_desc_patch)[expand_mask_all==1, :, : ].unsqueeze(1)   

        # # 画校准后的patch
        # for count in range(data_kp.shape[0]):            
        #     if count < 100:
        #         cv2.imwrite('demo/demo_patch_mesh_grid_new_' + str(count) + '.bmp', 255*data[count, 0, :, :].detach().cpu().numpy())
        #         cv2.imwrite('demo/demo_patch_kp_new_' + str(count) + '.bmp', 255*data_kp[count, 0, :, :].detach().cpu().numpy())
        # exit()
        # print(data.shape)
        # print(img_unfold_batch.shape)
        # results = data.permute(0, 2, 3, 1).reshape(-1, 1, self.desc_patch, self.desc_patch)   
        results = data_kp
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                self.descriptor_net.eval()
                outs = self.descriptor_net(results_batch)
                del results_batch

        return outs.squeeze(-1)   

    def cut_patch_unfold_patch_map_interpolation_aligned_batch_ext93(self, img_batch, img_ext_batch, points, train_flag=False, pt_norm_flag=True, fixed_angle=-200, sift_ori=None, cal_angle=0):
        _, _, hm, wm = img_batch.shape          # 122 x 40
        b, _, h, w = img_ext_batch.shape        # 128 x 52

        w_pad = 7 # 20 # 4 # 7 # 14
        h_pad = 12 # 25 # 9 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext_batch, pad_size, "constant", 0)
        # inner
        # img_pad_batch = deepcopy(img_resize)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 200
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 
        
        if not pt_norm_flag:
            keypoints_expand = keypoints_expand / keypoints_expand.new_tensor([wm - 5, hm - 1]).to(img_batch.device) * 2 - 1
      
        if fixed_angle >=-180 and fixed_angle < 180:
            orient_batch = torch.ones(b, 1, hm*(wm-4)) * fixed_angle
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            # orient_batch = self.get_orientation_batch(img_batch[:, :, :, 2:-2], device=img_batch.device)      # Bx1x(hm x(wm-4))
            # orient_batch_T = self.get_orientation_batch(img_ext_batch[:, :, 3:-3, 8:-8], device=img_batch.device)      # Bx1x(hm x(wm-4))
            orient_batch_T = self.get_sift_orientation_batch(img_ext_batch[:, :, 3:-3, 8:-8])      # Bx1x(hm x(wm-4))
        ori_expand = torch.zeros(max_num, device=img_batch.device)
        
        if sift_ori is not None:
            # print(sift_ori.shape, ori_expand[expand_mask_all.view(-1)==0].shape)
            orient_batch_kp = torch.cat((sift_ori.to(img_batch.device), ori_expand[expand_mask_all.view(-1)==0]), dim=-1).view(b, 1, 1, -1)                                 # nx1
            # print(orient_batch_kp.shape)
        else:
            orient_batch_kp = F.grid_sample(orient_batch_T.view(b, 1, hm, wm - 4),
                            keypoints_expand.view(b, 1, -1, 2),
                            mode='bilinear', align_corners=True)        # B X 1 X 1 X (max_num)
        # orient_batch_kp_T = F.grid_sample(orient_batch_T.view(b, 1, hm, wm - 4),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (max_num)

        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([wm - 5, hm - 1]) 
        keypoints_expand_mid[:, 0] += 2 + 6 + w_pad     # 36->92
        keypoints_expand_mid[:, 1] += 3 + h_pad    # 122->178

        # keypoints_expand = (keypoints_expand_mid/ downsampe_ratio).to(img_batch.device) + res_pad_size
        
        # only 22/32
        if self.only1:
            if self.is_rec:
                # data_kp = inv_warp_patch_batch_rec(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp.view(-1) + cal_angle, sample_factor=1.25)[expand_mask_all==1, :, : ].unsqueeze(1) 
                if self.is_expand_rec:
                    data_kp = inv_warp_patch_batch_rec(img_ext_pad_batch, keypoints_expand_mid, torch.zeros_like(orient_batch_kp.view(-1)), patch_size=(32, 16), sample_factor=1.25)[expand_mask_all==1, :, : ].unsqueeze(1) 
                elif self.is_square_ncal:
                    data_kp = inv_warp_patch_batch_rec(img_ext_pad_batch, keypoints_expand_mid, torch.zeros_like(orient_batch_kp.view(-1)), patch_size=(16, 16), sample_factor=1.25)[expand_mask_all==1, :, : ].unsqueeze(1) 
                else:
                    data_kp = inv_warp_patch_batch_rec(img_ext_pad_batch, keypoints_expand_mid, torch.zeros_like(orient_batch_kp.view(-1)), sample_factor=1.25)[expand_mask_all==1, :, : ].unsqueeze(1) 
            else:
                data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp.view(-1) + cal_angle, 16, self.sample_desc_patch)[expand_mask_all==1, :, : ].unsqueeze(1) 
                # data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, torch.zeros_like(orient_batch_kp.view(-1)), 16, self.sample_desc_patch)[expand_mask_all==1, :, : ].unsqueeze(1)
                if self.Guass_flag:        
                    # 高斯距离衰减
                    data_kp = data_kp * self.gauss_kernel.repeat(data_kp.shape[0], 1, 1).to(data_kp.device).unsqueeze(1)
        else:
        # both 22 and 32   
            data_kp_high = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp.view(-1) + cal_angle + 45, 16, self.sample_desc_patch)[expand_mask_all==1, :, : ].unsqueeze(1) 
            data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp.view(-1) + cal_angle, 16, 22)[expand_mask_all==1, :, : ].unsqueeze(1) 
            if self.Guass_flag:        
                # 高斯距离衰减
                data_kp_high = data_kp_high * self.gauss_kernel.repeat(data_kp_high.shape[0], 1, 1).to(data_kp_high.device).unsqueeze(1)
            data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

    
        # data_kp_T = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp_T.view(-1) + cal_angle, 16, self.sample_desc_patch)[expand_mask_all==1, :, : ].unsqueeze(1)   
        # print('point x:', keypoints_expand_mid[0, 0] - (2 + 6 + w_pad))
        # print('point y:', keypoints_expand_mid[0, 1] - (3 + h_pad))
        # print('orient:', orient_batch_kp.view(-1)[0])
        # print('data_kp:', data_kp[0,0,:, :])
        # # 画校准后的patch
        # print(data_kp.shape[0])
        # for count in range(data_kp.shape[0]):            
        #     if count < 100:
        #         # print(count)
        #         # cv2.imwrite('demo/demo_patch_mesh_grid_new_' + str(count) + '.bmp', 255*data_kp[count, 0, :, :].detach().cpu().numpy())
        #         # sift_ori = orient_batch_kp.view(-1)[expand_mask_all==1][count]
        #         # net_ori = orient_batch_kp_T.view(-1)[expand_mask_all==1][count]
        #         # print(sift_ori, net_ori)
        #         cv2.imwrite('demo/demo_patch_kp_new_' + str(count) + '.bmp', 255*data_kp[count, 0, :, :].detach().cpu().numpy())
        # exit()
        # print(data.shape)
        # print(img_unfold_batch.shape)
        # results = data.permute(0, 2, 3, 1).reshape(-1, 1, self.desc_patch, self.desc_patch)   
        results = data_kp
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                self.descriptor_net.eval()

                outs = self.descriptor_net(results_batch)
                
                # # 角度MOE mask
                # angle_mask_kp = orient_batch_kp.view(-1) >= 0
                # outs = self.descriptor_net(results_batch, wb_mask=angle_mask_kp[expand_mask_all.bool().view(-1)])

                del results_batch
        # print('desc:', outs.squeeze(-1)[0, :])
        # exit()
        # print(torch.sum(outs.squeeze(-1)[:, :128] * outs.squeeze(-1)[:, 128:], dim=1))
        # exit()
        return outs.squeeze(-1), orient_batch_kp.view(-1)[expand_mask_all==1]

    def cut_patch_unfold_patch_map_interpolation_aligned_batch_ext93_wbmask(self, img_batch, img_ext_batch, points, train_flag=False, pt_norm_flag=True, fixed_angle=-200, sift_ori=None, cal_angle=0):
        _, _, hm, wm = img_batch.shape          # 122 x 40
        b, _, h, w = img_ext_batch.shape        # 128 x 52

        w_pad = 7 # 20 # 4 # 7 # 14
        h_pad = 12 # 25 # 9 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext_batch, pad_size, "constant", 0)
        # inner
        # img_pad_batch = deepcopy(img_resize)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 200
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 
        
        if not pt_norm_flag:
            keypoints_expand = keypoints_expand / keypoints_expand.new_tensor([wm - 5, hm - 1]).to(img_batch.device) * 2 - 1
      
        if fixed_angle >=-180 and fixed_angle < 180:
            orient_batch = torch.ones(b, 1, hm*(wm-4)) * fixed_angle
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            # orient_batch = self.get_orientation_batch(img_batch[:, :, :, 2:-2], device=img_batch.device)      # Bx1x(hm x(wm-4))
            # orient_batch_T = self.get_orientation_batch(img_ext_batch[:, :, 3:-3, 8:-8], device=img_batch.device)      # Bx1x(hm x(wm-4))
            orient_batch_T = self.get_sift_orientation_batch(img_ext_batch).view(b, 1, h, w)[:, :, 3:-3, 8:-8]      # Bx1x(hm x(wm-4))
        ori_expand = torch.zeros(max_num, device=img_batch.device)
        
        if sift_ori is not None:
            # print(sift_ori.shape, ori_expand[expand_mask_all.view(-1)==0].shape)
            orient_batch_kp = torch.cat((sift_ori.to(img_batch.device), ori_expand[expand_mask_all.view(-1)==0]), dim=-1).view(b, 1, 1, -1)                                 # nx1
            # print(orient_batch_kp.shape)
        else:
            orient_batch_kp = F.grid_sample(orient_batch_T.view(b, 1, hm, wm - 4),
                            keypoints_expand.view(b, 1, -1, 2),
                            mode='bilinear', align_corners=True)        # B X 1 X 1 X (max_num)
        # orient_batch_kp_T = F.grid_sample(orient_batch_T.view(b, 1, hm, wm - 4),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (max_num)

        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([wm - 5, hm - 1]) 
        keypoints_expand_mid[:, 0] += 2 + 6 + w_pad     # 36->92
        keypoints_expand_mid[:, 1] += 3 + h_pad    # 122->178

        # keypoints_expand = (keypoints_expand_mid/ downsampe_ratio).to(img_batch.device) + res_pad_size
        
        # only 22/32
        if self.only1:
            if self.is_rec:
                # data_kp = inv_warp_patch_batch_rec(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp.view(-1) + cal_angle, sample_factor=1.25)[expand_mask_all==1, :, : ].unsqueeze(1) 
                if self.is_expand_rec:
                    data_kp = inv_warp_patch_batch_rec(img_ext_pad_batch, keypoints_expand_mid, torch.zeros_like(orient_batch_kp.view(-1)), patch_size=(32, 16), sample_factor=1.25)[expand_mask_all==1, :, : ].unsqueeze(1) 
                elif self.is_square_ncal:
                    data_kp = inv_warp_patch_batch_rec(img_ext_pad_batch, keypoints_expand_mid, torch.zeros_like(orient_batch_kp.view(-1)), patch_size=(16, 16), sample_factor=1.25)[expand_mask_all==1, :, : ].unsqueeze(1) 
                else:
                    data_kp = inv_warp_patch_batch_rec(img_ext_pad_batch, keypoints_expand_mid, torch.zeros_like(orient_batch_kp.view(-1)), sample_factor=1.25)[expand_mask_all==1, :, : ].unsqueeze(1) 
            else:
                data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp.view(-1) + cal_angle, 16, self.sample_desc_patch)[expand_mask_all==1, :, : ].unsqueeze(1) 
                # data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, torch.zeros_like(orient_batch_kp.view(-1)), 16, self.sample_desc_patch)[expand_mask_all==1, :, : ].unsqueeze(1)
                if self.Guass_flag:        
                    # 高斯距离衰减
                    data_kp = data_kp * self.gauss_kernel.repeat(data_kp.shape[0], 1, 1).to(data_kp.device).unsqueeze(1)
        else:
        # both 22 and 32   
            data_kp_high = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp.view(-1) + cal_angle + 45, 16, self.sample_desc_patch)[expand_mask_all==1, :, : ].unsqueeze(1) 
            data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp.view(-1) + cal_angle, 16, 22)[expand_mask_all==1, :, : ].unsqueeze(1) 
            if self.Guass_flag:        
                # 高斯距离衰减
                data_kp_high = data_kp_high * self.gauss_kernel.repeat(data_kp_high.shape[0], 1, 1).to(data_kp_high.device).unsqueeze(1)
            data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

    
        # data_kp_T = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp_T.view(-1) + cal_angle, 16, self.sample_desc_patch)[expand_mask_all==1, :, : ].unsqueeze(1)   
        # print('point x:', keypoints_expand_mid[0, 0] - (2 + 6 + w_pad))
        # print('point y:', keypoints_expand_mid[0, 1] - (3 + h_pad))
        # print('orient:', orient_batch_kp.view(-1)[0])
        # print('data_kp:', data_kp[0,0,:, :])
        # # 画校准后的patch
        # print(data_kp.shape[0])
        # for count in range(data_kp.shape[0]):            
        #     if count < 100:
        #         # print(count)
        #         # cv2.imwrite('demo/demo_patch_mesh_grid_new_' + str(count) + '.bmp', 255*data_kp[count, 0, :, :].detach().cpu().numpy())
        #         # sift_ori = orient_batch_kp.view(-1)[expand_mask_all==1][count]
        #         # net_ori = orient_batch_kp_T.view(-1)[expand_mask_all==1][count]
        #         # print(sift_ori, net_ori)
        #         cv2.imwrite('demo/demo_patch_kp_new_' + str(count) + '.bmp', 255*data_kp[count, 0, :, :].detach().cpu().numpy())
        # exit()
        # print(data.shape)
        # print(img_unfold_batch.shape)
        # results = data.permute(0, 2, 3, 1).reshape(-1, 1, self.desc_patch, self.desc_patch)   
        results = data_kp
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                self.descriptor_net.eval()

                outs, wb_mask = self.descriptor_net(results_batch)
                
                # # 角度MOE mask
                # angle_mask_kp = orient_batch_kp.view(-1) >= 0
                # outs = self.descriptor_net(results_batch, wb_mask=angle_mask_kp[expand_mask_all.bool().view(-1)])

                del results_batch
        # print('desc:', outs.squeeze(-1)[0, :])
        # exit()
        # print(torch.sum(outs.squeeze(-1)[:, :128] * outs.squeeze(-1)[:, 128:], dim=1))
        # exit()
        return outs.squeeze(-1), orient_batch_kp.view(-1)[expand_mask_all==1], wb_mask


    def cut_patch_unfold_patch_map_interpolation_aligned_batch_ext93_assist(self, img_batch, img_ext_batch, points, train_flag=False, pt_norm_flag=True, fixed_angle=-200, sift_ori=None, cal_angle=0):
        _, _, hm, wm = img_batch.shape          # 122 x 40
        b, _, h, w = img_ext_batch.shape        # 128 x 52

        w_pad = 7 # 4 # 7 # 14
        h_pad = 12 # 9 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext_batch, pad_size, "constant", 0)
        # inner
        # img_pad_batch = deepcopy(img_resize)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 200
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 
        
        if not pt_norm_flag:
            keypoints_expand = keypoints_expand / keypoints_expand.new_tensor([wm - 5, hm - 1]).to(img_batch.device) * 2 - 1
      
        if fixed_angle >=-180 and fixed_angle < 180:
            orient_batch = torch.ones(b, 1, hm*(wm-4)) * fixed_angle
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            # orient_batch = self.get_orientation_batch(img_batch[:, :, :, 2:-2], device=img_batch.device)      # Bx1x(hm x(wm-4))
            orient_batch_T = self.get_orientation_batch(img_ext_batch[:, :, 3:-3, 8:-8], device=img_batch.device)      # Bx1x(hm x(wm-4))
        ori_expand = torch.zeros(max_num, device=img_batch.device)
        
        if sift_ori is not None:
            # print(sift_ori.shape, ori_expand[expand_mask_all.view(-1)==0].shape)
            orient_batch_kp = torch.cat((sift_ori.to(img_batch.device), ori_expand[expand_mask_all.view(-1)==0]), dim=-1).view(b, 1, 1, -1)                                 # nx1
            # print(orient_batch_kp.shape)
        else:
            orient_batch_kp = F.grid_sample(orient_batch_T.view(b, 1, hm, wm - 4),
                            keypoints_expand.view(b, 1, -1, 2),
                            mode='bilinear', align_corners=True)        # B X 1 X 1 X (max_num)
        # orient_batch_kp_T = F.grid_sample(orient_batch_T.view(b, 1, hm, wm - 4),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (max_num)

        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([wm - 5, hm - 1]) 
        keypoints_expand_mid[:, 0] += 2 + 6 + w_pad     # 36->92
        keypoints_expand_mid[:, 1] += 3 + h_pad    # 122->178

        # keypoints_expand = (keypoints_expand_mid/ downsampe_ratio).to(img_batch.device) + res_pad_size
        
        # only 22/32
        if self.only1:
            data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp.view(-1) + cal_angle, 16, self.sample_desc_patch)[expand_mask_all==1, :, : ].unsqueeze(1) 
            if self.Guass_flag:        
                # 高斯距离衰减
                data_kp = data_kp * self.gauss_kernel.repeat(data_kp.shape[0], 1, 1).to(data_kp.device).unsqueeze(1)
        else:
        # both 22 and 32   
            data_kp_high = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp.view(-1) + cal_angle + 45, 16, self.sample_desc_patch)[expand_mask_all==1, :, : ].unsqueeze(1) 
            data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp.view(-1) + cal_angle, 16, 22)[expand_mask_all==1, :, : ].unsqueeze(1) 
            if self.Guass_flag:        
                # 高斯距离衰减
                data_kp_high = data_kp_high * self.gauss_kernel.repeat(data_kp_high.shape[0], 1, 1).to(data_kp_high.device).unsqueeze(1)
            data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)
        
        # data_kp_T = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand_mid, orient_batch_kp_T.view(-1) + cal_angle, 16, 32)[expand_mask_all==1, :, : ].unsqueeze(1)   
        # print('point x:', keypoints_expand_mid[0, 0] - (2 + 6 + w_pad))
        # print('point y:', keypoints_expand_mid[0, 1] - (3 + h_pad))
        # print('orient:', orient_batch_kp.view(-1)[0])
        # print('data_kp:', data_kp[0,0,:, :])
        # # 画校准后的patch
        # for count in range(data_kp.shape[0]):            
        #     if count < 100:
        #         # cv2.imwrite('demo/demo_patch_mesh_grid_new_' + str(count) + '.bmp', 255*data_kp[count, 0, :, :].detach().cpu().numpy())
        #         sift_ori = orient_batch_kp.view(-1)[expand_mask_all==1][count]
        #         net_ori = orient_batch_kp_T.view(-1)[expand_mask_all==1][count]
        #         # print(sift_ori, net_ori)
        #         cv2.imwrite('demo/demo_patch_kp_new_' + str(count) + '_' + str(sift_ori.item()) + '_' + str(net_ori.item()) + '.bmp', torch.cat((255*data_kp[count, 0, :, :], 255*data_kp_T[count, 0, :, :]), dim=-1).detach().cpu().numpy())
        # exit()
        # print(data.shape)
        # print(img_unfold_batch.shape)
        # results = data.permute(0, 2, 3, 1).reshape(-1, 1, self.desc_patch, self.desc_patch)   
        results = data_kp
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_assist_net(results_batch)
        else:
            with torch.no_grad():
                self.descriptor_assist_net.eval()
                outs = self.descriptor_assist_net(results_batch)
                del results_batch
        # print('desc:', outs.squeeze(-1)[0, :])
        # exit()
        # print(torch.sum(outs.squeeze(-1)[:, :128] * outs.squeeze(-1)[:, 128:], dim=1))
        # exit()
        return outs.squeeze(-1)   

    def cut_patch_from_featuremap(self, img_batch, points, train_flag=False, pt_norm_flag=True, fixed_angle=-200, sift_ori=None):
        b, _, h, w = img_batch.shape

        img_pad_size = (2, 2)
        img_pad_batch = F.pad(img_batch, img_pad_size, "constant", 0)   # [136, 40]

        if train_flag:
            feats = self.descriptor_net(img_pad_batch) # bx8x34x10
        else:
            with torch.no_grad():
                self.descriptor_net.eval()
                feats = self.descriptor_net(img_pad_batch)  # bx34x10
                # del results_batch
        # with torch.no_grad():
        #     outs_map = self.descriptor_net(data, angle=flip_flag)
        #     del data

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 200
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 
        
        if not pt_norm_flag:
            keypoints_expand = keypoints_expand / keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
        if fixed_angle >=-180 and fixed_angle < 180:
            orient_batch = torch.ones(b, 1, h*w) * fixed_angle
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        ori_expand = torch.zeros(max_num, device=img_batch.device)

        if sift_ori is not None:
            orient_batch_kp = torch.cat((sift_ori.to(img_batch.device), ori_expand[expand_mask_all.view(-1)==0]), dim=-1)                                 # nx1
            # print(orient_batch_kp, orient_batch_kp.shape)
        else:
            orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                            keypoints_expand.view(b, 1, -1, 2),
                            mode='bilinear', align_corners=True)        # B X 1 X 1 X (HXW)

        # 切描述patch 
        downsampe_ratio = 4
        # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)

        # x = torch.linspace(0, w-1, w) / downsampe_ratio      # ex: [-2, -1, 0, 1, 2]
        # y = torch.linspace(0, h-1, h) / downsampe_ratio
        # mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + self.desc_patch_expand // 2
        keypoints_expand = (keypoints_expand + 1) / (downsampe_ratio * 2) * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device) + self.desc_patch_expand // 2

        pad_size = (self.desc_patch_expand // 2, self.desc_patch_expand // 2, self.desc_patch_expand // 2, self.desc_patch_expand // 2)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x(h//4+pad)x(w//4+pad)
        # print(feats_pad_batch.view(-1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]).shape)
        # print(keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2).shape)
        # outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.sample_desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        
        # valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        # print(valid_all_mask.shape, pmask_kp.shape)
        
        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1), self.desc_patch, self.sample_desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        # outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[expand_mask_all==1, :] 
        # outs_map = outs.reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w) 
        # print(outs_kp.shape, outs_map.shape, valid_all_mask.shape)
        return outs_kp

    def cut_patch_from_featuremap_pad(self, img_batch, points, train_flag=False, pt_norm_flag=True, fixed_angle=-200, sift_ori=None, cal_angle=0):
        b, _, h, w = img_batch.shape        # 136 x 36

        img_resize = TF.resize(F.pad(img_batch, (2, 2), "constant", 0), (h + 1, w + 5))     # 137x41 (h + 1, w + 5)

        # inner
        img_pad_batch = deepcopy(img_resize)

        # # Origin
        # img_pad_size = (2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        # img_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [153, 57]
        
        
        # img_rotate = TF.rotate(img_resize, 45, InterpolationMode.BILINEAR)
        # print(img_rotate.shape)
        # img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])

        if train_flag:
            feats = self.descriptor_net(img_pad_batch) # bx8x38x14
        else:
            with torch.no_grad():
                self.descriptor_net.eval()
                feats = self.descriptor_net(img_pad_batch)      #  bx8x39x15
                # feats_rotate = self.descriptor_net(img_rotate)
                # del results_batch
        # with torch.no_grad():
        #     outs_map = self.descriptor_net(data, angle=flip_flag)
        #     del data

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 200
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 
        
        if not pt_norm_flag:
            keypoints_expand = keypoints_expand / keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
        if fixed_angle >=-180 and fixed_angle < 180:
            orient_batch = torch.ones(b, 1, h*w) * fixed_angle
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        ori_expand = torch.zeros(max_num, device=img_batch.device)
        
        if sift_ori is not None:
            # print(sift_ori.shape, ori_expand[expand_mask_all.view(-1)==0].shape)
            orient_batch_kp = torch.cat((sift_ori.to(img_batch.device), ori_expand[expand_mask_all.view(-1)==0]), dim=-1).view(b, 1, 1, -1)                                 # nx1
            # print(orient_batch_kp.shape)
        else:
            orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                            keypoints_expand.view(b, 1, -1, 2),
                            mode='bilinear', align_corners=True)        # B X 1 X 1 X (max_num)

        # 切描述patch 
        downsampe_ratio = 4
        # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)

        # # 距离衰减核
        # coords = self.get_grid_coords(self.desc_patch)
        # dis_weight = torch.tensor(coords).to(img_batch.device).repeat(b*max_num, 8, 1).transpose(1, 2).contiguous().view(-1, 128) 
        # dis_weight_inv = torch.exp(-1 * dis_weight * dis_weight / 2) # 1 / dis_weight

        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 1
        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]) 
        keypoints_expand_mid[:, 0] += 2     # 36->40
        keypoints_expand_mid = keypoints_expand_mid * keypoints_expand_mid.new_tensor([(w + 4) / (w + 3), h / (h - 1)])
        keypoints_expand = ((keypoints_expand_mid + 2 * self.sample_desc_patch)/ downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x41x17
        # print(feats_pad_batch.view(-1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]).shape)
        # print(keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2).shape)      
        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        # print(outs_kp.dtype, dis_weight_inv.dtype)
        # outs_kp = outs_kp * dis_weight_inv

        outs_kp = L2Norm()(outs_kp)[expand_mask_all==1, :].float()
        # outs_map = outs.reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w) 
        # print(outs_kp.shape, outs_map.shape, valid_all_mask.shape)
        return outs_kp  

    def cut_patch_from_featuremap_pad_ext(self, img_batch, img_ext_batch, points, train_flag=False, pt_norm_flag=True, fixed_angle=-200, sift_ori=None, cal_angle=0):
        _, _, hm, wm = img_batch.shape
        b, _, h, w = img_ext_batch.shape        # 144 x 52

        img_resize = TF.resize(img_ext_batch, (h + 1, w + 1))     # 144x52->145x53

        # inner
        img_pad_batch = deepcopy(img_resize)

        # # Origin
        # img_pad_size = (2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        # img_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [153, 57]
        
        
        # img_rotate = TF.rotate(img_resize, 45, InterpolationMode.BILINEAR)
        # print(img_rotate.shape)
        # img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])

        if train_flag:
            feats = self.descriptor_net(img_pad_batch) # bx8x38x14
        else:
            with torch.no_grad():
                self.descriptor_net.eval()
                feats = self.descriptor_net(img_pad_batch)      #  bx8x41x16
                # feats_rotate = self.descriptor_net(img_rotate)
                # del results_batch
        # with torch.no_grad():
        #     outs_map = self.descriptor_net(data, angle=flip_flag)
        #     del data

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 200
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 
        
        if not pt_norm_flag:
            keypoints_expand = keypoints_expand / keypoints_expand.new_tensor([wm - 5, hm - 1]).to(img_batch.device) * 2 - 1
      
        if fixed_angle >=-180 and fixed_angle < 180:
            orient_batch = torch.ones(b, 1, hm*(wm-4)) * fixed_angle
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch[:, :, :, 2:-2], device=img_batch.device)      # Bx1x(hm x(wm-4))
        ori_expand = torch.zeros(max_num, device=img_batch.device)
        
        if sift_ori is not None:
            # print(sift_ori.shape, ori_expand[expand_mask_all.view(-1)==0].shape)
            orient_batch_kp = torch.cat((sift_ori.to(img_batch.device), ori_expand[expand_mask_all.view(-1)==0]), dim=-1).view(b, 1, 1, -1)                                 # nx1
            # print(orient_batch_kp.shape)
        else:
            orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, hm, wm - 4),
                            keypoints_expand.view(b, 1, -1, 2),
                            mode='bilinear', align_corners=True)        # B X 1 X 1 X (max_num)

        # 切描述patch 
        downsampe_ratio = 4
        # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)

        # # 距离衰减核
        # coords = self.get_grid_coords(self.desc_patch)
        # dis_weight = torch.tensor(coords).to(img_batch.device).repeat(b*max_num, 8, 1).transpose(1, 2).contiguous().view(-1, 128) 
        # dis_weight_inv = torch.exp(-1 * dis_weight * dis_weight / 2) # 1 / dis_weight

        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 1
        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([wm - 5, hm - 1]) 
        keypoints_expand_mid[:, 0] += 2 + 6     # 36->52
        keypoints_expand_mid[:, 1] += 4     # 136->144
        keypoints_expand_mid = keypoints_expand_mid * keypoints_expand_mid.new_tensor([w / (w-1), h / (h - 1)])  # 144x52->145x53
        keypoints_expand_mid[:, 0] += self.sample_desc_patch
        keypoints_expand_mid[:, 1] += self.sample_desc_patch + 4
        keypoints_expand = (keypoints_expand_mid/ downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x41x17
        # print(feats_pad_batch.view(-1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]).shape)
        # print(keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2).shape)      
        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  

        # print(outs_kp.dtype, dis_weight_inv.dtype)
        # outs_kp = outs_kp * dis_weight_inv

        outs_kp = L2Norm()(outs_kp)[expand_mask_all==1, :].float()
        # print(outs_kp[0, :])
        # exit()
        # outs_map = outs.reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w) 
        # print(outs_kp.shape, outs_map.shape, valid_all_mask.shape)
        return outs_kp  

    def cut_patch_from_featuremap_pad_ext93(self, img_batch, img_ext_batch, points, train_flag=False, pt_norm_flag=True, fixed_angle=-200, sift_ori=None, cal_angle=0):
        _, _, hm, wm = img_batch.shape          # 122 x 40
        b, _, h, w = img_ext_batch.shape        # 128 x 52

        img_resize = TF.resize(img_ext_batch, (h + 1, w + 1))     # 128x52->129x53

        # inner
        img_pad_batch = deepcopy(img_resize)

        if self.has_amp:
            img_amp = self.get_sift_amp_grad_batch(img_resize)
            img_pad_batch = torch.cat((img_pad_batch, img_amp), dim=1)

        # # Origin
        # img_pad_size = (2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        # img_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [153, 57]
        
        
        # img_rotate = TF.rotate(img_resize, 45, InterpolationMode.BILINEAR)
        # print(img_rotate.shape)
        # img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])

        if train_flag:
            feats = self.descriptor_net(img_pad_batch) # bx8x38x14
        else:
            with torch.no_grad():
                self.descriptor_net.eval()
                feats = self.descriptor_net(img_pad_batch)      #  bx8x37x16
                # feats_rotate = self.descriptor_net(img_rotate)
                # del results_batch
        # with torch.no_grad():
        #     outs_map = self.descriptor_net(data, angle=flip_flag)
        #     del data

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 200
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 
        
        if not pt_norm_flag:
            keypoints_expand = keypoints_expand / keypoints_expand.new_tensor([wm - 5, hm - 1]).to(img_batch.device) * 2 - 1
      
        if fixed_angle >=-180 and fixed_angle < 180:
            orient_batch = torch.ones(b, 1, hm*(wm-4)) * fixed_angle
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            # orient_batch = self.get_orientation_batch(img_batch[:, :, :, 2:-2], device=img_batch.device)      # Bx1x(hm x(wm-4))
            orient_batch_T = self.get_orientation_batch(img_ext_batch[:, :, 3:-3, 8:-8], device=img_batch.device)      # Bx1x(hm x(wm-4))
        ori_expand = torch.zeros(max_num, device=img_batch.device)
        
        if sift_ori is not None:
            # print(sift_ori.shape, ori_expand[expand_mask_all.view(-1)==0].shape)
            orient_batch_kp = torch.cat((sift_ori.to(img_batch.device), ori_expand[expand_mask_all.view(-1)==0]), dim=-1).view(b, 1, 1, -1)                                 # nx1
            # print(orient_batch_kp.shape)
        else:
            orient_batch_kp = F.grid_sample(orient_batch_T.view(b, 1, hm, wm - 4),
                            keypoints_expand.view(b, 1, -1, 2),
                            mode='bilinear', align_corners=True)        # B X 1 X 1 X (max_num)
        
        # orient_batch_kp_T = F.grid_sample(orient_batch_T.view(b, 1, hm, wm - 4),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (max_num)

        # 切描述patch 
        downsampe_ratio = 4
        # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)

        # # 距离衰减核
        # coords = self.get_grid_coords(self.desc_patch)
        # dis_weight = torch.tensor(coords).to(img_batch.device).repeat(b*max_num, 8, 1).transpose(1, 2).contiguous().view(-1, 128) 
        # dis_weight_inv = torch.exp(-1 * dis_weight * dis_weight / 2) # 1 / dis_weight

        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 1
        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([wm - 5, hm - 1]) 
        keypoints_expand_mid[:, 0] += 2 + 6     # 36->52
        keypoints_expand_mid[:, 1] += 3     # 122->128
        keypoints_expand_mid = keypoints_expand_mid * keypoints_expand_mid.new_tensor([w / (w-1), h / (h - 1)])  # 128x52->129x53
        keypoints_expand_mid[:, 0] += self.sample_desc_patch
        keypoints_expand_mid[:, 1] += self.sample_desc_patch + 4
        keypoints_expand = (keypoints_expand_mid/ downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x41x17
        # print(feats_pad_batch.view(-1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]).shape)
        # print(keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2).shape)      
        # bx8xNx4x4

        if self.is_rec:
            if feats_pad_batch.shape[1] == 8:
                outs_kp = inv_warp_patch_batch_rec(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), torch.zeros_like(orient_batch_kp.repeat(1, 1, 8, 1).view(-1)) + cal_angle, patch_size=(8, 2), sample_factor=1).reshape(b, 8, -1, 8, 2).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
            else:
                outs_kp = inv_warp_patch_batch_rec(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 16, 1, 1).view(-1, 2), torch.zeros_like(orient_batch_kp.repeat(1, 1, 16, 1).view(-1)) + cal_angle, patch_size=(8, 2), sample_factor=1).reshape(b, 2, 8, -1, 8, 2).permute(0, 3, 1, 4, 5, 2).contiguous().view(-1, 256)  
        else:
            if feats_pad_batch.shape[1] == 8:
                outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
            else:
                outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 16, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 16, 1).view(-1) + cal_angle, self.desc_patch, self.desc_patch).reshape(b, 2, 8, -1, 4, 4).permute(0, 3, 1, 4, 5, 2).contiguous().view(-1, 256)  

        # print(outs_kp.dtype, dis_weight_inv.dtype)
        # outs_kp = outs_kp * dis_weight_inv
        outs_kp = L2Norm()(outs_kp)[expand_mask_all==1, :].float()
        # print(outs_kp[0, :])
        # exit()
        # outs_map = outs.reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w) 
        # print(outs_kp.shape, outs_map.shape, valid_all_mask.shape)
        return outs_kp  

    # patch AT 扩边图 siftori校准 onlyP
    def cut_patch_unfold_patch_map_interpolation_aligned_batch_ext93_onlyP_scoresmap(self, img_batch, sub_pixel=False):
        descriptor_map, scores_map = self.extract_dense_map(img_batch[:, :, 1:-1, :])    # 120 x 40
        # keypoints, _, scores, scoredispersitys = self.dkd(scores_map, descriptor_map,
        #                                                     sub_pixel=sub_pixel)
        return scores_map
    
    # patch AT 扩边图 siftori校准 onlyP
    def cut_patch_unfold_patch_map_interpolation_aligned_batch_ext93_onlyP_scoresmap_9800(self, img_batch, sub_pixel=False):
        # print(img_batch.shape)
        descriptor_map, scores_map = self.extract_dense_map(img_batch)    # 128 x 40
        # keypoints, _, scores, scoredispersitys = self.dkd(scores_map, descriptor_map,
        #                                                     sub_pixel=sub_pixel)
        return scores_map
    
     # patch AT 扩边图 siftori校准 onlyP
    def cut_patch_unfold_patch_map_interpolation_aligned_batch_ext93_onlyP(self, img_batch, sub_pixel=False):
        b, c, h, w = img_batch[:, :, 1:-1, :].shape
        descriptor_map, scores_map = self.extract_dense_map(img_batch[:, :, 1:-1, :])    # 120 x 40
        keypoints, _, scores, scoredispersitys = self.dkd(scores_map, descriptor_map, sub_pixel=sub_pixel)
        assert len(keypoints) == 1
        keypoints_xy =  (keypoints[0] + 1) / 2 * keypoints[0].new_tensor([w - 1, h - 1])
        return keypoints_xy.transpose(0, 1)

    # match
    def get_match_predict(self, data):

        with torch.no_grad():
            pred_match = self.match_net(data)         # 1 x 1x（NA + 1）x (NB + 1)
        pred_match = pred_match.squeeze()[:-1, :-1]   

        # show_result = find_match_idx_score(pred_match.unsqueeze(0), match_thr=self.match_thr)
        # show_pred = show_result['matches0']
        return  pred_match
    
    # match
    def get_match_predict_all(self, data):

        with torch.no_grad():
            pred_match = self.match_net(data)         # 1 x 1x（NA + 1）x (NB + 1)
        pred_match = pred_match.squeeze()   

        # show_result = find_match_idx_score(pred_match.unsqueeze(0), match_thr=self.match_thr)
        # show_pred = show_result['matches0']
        return  pred_match
    
        # match
    def get_match_predict_dict(self, data):

        with torch.no_grad():
            pred_match_dict = self.match_net(data)         # 1 x 1x（NA + 1）x (NB + 1)
        # pred_match = pred_match.squeeze()   

        # show_result = find_match_idx_score(pred_match.unsqueeze(0), match_thr=self.match_thr)
        # show_pred = show_result['matches0']
        return  pred_match_dict


    def forward(self, img, image_size_max=99999, sort=False, sub_pixel=False):
        """
        :param img: np.array HxWx3, RGB
        :param image_size_max: maximum image size, otherwise, the image will be resized
        :param sort: sort keypoints by scores
        :param sub_pixel: whether to use sub-pixel accuracy
        :return: a dictionary with 'keypoints', 'descriptors', 'scores', and 'time'
        """
        B, C, H, W = img.shape
        # assert three == 3, "input image shape should be [HxWx3]"

        # # ==================== image size constraint
        # image = deepcopy(img)
        # max_hw = max(H, W)
        # if max_hw > image_size_max:
        #     ratio = float(image_size_max / max_hw)
        #     image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)

        # # ==================== convert image to tensor
        # image = torch.from_numpy(image).to(self.device).to(torch.float32).permute(2, 0, 1)[None] / 255.0

        # ==================== extract keypoints
        start = time.time()
        if self.phase == 'train':
            descriptor_map, scores_map = self.extract_dense_map(img)
            keypoints, _, scores, scoredispersitys = self.dkd(scores_map, descriptor_map,
                                                                sub_pixel=sub_pixel)
            # descriptors = self.cut_patch(img, keypoints, self.desc_patch, True)
            # descriptors = self.cut_patch_unfold(img, keypoints, True)
            descriptors = self.cut_patch_unfold_interpolation(img, keypoints, True)
            # keypoints, descriptors, scores = keypoints, descriptors, scores
            # print(len(keypoints), keypoints[0].shape, descriptors[0].shape, scores[0].shape)
            # keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W - 1, H - 1]])    # 归一化点坐标
            pass
        else:
            with torch.no_grad():
                descriptor_map, scores_map = self.extract_dense_map(img)
                keypoints, descriptors, scores, scoredispersitys = self.dkd(scores_map, descriptor_map,
                                                                sub_pixel=sub_pixel)
                keypoints, descriptors, scores = keypoints[0], descriptors[0], scores[0]
                keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W - 1, H - 1]])    # 归一化点坐标->尺寸坐标

        # if sort:
        #     indices = torch.argsort(scores, descending=True)        # 置信度降序排序 
        #     keypoints = keypoints[indices]
        #     descriptors = descriptors[indices]
        #     scores = scores[indices]

        end = time.time()

        return {'keypoints': keypoints,
                'descriptors': descriptors,
                'descriptor_map': descriptor_map,
                'scores': scores,
                'scores_map': scores_map,
                'scoredispersitys': scoredispersitys,
                'time': end - start, }


if __name__ == '__main__':
    import numpy as np
    from thop import profile

    net = ALikeWithHard(c1=32, c2=64, c3=128, c4=128, dim=128, single_head=False)

    image = np.random.random((640, 480, 3)).astype(np.float32)
    flops, params = profile(net, inputs=(image, 9999, False), verbose=False)
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))
