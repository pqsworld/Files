import logging
import os
import cv2
import torch
import torch.nn as nn

from copy import deepcopy
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import math
import numpy as np
import random
from collections import OrderedDict

from models.ALNet import ALNet, ALNet_New, ALNet_Angle, ALNet_Angle_New, ALNet_Angle_Distill, ALNet_Angle_Deep, ALNet_Angle_Deform
from models.modules import DKD, DKDR
from models.hardnet_model import HardNet_fast_twice_half3_short, HardNet_fast_twice_half3_MO_MOE, HardNet_fast_half_MOE, HardNet_fast_twice_big_MO, HardNet_small, HardNet, HardNet_smaller, HardNet_smaller_patch, HardNet_Sift, HardNet_fast, HardNet_fast_half, HardNet_fast_twice_half3_MO, HardNet_fast_twice_half3_MOA, HardNet_fast_twice, HardNet_fast_twice_half, HardNet_fast_twice_half3, HardNet_fast_big, HardNet_fast_big_ap, HardNet_fast_big_mp, HardNet_fast_big_Deform, HardNet_fast_twice_half_cr, HardNet_fast_twice_half2, HardNet_fast_twice_half_cr2, HardNet_fast_twice_half_norm2, HardNet_fast_twice_big, HardNet_fast_twice_big_BTL, HardNet_fast_twice_big2, HardNet_fast_Pconv, HardNet_Sift_Deform, HardNet_fast_deform, HardNet_fast_deform_one, HardNet_fast_deform_last, HardNet_fast_deform_offset, HardNet_fast_Group,  HardNet_fast_twice_big_vanila, HardNet_fast_twice_half3_vanila, HardNet_fast_twice_mid_vanila, HardNet_fast_half_MO, L2Norm
from models.hardnet_model_quant import HardNet_fast_quant
from models.PTQ.ptq_config import Config
# from utils.loss_functions.sparse_loss import D
from utils.utils import inv_warp_image, batch_inv_warp_image, inv_warp_patch_batch, inv_warp_patch_batch_rec, warp_points, filter_points
from utils.homographies import sample_homography_cv
import time
# from mmcv.cnn import constant_init, kaiming_init

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

def sample_descriptor(descriptor_map, kpts, bilinear_interp=False):
    """
    :param descriptor_map: BxCxHxW
    :param kpts: list, len=B, each is Nx2 (keypoints) [h,w]
    :param bilinear_interp: bool, whether to use bilinear interpolation
    :return: descriptors: list, len=B, each is NxD
    """
    batch_size, channel, height, width = descriptor_map.shape

    descriptors = []
    for index in range(batch_size):
        kptsi = kpts[index]  # Nx2,(x,y)

        if bilinear_interp:
            descriptors_ = F.grid_sample(descriptor_map[index].unsqueeze(0), kptsi.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, :, 0, :]  # CxN
        else:
            kptsi = (kptsi + 1) / 2 * kptsi.new_tensor([[width - 1, height - 1]])
            kptsi = kptsi.long()
            descriptors_ = descriptor_map[index, :, kptsi[:, 1], kptsi[:, 0]]  # CxN

        # descriptors_ = F.normalize(descriptors_, p=2, dim=0)
        descriptors.append(descriptors_.t())

    return descriptors

# FGD
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

# ALNet_Angle_New ALNet_Angle ALNet_Angle_Distill ALNet_Angle_Deep ALNet_Angle_Deform
class ALikeKptAngle(ALNet_Angle):
    def __init__(self,
                 # ================================== feature encoder
                 c1: int = 8, c2: int = 16, c3: int = 32, c4: int = 64, dim: int = 32,
                 # c1: int = 4, c2: int = 8, c3: int = 16, c4: int = 32, dim: int = 16,
                 # c1: int = 16, c2: int = 32, c3: int = 64, c4: int = 128, dim: int = 64,
                 # c1: int = 8, c2: int = 8, c3: int = 16, c4: int = 32, dim: int = 16,
                 single_head: bool = True, # 原来用的False，峰值空间太大
                 # ================================== detect parameterss
                 radius: int = 2,   # 2
                 top_k: int = -1, scores_th: float = 0.2,      # default: 1 / ((2*raduis+1) * (2*raduis+1))
                 n_limit: int = 200,  # 400, 130, 80
                 device: str = 'cpu',
                 model_path: str = '',
                 phase: str = 'train'
                 ):
        super().__init__(c1, c2, c3, c4, dim, single_head)
        self.radius = radius            # nms radius
        self.top_k = top_k
        self.n_limit = n_limit
        self.scores_th = scores_th
        self.dkd = DKD(radius=self.radius, top_k=self.top_k,
                       scores_th=self.scores_th, n_limit=self.n_limit)
        # self.descriptor_net = nn.DataParallel(HardNet_smaller(train_flag=(phase == 'train')), device_ids=[0, 1, 2])# HardNet_small(train_flag=(phase == 'train'))  # HardNet(train_flag=(phase == 'train'))   
        # self.descriptor_net = nn.DataParallel(HardNet_smaller_patch(train_flag=(phase == 'train')), device_ids=[0, 1, 2])
        # self.descriptor_net = nn.DataParallel(HardNet_Sift(train_flag=(phase == 'train')), device_ids=[0, 1, 2])
        # self.descriptor_net = nn.DataParallel(HardNet_fast(train_flag=(phase == 'train')), device_ids=[0, 1, 2])

        # patch
        # self.descriptor_net = HardNet_fast(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half2(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half3(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half3_MO(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half3_MO_MOE(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half3_MOA(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half_cr(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half_cr2(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_half_norm2(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_big(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_twice_big_MO(train_flag=(phase == 'test'))
        # # self.descriptor_net = HardNet_fast_twice_half3_vanila(train_flag=(phase == 'train'))
        # # self.descriptor_net = HardNet_fast_twice_mid_vanila(train_flag=(phase == 'train'))
        # # self.descriptor_net = HardNet_fast_twice_big_BTL(train_flag=(phase == 'train'))
        # # self.descriptor_net = HardNet_fast_big(train_flag=(phase == 'train'))
        # # self.descriptor_net = HardNet_fast_big_ap(train_flag=(phase == 'train'))
        # # self.descriptor_net = HardNet_fast_big_mp(train_flag=(phase == 'train'))
        # # self.descriptor_net = HardNet_fast_big_Deform(train_flag=(phase == 'train'))
        # # self.descriptor_net = HardNet_fast_half(train_flag=(phase == 'train'))
        # # self.descriptor_net = HardNet_fast_half_MOE(train_flag=(phase == 'train'))
        # # self.descriptor_net = HardNet_fast_half_MO(train_flag=(phase == 'train'))
        # # self.descriptor_net = HardNet_fast_twice_big2(train_flag=(phase == 'train'))
        # # self.descriptor_net = HardNet_fast_Pconv(train_flag=(phase == 'train'))

        # # 描述教师模型
        self.descriptor_tea_net = HardNet_fast_twice_half3_short(train_flag=(phase == 'test'))
        # HardNet_fast_twice_big_MO(train_flag=(phase == 'test'))
        teacher_net_weight = './checkpoints/Des/93061_short.pth.tar'
        # '/hdd/file-input/linwc/Descriptor/code/Test_sys/checkpoints/Des/0703m_ALikeWithHard_patch_group_inner_re_smallc4_ext_rt_nocut_nort_ne_58400_checkpoint.pth.tar'

        # '/hdd/file-input/linwc/Descriptor/code/Test_sys/checkpoints/Des/0522m_ALikeWithHard_patch_group_inner_re_smallc4_ext_rt_nocut_nort_ne_234400_checkpoint.pth.tar'
        # '/hdd/file-input/linwc/Descriptor/code/Test_sys/checkpoints/Des/0614a_ALikeWithHard_patch_group_inner_re_smallc4_ext_rt_nocut_nort_ne_132400_checkpoint.pth.tar'
        # '/hdd/file-input/linwc/Descriptor/code/Test_sys/checkpoints/Des/0602a_ALikeWithHard_patch_group_inner_re_smallc4_ext_rt_nocut_nort_ne_115800_checkpoint.pth.tar'
        # '/hdd/file-input/linwc/Descriptor/code/Test_sys/checkpoints/Des/0522a_ALikeWithHard_patch_group_inner_re_smallc4_ext_rt_nocut_nort_ne_238400_checkpoint.pth.tar'
        # '/hdd/file-input/linwc/Descriptor/code/Test_sys/checkpoints/Des/0522m_ALikeWithHard_patch_group_inner_re_smallc4_ext_rt_nocut_nort_ne_32000_checkpoint.pth.tar'
        # '/hdd/file-input/linwc/Descriptor/code/Test_sys/checkpoints/Des/0505m_ALikeWithHard_patch_group_inner_re_smallc4_ext_rt_nocut_nort_ne_33200_checkpoint.pth.tar'
        checkpoint_teacher = torch.load(teacher_net_weight, map_location=lambda storage, loc: storage)

        # NoneDP_param = OrderedDict()
        # for k in checkpoint_teacher['model_state_dict'].keys():
        #     if 'descriptor_net' in k:
        #         NoneDP_param[k.replace('.module', '').replace('descriptor_net.', '')] = checkpoint_teacher['model_state_dict'][k]
        # self.descriptor_tea_net.load_state_dict(NoneDP_param)

        self.descriptor_tea_net.load_state_dict(checkpoint_teacher['model_state_dict'])
        self.descriptor_tea_net.eval()

        self.has_point_teacher = False # False
        if self.has_point_teacher:
            # 点教师模型
            self.point_tea_net = ALNet_Angle_Distill(c1=16, c2=32, c3=64, c4=128, dim=64, single_head=True)
            point_teacher_net_weight = '/home/linwc/match/code/Test_sys/checkpoints/Pnt/0816n_346000_checkpoint.pth.tar'
            point_checkpoint_teacher = torch.load(point_teacher_net_weight, map_location=lambda storage, loc: storage)
            Point_NoneDP_param = OrderedDict()
            for k in point_checkpoint_teacher['model_state_dict'].keys():
                if 'descriptor' not in k:
                    Point_NoneDP_param[k.replace('.module', '')] = point_checkpoint_teacher['model_state_dict'][k]
            self.point_tea_net.load_state_dict(Point_NoneDP_param)
            self.point_tea_net.eval()

            # FGD loss
            st_fea_dim = 16 # 8 16
            tea_fea_dim = 64
            self.point_fgd_loss = FeatureLoss(
                st_fea_dim, 
                tea_fea_dim, 
                temp=0.5,       # 斜率超参，论文里表明与性能相关性不强
                alpha_fgd=0.001,
                beta_fgd=0, # 0.0005,
                gamma_fgd=0.001,
                lambda_fgd=0.000005)

        # dense
        # self.descriptor_net = HardNet_fast_Ecnn(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_Ecnn_twice(train_flag=(phase == 'train'))
        # self.descriptor_net = HardNet_fast_Ecnn_third(train_flag=(phase == 'train'))

        # self.descriptor_net = nn.DataParallel(HardNet_Sift_Deform(train_flag=(phase == 'train')), device_ids=[0, 1, 2, 3])
        # self.descriptor_net =  nn.DataParallel(HardNet_fast_deform(train_flag=(phase == 'train')), device_ids=[0, 1, 2])
        # self.descriptor_net =  nn.DataParallel(HardNet_fast_deform_one(train_flag=(phase == 'train')), device_ids=[0, 1, 2])
        # self.descriptor_net =  nn.DataParallel(HardNet_fast_deform_last(train_flag=(phase == 'train')), device_ids=[0, 1, 2])
        # self.descriptor_net =  nn.DataParallel(HardNet_fast_deform_offset(train_flag=(phase == 'train')), device_ids=[0, 1, 2])
        # self.descriptor_net =  nn.DataParallel(HardNet_fast_Group(train_flag=(phase == 'train')), device_ids=[0, 1, 2])

        # cfg = Config(lis=False)
        # self.descriptor_net = nn.DataParallel(HardNet_fast_quant(cfg=cfg, train_flag=(phase == 'train')), device_ids=[0, 1, 2])

        # self.descriptor_net = nn.DataParallel(HardNet_fast_Ecnn(train_flag=(phase == 'train')), device_ids=[0, 1, 2])

        self.patch_cal = True # True

        self.desc_patch = 4 # 4 # 16
        self.sample_desc_patch = 22  # 18 # 32 # 12 # 6 # 4  # 32
        self.rect_patch_size = 32 * 8 # 16 * 16
        self.rpatch = (32, 8) # (16, 16)
        self.desc_patch_expand = 4 # 12 # 6 # 8  # 48
        self.is256 = True
        self.has45 = True
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

        # """
        # dis_weight:
        #     s1 s2 s2 s1
        #     s2 s3 s3 s2  x 8
        #     s2 s3 s3 s2
        #     s1 s2 s2 s1
        # """
        # dis_sigma1 = nn.Parameter(torch.ones(1, 8))
        # dis_sigma2 = nn.Parameter(torch.ones(1, 8))
        # dis_sigma3 = nn.Parameter(torch.ones(1, 8))
        # dis_sigma_one = torch.cat((dis_sigma1, dis_sigma2.repeat(2, 1), dis_sigma1), dim=0)
        # dis_sigma_two = torch.cat((dis_sigma2, dis_sigma3.repeat(2, 1), dis_sigma2), dim=0)
        # dis_sigma_weight = torch.cat((dis_sigma_one, dis_sigma_two, dis_sigma_two, dis_sigma_one), dim=0).view(1, -1)
        # self.register_buffer('dis_sigma_weight', dis_sigma_weight)

        # # FGD loss
        # fea_dim = 16 # 8 16
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
        #     alpha_mgd=0.0002, # 0.002,
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
        if self.has_point_teacher:
            scores_map, descriptor_map, mid_fea = super().forward(image)
        else:
            scores_map, descriptor_map = super().forward(image)
            mid_fea = None

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
            return descriptor_map, scores_map, mid_fea

    def generate_homograhy_by_angle(self, H, W, angles):
        scale = 1
        M = [cv2.getRotationMatrix2D((W / 2, H / 2), i, scale) for i in angles]
        # center = np.mean(pts2, axis=0, keepdims=True)
        homo = [np.concatenate((m, [[0, 0, 1.]]), axis=0) for m in M]

        # valid = np.arange(n_angles)
        # idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        # homo = M[idx]

        return homo

    def get_orientation(self, img, keypoints=None, device='cpu'):
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
        Gxx_unfold = self.patch_unfold_orient(torch.tensor(Gxx).unsqueeze(0).unsqueeze(0).to(device)).view(1, -1, h, w)
        Gyy_unfold = self.patch_unfold_orient(torch.tensor(Gyy).unsqueeze(0).unsqueeze(0).to(device)).view(1, -1, h, w)
        Gxy_unfold = self.patch_unfold_orient(torch.tensor(Gxy).unsqueeze(0).unsqueeze(0).to(device)).view(1, -1, h, w)
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
            return angle_all.view(1, 1, -1)
        else:
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


        return H_p.view(batch, -1)

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


    def get_big_mask(self, ori_batch, trans_rot=None):
        # ori_batch: b, 1, n:
        # print(trans_rot.shape)
        ori_batch_after = ori_batch - trans_rot.unsqueeze(1).unsqueeze(1).repeat(1, 1, ori_batch.shape[-1])
        ori_batch_after[ori_batch_after < 0] = ori_batch_after[ori_batch_after < 0] + 360   # [-270, 0] -> [90, 360]; [0, 270] -> [0, 270]
        cond = torch.logical_and(ori_batch_after > 90, ori_batch_after < 270).int()
        # print(cond.shape)
        return cond

    def transform_keypoints(self, keypoints_expand, w, h, we, he, wm, hm):
        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1])
        keypoints_expand_mid[:, 0] += (we - w) / 2              # 6
        keypoints_expand_mid[:, 1] += (he - h) / 2              # 4
        keypoints_expand_resize = keypoints_expand_mid * keypoints_expand_mid.new_tensor([we / (we - 1), he / (he - 1)])  # 129x53
        keypoints_expand_resize[:, 0] += self.sample_desc_patch
        keypoints_expand_resize[:, 1] += self.sample_desc_patch + 4
        keypoints_expand_resize_norm = keypoints_expand_resize / keypoints_expand_resize.new_tensor([wm - 1, hm - 1]).to(keypoints_expand_resize.device) * 2 - 1
        return keypoints_expand_resize, keypoints_expand_resize_norm

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

    def cut_patch_unfold(self, img_batch, points, train_flag=False):
        b, _, h, w = img_batch.shape
        results = None 
        # Padding Zero
        # homography = sample_homography_cv(self.desc_patch, self.desc_patch, max_angle=180, n_angles=360)
        # homography = torch.tensor(homography, dtype=torch.float32)
        img_unfold_batch = self.patch_unfold(img_batch).view(b, -1, h+1, w+1)[:, :, :-1, :-1]     # Bx(patch_size x patch_size) x H x W
        for batch_idx in range(b):
            keypoints = points[batch_idx]
            keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]]).to(keypoints.device)
            img_patches = img_unfold_batch[batch_idx].transpose(0, 2)     # w x h x (patch_size x patch_size)
            # print(img_patches_rotate.shape)
            keypoints = (keypoints + 0.5).long()

            # no rotate
            data = img_patches[keypoints[:, 0], keypoints[:, 1]].view(-1, self.desc_patch, self.desc_patch).unsqueeze(1)    # M x patch_size x patch_size
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

    def cut_patch_unfold_interpolation(self, img_batch, points, train_flag=False):
        b, _, h, w = img_batch.shape
        results = None 
        # Padding Zero
        # homography = sample_homography_cv(self.desc_patch, self.desc_patch, max_angle=180, n_angles=360)
        # homography = torch.tensor(homography, dtype=torch.float32)
        img_unfold_batch = self.patch_unfold(img_batch).view(b, -1, h+1, w+1)     # Bx(patch_size x patch_size) x (H + 1) x (W + 1)
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_offset = keypoints * keypoints.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
            img_patches = F.grid_sample(img_unfold_batch[batch_idx].unsqueeze(0),
                                    keypoints_offset.view(1, 1, -1, 2),
                                    mode='bilinear', align_corners=True).squeeze(2).squeeze(0)
            # no rotate
            data = img_patches.transpose(0, 1).view(-1, self.desc_patch, self.desc_patch).unsqueeze(1)  # M x 1 x patch_size x patch_size
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
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
        # outs = torch.chunk(outs, b, dim=0)
    
        return outs.squeeze(-1)   

    def cut_patch_unfold_interpolation_aligned(self, img_batch, points, train_flag=False):
        b, _, h, w = img_batch.shape
        results = None 
        # Padding Zero
        # homography = sample_homography_cv(self.desc_patch, self.desc_patch, max_angle=180, n_angles=360)
        # homography = torch.tensor(homography, dtype=torch.float32)
        img_unfold_batch_expand = self.patch_unfold_expand(img_batch).view(b, -1, h+1, w+1)     # Bx(expand_patch_size x expand_patch_size) x (H + 1) x (W + 1)
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_offset = keypoints * keypoints.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
            img_patches = F.grid_sample(img_unfold_batch_expand[batch_idx].unsqueeze(0),
                                    keypoints_offset.view(1, 1, -1, 2),
                                    mode='bilinear', align_corners=True).squeeze(2).squeeze(0)      # (expand_patch_size x expand_patch_size) x M
            orientation_patch = self.get_orientation(img_batch[batch_idx].squeeze().cpu().numpy(), keypoints, keypoints.device)
            homography_expand_patch = self.generate_homograhy_by_angle(self.desc_patch_expand, self.desc_patch_expand, -orientation_patch.squeeze(0).detach().cpu().numpy())
            homography_expand_patch = torch.tensor(homography_expand_patch, dtype=torch.float32)        # Mx3x3
            data_expand = img_patches.transpose(0, 1).view(-1, self.desc_patch_expand, self.desc_patch_expand).unsqueeze(1) # M x 1 x expand_patch_size x expand_patch_size
            data_expand_rotate = batch_inv_warp_image(data_expand.cpu(), homography_expand_patch, mode="bilinear").to(data_expand.device)
            
            # 24x24 -> 16x16
            data = data_expand_rotate[:, :, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2].contiguous()
  
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

    def cut_patch_unfold_map_interpolation(self, img_batch, train_flag=False):
        b, _, h, w = img_batch.shape
        # Padding Zero
        img_unfold_batch = self.patch_unfold(img_batch).view(b, -1, h+1, w+1)    # Bx(patch_size x patch_size)x H x W 
        x = torch.linspace(0, w-1, w)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        # print(mesh_points.shape, mesh_points)
        mesh_grid = mesh_points / mesh_points.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
        keypoints_offset = mesh_grid * mesh_grid.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
        # print(keypoints_offset.shape)
        data = F.grid_sample(img_unfold_batch,
                        keypoints_offset.unsqueeze(0).repeat(b, 1, 1, 1),
                        mode='bilinear', align_corners=True)            # bx(patch_size x patch_size)x1x(hxw)
        results = data.transpose(1, 3).reshape(-1, self.desc_patch, self.desc_patch).unsqueeze(1).contiguous()
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
                del results_batch
        outs = outs.squeeze(-1).reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w)          # B X dim x H X W
        return outs  

    def cut_patch_unfold_map_interpolation_aligned(self, img_batch, train_flag=False):
        b, _, h, w = img_batch.shape
        # Padding Zero
        img_unfold_batch = self.patch_unfold_expand(img_batch).view(b, -1, h+1, w+1)    # Bx(expand_patch_size x expand_patch_size)x H x W 
        x = torch.linspace(0, w-1, w)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        # print(mesh_points.shape, mesh_points)
        mesh_grid = mesh_points / mesh_points.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
        keypoints_offset = mesh_grid * mesh_grid.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
        # print(keypoints_offset.shape)
        img_patches = F.grid_sample(img_unfold_batch,
                        keypoints_offset.unsqueeze(0).repeat(b, 1, 1, 1),
                        mode='bilinear', align_corners=True)        # B X (expand_patch_size x expand_patch_size) X 1 X (HXW)

        orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        homography_expand_patch = self.generate_homograhy_by_angle(self.desc_patch_expand, self.desc_patch_expand, -orient_batch.view(-1).squeeze().detach().cpu().numpy())
        homography_expand_patch_all = torch.tensor(homography_expand_patch, dtype=torch.float32)        # (BxM)x3x3
        data_expand = img_patches.transpose(1, 3).reshape(-1, self.desc_patch_expand, self.desc_patch_expand).unsqueeze(1) # (BxM) x 1 x expand_patch_size x expand_patch_size
        data_expand_rotate = inv_warp_image(data_expand.cpu(), homography_expand_patch_all)
        # data_expand_rotate = batch_inv_warp_image(data_expand.cpu(), homography_expand_patch_all, mode="bilinear").to(data_expand.device)    
        # 24x24 -> 16x16
        data = data_expand_rotate[:, :, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2].contiguous()
        # print(data.shape)
        # print(img_unfold_batch.shape)
        # results = data.permute(0, 2, 3, 1).reshape(-1, 1, self.desc_patch, self.desc_patch)   
        results = data   
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
                del results_batch
        outs = outs.squeeze(-1).reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w)          # B X dim x H X W
        return outs      

    def cut_patch_unfold_patch_map_interpolation_aligned(self, img_batch,  points, train_flag=False):
        b, _, h, w = img_batch.shape
        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 250
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
    
        orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X (expand_patch_size x expand_patch_size) X 1 X (HXW)
        homography_expand_map = self.generate_homograhy_by_angle(self.desc_patch_expand, self.desc_patch_expand, -orient_batch.view(-1).squeeze().detach().cpu().numpy())
        homography_expand_map_all = torch.tensor(homography_expand_map, dtype=torch.float32)        # (BxHxW)x3x3
        homography_expand_patch = self.generate_homograhy_by_angle(self.desc_patch_expand, self.desc_patch_expand, -orient_batch_kp.view(-1).squeeze().detach().cpu().numpy())
        homography_expand_patch_all = torch.tensor(homography_expand_patch, dtype=torch.float32)        # (BxM)x3x3
        
        # Padding Zero
        img_unfold_batch = self.patch_unfold_expand(img_batch).view(b, -1, h+1, w+1)    # Bx(expand_patch_size x expand_patch_size)x H x W 
        x = torch.linspace(0, w-1, w)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        # print(mesh_points.shape, mesh_points)
        mesh_grid = mesh_points / mesh_points.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
        mesh_grid_offset = mesh_grid * mesh_grid.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
        keypoints_expand_offset = keypoints_expand * keypoints_expand.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
        # print(keypoints_offset.shape)
        img_patches = F.grid_sample(img_unfold_batch,
                        mesh_grid_offset.unsqueeze(0).repeat(b, 1, 1, 1),
                        mode='bilinear', align_corners=True)        # B X (expand_patch_size x expand_patch_size) X 1 X (HXW)
        
        img_patches_kp = F.grid_sample(img_unfold_batch,
                        keypoints_expand_offset.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X (expand_patch_size x expand_patch_size) X 1 X M

        data_expand = img_patches.transpose(1, 3).reshape(-1, self.desc_patch_expand, self.desc_patch_expand).unsqueeze(1) # (BxHxW) x 1 x expand_patch_size x expand_patch_size
        data_expand_rotate = batch_inv_warp_image(data_expand.cpu(), homography_expand_map_all, device=data_expand.device)
        data_expand_kp = img_patches_kp.transpose(1, 3).reshape(-1, self.desc_patch_expand, self.desc_patch_expand).unsqueeze(1) # (BxM) x 1 x expand_patch_size x expand_patch_size
        data_expand_rotate_kp = batch_inv_warp_image(data_expand_kp.cpu(), homography_expand_patch_all, device=data_expand_kp.device)
        # data_expand_rotate = batch_inv_warp_image(data_expand.cpu(), homography_expand_patch_all, mode="bilinear").to(data_expand.device)    
        # 24x24 -> 16x16
        data = data_expand_rotate[:, :, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2].contiguous()
        data_kp = data_expand_rotate_kp[expand_mask_all==1,  :, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2].contiguous()
        
        # # 画校准后的patch
        # for count in range(data_kp.shape[0]):
        #     if count < 100:
        #         cv2.imwrite('demo/demo_patch_mesh_grid_' + str(count) + '.bmp', 255*data[count, 0, :, :].detach().cpu().numpy())
        #         cv2.imwrite('demo/demo_patch_kp_' + str(count) + '.bmp', 255*data_kp[count, 0, :, :].detach().cpu().numpy())
                
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
                outs = self.descriptor_net(results_batch)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w)          # B X dim x H X W
        return outs.squeeze(-1), outs_map     

    def cut_patch_unfold_patch_map_interpolation_aligned_batch(self, img_batch, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape
        pad_size = (self.desc_patch_expand // 2, self.desc_patch_expand // 2, self.desc_patch_expand // 2, self.desc_patch_expand // 2)
        img_pad_batch = F.pad(img_batch, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 400
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
        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            # print(orient_batch)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)

        # orient_batch = orient_batch + big_mask * 180
        # orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)

        # new 

        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch + big_mask * 180
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(0, w-1, w)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + self.desc_patch_expand // 2
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device) + self.desc_patch_expand // 2
        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # 随机±5度增强，兼容sift计算的角度
        rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        # rand_angle_enhance = 0
        # print(flip_flag)
        data = inv_warp_patch_batch(img_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, self.desc_patch, self.sample_desc_patch).unsqueeze(1)   
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        # print(valid_all_mask.shape, pmask_kp.shape)
        data_kp = inv_warp_patch_batch(img_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, self.desc_patch, self.sample_desc_patch)[valid_all_mask.bool(), :, : ].unsqueeze(1)   

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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w)          # B X dim x H X W

        return outs.squeeze(-1), outs_map, valid_all_mask     

    def cut_patch_unfold_patch_map_interpolation_aligned_batch93(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 14
        h_pad = 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            # # 点转到AT上
            # keypoints_T = warp_points(keypoints_ori[:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            # kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            # kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            # keypoints_T = keypoints_T[kmask_T]
            # keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1

            # keypoints = keypoints_ori[kmask_T]
            # keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            # keypoints_new.append(keypoints)
            # kmask_T_all.append(kmask_T)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
            orient_batch_ext = self.get_orientation_batch(img_ext_pad_batch, device=img_batch.device)      # Bx1x(haxwa)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)
            big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 20         # 40 -> 80
        mesh_points[:, 1] += 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 20         # 40 -> 80
        keypoints_expand[:, 1] += 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        data = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16

        # 4x4主方向
        ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16

        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data, ori_data_kp, bin_data, bin_data_kp     

    # patch
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 7 # 7 # 14
        h_pad = 12 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            # # 点转到AT上
            # keypoints_T = warp_points(keypoints_ori[:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            # kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            # kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            # keypoints_T = keypoints_T[kmask_T]
            # keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1

            # keypoints = keypoints_ori[kmask_T]
            # keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            # keypoints_new.append(keypoints)
            # kmask_T_all.append(kmask_T)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
            # orient_batch_ext = self.get_orientation_batch(img_ext_pad_batch, device=img_batch.device)      # Bx1x(haxwa)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 13 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 16 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 13 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 16 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        data = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch)
        ori_data_kp = orient_batch_kp.view(-1).unsqueeze(1)[valid_all_mask.bool(), :]           

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data, ori_data_kp, bin_data, bin_data_kp     

    # patch AT
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_AT(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 7 # 7 # 14
        h_pad = 12 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            # # 点转到AT上
            # keypoints_T = warp_points(keypoints_ori[:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            # kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            # kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            # keypoints_T = keypoints_T[kmask_T]
            # keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1

            # keypoints = keypoints_ori[kmask_T]
            # keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            # keypoints_new.append(keypoints)
            # kmask_T_all.append(kmask_T)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hmxwm)
            
            # orient_batch_ext = self.get_orientation_batch(img_ext_pad_batch, device=img_batch.device)      # Bx1x(haxwa)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        orient_batch = orient_batch_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        
        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, hm, wm)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36

        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 13 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 16 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 13 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 16 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        data = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)         

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()     

    # patch AT 扩边图算角度
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_AT_ext(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 7 # 7 # 14
        h_pad = 12 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            # # 点转到AT上
            # keypoints_T = warp_points(keypoints_ori[:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            # kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            # kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            # keypoints_T = keypoints_T[kmask_T]
            # keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1

            # keypoints = keypoints_ori[kmask_T]
            # keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            # keypoints_new.append(keypoints)
            # kmask_T_all.append(kmask_T)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
            
            # orient_batch_ext = self.get_orientation_batch(img_ext_pad_batch, device=img_batch.device)      # Bx1x(haxwa)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        orient_batch = orient_batch_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        
        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, hm, wm)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36

        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 13 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 16 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 13 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 16 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        data = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)         

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()     

    # patch AT 扩边图 siftori校准
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 7 # 4 # 7 # 14
        h_pad = 12 # 9 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            # sift角度
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        orient_batch = orient_batch_sift_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36

        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, hm, wm)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 13 # 10 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 16 # 13 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 13 # 10 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 16 # 13 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        # # 随机20%切变角度 [-10°, 10°]
        # if trans_rot_angle is not None:
        #     rand_shear_ori = torch.rand(orient_batch_kp.view(-1).shape[0], device=orient_batch.device) * 20 - 10
        #     rand_shear = torch.where(torch.rand(orient_batch_kp.view(-1).shape[0], device=orient_batch.device) < 0.2, rand_shear_ori, torch.zeros_like(rand_shear_ori, device=rand_shear_ori.device))
        # else:
        #     rand_shear = torch.zeros_like(orient_batch_kp.view(-1), device=orient_batch.device)

        data = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        # data_low = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1)   # BHWX1X16X16
        # data = torch.cat((data_low, data_high), dim=1)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)               

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()    

    # patch AT siftori校准
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_onlyP(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 120 x 40  128 x 56
        w_pad = 7 # 4 # 7 # 14
        h_pad = 12 # 9 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([4, 1], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-5, hp-2], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # # 128x56 -> 120x40
            # keypoints_ori[:, 0] -= 8
            # keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            # sift角度
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        orient_batch = orient_batch_sift_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36

        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, hm, wm)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 13 # 10 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 16 # 13 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 13 # 10 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 16 # 13 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        # # 随机20%切变角度 [-10°, 10°]
        # if trans_rot_angle is not None:
        #     rand_shear_ori = torch.rand(orient_batch_kp.view(-1).shape[0], device=orient_batch.device) * 20 - 10
        #     rand_shear = torch.where(torch.rand(orient_batch_kp.view(-1).shape[0], device=orient_batch.device) < 0.2, rand_shear_ori, torch.zeros_like(rand_shear_ori, device=rand_shear_ori.device))
        # else:
        #     rand_shear = torch.zeros_like(orient_batch_kp.view(-1), device=orient_batch.device)

        data = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        # data_low = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1)   # BHWX1X16X16
        # data = torch.cat((data_low, data_high), dim=1)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)               

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
        # # 画校准后的patch
        # for count in range(data_kp.shape[0]):            
        #     if count < 100:
        #         cv2.imwrite('demo/demo_patch_mesh_grid_new_' + str(count) + '.bmp', 255*data[count, 0, :, :].detach().cpu().numpy())
        #         cv2.imwrite('demo/demo_patch_kp_new_' + str(count) + '.bmp', 255*data_kp[count, 0, :, :].detach().cpu().numpy())
        # exit()
        # print(data.shape)
        # print(img_unfold_batch.shape)
        # results = data.permute(0, 2, 3, 1).reshape(-1, 1, self.desc_patch, self.desc_patch)   
        # results = data_kp
        # # compute output for patch a
        # results_batch = Variable(results)   
        # print(results_batch.shape)
        with torch.no_grad():
            outs = self.descriptor_tea_net(data_kp, angle=flip_flag)
            # del results_batch
            outs_map = self.descriptor_tea_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()    

    # patch AT 扩边图 siftori校准
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_onlyP(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 40  128 x 56
        w_pad = 16 # 4 # 7 # 14
        h_pad = 16 # 9 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([0, 0], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-1, hp-1], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x40-> 120x40
            # keypoints_ori[:, 0] -= 8
            # keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            # sift角度
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm).contiguous().view(b, 1, -1) # orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm).contiguous().view(b, 1, -1)  # orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        orient_batch = orient_batch_sift_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w).contiguous() # orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36

        orient_batch_ori = orient_batch_ori.view(b, 1, h, w).contiguous()    # orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, hm, wm).contiguous()   # orient_batch_T.view(b, 1, hm, wm)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(0, w-1, w)# torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += w_pad  #  13 # 10 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += h_pad  # 16 # 13 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += w_pad # 13 # 10 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += h_pad # 16 # 13 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        # # 随机20%切变角度 [-10°, 10°]
        # if trans_rot_angle is not None:
        #     rand_shear_ori = torch.rand(orient_batch_kp.view(-1).shape[0], device=orient_batch.device) * 20 - 10
        #     rand_shear = torch.where(torch.rand(orient_batch_kp.view(-1).shape[0], device=orient_batch.device) < 0.2, rand_shear_ori, torch.zeros_like(rand_shear_ori, device=rand_shear_ori.device))
        # else:
        #     rand_shear = torch.zeros_like(orient_batch_kp.view(-1), device=orient_batch.device)

        data = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        # data_low = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1)   # BHWX1X16X16
        # data = torch.cat((data_low, data_high), dim=1)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)               

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w) # bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
        # # 画校准后的patch
        # for count in range(data_kp.shape[0]):            
        #     if count < 100:
        #         cv2.imwrite('demo/demo_patch_mesh_grid_new_' + str(count) + '.bmp', 255*data[count, 0, :, :].detach().cpu().numpy())
        #         cv2.imwrite('demo/demo_patch_kp_new_' + str(count) + '.bmp', 255*data_kp[count, 0, :, :].detach().cpu().numpy())
        # exit()
        # print(data.shape)
        # print(img_unfold_batch.shape)
        # results = data.permute(0, 2, 3, 1).reshape(-1, 1, self.desc_patch, self.desc_patch)   
        # results = data_kp
        # # compute output for patch a
        # results_batch = Variable(results)   
        # print(results_batch.shape)
        with torch.no_grad():
            outs = self.descriptor_tea_net(data_kp, angle=flip_flag)
            # del results_batch
            outs_map = self.descriptor_tea_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w) # outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()    

    # patch AT 扩边图 siftori校准 begin_img
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_onlyP_bg(self, img_point, points, pmask=None):
        # b, _, h, w = img_batch.shape    # 120 x 40
        # _, _, hm, wm = img_ext.shape    # 128 x 52
        b, _, hp, wp = img_point.shape  # 128 x 40  128 x 56
        h, w = hp-8, wp                 # 120 x 40
        # w_pad = 7 # 4 # 7 # 14
        # h_pad = 12 # 9 # 12 # 19
        # pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        # img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        # bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_point.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_point.device)
            kmask = (keypoints_ori >= torch.tensor([4, 5], device=img_point.device)) * (keypoints_ori <= (torch.tensor([wp-5, hp-6], device=img_point.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x40-> 120x40
            # keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_point.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_point.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        return valid_all_mask, kmask_all, keypoints_new 


    # patch AT 扩边图 NE siftori校准
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_NE(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 7 # 4 # 7 # 14
        h_pad = 12 # 9 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            # # 点转到AT上
            # keypoints_T = warp_points(keypoints_ori[:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            # kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            # kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            # keypoints_T = keypoints_T[kmask_T]
            # keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1

            # keypoints = keypoints_ori[kmask_T]
            # keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            # keypoints_new.append(keypoints)
            # kmask_T_all.append(kmask_T)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            # orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        # orient_batch_ori = orient_batch_sift_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正

        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 13 # 10 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 16 # 13 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)
        
        # # 主方向
        # ori_data = deepcopy(orient_batch_ori)
        # ori_data_kp = deepcopy(orient_batch_T)               

        # # 黑白相似度
        # bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        # bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16

        results = data_kp
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        if train_flag:
            outs, _ = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs, _ = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch

        return outs.squeeze(-1), valid_all_mask, kmask_all, keypoints_new, bin_data_kp.detach()    

    # patch AT 扩边图 siftori校准 + 教师模型
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_tea(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 7 # 4 # 7 # 14
        h_pad = 12 # 9 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            # sift角度
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        orient_batch = orient_batch_sift_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36

        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, hm, wm)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 13 # 10 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 16 # 13 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 13 # 10 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 16 # 13 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        data = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        # data_low = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1)   # BHWX1X16X16
        # data = torch.cat((data_low, data_high), dim=1)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)               

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
       
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            outs_teacher = self.descriptor_tea_net(data_kp, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, outs_teacher.squeeze(-1), valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()    


    # patch AT 扩边图 siftori校准 + 教师模型 FGD
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_tea_fgd(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 7 # 4 # 7 # 14
        h_pad = 12 # 9 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            # sift角度
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        orient_batch = orient_batch_sift_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36

        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, hm, wm)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 13 # 10 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 16 # 13 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 13 # 10 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 16 # 13 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        data = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        # data_low = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1)   # BHWX1X16X16
        # data = torch.cat((data_low, data_high), dim=1)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)               

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
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
            outs, outs_fea = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs, outs_fea = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        
        with torch.no_grad():
            outs_map, _ = self.descriptor_net(data, angle=flip_flag)
            outs_teacher, outs_teacher_fea = self.descriptor_tea_net(data_kp, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        # fgd loss
        fea_loss = self.fgd_loss(outs_fea, outs_teacher_fea)
        
        # # mgd loss
        # fea_loss = self.mgd_loss(outs_fea, outs_teacher_fea)

        return outs.squeeze(-1), outs_map, outs_teacher.squeeze(-1), valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach(), fea_loss    


    # patch AT 扩边图 siftori校准 长条形描述
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_rec(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 20 # 4 # 7 # 14
        h_pad = 25 # 9 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->178x92
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            # sift角度
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        orient_batch = orient_batch_sift_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36

        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, hm, wm)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 26 # 10 # 13 # 20         # 40 -> 92
        mesh_points[:, 1] += 29 # 13 # 16 # 23         # 120 -> 178
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 26 # 10 # 13 # 20         # 40 -> 92
        keypoints_expand[:, 1] += 29 # 13 # 16 # 23         # 120 -> 178

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        if self.patch_cal:
            cal_theta = orient_batch.view(-1)
            cal_theta_kp = orient_batch_kp.view(-1)
        else:
            cal_theta = torch.zeros_like(orient_batch.view(-1))
            cal_theta_kp = torch.zeros_like(orient_batch_kp.view(-1))

        data = inv_warp_patch_batch_rec(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), cal_theta + flip_flag * 180 + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25).unsqueeze(1)   # BHWX1X16X16
        # data_low = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1)   # BHWX1X16X16
        # data = torch.cat((data_low, data_high), dim=1)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp = inv_warp_patch_batch_rec(img_ext_pad_batch, keypoints_expand, cal_theta_kp + flip_flag * 180 + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)               

        # 黑白相似度
        bin_data = inv_warp_patch_batch_rec(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), cal_theta + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25).unsqueeze(1).view(-1, self.rect_patch_size)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch_rec(bin_img_ext_pad_batch, keypoints_expand, cal_theta_kp + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, self.rect_patch_size)   # BNX1X16X16
        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()    

    # patch AT 扩边图 NE siftori校准 长条形描述
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_rec_NE(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 20 # 4 # 7 # 14
        h_pad = 25 # 9 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            # # 点转到AT上
            # keypoints_T = warp_points(keypoints_ori[:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            # kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            # kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            # keypoints_T = keypoints_T[kmask_T]
            # keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1

            # keypoints = keypoints_ori[kmask_T]
            # keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            # keypoints_new.append(keypoints)
            # kmask_T_all.append(kmask_T)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            # orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        # orient_batch_ori = orient_batch_sift_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正

        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 26 # 10 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 29 # 13 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        if self.patch_cal:
            # cal_theta = orient_batch.view(-1)
            cal_theta_kp = orient_batch_kp.view(-1)
        else:
            # cal_theta = torch.zeros_like(orient_batch.view(-1))
            cal_theta_kp = torch.zeros_like(orient_batch_kp.view(-1))

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp = inv_warp_patch_batch_rec(img_ext_pad_batch, keypoints_expand, cal_theta_kp + flip_flag * 180 + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)
        
        # # 主方向
        # ori_data = deepcopy(orient_batch_ori)
        # ori_data_kp = deepcopy(orient_batch_T)               

        # # 黑白相似度
        # bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        # bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch_rec(bin_img_ext_pad_batch, keypoints_expand, cal_theta_kp + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, self.rect_patch_size)   # BNX1X16X16

        results = data_kp
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        if train_flag:
            outs, _ = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs, _ = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch

        return outs.squeeze(-1), valid_all_mask, kmask_all, keypoints_new, bin_data_kp.detach()    


    # patch AT 扩边图 siftori校准 长条形描述 + 教师模型
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_rec_tea(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 20 # 4 # 7 # 14
        h_pad = 25 # 9 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->178x92
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            # sift角度
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        orient_batch = orient_batch_sift_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36

        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, hm, wm)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 26 # 10 # 13 # 20         # 40 -> 92
        mesh_points[:, 1] += 29 # 13 # 16 # 23         # 120 -> 178
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 26 # 10 # 13 # 20         # 40 -> 92
        keypoints_expand[:, 1] += 29 # 13 # 16 # 23         # 120 -> 178

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        if self.patch_cal:
            cal_theta = orient_batch.view(-1)
            cal_theta_kp = orient_batch_kp.view(-1)
        else:
            cal_theta = torch.zeros_like(orient_batch.view(-1))
            cal_theta_kp = torch.zeros_like(orient_batch_kp.view(-1))

        data = inv_warp_patch_batch_rec(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), cal_theta + flip_flag * 180 + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25).unsqueeze(1)   # BHWX1X16X16
        # data_low = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1)   # BHWX1X16X16
        # data = torch.cat((data_low, data_high), dim=1)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp = inv_warp_patch_batch_rec(img_ext_pad_batch, keypoints_expand, cal_theta_kp + flip_flag * 180 + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)               

        # 黑白相似度
        bin_data = inv_warp_patch_batch_rec(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), cal_theta + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25).unsqueeze(1).view(-1, self.rect_patch_size)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch_rec(bin_img_ext_pad_batch, keypoints_expand, cal_theta_kp + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, self.rect_patch_size)   # BNX1X16X16
        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            outs_teacher = self.descriptor_tea_net(data_kp, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
    
        return outs.squeeze(-1), outs_map, outs_teacher.squeeze(-1), valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()     

    # patch AT 扩边图 siftori校准 长条形描述 + 教师模型 FGD
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_rec_tea_fgd(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 20 # 4 # 7 # 14
        h_pad = 25 # 9 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->178x92
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            # sift角度
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        orient_batch = orient_batch_sift_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36

        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, hm, wm)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 26 # 10 # 13 # 20         # 40 -> 92
        mesh_points[:, 1] += 29 # 13 # 16 # 23         # 120 -> 178
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 26 # 10 # 13 # 20         # 40 -> 92
        keypoints_expand[:, 1] += 29 # 13 # 16 # 23         # 120 -> 178

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        if self.patch_cal:
            cal_theta = orient_batch.view(-1)
            cal_theta_kp = orient_batch_kp.view(-1)
        else:
            cal_theta = torch.zeros_like(orient_batch.view(-1))
            cal_theta_kp = torch.zeros_like(orient_batch_kp.view(-1))

        data = inv_warp_patch_batch_rec(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), cal_theta + flip_flag * 180 + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25).unsqueeze(1)   # BHWX1X16X16
        # data_low = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1)   # BHWX1X16X16
        # data = torch.cat((data_low, data_high), dim=1)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp = inv_warp_patch_batch_rec(img_ext_pad_batch, keypoints_expand, cal_theta_kp + flip_flag * 180 + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        # data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)               

        # 黑白相似度
        bin_data = inv_warp_patch_batch_rec(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), cal_theta + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25).unsqueeze(1).view(-1, self.rect_patch_size)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch_rec(bin_img_ext_pad_batch, keypoints_expand, cal_theta_kp + cal_angle + rand_angle_enhance, patch_size=self.rpatch, sample_factor=1.25)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, self.rect_patch_size)   # BNX1X16X16
        
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
            outs, outs_fea = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs, outs_fea = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        
        with torch.no_grad():
            outs_map, _ = self.descriptor_net(data, angle=flip_flag)
            outs_teacher, outs_teacher_fea = self.descriptor_tea_net(data_kp, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        # fgd loss
        fea_loss = self.fgd_loss(outs_fea, outs_teacher_fea)
        
        # # mgd loss
        # fea_loss = self.mgd_loss(outs_fea, outs_teacher_fea)
    
        return outs.squeeze(-1), outs_map, outs_teacher.squeeze(-1), valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach(), fea_loss     


    # patch Gauss
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_Gauss(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 7 # 7 # 14
        h_pad = 12 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            # # 点转到AT上
            # keypoints_T = warp_points(keypoints_ori[:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            # kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            # kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            # keypoints_T = keypoints_T[kmask_T]
            # keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1

            # keypoints = keypoints_ori[kmask_T]
            # keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            # keypoints_new.append(keypoints)
            # kmask_T_all.append(kmask_T)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
            # orient_batch_ext = self.get_orientation_batch(img_ext_pad_batch, device=img_batch.device)      # Bx1x(haxwa)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 13 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 16 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 13 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 16 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        data = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16

        # 高斯距离衰减
        data = data * self.gauss_kernel.repeat(data.shape[0], 1, 1).to(data.device).unsqueeze(1)
        data_kp = data_kp * self.gauss_kernel.repeat(data_kp.shape[0], 1, 1).to(data_kp.device).unsqueeze(1)
        # print(data.shape, data_kp.shape)
        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch)
        ori_data_kp = orient_batch_kp.view(-1).unsqueeze(1)[valid_all_mask.bool(), :]           

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data, ori_data_kp, bin_data, bin_data_kp     


    # patch 拼接22和32
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_ms(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 14 # 7 # 14
        h_pad = 19 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            # # 点转到AT上
            # keypoints_T = warp_points(keypoints_ori[:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            # kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            # kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            # keypoints_T = keypoints_T[kmask_T]
            # keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1

            # keypoints = keypoints_ori[kmask_T]
            # keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            # keypoints_new.append(keypoints)
            # kmask_T_all.append(kmask_T)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
            # orient_batch_ext = self.get_orientation_batch(img_ext_pad_batch, device=img_batch.device)      # Bx1x(haxwa)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 20 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 23 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 20 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 23 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        data_high = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        data_low = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1)   # BHWX1X16X16
        data = torch.cat((data_low, data_high), dim=1)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp_high = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch)
        ori_data_kp = orient_batch_kp.view(-1).unsqueeze(1)[valid_all_mask.bool(), :]           

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()     


    # patch 拼接22和32 AT 扩边图
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_ms_AT_ext(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 14 # 7 # 14
        h_pad = 19 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        orient_batch = orient_batch_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        
        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, hm, wm)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 20 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 23 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 20 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 23 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        data_high = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        data_low = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1)   # BHWX1X16X16
        data = torch.cat((data_low, data_high), dim=1)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp_high = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)               

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()    

    # patch 拼接22和32 AT 扩边图 NE
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_ms_AT_ext_NE(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 14 # 7 # 14
        h_pad = 19 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            # # 点转到AT上
            # keypoints_T = warp_points(keypoints_ori[:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            # kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            # kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            # keypoints_T = keypoints_T[kmask_T]
            # keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1

            # keypoints = keypoints_ori[kmask_T]
            # keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            # keypoints_new.append(keypoints)
            # kmask_T_all.append(kmask_T)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            # orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        # orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正

        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 20 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 23 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp_high = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)
        
        # # 主方向
        # ori_data = deepcopy(orient_batch_ori)
        # ori_data_kp = deepcopy(orient_batch_T)               

        # # 黑白相似度
        # bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        # bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16

        results = data_kp
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        if train_flag:
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch

        return outs.squeeze(-1), valid_all_mask, kmask_all, keypoints_new, bin_data_kp.detach()    

    # patch 拼接22和32 AT 扩边图 siftori校准
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_ms_AT_ext(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 14 # 7 # 14
        h_pad = 19 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            # sift角度
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        orient_batch = orient_batch_sift_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36

        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, hm, wm)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 20 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 23 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 20 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 23 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        data_high = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        data_low = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1)   # BHWX1X16X16
        data = torch.cat((data_low, data_high), dim=1)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp_high = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)               

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()    

    # patch 拼接22和32 AT 扩边图 NE siftori校准
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_ms_AT_ext_NE(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 14 # 7 # 14
        h_pad = 19 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            # # 点转到AT上
            # keypoints_T = warp_points(keypoints_ori[:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            # kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            # kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            # keypoints_T = keypoints_T[kmask_T]
            # keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1

            # keypoints = keypoints_ori[kmask_T]
            # keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            # keypoints_new.append(keypoints)
            # kmask_T_all.append(kmask_T)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            # orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        # orient_batch_ori = orient_batch_sift_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正

        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 20 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 23 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp_high = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)
        
        # # 主方向
        # ori_data = deepcopy(orient_batch_ori)
        # ori_data_kp = deepcopy(orient_batch_T)               

        # # 黑白相似度
        # bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        # bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16

        results = data_kp
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        if train_flag:
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch

        return outs.squeeze(-1), valid_all_mask, kmask_all, keypoints_new, bin_data_kp.detach()    

    # patch 拼接22和22的45度 AT 扩边图 siftori校准
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_ms_AT_ext_cat45(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 14 # 7 # 14
        h_pad = 19 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            # sift角度
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        orient_batch = orient_batch_sift_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36

        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, hm, wm)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 20 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 23 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 20 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 23 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        data_high = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance + 45, 16, 22).unsqueeze(1)   # BHWX1X16X16
        data_low = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1)   # BHWX1X16X16
        data = torch.cat((data_low, data_high), dim=1)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp_high = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance + 45, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)               

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()    

    # patch 拼接22和22的45度 AT 扩边图 NE siftori校准
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_ms_AT_ext_NE_cat45(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 14 # 7 # 14
        h_pad = 19 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            # # 点转到AT上
            # keypoints_T = warp_points(keypoints_ori[:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            # kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            # kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            # keypoints_T = keypoints_T[kmask_T]
            # keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1

            # keypoints = keypoints_ori[kmask_T]
            # keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            # keypoints_new.append(keypoints)
            # kmask_T_all.append(kmask_T)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_ext.device)      # Bx1x(hm x wm)
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, hm, wm)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            # orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hm x wm)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        
        # orient_batch_ori = orient_batch_sift_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正

        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 20 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 23 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp_high = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance + 45, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)
        
        # # 主方向
        # ori_data = deepcopy(orient_batch_ori)
        # ori_data_kp = deepcopy(orient_batch_T)               

        # # 黑白相似度
        # bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        # bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16

        results = data_kp
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        if train_flag:
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch

        return outs.squeeze(-1), valid_all_mask, kmask_all, keypoints_new, bin_data_kp.detach()    


    # patch 拼接22和32 gauss
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_ms_Gauss(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 14 # 7 # 14
        h_pad = 19 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        # kmask_T_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            # # 点转到AT上
            # keypoints_T = warp_points(keypoints_ori[:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            # kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            # kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            # keypoints_T = keypoints_T[kmask_T]
            # keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1

            # keypoints = keypoints_ori[kmask_T]
            # keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            # keypoints_new.append(keypoints)
            # kmask_T_all.append(kmask_T)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            # expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            # expand_mask_T[:keypoints_T.shape[0]] = 1
            # if keypoints_expand_T is None: 
            #     keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = expand_mask_T
            # else:
            #     keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
            #     expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
            # orient_batch_ext = self.get_orientation_batch(img_ext_pad_batch, device=img_batch.device)      # Bx1x(haxwa)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 20 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 23 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 20 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 23 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        data_high = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        data_low = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1)   # BHWX1X16X16
        # 高斯距离衰减
        data_high = data_high * self.gauss_kernel.repeat(data_high.shape[0], 1, 1).to(data_high.device).unsqueeze(1)
        data = torch.cat((data_low, data_high), dim=1)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        data_kp_high = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        data_kp_low = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
        
        data_kp_high = data_kp_high * self.gauss_kernel.repeat(data_kp_high.shape[0], 1, 1).to(data_kp_high.device).unsqueeze(1)
        data_kp = torch.cat((data_kp_low, data_kp_high), dim=1)

        # # 4x4主方向
        # ori_data = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)   # BHWX64
        # ori_data_kp = inv_warp_patch_batch(orient_batch_ext.view(b, 1, hm+2*h_pad, wm+2*w_pad), keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 8, self.sample_desc_patch).unsqueeze(1).view(-1, 64)
        # ori_data = ori_data - orient_batch.view(-1).unsqueeze(1)
        # ori_data = ori_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        # ori_data_kp = ori_data_kp - orient_batch_kp.view(-1).unsqueeze(1)
        # ori_data_kp = ori_data_kp[valid_all_mask.bool(), :]
        
        # 主方向
        ori_data = deepcopy(orient_batch)
        ori_data_kp = orient_batch_kp.view(-1).unsqueeze(1)[valid_all_mask.bool(), :]           

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
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
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()     



    # dense
    def cut_patch_unfold_dense_map_interpolation_aligned_batch93_newOri(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, he, we = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
  
        # Resize 成奇数 
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        # inner 
        img_pad_batch = deepcopy(img_resize)
        img_pad_size = (self.sample_desc_patch, self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_ext_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [145, 61]
        hm, wm = img_ext_pad_batch.shape[2], img_ext_pad_batch.shape[3]  # [145, 61]

        # 0/180°随机增强
        flip_flag = random.choice([0, 1])
        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        # 计算描述子特征图
        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x37x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
    
        w_pad = 7 # 7 # 14
        h_pad = 12 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        # img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
            # orient_batch_ext = self.get_orientation_batch(img_ext_pad_batch, device=img_batch.device)      # Bx1x(haxwa)

        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        downsampe_ratio = 4
        # 用于计算黑白相似度和方向
        xb = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        yb = torch.linspace(0, h-1, h) 
        mesh_points_b = torch.stack(torch.meshgrid([yb, xb])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        mesh_points_b[:, 1] += 16 # 16 # 23         # 120 -> 166
        keypoints_expand_b = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        keypoints_expand_b[:, 1] += 16 # 16 # 23         # 120 -> 166

        # 用于计算描述字
        x = (torch.linspace(2, w-3, w-4) + (we - w) / 2) * (we / (we - 1)) + self.sample_desc_patch    # ex: [-2, -1, 0, 1, 2]
        xd = x / downsampe_ratio 
        y = (torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he - 1)) + self.sample_desc_patch + 4
        yd = y / downsampe_ratio
        mesh_points = torch.stack(torch.meshgrid([yd, xd])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)

        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += (we - w) / 2 # 6         # 40 -> 52
        keypoints_expand[:, 1] += (he - h) / 2 # 3         # 120 -> 128
        keypoints_expand[:, 0] = keypoints_expand[:, 0] * (we / (we - 1)) + self.sample_desc_patch
        keypoints_expand[:, 1] = keypoints_expand[:, 1] * (he / (he - 1)) + self.sample_desc_patch + 4
        keypoints_expand[:, 0] = keypoints_expand[:, 0] / downsampe_ratio
        keypoints_expand[:, 1] = keypoints_expand[:, 1] / downsampe_ratio

        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        # 切校准后描述子
        # outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, 3.5).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 

        # bx8xNx4x4
        # outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, 3.5).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        
        # 主方向
        ori_data = deepcopy(orient_batch)
        ori_data_kp = orient_batch_kp.view(-1).unsqueeze(1)[valid_all_mask.bool(), :]           

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points_b.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand_b, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 

        return outs_kp, outs_map.detach(), valid_all_mask, kmask_all, keypoints_new, ori_data, ori_data_kp, bin_data, bin_data_kp     

    # dense AT 扩边图
    def cut_patch_unfold_dense_map_interpolation_aligned_batch93_newOri_AT_ext(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, he, we = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
  
        # Resize 成奇数 
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        # inner 
        img_pad_batch = deepcopy(img_resize)
        img_pad_size = (self.sample_desc_patch, self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_ext_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [145, 61]
        hm, wm = img_ext_pad_batch.shape[2], img_ext_pad_batch.shape[3]  # [145, 61]

        # 0/180°随机增强
        flip_flag = random.choice([0, 1])
        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        # 计算描述子特征图
        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x37x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
    
        w_pad = 7 # 7 # 14
        h_pad = 12 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        # img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_batch.device)      # Bx1x(hxw)
            orient_batch_ori = orient_batch_ori.view(b, 1, he, we)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hmxwm)

        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        
        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, he, we)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36

        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        downsampe_ratio = 4
        # 用于计算黑白相似度和方向
        xb = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        yb = torch.linspace(0, h-1, h) 
        mesh_points_b = torch.stack(torch.meshgrid([yb, xb])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        mesh_points_b[:, 1] += 16 # 16 # 23         # 120 -> 166
        keypoints_expand_b = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        keypoints_expand_b[:, 1] += 16 # 16 # 23         # 120 -> 166

        # 用于计算描述字
        x = (torch.linspace(2, w-3, w-4) + (we - w) / 2) * (we / (we - 1)) + self.sample_desc_patch    # ex: [-2, -1, 0, 1, 2]
        xd = x / downsampe_ratio 
        y = (torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he - 1)) + self.sample_desc_patch + 4
        yd = y / downsampe_ratio
        mesh_points = torch.stack(torch.meshgrid([yd, xd])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)

        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += (we - w) / 2 # 6         # 40 -> 52
        keypoints_expand[:, 1] += (he - h) / 2 # 3         # 120 -> 128
        keypoints_expand[:, 0] = keypoints_expand[:, 0] * (we / (we - 1)) + self.sample_desc_patch
        keypoints_expand[:, 1] = keypoints_expand[:, 1] * (he / (he - 1)) + self.sample_desc_patch + 4
        keypoints_expand[:, 0] = keypoints_expand[:, 0] / downsampe_ratio
        keypoints_expand[:, 1] = keypoints_expand[:, 1] / downsampe_ratio

        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        if self.is256:
            outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*16, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 16, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 2, 8, -1, 4, 4).permute(0, 3, 1, 4, 5, 2).contiguous().view(-1, 256) 

            # bx8xNx4x4
            # outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
            outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 16, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 16, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 2, 8, -1, 4, 4).permute(0, 3, 1, 4, 5, 2).contiguous().view(-1, 256)  

        else:
            # 切校准后描述子
            # outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
            outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 

            # bx8xNx4x4
            # outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
            outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)         
           

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points_b.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand_b, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 

        return outs_kp, outs_map.detach(), valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()     

    # dense AT 扩边图 NE
    def cut_patch_unfold_dense_map_interpolation_aligned_batch93_newOri_AT_ext_NE(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, he, we = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
  
        # Resize 成奇数 
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        # inner 
        img_pad_batch = deepcopy(img_resize)
        img_pad_size = (self.sample_desc_patch, self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_ext_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [145, 61]
        hm, wm = img_ext_pad_batch.shape[2], img_ext_pad_batch.shape[3]  # [145, 61]

        # 0/180°随机增强
        flip_flag = random.choice([0, 1])
        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        # 计算描述子特征图
        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x37x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
    
        w_pad = 7 # 7 # 14
        h_pad = 12 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        # img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_batch.device)      # Bx1x(hxw)
            orient_batch_ori = orient_batch_ori.view(b, 1, he, we)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            # orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hmxwm)

        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)

        # orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正

        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        downsampe_ratio = 4
        # # 用于计算黑白相似度和方向
        # xb = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        # yb = torch.linspace(0, h-1, h) 
        # mesh_points_b = torch.stack(torch.meshgrid([yb, xb])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        # mesh_points_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        # mesh_points_b[:, 1] += 16 # 16 # 23         # 120 -> 166
        keypoints_expand_b = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        keypoints_expand_b[:, 1] += 16 # 16 # 23         # 120 -> 166

        # # 用于计算描述字
        # x = (torch.linspace(2, w-3, w-4) + (we - w) / 2) * (we / (we - 1)) + self.sample_desc_patch    # ex: [-2, -1, 0, 1, 2]
        # xd = x / downsampe_ratio 
        # y = (torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he - 1)) + self.sample_desc_patch + 4
        # yd = y / downsampe_ratio
        # mesh_points = torch.stack(torch.meshgrid([yd, xd])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)

        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += (we - w) / 2 # 6         # 40 -> 52
        keypoints_expand[:, 1] += (he - h) / 2 # 3         # 120 -> 128
        keypoints_expand[:, 0] = keypoints_expand[:, 0] * (we / (we - 1)) + self.sample_desc_patch
        keypoints_expand[:, 1] = keypoints_expand[:, 1] * (he / (he - 1)) + self.sample_desc_patch + 4
        keypoints_expand[:, 0] = keypoints_expand[:, 0] / downsampe_ratio
        keypoints_expand[:, 1] = keypoints_expand[:, 1] / downsampe_ratio

        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        # # 切校准后描述子
        # # outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        # outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, 3.5).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 

        if self.is256:
            outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 16, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 16, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 2, 8, -1, 4, 4).permute(0, 3, 1, 4, 5, 2).contiguous().view(-1, 256)  
        else:
        # bx8xNx4x4
            outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        
        # # 主方向
        # ori_data = deepcopy(orient_batch_ori)
        # ori_data_kp = deepcopy(orient_batch_T)         
           

        # # 黑白相似度
        # bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points_b.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        # bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand_b, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 

        return outs_kp, valid_all_mask, kmask_all, keypoints_new, bin_data_kp.detach()     

    # dense AT 扩边图 sift类似计算角度  +45°描述
    def cut_patch_unfold_dense_map_interpolation_aligned_batch93_siftOri_AT_ext_a45(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, he, we = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
  
        # Resize 成奇数 
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        # inner 
        img_pad_batch = deepcopy(img_resize)
        img_pad_size = (self.sample_desc_patch, self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_ext_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [145, 61]
        hm, wm = img_ext_pad_batch.shape[2], img_ext_pad_batch.shape[3]  # [145, 61]

        # 0/180°随机增强
        flip_flag = random.choice([0, 1])
        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        # 计算描述子特征图
        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x37x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
    
        w_pad = 7 # 7 # 14
        h_pad = 12 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        # img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_batch.device)      # Bx1x(hxw)
            # sift角度
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, he, we)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, he, we)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hmxwm)

        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch_sift_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        
        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, he, we)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36

        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        downsampe_ratio = 4
        # 用于计算黑白相似度和方向
        xb = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        yb = torch.linspace(0, h-1, h) 
        mesh_points_b = torch.stack(torch.meshgrid([yb, xb])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        mesh_points_b[:, 1] += 16 # 16 # 23         # 120 -> 166
        keypoints_expand_b = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        keypoints_expand_b[:, 1] += 16 # 16 # 23         # 120 -> 166

        # 用于计算描述字
        x = (torch.linspace(2, w-3, w-4) + (we - w) / 2) * (we / (we - 1)) + self.sample_desc_patch    # ex: [-2, -1, 0, 1, 2]
        xd = x / downsampe_ratio 
        y = (torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he - 1)) + self.sample_desc_patch + 4
        yd = y / downsampe_ratio
        mesh_points = torch.stack(torch.meshgrid([yd, xd])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)

        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += (we - w) / 2 # 6         # 40 -> 52
        keypoints_expand[:, 1] += (he - h) / 2 # 3         # 120 -> 128
        keypoints_expand[:, 0] = keypoints_expand[:, 0] * (we / (we - 1)) + self.sample_desc_patch
        keypoints_expand[:, 1] = keypoints_expand[:, 1] * (he / (he - 1)) + self.sample_desc_patch + 4
        keypoints_expand[:, 0] = keypoints_expand[:, 0] / downsampe_ratio
        keypoints_expand[:, 1] = keypoints_expand[:, 1] / downsampe_ratio

        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        if self.is256:
            if self.has45:
                outs_0 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
                outs_45 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance + 45, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
                outs = torch.cat((outs_0, outs_45), dim=-1)
                # bx8xNx4x4
                outs_kp_0 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
                outs_kp_45 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance + 45, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
                outs_kp = torch.cat((outs_kp_0, outs_kp_45), dim=-1)
            else:
                outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*16, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 16, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 2, 8, -1, 4, 4).permute(0, 3, 1, 4, 5, 2).contiguous().view(-1, 256) 

                # bx8xNx4x4
                # outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
                outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 16, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 16, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 2, 8, -1, 4, 4).permute(0, 3, 1, 4, 5, 2).contiguous().view(-1, 256)  

        else:
            # 切校准后描述子
            # outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
            outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 

            # bx8xNx4x4
            # outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
            outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)         
           

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points_b.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand_b, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 

        return outs_kp, outs_map.detach(), valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()     

    # dense AT 扩边图 sift类似计算角度 NE +45°描述
    def cut_patch_unfold_dense_map_interpolation_aligned_batch93_siftOri_AT_ext_NE_a45(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, he, we = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
  
        # Resize 成奇数 
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        # inner 
        img_pad_batch = deepcopy(img_resize)
        img_pad_size = (self.sample_desc_patch, self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_ext_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [145, 61]
        hm, wm = img_ext_pad_batch.shape[2], img_ext_pad_batch.shape[3]  # [145, 61]

        # 0/180°随机增强
        flip_flag = random.choice([0, 1])
        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        # 计算描述子特征图
        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x37x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
    
        w_pad = 7 # 7 # 14
        h_pad = 12 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        # img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_batch.device)      # Bx1x(hxw)
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, he, we)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, he, we)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            # orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hmxwm)

        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)

        # orient_batch_ori = orient_batch_sift_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正

        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        downsampe_ratio = 4
        # # 用于计算黑白相似度和方向
        # xb = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        # yb = torch.linspace(0, h-1, h) 
        # mesh_points_b = torch.stack(torch.meshgrid([yb, xb])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        # mesh_points_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        # mesh_points_b[:, 1] += 16 # 16 # 23         # 120 -> 166
        keypoints_expand_b = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        keypoints_expand_b[:, 1] += 16 # 16 # 23         # 120 -> 166

        # # 用于计算描述字
        # x = (torch.linspace(2, w-3, w-4) + (we - w) / 2) * (we / (we - 1)) + self.sample_desc_patch    # ex: [-2, -1, 0, 1, 2]
        # xd = x / downsampe_ratio 
        # y = (torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he - 1)) + self.sample_desc_patch + 4
        # yd = y / downsampe_ratio
        # mesh_points = torch.stack(torch.meshgrid([yd, xd])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)

        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += (we - w) / 2 # 6         # 40 -> 52
        keypoints_expand[:, 1] += (he - h) / 2 # 3         # 120 -> 128
        keypoints_expand[:, 0] = keypoints_expand[:, 0] * (we / (we - 1)) + self.sample_desc_patch
        keypoints_expand[:, 1] = keypoints_expand[:, 1] * (he / (he - 1)) + self.sample_desc_patch + 4
        keypoints_expand[:, 0] = keypoints_expand[:, 0] / downsampe_ratio
        keypoints_expand[:, 1] = keypoints_expand[:, 1] / downsampe_ratio

        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        # # 切校准后描述子
        # # outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        # outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, 3.5).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 

        if self.is256:
            if self.has45:
                outs_kp_0 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
                outs_kp_45 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance + 45, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
                outs_kp = torch.cat((outs_kp_0, outs_kp_45), dim=-1)
            else:
                outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 16, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 16, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 2, 8, -1, 4, 4).permute(0, 3, 1, 4, 5, 2).contiguous().view(-1, 256)  
        else:
        # bx8xNx4x4
            outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        
        # # 主方向
        # ori_data = deepcopy(orient_batch_ori)
        # ori_data_kp = deepcopy(orient_batch_T)         
           

        # # 黑白相似度
        # bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points_b.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        # bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand_b, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 

        return outs_kp, valid_all_mask, kmask_all, keypoints_new, bin_data_kp.detach()     

    # dense AT 扩边图 sift类似计算角度  +45°描述
    def cut_patch_unfold_dense_map_interpolation_aligned_batch93_siftOri_AT_ext_a45_ms(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, he, we = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
  
        # Resize 成奇数 
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        # inner 
        img_pad_batch = deepcopy(img_resize)
        img_pad_size = (self.sample_desc_patch, self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_ext_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [145, 61]
        hm, wm = img_ext_pad_batch.shape[2], img_ext_pad_batch.shape[3]  # [145, 61]

        # 0/180°随机增强
        flip_flag = random.choice([0, 1])
        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        # 计算描述子特征图
        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x37x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
    
        w_pad = 7 # 7 # 14
        h_pad = 12 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        # img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_batch.device)      # Bx1x(hxw)
            # sift角度
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, he, we)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, he, we)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hmxwm)

        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch_sift_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        
        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, he, we)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36

        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        downsampe_ratio = 4
        # 用于计算黑白相似度和方向
        xb = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        yb = torch.linspace(0, h-1, h) 
        mesh_points_b = torch.stack(torch.meshgrid([yb, xb])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        mesh_points_b[:, 1] += 16 # 16 # 23         # 120 -> 166
        keypoints_expand_b = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        keypoints_expand_b[:, 1] += 16 # 16 # 23         # 120 -> 166

        # 用于计算描述字
        x = (torch.linspace(2, w-3, w-4) + (we - w) / 2) * (we / (we - 1)) + self.sample_desc_patch    # ex: [-2, -1, 0, 1, 2]
        xd = x / downsampe_ratio 
        y = (torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he - 1)) + self.sample_desc_patch + 4
        yd = y / downsampe_ratio
        mesh_points = torch.stack(torch.meshgrid([yd, xd])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)

        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += (we - w) / 2 # 6         # 40 -> 52
        keypoints_expand[:, 1] += (he - h) / 2 # 3         # 120 -> 128
        keypoints_expand[:, 0] = keypoints_expand[:, 0] * (we / (we - 1)) + self.sample_desc_patch
        keypoints_expand[:, 1] = keypoints_expand[:, 1] * (he / (he - 1)) + self.sample_desc_patch + 4
        keypoints_expand[:, 0] = keypoints_expand[:, 0] / downsampe_ratio
        keypoints_expand[:, 1] = keypoints_expand[:, 1] / downsampe_ratio

        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        if self.is256:
            if self.has45:
                outs_0 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
                outs_45 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance + 45, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
                outs = torch.cat((outs_0, outs_45), dim=-1)
                # bx8xNx4x4
                outs_kp_0 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
                outs_kp_45 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance + 45, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
                outs_kp = torch.cat((outs_kp_0, outs_kp_45), dim=-1)
            else:
                outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*16, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 16, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 2, 8, -1, 4, 4).permute(0, 3, 1, 4, 5, 2).contiguous().view(-1, 256) 

                # bx8xNx4x4
                # outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
                outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 16, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 16, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 2, 8, -1, 4, 4).permute(0, 3, 1, 4, 5, 2).contiguous().view(-1, 256)  

        else:
            # 切校准后描述子
            # outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
            outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 

            # bx8xNx4x4
            # outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
            outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)         
           

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points_b.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand_b, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 

        return outs_kp, outs_map.detach(), valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()     



    # dense AT 扩边图 sift类似计算角度  +45°描述 amp
    def cut_patch_unfold_dense_map_interpolation_aligned_batch93_siftOri_AT_ext_a45_amp(self, img_batch, img_ext, img_ext_T, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, he, we = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
  
        # Resize 成奇数 
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        img_amp = self.get_sift_amp_grad_batch(img_resize)
        # inner 
        img_pad_batch = torch.cat((img_resize, img_amp), dim=1)
        img_pad_size = (self.sample_desc_patch, self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_ext_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [145, 61]
        hm, wm = img_ext_pad_batch.shape[2], img_ext_pad_batch.shape[3]  # [145, 61]

        # 0/180°随机增强
        flip_flag = random.choice([0, 1])
        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        # 计算描述子特征图
        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x37x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
    
        w_pad = 7 # 7 # 14
        h_pad = 12 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        # img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_batch.device)      # Bx1x(hxw)
            # sift角度
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, he, we)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, he, we)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hmxwm)

        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch_sift_ori + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        
        orient_batch_ori = orient_batch_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正
        orient_batch_T = orient_batch_T.view(b, 1, he, we)[:, :, 4:-4, 8:-8].contiguous()     # 120 x 36

        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        downsampe_ratio = 4
        # 用于计算黑白相似度和方向
        xb = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        yb = torch.linspace(0, h-1, h) 
        mesh_points_b = torch.stack(torch.meshgrid([yb, xb])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        mesh_points_b[:, 1] += 16 # 16 # 23         # 120 -> 166
        keypoints_expand_b = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        keypoints_expand_b[:, 1] += 16 # 16 # 23         # 120 -> 166

        # 用于计算描述字
        x = (torch.linspace(2, w-3, w-4) + (we - w) / 2) * (we / (we - 1)) + self.sample_desc_patch    # ex: [-2, -1, 0, 1, 2]
        xd = x / downsampe_ratio 
        y = (torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he - 1)) + self.sample_desc_patch + 4
        yd = y / downsampe_ratio
        mesh_points = torch.stack(torch.meshgrid([yd, xd])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)

        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += (we - w) / 2 # 6         # 40 -> 52
        keypoints_expand[:, 1] += (he - h) / 2 # 3         # 120 -> 128
        keypoints_expand[:, 0] = keypoints_expand[:, 0] * (we / (we - 1)) + self.sample_desc_patch
        keypoints_expand[:, 1] = keypoints_expand[:, 1] * (he / (he - 1)) + self.sample_desc_patch + 4
        keypoints_expand[:, 0] = keypoints_expand[:, 0] / downsampe_ratio
        keypoints_expand[:, 1] = keypoints_expand[:, 1] / downsampe_ratio

        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        if self.is256:
            if self.has45:
                outs_0 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
                outs_45 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance + 45, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
                outs = torch.cat((outs_0, outs_45), dim=-1)
                # bx8xNx4x4
                outs_kp_0 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
                outs_kp_45 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance + 45, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
                outs_kp = torch.cat((outs_kp_0, outs_kp_45), dim=-1)
            else:
                outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*16, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 16, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 2, 8, -1, 4, 4).permute(0, 3, 1, 4, 5, 2).contiguous().view(-1, 256) 

                # bx8xNx4x4
                # outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
                outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 16, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 16, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 2, 8, -1, 4, 4).permute(0, 3, 1, 4, 5, 2).contiguous().view(-1, 256)  

        else:
            # 切校准后描述子
            # outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
            outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 

            # bx8xNx4x4
            # outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
            outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        
        # 主方向
        ori_data = deepcopy(orient_batch_ori)
        ori_data_kp = deepcopy(orient_batch_T)         
           

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points_b.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand_b, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 

        return outs_kp, outs_map.detach(), valid_all_mask, kmask_all, keypoints_new, ori_data.detach(), ori_data_kp.detach(), bin_data.detach(), bin_data_kp.detach()     

    # dense AT 扩边图 sift类似计算角度 NE +45°描述 amp
    def cut_patch_unfold_dense_map_interpolation_aligned_batch93_siftOri_AT_ext_NE_a45_amp(self, img_batch, img_ext, img_point, bin_img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, he, we = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
  
        # Resize 成奇数 
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        img_amp = self.get_sift_amp_grad_batch(img_resize)
        # inner 
        img_pad_batch = torch.cat((img_resize, img_amp), dim=1)

        img_pad_size = (self.sample_desc_patch, self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_ext_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [145, 61]
        hm, wm = img_ext_pad_batch.shape[2], img_ext_pad_batch.shape[3]  # [145, 61]

        # 0/180°随机增强
        flip_flag = random.choice([0, 1])
        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        # 计算描述子特征图
        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x37x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
    
        w_pad = 7 # 7 # 14
        h_pad = 12 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        # img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        kmask_all = []
        keypoints_new = []
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ori = ((keypoints + 1) / 2 * keypoints.new_tensor([[wp - 1, hp - 1]])).to(img_batch.device)
            kmask = (keypoints_ori >= torch.tensor([12, 5], device=img_batch.device)) * (keypoints_ori <= (torch.tensor([wp-13, hp-6], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            keypoints_ori = keypoints_ori[kmask]
            # 128x56 -> 120x40
            keypoints_ori[:, 0] -= 8
            keypoints_ori[:, 1] -= 4
            keypoints = keypoints_ori / keypoints_ori.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)

            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch_ori = self.get_orientation_batch(img_ext, device=img_batch.device)      # Bx1x(hxw)
            orient_batch_sift_ori = self.get_sift_orientation_batch(img_ext)      # Bx1x(hm x wm)
            orient_batch_ori = orient_batch_ori.view(b, 1, he, we)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            orient_batch_sift_ori = orient_batch_sift_ori.view(b, 1, he, we)[:, :, 4:-4, 6:-6].contiguous().view(b, 1, -1) # Bx1x(h x w)
            # orient_batch_T = self.get_orientation_batch(img_ext_T, device=img_ext_T.device)      # Bx1x(hmxwm)

        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch_ori).to(orient_batch_ori.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch_ori, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch_sift_ori.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)

        # orient_batch_ori = orient_batch_sift_ori.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36 不加trans矫正

        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        downsampe_ratio = 4
        # # 用于计算黑白相似度和方向
        # xb = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        # yb = torch.linspace(0, h-1, h) 
        # mesh_points_b = torch.stack(torch.meshgrid([yb, xb])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        # mesh_points_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        # mesh_points_b[:, 1] += 16 # 16 # 23         # 120 -> 166
        keypoints_expand_b = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand_b[:, 0] += 13 # 13 # 20         # 40 -> 80
        keypoints_expand_b[:, 1] += 16 # 16 # 23         # 120 -> 166

        # # 用于计算描述字
        # x = (torch.linspace(2, w-3, w-4) + (we - w) / 2) * (we / (we - 1)) + self.sample_desc_patch    # ex: [-2, -1, 0, 1, 2]
        # xd = x / downsampe_ratio 
        # y = (torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he - 1)) + self.sample_desc_patch + 4
        # yd = y / downsampe_ratio
        # mesh_points = torch.stack(torch.meshgrid([yd, xd])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)

        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += (we - w) / 2 # 6         # 40 -> 52
        keypoints_expand[:, 1] += (he - h) / 2 # 3         # 120 -> 128
        keypoints_expand[:, 0] = keypoints_expand[:, 0] * (we / (we - 1)) + self.sample_desc_patch
        keypoints_expand[:, 1] = keypoints_expand[:, 1] * (he / (he - 1)) + self.sample_desc_patch + 4
        keypoints_expand[:, 0] = keypoints_expand[:, 0] / downsampe_ratio
        keypoints_expand[:, 1] = keypoints_expand[:, 1] / downsampe_ratio

        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        # # 切校准后描述子
        # # outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        # outs = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, 3.5).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 

        if self.is256:
            if self.has45:
                outs_kp_0 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
                outs_kp_45 = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance + 45, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
                outs_kp = torch.cat((outs_kp_0, outs_kp_45), dim=-1)
            else:
                outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 16, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 16, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 2, 8, -1, 4, 4).permute(0, 3, 1, 4, 5, 2).contiguous().view(-1, 256)  
        else:
        # bx8xNx4x4
            outs_kp = inv_warp_patch_batch(feats.contiguous().view(-1, 1, feats.shape[-2], feats.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        
        # # 主方向
        # ori_data = deepcopy(orient_batch_ori)
        # ori_data_kp = deepcopy(orient_batch_T)         
           

        # # 黑白相似度
        # bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points_b.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, 22).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        # bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand_b, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, 22)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 

        return outs_kp, valid_all_mask, kmask_all, keypoints_new, bin_data_kp.detach()     



    # sift
    # patch
    def cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_sift(self, img_batch, img_ext, img_point, bin_img_ext, points, siftm, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape    # 120 x 40
        _, _, hm, wm = img_ext.shape    # 128 x 52
        _, _, hp, wp = img_point.shape  # 128 x 56
        w_pad = 7 # 7 # 14
        h_pad = 12 # 12 # 19
        pad_size = (w_pad, w_pad, h_pad, h_pad)     # 128x52 pad->166x80
        img_ext_pad_batch = F.pad(img_ext, pad_size, "constant", 0)
        bin_img_ext_pad_batch = F.pad(bin_img_ext, pad_size, "constant", 0)

        keypoints = [points[i, :, :][siftm[i] == 1, :] for i in range(b)]
        max_num = points.shape[1]
        keypoints_expand = points.view(-1, 2)
        expand_mask_all = siftm.view(-1)

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
            # orient_batch_ext = self.get_orientation_batch(img_ext_pad_batch, device=img_batch.device)      # Bx1x(haxwa)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        # big_mask_ext = torch.zeros_like(orient_batch_ext).to(orient_batch_ext.device)
        if trans_rot_angle is not None:
            # 中心裁剪不影响trans角
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)
            # big_mask_ext = self.get_big_mask(orient_batch_ext, trans_rot=trans_rot_angle)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch + big_mask * 180
        orient_batch = orient_batch.view(b, 1, h, w)[:, :, :, 2:-2].contiguous()     # 120 x 36
        # orient_batch_ext = orient_batch_ext + big_mask_ext * 180        # Bx1x(haxwa)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x = torch.linspace(2, w-3, w-4)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points[:, 0] += 13 # 13 # 20         # 40 -> 80
        mesh_points[:, 1] += 16 # 16 # 23         # 120 -> 166
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device)
        keypoints_expand[:, 0] += 13 # 13 # 20         # 40 -> 80
        keypoints_expand[:, 1] += 16 # 16 # 23         # 120 -> 166

        # 0,180随机旋转增强，描述对应翻转
        flip_flag = random.choice([0, 1])
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        rand_angle_enhance = 0

        data = inv_warp_patch_batch(img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1)   # BHWX1X16X16
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        data_kp = inv_warp_patch_batch(img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + flip_flag * 180 + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1)   # BNX1X16X16
  
        # 主方向
        ori_data = deepcopy(orient_batch)
        ori_data_kp = orient_batch_kp.view(-1).unsqueeze(1)[valid_all_mask.bool(), :]           

        # 黑白相似度
        bin_data = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch).unsqueeze(1).view(-1, 256)   # BHWX1X16X16
        bin_data = bin_data.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)
        bin_data_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, 16, self.sample_desc_patch)[valid_all_mask.bool(), :, :].unsqueeze(1).view(-1, 256)   # BNX1X16X16
        
        results = data_kp
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        if train_flag:
            outs = self.descriptor_net(results_batch, angle=flip_flag)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch, angle=flip_flag)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data, angle=flip_flag)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)          # B X dim x H X (W-4)

        return outs.squeeze(-1), outs_map, valid_all_mask, keypoints, ori_data, ori_data_kp, bin_data, bin_data_kp     



    def cut_patch_from_featuremap(self, img_batch, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape
        flip_flag = random.choice([0, 1])

        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°
        else:
            img_batch_flip = deepcopy(img_batch)

        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x34x10
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
                # del results_batch
        # with torch.no_grad():
        #     outs_map = self.descriptor_net(data, angle=flip_flag)
        #     del data

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 400
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
        
        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            # print(orient_batch)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)

        # orient_batch = orient_batch + big_mask * 180
        # orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch + big_mask * 180
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        # 切描述patch 
        downsampe_ratio = 4
        # 随机±5度增强，兼容sift计算的角度
        rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)

        x = torch.linspace(0, w-1, w) / downsampe_ratio      # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) / downsampe_ratio
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + self.desc_patch_expand // 2
        keypoints_expand = (keypoints_expand + 1) / (downsampe_ratio * 2) * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device) + self.desc_patch_expand // 2

        pad_size = (self.desc_patch_expand // 2, self.desc_patch_expand // 2, self.desc_patch_expand // 2, self.desc_patch_expand // 2)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x(h//4+pad)x(w//4+pad)
        # print(feats_pad_batch.view(-1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]).shape)
        # print(keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2).shape)
        outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.sample_desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        # print(valid_all_mask.shape, pmask_kp.shape)
        
        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.sample_desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w) 
        # print(outs_kp.shape, outs_map.shape, valid_all_mask.shape)
        return outs_kp, outs_map.detach(), valid_all_mask  

    def cut_patch_from_featuremap_pad(self, img_batch, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape        # 136x40
        img_resize = TF.resize(img_batch, [137, 41])    # 137x41
        # b_r, _, h_r, w_r = img_resize.shape
        flip_flag = random.choice([0, 1])

        img_pad_size = (2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [153, 57]

        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x39x15
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
                # del results_batch
        # with torch.no_grad():
        #     outs_map = self.descriptor_net(data, angle=flip_flag)
        #     del data

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            # kmask = keypoints[:, 0] >= (2 + 1) / 2 * (w-1) and keypoints[:, 0] <= (w - 3 + 1) / 2 * (w-1)
            # keypoints = keypoints[kmask]
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 
        
        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            # print(orient_batch)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)

        # orient_batch = orient_batch + big_mask * 180
        # orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch + big_mask * 180
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        # pad_zero_mask = torch.zeros_like(img_batch, device=img_batch.device)
        # pad_zero_mask[:, :, :, 2:-2] = 1        # bxcxhxw

        # 切描述patch 
        downsampe_ratio = 4
        # 随机±5度增强，兼容sift计算的角度
        rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)

        x = (torch.linspace(0, w-1, w) * (w / (w - 1)) + 2 * self.sample_desc_patch) / downsampe_ratio      # ex: [-2, -1, 0, 1, 2] 137x41->153x57
        y = (torch.linspace(0, h-1, h) * (h / (h - 1)) + 2 * self.sample_desc_patch) / downsampe_ratio
        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 2
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + res_pad_size
        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1])
        keypoints_expand_reszie = keypoints_expand_mid * keypoints_expand.new_tensor([w / (w - 1), h / (h - 1)])  # 137x41
        keypoints_expand = ((keypoints_expand_reszie + 2 * self.sample_desc_patch) / downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x41x17

        outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        
        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        
        # outs = self.dis_sigma_weight.repeat(outs.shape[0], 1) * outs
        # outs_kp = self.dis_sigma_weight.repeat(outs_kp.shape[0], 1) * outs_kp

        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w) 
        # print(outs_kp.shape, outs_map.shape, valid_all_mask.shape)
        return outs_kp, outs_map.detach(), valid_all_mask  

    def cut_patch_from_featuremap_pad_flip(self, img_batch, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape
        flip_flag = random.choice([0, 1])

        img_pad_size = (2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_pad_batch = F.pad(img_batch, img_pad_size, "constant", 0)   # [152, 56]

        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        if train_flag:
            feats = self.descriptor_net(img_batch_flip) # bx8x38x14
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip)
                # del results_batch
        # with torch.no_grad():
        #     outs_map = self.descriptor_net(data, angle=flip_flag)
        #     del data

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 400
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
        
        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            # print(orient_batch)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)

        # orient_batch = orient_batch + big_mask * 180
        # orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch + big_mask * 180
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        # 切描述patch 
        downsampe_ratio = 4
        # 随机±5度增强，兼容sift计算的角度
        rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        res_pad_size = (self.desc_patch_expand - self.desc_patch) // 2          # 1
        if flip_flag == 1:
            x = (w - 1 - torch.linspace(0, w-1, w) + 2 * self.sample_desc_patch) / downsampe_ratio      # ex: [-2, -1, 0, 1, 2] 136x40->152x56
            y = (h - 1 - torch.linspace(0, h-1, h) + 2 * self.sample_desc_patch) / downsampe_ratio
            keypoints_expand = (((1 - keypoints_expand) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]) + 2 * self.sample_desc_patch)/ downsampe_ratio).to(img_batch.device) + res_pad_size
        else:
            x = (torch.linspace(0, w-1, w) + 2 * self.sample_desc_patch) / downsampe_ratio      # ex: [-2, -1, 0, 1, 2] 136x40->152x56
            y = (torch.linspace(0, h-1, h) + 2 * self.sample_desc_patch) / downsampe_ratio
            keypoints_expand = (((1 + keypoints_expand) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]) + 2 * self.sample_desc_patch)/ downsampe_ratio).to(img_batch.device) + res_pad_size
      
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + res_pad_size
        
        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x40x16
        # print(feats_pad_batch.view(-1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]).shape)
        # print(keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2).shape)
        # outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 8, 1).view(-1) + cal_angle + rand_angle_enhance + flip_flag * 180, self.desc_patch, self.sample_desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.sample_desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 16, 8) 
        if flip_flag == 1:
            outs = torch.flip(outs, dims=[1])
        outs = outs.view(-1, 128)
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        # print(valid_all_mask.shape, pmask_kp.shape)
        
        # bx8xNx4x4
        # outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance + flip_flag * 180, self.desc_patch, self.sample_desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.sample_desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 16, 8)  
        if flip_flag == 1:
            outs_kp = torch.flip(outs_kp, dims=[1])
        outs_kp = outs_kp.view(-1, 128)

        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w) 
        # print(outs_kp.shape, outs_map.shape, valid_all_mask.shape)
        return outs_kp, outs_map.detach(), valid_all_mask 

    def cut_patch_from_featuremap_pad_inner(self, img_batch, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape        # 136x40
        img_resize = TF.resize(img_batch, [h + 1, w + 1])    # 137x41
        # b_r, _, h_r, w_r = img_resize.shape
        flip_flag = random.choice([0, 1])

        # inner 
        img_pad_batch = deepcopy(img_resize)

        # origin
        # img_pad_size = (2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        # img_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [153, 57]

        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x39x15
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
                # del results_batch
        # with torch.no_grad():
        #     outs_map = self.descriptor_net(data, angle=flip_flag)
        #     del data

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            # kmask = keypoints[:, 0] >= (2 + 1) / 2 * (w-1) and keypoints[:, 0] <= (w - 3 + 1) / 2 * (w-1)
            # keypoints = keypoints[kmask]
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 
        
        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            # print(orient_batch)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)

        # orient_batch = orient_batch + big_mask * 180
        # orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch + big_mask * 180
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        # pad_zero_mask = torch.zeros_like(img_batch, device=img_batch.device)
        # pad_zero_mask[:, :, :, 2:-2] = 1        # bxcxhxw

        # 切描述patch 
        downsampe_ratio = 4
        # 随机±5度增强，兼容sift计算的角度
        rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)

        x = (torch.linspace(0, w-1, w) * (w / (w - 1)) + 2 * self.sample_desc_patch) / downsampe_ratio      # ex: [-2, -1, 0, 1, 2] 137x41->161x65
        y = (torch.linspace(0, h-1, h) * (h / (h - 1)) + 2 * self.sample_desc_patch) / downsampe_ratio
        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 2
        # res_pad_size = 0
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + res_pad_size
        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1])
        keypoints_expand_resize = keypoints_expand_mid * keypoints_expand.new_tensor([w / (w - 1), h / (h - 1)])  # 137x41
        keypoints_expand = ((keypoints_expand_resize + 2 * self.sample_desc_patch) / downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x41x17
        # print(feats_pad_batch.shape, keypoints_expand.max(0))
        outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        # print(feats_pad_batch.shape)
        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        
        # outs = self.dis_sigma_weight.repeat(outs.shape[0], 1) * outs
        # outs_kp = self.dis_sigma_weight.repeat(outs_kp.shape[0], 1) * outs_kp

        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w) 
        # print(outs_kp.shape, outs_map.shape, valid_all_mask.shape)
        return outs_kp, outs_map.detach(), valid_all_mask 

    def cut_patch_from_featuremap_pad_inner_ext(self, img_batch, img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None, part_train_mask=None):
        b, _, h, w = img_batch.shape        # 136x40
        _, _, he, we = img_ext.shape        # 144x52
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 145x53
        # b_r, _, h_r, w_r = img_resize.shape
        flip_flag = random.choice([0, 1])

        # inner 
        img_pad_batch = deepcopy(img_resize)

        # origin
        # img_pad_size = (2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        # img_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [153, 57]

        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x41x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
                # del results_batch
        # with torch.no_grad():
        #     outs_map = self.descriptor_net(data, angle=flip_flag)
        #     del data

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        keypoints_new = []
        kmask_all = []
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ex = ((keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]])).to(img_batch.device)
            # kmask = (keypoints[:, 0] >= (2 / (w-1) * 2 - 1)) * (keypoints[:, 0] <= ((w-3) / (w-1) * 2 - 1))       # 在中心136 x 36 中
            kmask = (keypoints_ex >= torch.tensor([4, 2], device=img_batch.device)) * (keypoints_ex <= (torch.tensor([w-5, h-3], device=img_batch.device)))    # 132x32
            kmask = (torch.prod(kmask, dim=-1) == 1)

            keypoints = keypoints[kmask]
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 
        
        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            # print(orient_batch)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)

        # orient_batch = orient_batch + big_mask * 180
        # orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x_o = torch.linspace(0, w-5, w-4) + 2       # ex: [-2, -1, 0, 1, 2] 中心136x36的网格点
        y_o = torch.linspace(0, h-1, h)
        mesh_points_o = torch.stack(torch.meshgrid([y_o, x_o])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points_o = mesh_points_o / mesh_points_o.new_tensor([w - 1, h - 1]) * 2 - 1
        big_mask_o = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        mesh_points_o.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_o = (big_mask_o >= 0.5).int()         # {0, 1}
        orient_batch_o = F.grid_sample(orient_batch.view(b, 1, h, w),
                        mesh_points_o.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_o = orient_batch_o + big_mask_o * 180

        # pad_zero_mask = torch.zeros_like(img_batch, device=img_batch.device)
        # pad_zero_mask[:, :, :, 2:-2] = 1        # bxcxhxw

        # 切描述patch 
        downsampe_ratio = 4
        # 随机±5度增强，兼容sift计算的角度
        rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        # print(h, w, he, we)
        x = ((torch.linspace(0, w-5, w-4) + (we - w + 4) / 2) * (we / (we-1)) + self.sample_desc_patch) / downsampe_ratio      # ex: [-2, -1, 0, 1, 2] 144x52->161x61
        y = ((torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he-1)) + self.sample_desc_patch + 4) / downsampe_ratio
        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 2
        # res_pad_size = 0
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + res_pad_size
        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1])
        keypoints_expand_mid[:, 0] += (we - w) / 2              # 6
        keypoints_expand_mid[:, 1] += (he - h) / 2              # 4
        keypoints_expand_resize = keypoints_expand_mid * keypoints_expand_mid.new_tensor([we / (we - 1), he / (he - 1)])  # 137x41
        keypoints_expand_resize[:, 0] += self.sample_desc_patch
        keypoints_expand_resize[:, 1] += self.sample_desc_patch + 4
        keypoints_expand = (keypoints_expand_resize / downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x41x16
        # print(feats_pad_batch.shape, keypoints_expand.max(0))
        outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch_o.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        # print(feats_pad_batch.shape)
        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        
        # outs = self.dis_sigma_weight.repeat(outs.shape[0], 1) * outs
        # outs_kp = self.dis_sigma_weight.repeat(outs_kp.shape[0], 1) * outs_kp

        # 标准差 类似于信息熵
        # outs_kp_entropy = torch.std(outs_kp, dim=1, unbiased=False, keepdim=True)[valid_all_mask.bool(), :] 

        # # mask64
        # outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        # outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        # mask96
        outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 
        # print(outs_kp.shape, outs_map.shape, valid_all_mask.shape)
        return outs_kp, outs_map.detach(), valid_all_mask, keypoints_new, kmask_all
        # return outs_kp, outs_map.detach(), valid_all_mask, keypoints_new, kmask_all, outs_kp_entropy

    def cut_patch_from_featuremap_pad_inner_ext93(self, img_batch, img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None, part_train_mask=None):
        b, _, h, w = img_batch.shape        # 120x40
        _, _, he, we = img_ext.shape        # 128x52
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        # b_r, _, h_r, w_r = img_resize.shape
        flip_flag = random.choice([0, 1])

        # inner 
        img_pad_batch = deepcopy(img_resize)

        # origin
        # img_pad_size = (2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        # img_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [153, 57]

        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x41x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
                # del results_batch
        # with torch.no_grad():
        #     outs_map = self.descriptor_net(data, angle=flip_flag)
        #     del data

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        keypoints_new = []
        kmask_all = []
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ex = ((keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]])).to(img_batch.device)
            # kmask = (keypoints[:, 0] >= (2 / (w-1) * 2 - 1)) * (keypoints[:, 0] <= ((w-3) / (w-1) * 2 - 1))       # 在中心136 x 36 中
            kmask = (keypoints_ex >= torch.tensor([4, 1], device=img_batch.device)) * (keypoints_ex <= (torch.tensor([w-5, h-2], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)

            keypoints = keypoints[kmask]
            keypoints_new.append(keypoints)
            kmask_all.append(kmask)
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 
        
        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            # print(orient_batch)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)

        # orient_batch = orient_batch + big_mask * 180
        # orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x_o = torch.linspace(0, w-5, w-4) + 2       # ex: [-2, -1, 0, 1, 2] 中心120x36的网格点
        y_o = torch.linspace(0, h-1, h)
        mesh_points_o = torch.stack(torch.meshgrid([y_o, x_o])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points_o = mesh_points_o / mesh_points_o.new_tensor([w - 1, h - 1]) * 2 - 1
        big_mask_o = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        mesh_points_o.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_o = (big_mask_o >= 0.5).int()         # {0, 1}
        orient_batch_o = F.grid_sample(orient_batch.view(b, 1, h, w),
                        mesh_points_o.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_o = orient_batch_o + big_mask_o * 180

        # pad_zero_mask = torch.zeros_like(img_batch, device=img_batch.device)
        # pad_zero_mask[:, :, :, 2:-2] = 1        # bxcxhxw

        # 切描述patch 
        downsampe_ratio = 4
        # 随机±5度增强，兼容sift计算的角度
        rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        # print(h, w, he, we)
        x = ((torch.linspace(0, w-5, w-4) + (we - w + 4) / 2) * (we / (we-1)) + self.sample_desc_patch) / downsampe_ratio      # ex: [-2, -1, 0, 1, 2] 128x52->145x61
        y = ((torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he-1)) + self.sample_desc_patch + 4) / downsampe_ratio
        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 2
        # res_pad_size = 0
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + res_pad_size
        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1])
        keypoints_expand_mid[:, 0] += (we - w) / 2              # 6
        keypoints_expand_mid[:, 1] += (he - h) / 2              # 4
        keypoints_expand_resize = keypoints_expand_mid * keypoints_expand_mid.new_tensor([we / (we - 1), he / (he - 1)])  # 129x53
        keypoints_expand_resize[:, 0] += self.sample_desc_patch
        keypoints_expand_resize[:, 1] += self.sample_desc_patch + 4
        keypoints_expand = (keypoints_expand_resize / downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x41x16
        # print(feats_pad_batch.shape, keypoints_expand.max(0))
        outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch_o.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        # print(feats_pad_batch.shape)
        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        
        # outs = self.dis_sigma_weight.repeat(outs.shape[0], 1) * outs
        # outs_kp = self.dis_sigma_weight.repeat(outs_kp.shape[0], 1) * outs_kp

        # 标准差 类似于信息熵
        # outs_kp_entropy = torch.std(outs_kp, dim=1, unbiased=False, keepdim=True)[valid_all_mask.bool(), :] 

        # # mask64
        # outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        # outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        # mask96
        outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 
        # print(outs_kp.shape, outs_map.shape, valid_all_mask.shape)
        return outs_kp, outs_map.detach(), valid_all_mask, keypoints_new, kmask_all
        # return outs_kp, outs_map.detach(), valid_all_mask, keypoints_new, kmask_all, outs_kp_entropy

    def cut_patch_from_featuremap_pad_inner_ext93_oriR(self, img_batch, img_ext, img_ext_T, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None, part_train_mask=None, h_ATB=None):
        b, _, h, w = img_batch.shape        # 120x40
        _, _, he, we = img_ext.shape        # 128x52
        
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        img_resize_T = TF.resize(img_ext_T, [he + 1, we + 1])    # 129x53

        flip_flag = random.choice([0, 1])

        # inner 
        img_pad_batch = deepcopy(img_resize)  # 129 x 53

        # origin
        img_pad_size = (self.sample_desc_patch, self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_ext_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [145, 61]
        img_ext_T_pad_batch = F.pad(img_resize_T, img_pad_size, "constant", 0)   # [145, 61]

        hm, wm = img_ext_pad_batch.shape[2], img_ext_pad_batch.shape[3]   # [145, 61]

        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x37x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        keypoints_expand_T = None         # (b x max_num)x2
        expand_mask_all_T = None          # (b x max_mum)x1
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        keypoints_new = []
        kmask_all = []
        keypoints_new_T = []
        kmask_all_T = []
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ex = ((keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]])).to(img_batch.device)
           
            # 中心118x32
            kmask = (keypoints_ex >= torch.tensor([4, 1], device=img_batch.device)) * (keypoints_ex <= (torch.tensor([w-5, h-2], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            
            # 点转到AT上
            keypoints_T = warp_points(keypoints_ex[kmask][:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            keypoints_T = keypoints_T[kmask_T]
            keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(keypoints_T.device) * 2 - 1
            
            keypoints = keypoints[kmask][kmask_T]

            keypoints_new.append(keypoints)
            kmask_all.append(kmask)
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            keypoints_new_T.append(keypoints_T)
            kmask_all_T.append(kmask_T)
            expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask_T[:keypoints_T.shape[0]] = 1
            if keypoints_expand_T is None: 
                keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
                expand_mask_all_T = expand_mask_T
            else:
                keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
                expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (230 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_ext_pad_batch, device=img_ext_pad_batch.device)      # Bx1x(hexwe)
            orient_batch_T = self.get_orientation_batch(img_ext_T_pad_batch, device=img_ext_T_pad_batch.device)      # Bx1x(hexwe)

        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)           # Bx1x(hexwe)

        # 切描述patch 
        downsampe_ratio = 4
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-5, 5)
        rand_angle_enhance = 0

        x = (torch.linspace(0, w-5, w-4) + (we - w + 4) / 2) * (we / (we-1)) + self.sample_desc_patch
        xd = x / downsampe_ratio      # ex: [-2, -1, 0, 1, 2] 128x52->145x61
        y = (torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he-1)) + self.sample_desc_patch + 4
        yd = y / downsampe_ratio
        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 0

        mesh_points_o = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points_o_norm = mesh_points_o / mesh_points_o.new_tensor([wm - 1, hm - 1]).to(mesh_points_o.device) * 2 - 1

        mesh_points = torch.stack(torch.meshgrid([yd, xd])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + res_pad_size

        # 转换点 [w, h] -> [wm, hm]
        keypoints_expand_resize, keypoints_expand_resize_norm = self.transform_keypoints(keypoints_expand, w, h, we, he, wm, hm)
        keypoints_expand_resize_T, keypoints_expand_resize_norm_T = self.transform_keypoints(keypoints_expand_T, w, h, we, he, wm, hm)
        # new 
        # key points
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, hm, wm),
                        keypoints_expand_resize_norm.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, hm, wm),
                        keypoints_expand_resize_norm.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180
        # grid points
        big_mask_o = F.grid_sample(big_mask.float().view(b, 1, hm, wm),
                        mesh_points_o_norm.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_o = (big_mask_o >= 0.5).int()         # {0, 1}
        orient_batch_o = F.grid_sample(orient_batch.view(b, 1, hm, wm),
                        mesh_points_o_norm.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_T_o = orient_batch_o.view(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 

        orient_batch_o = orient_batch_o + big_mask_o * 180
        
        # # AT图上 
        # # 中心点主方向
        # orient_batch_T_o = F.grid_sample(orient_batch_T.view(b, 1, hm, wm),
        #                 mesh_points_o_norm.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        # orient_batch_T_o = orient_batch_T_o.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 

        orient_batch_T_kp = F.grid_sample(orient_batch_T.view(b, 1, hm, wm),
                                keypoints_expand_resize_norm_T.view(b, 1, -1, 2),
                                mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        # 4x4邻域方向统计
        ori_zeros = torch.zeros_like(orient_batch_o).to(img_batch.device).view(-1)
        ori_zeros_kp = torch.zeros_like(orient_batch_T_kp).to(img_batch.device).view(-1)
        outs_ori = inv_warp_patch_batch(orient_batch.contiguous().view(-1, 1, hm, wm), mesh_points_o.repeat(b, 1, 1).view(-1, 2), ori_zeros, self.desc_patch, self.desc_patch * downsampe_ratio).reshape(b, -1, self.desc_patch, self.desc_patch).contiguous().view(-1, self.desc_patch * self.desc_patch)  
        outs_ori = outs_ori.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)     # B x 16 x 120 x 36

        outs_kp_ori = inv_warp_patch_batch(orient_batch_T.contiguous().view(-1, 1, hm, wm), keypoints_expand_resize_T.view(b, 1, -1, 2).view(-1, 2), ori_zeros_kp, self.desc_patch, self.desc_patch * downsampe_ratio).reshape(b, -1, self.desc_patch, self.desc_patch).contiguous().view(-1, self.desc_patch * self.desc_patch)  
        
        valid_all_mask_T = (expand_mask_all_T == 1)
        orient_batch_T_kp = orient_batch_T_kp.view(-1)[valid_all_mask_T]
        outs_kp_ori = outs_kp_ori[valid_all_mask_T, :]

        # 描述子计算
        keypoints_expand = (keypoints_expand_resize / downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x37x16
        outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch_o.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  

        # 标准差 类似于信息熵
        # outs_kp_entropy = torch.std(outs_kp, dim=1, unbiased=False, keepdim=True)[valid_all_mask.bool(), :] 

        # # mask64
        # outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        # outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        # # mask96
        # outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        # outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 

        return outs_kp, outs_map.detach(), valid_all_mask, keypoints_new, kmask_all, orient_batch_T_o, outs_ori, orient_batch_T_kp, outs_kp_ori, kmask_all_T


    def cut_patch_from_featuremap_pad_inner_ext93_oriR_bw(self, img_batch, img_ext, img_ext_T, bin_img_ext, bin_img_ext_T, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None, part_train_mask=None, h_ATB=None):
        b, _, h, w = img_batch.shape        # 120x40
        _, _, he, we = img_ext.shape        # 128x52
        
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        img_resize_T = TF.resize(img_ext_T, [he + 1, we + 1])    # 129x53

        bin_img_resize = TF.resize(bin_img_ext, [he + 1, we + 1])    # 129x53
        bin_img_resize_T = TF.resize(bin_img_ext_T, [he + 1, we + 1])    # 129x53

        flip_flag = random.choice([0, 1])

        # inner 
        img_pad_batch = deepcopy(img_resize)  # 129 x 53

        # origin
        img_pad_size = (self.sample_desc_patch, self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_ext_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [145, 61]
        img_ext_T_pad_batch = F.pad(img_resize_T, img_pad_size, "constant", 0)   # [145, 61]

        bin_img_ext_pad_batch = F.pad(bin_img_resize, img_pad_size, "constant", 0)   # [145, 61]
        bin_img_ext_T_pad_batch = F.pad(bin_img_resize_T, img_pad_size, "constant", 0)   # [145, 61]

        hm, wm = img_ext_pad_batch.shape[2], img_ext_pad_batch.shape[3]   # [145, 61]

        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x37x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        keypoints_expand_T = None         # (b x max_num)x2
        expand_mask_all_T = None          # (b x max_mum)x1
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        keypoints_new = []
        kmask_all = []
        keypoints_new_T = []
        kmask_all_T = []
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ex = ((keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]])).to(img_batch.device)
           
            # 中心118x32
            kmask = (keypoints_ex >= torch.tensor([4, 1], device=img_batch.device)) * (keypoints_ex <= (torch.tensor([w-5, h-2], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            
            # 点转到AT上
            keypoints_T = warp_points(keypoints_ex[kmask][:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            keypoints_T = keypoints_T[kmask_T]
            keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(keypoints_T.device) * 2 - 1
            
            keypoints = keypoints[kmask][kmask_T]

            keypoints_new.append(keypoints)
            kmask_all.append(kmask)
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            keypoints_new_T.append(keypoints_T)
            kmask_all_T.append(kmask_T)
            expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask_T[:keypoints_T.shape[0]] = 1
            if keypoints_expand_T is None: 
                keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
                expand_mask_all_T = expand_mask_T
            else:
                keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
                expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 

        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (230 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_ext_pad_batch, device=img_ext_pad_batch.device)      # Bx1x(hexwe)
            orient_batch_T = self.get_orientation_batch(img_ext_T_pad_batch, device=img_ext_T_pad_batch.device)      # Bx1x(hexwe)

        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)           # Bx1x(hexwe)

        # 切描述patch 
        downsampe_ratio = 4
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-5, 5)
        rand_angle_enhance = 0

        x = (torch.linspace(0, w-5, w-4) + (we - w + 4) / 2) * (we / (we-1)) + self.sample_desc_patch
        xd = x / downsampe_ratio      # ex: [-2, -1, 0, 1, 2] 128x52->145x61
        y = (torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he-1)) + self.sample_desc_patch + 4
        yd = y / downsampe_ratio
        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 0

        mesh_points_o = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points_o_norm = mesh_points_o / mesh_points_o.new_tensor([wm - 1, hm - 1]).to(mesh_points_o.device) * 2 - 1

        mesh_points = torch.stack(torch.meshgrid([yd, xd])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + res_pad_size

        # 转换点 [w, h] -> [wm, hm]
        keypoints_expand_resize, keypoints_expand_resize_norm = self.transform_keypoints(keypoints_expand, w, h, we, he, wm, hm)
        keypoints_expand_resize_T, keypoints_expand_resize_norm_T = self.transform_keypoints(keypoints_expand_T, w, h, we, he, wm, hm)
        # new 
        # key points
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, hm, wm),
                        keypoints_expand_resize_norm.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, hm, wm),
                        keypoints_expand_resize_norm.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180
        # grid points
        big_mask_o = F.grid_sample(big_mask.float().view(b, 1, hm, wm),
                        mesh_points_o_norm.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_o = (big_mask_o >= 0.5).int()         # {0, 1}
        orient_batch_o = F.grid_sample(orient_batch.view(b, 1, hm, wm),
                        mesh_points_o_norm.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_T_o = orient_batch_o.view(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 

        orient_batch_o = orient_batch_o + big_mask_o * 180
        
        # # AT图上 
        # # 中心点主方向
        # orient_batch_T_o = F.grid_sample(orient_batch_T.view(b, 1, hm, wm),
        #                 mesh_points_o_norm.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        # orient_batch_T_o = orient_batch_T_o.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 

        orient_batch_T_kp = F.grid_sample(orient_batch_T.view(b, 1, hm, wm),
                                keypoints_expand_resize_norm_T.view(b, 1, -1, 2),
                                mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        # 4x4邻域方向统计
        ori_zeros = torch.zeros_like(orient_batch_o).to(img_batch.device).view(-1)
        ori_zeros_kp = torch.zeros_like(orient_batch_T_kp).to(img_batch.device).view(-1)
        outs_ori = inv_warp_patch_batch(orient_batch.contiguous().view(-1, 1, hm, wm), mesh_points_o.repeat(b, 1, 1).view(-1, 2), ori_zeros, self.desc_patch, self.desc_patch * downsampe_ratio).reshape(b, -1, self.desc_patch, self.desc_patch).contiguous().view(-1, self.desc_patch * self.desc_patch)  
        outs_ori = outs_ori.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)     # B x 16 x 120 x 36

        outs_kp_ori = inv_warp_patch_batch(orient_batch_T.contiguous().view(-1, 1, hm, wm), keypoints_expand_resize_T.view(b, 1, -1, 2).view(-1, 2), ori_zeros_kp, self.desc_patch, self.desc_patch * downsampe_ratio).reshape(b, -1, self.desc_patch, self.desc_patch).contiguous().view(-1, self.desc_patch * self.desc_patch)  
        
        valid_all_mask_T = (expand_mask_all_T == 1)
        orient_batch_T_kp = orient_batch_T_kp.view(-1)[valid_all_mask_T]
        outs_kp_ori = outs_kp_ori[valid_all_mask_T, :]

        # 描述子计算
        keypoints_expand = (keypoints_expand_resize / downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x37x16
        outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch_o.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  

        # 切二值块
        bin_all = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points_o.repeat(b, 1, 1).view(-1, 2), orient_batch_o.view(-1) + cal_angle + rand_angle_enhance, self.desc_patch * downsampe_ratio, self.desc_patch * downsampe_ratio).unsqueeze(1).view(-1, self.desc_patch * downsampe_ratio * self.desc_patch * downsampe_ratio)  
        # bin_T_all = inv_warp_patch_batch(bin_img_ext_T_pad_batch, mesh_points_o.repeat(b, 1, 1).view(-1, 2), torch.zeros_like(orient_batch_o, device=orient_batch_o.device).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch * downsampe_ratio, self.desc_patch * downsampe_ratio).unsqueeze(1).view(-1, self.desc_patch * downsampe_ratio * self.desc_patch * downsampe_ratio)  
        bin_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand_resize, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, self.desc_patch * downsampe_ratio, self.desc_patch * downsampe_ratio)[valid_all_mask.bool(), :, : ].unsqueeze(1).view(-1, self.desc_patch * downsampe_ratio * self.desc_patch * downsampe_ratio)   
        bin_T_kp = inv_warp_patch_batch(bin_img_ext_T_pad_batch, keypoints_expand_resize_T, ori_zeros_kp + cal_angle + rand_angle_enhance, self.desc_patch * downsampe_ratio, self.desc_patch * downsampe_ratio)[valid_all_mask_T.bool(), :, : ].unsqueeze(1).view(-1, self.desc_patch * downsampe_ratio * self.desc_patch * downsampe_ratio)   
        # print(bin_T_kp.shape, bin_T_kp[0, :])
        bin_all = bin_all.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)
        # bin_T_all = bin_T_all.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)

        # 标准差 类似于信息熵
        # outs_kp_entropy = torch.std(outs_kp, dim=1, unbiased=False, keepdim=True)[valid_all_mask.bool(), :] 

        # # mask64
        # outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        # outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        # # mask96
        # outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        # outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 

        return outs_kp, outs_map.detach(), valid_all_mask, keypoints_new, kmask_all, orient_batch_T_o, outs_ori, orient_batch_T_kp, outs_kp_ori, kmask_all_T, bin_all, bin_kp, bin_T_kp



    def cut_patch_from_featuremap_pad_inner_ext_modify(self, img_batch, img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None, h_en=None, part_train_mask=None):
        b, _, h, w = img_batch.shape        # 136x40
        _, _, he, we = img_ext.shape        # 144x52
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 145x53
        # b_r, _, h_r, w_r = img_resize.shape
        flip_flag = random.choice([0, 1])

        # inner 
        img_pad_batch = deepcopy(img_resize)

        # origin
        # img_pad_size = (2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        # img_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [153, 57]

        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x41x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
                # del results_batch
        # with torch.no_grad():
        #     outs_map = self.descriptor_net(data, angle=flip_flag)
        #     del data

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        keypoints_new = []
        kmask_all = []
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ex = ((keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]])).to(img_batch.device)
            keypoints_H = warp_points(keypoints_ex[:, :2], h_en[batch_idx].squeeze(), device=img_batch.device)  # 利用变换矩阵变换坐标点

            kmask = (keypoints_H >= torch.tensor([4, 2], device=img_batch.device)) * (keypoints_H <= (torch.tensor([w-5, h-3], device=img_batch.device)))    # 132x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            
            keypoints = keypoints_H[kmask]
            keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(keypoints.device) * 2 - 1

            keypoints_new.append(keypoints)
            kmask_all.append(kmask)
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 
        
        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            # print(orient_batch)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)

        # orient_batch = orient_batch + big_mask * 180
        # orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x_o = torch.linspace(0, w-5, w-4) + 2       # ex: [-2, -1, 0, 1, 2] 中心136x36的网格点
        y_o = torch.linspace(0, h-1, h)
        mesh_points_o = torch.stack(torch.meshgrid([y_o, x_o])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points_o = mesh_points_o / mesh_points_o.new_tensor([w - 1, h - 1]) * 2 - 1
        big_mask_o = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        mesh_points_o.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_o = (big_mask_o >= 0.5).int()         # {0, 1}
        orient_batch_o = F.grid_sample(orient_batch.view(b, 1, h, w),
                        mesh_points_o.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_o = orient_batch_o + big_mask_o * 180

        # pad_zero_mask = torch.zeros_like(img_batch, device=img_batch.device)
        # pad_zero_mask[:, :, :, 2:-2] = 1        # bxcxhxw

        # 切描述patch 
        downsampe_ratio = 4
        # 随机±5度增强，兼容sift计算的角度
        rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        # print(h, w, he, we)
        x = ((torch.linspace(0, w-5, w-4) + (we - w + 4) / 2) * (we / (we-1)) + self.sample_desc_patch) / downsampe_ratio      # ex: [-2, -1, 0, 1, 2] 144x52->161x61
        y = ((torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he-1)) + self.sample_desc_patch + 4) / downsampe_ratio
        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 2
        # res_pad_size = 0
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + res_pad_size
        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1])
        keypoints_expand_mid[:, 0] += (we - w) / 2              # 6
        keypoints_expand_mid[:, 1] += (he - h) / 2              # 4
        keypoints_expand_resize = keypoints_expand_mid * keypoints_expand_mid.new_tensor([we / (we - 1), he / (he - 1)])  # 137x41
        keypoints_expand_resize[:, 0] += self.sample_desc_patch
        keypoints_expand_resize[:, 1] += self.sample_desc_patch + 4
        keypoints_expand = (keypoints_expand_resize / downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x41x16
        # print(feats_pad_batch.shape, keypoints_expand.max(0))
        outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch_o.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        # print(feats_pad_batch.shape)
        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        
        # outs = self.dis_sigma_weight.repeat(outs.shape[0], 1) * outs
        # outs_kp = self.dis_sigma_weight.repeat(outs_kp.shape[0], 1) * outs_kp

        # 标准差 类似于信息熵
        # outs_kp_entropy = torch.std(outs_kp, dim=1, unbiased=False, keepdim=True)[valid_all_mask.bool(), :] 

        # mask64
        # outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        # outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        # mask96
        outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 
        # print(outs_kp.shape, outs_map.shape, valid_all_mask.shape)

        return outs_kp, outs_map.detach(), valid_all_mask, keypoints_new, kmask_all
        # return outs_kp, outs_map.detach(), valid_all_mask, keypoints_new, kmask_all, outs_kp_entropy

    def cut_patch_from_featuremap_pad_inner_ext_modify93(self, img_batch, img_ext, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None, h_en=None, part_train_mask=None):
        b, _, h, w = img_batch.shape        # 120x40
        _, _, he, we = img_ext.shape        # 128x52
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        # b_r, _, h_r, w_r = img_resize.shape
        flip_flag = random.choice([0, 1])

        # inner 
        img_pad_batch = deepcopy(img_resize)

        # origin
        # img_pad_size = (2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        # img_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [153, 57]

        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x37x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
                # del results_batch
        # with torch.no_grad():
        #     outs_map = self.descriptor_net(data, angle=flip_flag)
        #     del data

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        keypoints_new = []
        kmask_all = []
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ex = ((keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]])).to(img_batch.device)
            keypoints_H = warp_points(keypoints_ex[:, :2], h_en[batch_idx].squeeze(), device=img_batch.device)  # 利用变换矩阵变换坐标点

            kmask = (keypoints_H >= torch.tensor([4, 1], device=img_batch.device)) * (keypoints_H <= (torch.tensor([w-5, h-2], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)
            
            keypoints = keypoints_H[kmask]
            keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(keypoints.device) * 2 - 1

            keypoints_new.append(keypoints)
            kmask_all.append(kmask)
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 
        
        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            # print(orient_batch)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)

        # orient_batch = orient_batch + big_mask * 180
        # orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        x_o = torch.linspace(0, w-5, w-4) + 2       # ex: [-2, -1, 0, 1, 2] 中心136x36的网格点
        y_o = torch.linspace(0, h-1, h)
        mesh_points_o = torch.stack(torch.meshgrid([y_o, x_o])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points_o = mesh_points_o / mesh_points_o.new_tensor([w - 1, h - 1]) * 2 - 1
        big_mask_o = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        mesh_points_o.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_o = (big_mask_o >= 0.5).int()         # {0, 1}
        orient_batch_o = F.grid_sample(orient_batch.view(b, 1, h, w),
                        mesh_points_o.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_o = orient_batch_o + big_mask_o * 180

        # pad_zero_mask = torch.zeros_like(img_batch, device=img_batch.device)
        # pad_zero_mask[:, :, :, 2:-2] = 1        # bxcxhxw

        # 切描述patch 
        downsampe_ratio = 4
        # 随机±5度增强，兼容sift计算的角度
        rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)
        # print(h, w, he, we)
        x = ((torch.linspace(0, w-5, w-4) + (we - w + 4) / 2) * (we / (we-1)) + self.sample_desc_patch) / downsampe_ratio      # ex: [-2, -1, 0, 1, 2] 128x52->145x61
        y = ((torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he-1)) + self.sample_desc_patch + 4) / downsampe_ratio
        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 2
        # res_pad_size = 0
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + res_pad_size
        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1])
        keypoints_expand_mid[:, 0] += (we - w) / 2              # 6
        keypoints_expand_mid[:, 1] += (he - h) / 2              # 4
        keypoints_expand_resize = keypoints_expand_mid * keypoints_expand_mid.new_tensor([we / (we - 1), he / (he - 1)])  # 137x41
        keypoints_expand_resize[:, 0] += self.sample_desc_patch
        keypoints_expand_resize[:, 1] += self.sample_desc_patch + 4
        keypoints_expand = (keypoints_expand_resize / downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x41x16
        # print(feats_pad_batch.shape, keypoints_expand.max(0))
        outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch_o.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        # print(feats_pad_batch.shape)
        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        
        # outs = self.dis_sigma_weight.repeat(outs.shape[0], 1) * outs
        # outs_kp = self.dis_sigma_weight.repeat(outs_kp.shape[0], 1) * outs_kp

        # 标准差 类似于信息熵
        # outs_kp_entropy = torch.std(outs_kp, dim=1, unbiased=False, keepdim=True)[valid_all_mask.bool(), :] 

        # # mask64
        # outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        # outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        # mask96
        outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 
        # print(outs_kp.shape, outs_map.shape, valid_all_mask.shape)

        return outs_kp, outs_map.detach(), valid_all_mask, keypoints_new, kmask_all
        # return outs_kp, outs_map.detach(), valid_all_mask, keypoints_new, kmask_all, outs_kp_entropy

    def cut_patch_from_featuremap_pad_inner_ext_modify93_oriR(self, img_batch, img_ext, img_ext_T, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None, h_en=None, part_train_mask=None, h_ATB=None):
        b, _, h, w = img_batch.shape        # 120x40
        _, _, he, we = img_ext.shape        # 128x52
        
        # Resize 成奇数 
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        img_resize_T = TF.resize(img_ext_T, [he + 1, we + 1])    # 129x53

        flip_flag = random.choice([0, 1])

        # inner 
        img_pad_batch = deepcopy(img_resize)

        # origin
        img_pad_size = (self.sample_desc_patch, self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_ext_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [145, 61]
        img_ext_T_pad_batch = F.pad(img_resize_T, img_pad_size, "constant", 0)   # [145, 61]

        hm, wm = img_ext_pad_batch.shape[2], img_ext_pad_batch.shape[3]  # [145, 61]

        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x37x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        keypoints_expand_T = None         # (b x max_num)x2
        expand_mask_all_T = None          # (b x max_mum)x1
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        keypoints_new = []
        kmask_all = []
        keypoints_new_T = []
        kmask_all_T = []
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ex = ((keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]])).to(img_batch.device)

            # 点转到AE上
            keypoints_H = warp_points(keypoints_ex[:, :2], h_en[batch_idx].squeeze(), device=img_batch.device)  # 利用变换矩阵变换坐标点
            # 中心118 x 32
            kmask = (keypoints_H >= torch.tensor([4, 1], device=img_batch.device)) * (keypoints_H <= (torch.tensor([w-5, h-2], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)

            # 点转到AT上
            keypoints_T = warp_points(keypoints_H[kmask][:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            keypoints_T = keypoints_T[kmask_T]

            keypoints = keypoints_H[kmask][kmask_T]
            keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(keypoints.device) * 2 - 1
            keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(keypoints_T.device) * 2 - 1

            keypoints_new.append(keypoints)
            kmask_all.append(kmask)
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            keypoints_new_T.append(keypoints_T)
            kmask_all_T.append(kmask_T)
            expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask_T[:keypoints_T.shape[0]] = 1
            if keypoints_expand_T is None: 
                keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
                expand_mask_all_T = expand_mask_T
            else:
                keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
                expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 
        
        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (230 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            # print(orient_batch)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_ext_pad_batch, device=img_ext_pad_batch.device)      # Bx1x(hexwe)
            orient_batch_T = self.get_orientation_batch(img_ext_T_pad_batch, device=img_ext_T_pad_batch.device)      # Bx1x(hexwe)

        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)           # Bx1x(hexwe)

        # 切描述patch 
        downsampe_ratio = 4
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-5, 5)
        rand_angle_enhance = 0

        x = (torch.linspace(0, w-5, w-4) + (we - w + 4) / 2) * (we / (we-1)) + self.sample_desc_patch
        xd = x / downsampe_ratio      # ex: [-2, -1, 0, 1, 2] 128x52->145x61
        y = (torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he-1)) + self.sample_desc_patch + 4
        yd = y / downsampe_ratio
        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 2

        mesh_points_o = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points_o_norm = mesh_points_o / mesh_points_o.new_tensor([wm - 1, hm - 1]).to(mesh_points_o.device) * 2 - 1

        mesh_points = torch.stack(torch.meshgrid([yd, xd])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + res_pad_size

        # 转换点 [w, h] -> [wm, hm]
        keypoints_expand_resize, keypoints_expand_resize_norm = self.transform_keypoints(keypoints_expand, w, h, we, he, wm, hm)
        keypoints_expand_resize_T, keypoints_expand_resize_norm_T = self.transform_keypoints(keypoints_expand_T, w, h, we, he, wm, hm)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        # new 
        # key points
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, hm, wm),
                        keypoints_expand_resize_norm.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, hm, wm),
                        keypoints_expand_resize_norm.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        # grid points
        big_mask_o = F.grid_sample(big_mask.float().view(b, 1, hm, wm),
                        mesh_points_o_norm.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_o = (big_mask_o >= 0.5).int()         # {0, 1}
        orient_batch_o = F.grid_sample(orient_batch.view(b, 1, hm, wm),
                        mesh_points_o_norm.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_T_o = orient_batch_o.view(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)

        orient_batch_o = orient_batch_o + big_mask_o * 180

        # 中心点主方向
        # orient_batch_T_o = F.grid_sample(orient_batch_T.view(b, 1, hm, wm),
        #                 mesh_points_o_norm.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        # orient_batch_T_o = orient_batch_T_o.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)

        orient_batch_T_kp = F.grid_sample(orient_batch_T.view(b, 1, hm, wm),
                                keypoints_expand_resize_norm_T.view(b, 1, -1, 2),
                                mode='bilinear', align_corners=True)       # B X 1 X 1 X (n)
        
        # 4x4邻域方向计算
        ori_zeros = torch.zeros_like(orient_batch_o).to(img_batch.device).view(-1)
        ori_zeros_kp = torch.zeros_like(orient_batch_T_kp).to(img_batch.device).view(-1)
        outs_ori = inv_warp_patch_batch(orient_batch.contiguous().view(-1, 1, hm, wm), mesh_points_o.repeat(b, 1, 1).view(-1, 2), ori_zeros, self.desc_patch, self.desc_patch * downsampe_ratio).reshape(b, -1, self.desc_patch, self.desc_patch).contiguous().view(-1, self.desc_patch * self.desc_patch)  
        outs_ori = outs_ori.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)     # B x 16 x 120 x 36

        outs_kp_ori = inv_warp_patch_batch(orient_batch_T.contiguous().view(-1, 1, hm, wm), keypoints_expand_resize_T.view(b, 1, -1, 2).view(-1, 2), ori_zeros_kp, self.desc_patch, self.desc_patch * downsampe_ratio).reshape(b, -1, self.desc_patch, self.desc_patch).contiguous().view(-1, self.desc_patch * self.desc_patch)  
        
        valid_all_mask_T = (expand_mask_all_T == 1)
        orient_batch_T_kp = orient_batch_T_kp.view(-1)[valid_all_mask_T]
        outs_kp_ori = outs_kp_ori[valid_all_mask_T, :]

        # 计算校准后描述子
        keypoints_expand = (keypoints_expand_resize / downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x41x16
        
        outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch_o.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
       
        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        
        # 标准差 类似于信息熵
        # outs_kp_entropy = torch.std(outs_kp, dim=1, unbiased=False, keepdim=True)[valid_all_mask.bool(), :] 

        # # mask64
        # outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        # outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        # # mask96
        # outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        # outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 

        return outs_kp, outs_map.detach(), valid_all_mask, keypoints_new, kmask_all, orient_batch_T_o, outs_ori, orient_batch_T_kp, outs_kp_ori, kmask_all_T


    def cut_patch_from_featuremap_pad_inner_ext_modify93_oriR_bw(self, img_batch, img_ext, img_ext_T, bin_img_ext, bin_img_ext_T, points, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None, h_en=None, part_train_mask=None, h_ATB=None):
        b, _, h, w = img_batch.shape        # 120x40
        _, _, he, we = img_ext.shape        # 128x52
        
        # Resize 成奇数 
        img_resize = TF.resize(img_ext, [he + 1, we + 1])    # 129x53
        img_resize_T = TF.resize(img_ext_T, [he + 1, we + 1])    # 129x53

        bin_img_resize = TF.resize(bin_img_ext, [he + 1, we + 1])    # 129x53
        bin_img_resize_T = TF.resize(bin_img_ext_T, [he + 1, we + 1])    # 129x53
        
        flip_flag = random.choice([0, 1])

        # inner 
        img_pad_batch = deepcopy(img_resize)

        # origin
        img_pad_size = (self.sample_desc_patch, self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        img_ext_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [145, 61]
        img_ext_T_pad_batch = F.pad(img_resize_T, img_pad_size, "constant", 0)   # [145, 61]

        bin_img_ext_pad_batch = F.pad(bin_img_resize, img_pad_size, "constant", 0)   # [145, 61]
        bin_img_ext_T_pad_batch = F.pad(bin_img_resize_T, img_pad_size, "constant", 0)   # [145, 61]

        hm, wm = img_ext_pad_batch.shape[2], img_ext_pad_batch.shape[3]  # [145, 61]

        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x37x16
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        keypoints_expand_T = None         # (b x max_num)x2
        expand_mask_all_T = None          # (b x max_mum)x1
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        keypoints_new = []
        kmask_all = []
        keypoints_new_T = []
        kmask_all_T = []
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_ex = ((keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]])).to(img_batch.device)

            # 点转到AE上
            keypoints_H = warp_points(keypoints_ex[:, :2], h_en[batch_idx].squeeze(), device=img_batch.device)  # 利用变换矩阵变换坐标点
            # 中心118 x 32
            kmask = (keypoints_H >= torch.tensor([4, 1], device=img_batch.device)) * (keypoints_H <= (torch.tensor([w-5, h-2], device=img_batch.device)))    # 118x32
            kmask = (torch.prod(kmask, dim=-1) == 1)

            # 点转到AT上
            keypoints_T = warp_points(keypoints_H[kmask][:, :2], h_ATB[batch_idx].squeeze(), device=img_batch.device)
            kmask_T = (keypoints_T >= torch.tensor([2, 0], device=img_batch.device)) * (keypoints_T <= (torch.tensor([w-3, h-1], device=img_batch.device)))    # 120x36
            kmask_T = (torch.prod(kmask_T, dim=-1) == 1)
            keypoints_T = keypoints_T[kmask_T]

            keypoints = keypoints_H[kmask][kmask_T]
            keypoints = keypoints / keypoints.new_tensor([w - 1, h - 1]).to(keypoints.device) * 2 - 1
            keypoints_T = keypoints_T / keypoints_T.new_tensor([w - 1, h - 1]).to(keypoints_T.device) * 2 - 1

            keypoints_new.append(keypoints)
            kmask_all.append(kmask)
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

            keypoints_new_T.append(keypoints_T)
            kmask_all_T.append(kmask_T)
            expand_mask_T = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask_T[:keypoints_T.shape[0]] = 1
            if keypoints_expand_T is None: 
                keypoints_expand_T = torch.cat((keypoints_T, expand[expand_mask_T==0]), dim=0)
                expand_mask_all_T = expand_mask_T
            else:
                keypoints_expand_T = torch.cat((keypoints_expand_T, keypoints_T, expand[expand_mask_T==0]), dim=0)
                expand_mask_all_T = torch.cat((expand_mask_all_T, expand_mask_T), dim=0) 
        
        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (230 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            # print(orient_batch)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_ext_pad_batch, device=img_ext_pad_batch.device)      # Bx1x(hexwe)
            orient_batch_T = self.get_orientation_batch(img_ext_T_pad_batch, device=img_ext_T_pad_batch.device)      # Bx1x(hexwe)

        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)           # Bx1x(hexwe)

        # 切描述patch 
        downsampe_ratio = 4
        # # 随机±5度增强，兼容sift计算的角度
        # rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-5, 5)
        rand_angle_enhance = 0

        x = (torch.linspace(0, w-5, w-4) + (we - w + 4) / 2) * (we / (we-1)) + self.sample_desc_patch
        xd = x / downsampe_ratio      # ex: [-2, -1, 0, 1, 2] 128x52->145x61
        y = (torch.linspace(0, h-1, h) + (he - h) / 2) * (he / (he-1)) + self.sample_desc_patch + 4
        yd = y / downsampe_ratio
        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 2

        mesh_points_o = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        mesh_points_o_norm = mesh_points_o / mesh_points_o.new_tensor([wm - 1, hm - 1]).to(mesh_points_o.device) * 2 - 1

        mesh_points = torch.stack(torch.meshgrid([yd, xd])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + res_pad_size

        # 转换点 [w, h] -> [wm, hm]
        keypoints_expand_resize, keypoints_expand_resize_norm = self.transform_keypoints(keypoints_expand, w, h, we, he, wm, hm)
        keypoints_expand_resize_T, keypoints_expand_resize_norm_T = self.transform_keypoints(keypoints_expand_T, w, h, we, he, wm, hm)

        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))

        # new 
        # key points
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, hm, wm),
                        keypoints_expand_resize_norm.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, hm, wm),
                        keypoints_expand_resize_norm.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        # grid points
        big_mask_o = F.grid_sample(big_mask.float().view(b, 1, hm, wm),
                        mesh_points_o_norm.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_o = (big_mask_o >= 0.5).int()         # {0, 1}
        orient_batch_o = F.grid_sample(orient_batch.view(b, 1, hm, wm),
                        mesh_points_o_norm.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch_T_o = orient_batch_o.view(-1).reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)

        orient_batch_o = orient_batch_o + big_mask_o * 180

        # 中心点主方向
        # orient_batch_T_o = F.grid_sample(orient_batch_T.view(b, 1, hm, wm),
        #                 mesh_points_o_norm.unsqueeze(0).repeat(b, 1, 1).view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        # orient_batch_T_o = orient_batch_T_o.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)

        orient_batch_T_kp = F.grid_sample(orient_batch_T.view(b, 1, hm, wm),
                                keypoints_expand_resize_norm_T.view(b, 1, -1, 2),
                                mode='bilinear', align_corners=True)       # B X 1 X 1 X (n)
        
        # 4x4邻域方向计算
        ori_zeros = torch.zeros_like(orient_batch_o).to(img_batch.device).view(-1)
        ori_zeros_kp = torch.zeros_like(orient_batch_T_kp).to(img_batch.device).view(-1)
        outs_ori = inv_warp_patch_batch(orient_batch.contiguous().view(-1, 1, hm, wm), mesh_points_o.repeat(b, 1, 1).view(-1, 2), ori_zeros, self.desc_patch, self.desc_patch * downsampe_ratio).reshape(b, -1, self.desc_patch, self.desc_patch).contiguous().view(-1, self.desc_patch * self.desc_patch)  
        outs_ori = outs_ori.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)     # B x 16 x 120 x 36

        outs_kp_ori = inv_warp_patch_batch(orient_batch_T.contiguous().view(-1, 1, hm, wm), keypoints_expand_resize_T.view(b, 1, -1, 2).view(-1, 2), ori_zeros_kp, self.desc_patch, self.desc_patch * downsampe_ratio).reshape(b, -1, self.desc_patch, self.desc_patch).contiguous().view(-1, self.desc_patch * self.desc_patch)  
        
        valid_all_mask_T = (expand_mask_all_T == 1)
        orient_batch_T_kp = orient_batch_T_kp.view(-1)[valid_all_mask_T]
        outs_kp_ori = outs_kp_ori[valid_all_mask_T, :]

        # 计算校准后描述子
        keypoints_expand = (keypoints_expand_resize / downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x41x16
        
        outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch_o.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
       
        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        
        # 切二值块
        bin_all = inv_warp_patch_batch(bin_img_ext_pad_batch, mesh_points_o.repeat(b, 1, 1).view(-1, 2), orient_batch_o.view(-1) + cal_angle + rand_angle_enhance, self.desc_patch * downsampe_ratio, self.desc_patch * downsampe_ratio).unsqueeze(1).view(-1, self.desc_patch * downsampe_ratio * self.desc_patch * downsampe_ratio)  
        # bin_T_all = inv_warp_patch_batch(bin_img_ext_T_pad_batch, mesh_points_o.repeat(b, 1, 1).view(-1, 2), torch.zeros_like(orient_batch_o, device=orient_batch_o.device).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch * downsampe_ratio, self.desc_patch * downsampe_ratio).unsqueeze(1).view(-1, self.desc_patch * downsampe_ratio * self.desc_patch * downsampe_ratio)  
        bin_kp = inv_warp_patch_batch(bin_img_ext_pad_batch, keypoints_expand_resize, orient_batch_kp.view(-1) + cal_angle + rand_angle_enhance, self.desc_patch * downsampe_ratio, self.desc_patch * downsampe_ratio)[valid_all_mask.bool(), :, : ].unsqueeze(1).view(-1, self.desc_patch * downsampe_ratio * self.desc_patch * downsampe_ratio)   
        bin_T_kp = inv_warp_patch_batch(bin_img_ext_T_pad_batch, keypoints_expand_resize_T, ori_zeros_kp + cal_angle + rand_angle_enhance, self.desc_patch * downsampe_ratio, self.desc_patch * downsampe_ratio)[valid_all_mask_T.bool(), :, : ].unsqueeze(1).view(-1, self.desc_patch * downsampe_ratio * self.desc_patch * downsampe_ratio)   

        bin_all = bin_all.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)
        # bin_T_all = bin_T_all.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4)

        
        # 标准差 类似于信息熵
        # outs_kp_entropy = torch.std(outs_kp, dim=1, unbiased=False, keepdim=True)[valid_all_mask.bool(), :] 

        # # mask64
        # outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        # outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        # # mask96
        # outs = torch.cat((outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask], outs[:, part_train_mask]), dim=-1)
        # outs_kp = torch.cat((outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask], outs_kp[:, part_train_mask]), dim=-1)

        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*(w-4), -1).transpose(1, 2).view(b, -1, h, w-4) 

        return outs_kp, outs_map.detach(), valid_all_mask, keypoints_new, kmask_all, orient_batch_T_o, outs_ori, orient_batch_T_kp, outs_kp_ori, kmask_all_T, bin_all, bin_kp, bin_T_kp



    def cut_patch_from_featuremap_pad_inner_sift(self, img_batch, points, siftm, train_flag=False, fixed_angle=None, cal_angle=0, trans_rot_angle=None, pmask=None):
        b, _, h, w = img_batch.shape        # 136x40
        keypoints = [points[i, :, :][siftm[i] == 1, :] for i in range(b)]
        img_resize = TF.resize(img_batch, [h + 1, w + 1])    # 137x41
        # b_r, _, h_r, w_r = img_resize.shape
        flip_flag = random.choice([0, 1])

        # inner 
        img_pad_batch = deepcopy(img_resize)

        # # partial pad
        # img_pad_size = (2 * self.sample_desc_patch - 4, 2 * self.sample_desc_patch - 4, 2 * self.sample_desc_patch - 4, 2 * self.sample_desc_patch - 4)
        # img_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [153, 57]

        # # origin
        # img_pad_size = (2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch, 2 * self.sample_desc_patch)
        # img_pad_batch = F.pad(img_resize, img_pad_size, "constant", 0)   # [161, 65]

        if flip_flag == 1:
            img_batch_flip = torch.flip(torch.flip(img_pad_batch, dims=[2]), dims=[3])  # 输入图像两次镜像等价于旋转180°         
        else:
            img_batch_flip = deepcopy(img_pad_batch)

        if train_flag:
            feats = self.descriptor_net(img_batch_flip, angle=flip_flag) # bx8x39x15
        else:
            with torch.no_grad():
                feats = self.descriptor_net(img_batch_flip, angle=flip_flag)
                # del results_batch
        # with torch.no_grad():
        #     outs_map = self.descriptor_net(data, angle=flip_flag)
        #     del data

        # keypoints_expand = None         # (b x max_num)x2
        # expand_mask_all = None          # (b x max_mum)x1
        # max_num = 400
        # expand = torch.zeros((max_num, 2), device=img_batch.device) 
        # for batch_idx in range(b):
        #     keypoints = points[batch_idx].float()
        #     # kmask = keypoints[:, 0] >= (2 + 1) / 2 * (w-1) and keypoints[:, 0] <= (w - 3 + 1) / 2 * (w-1)
        #     # keypoints = keypoints[kmask]
        #     expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
        #     expand_mask[:keypoints.shape[0]] = 1
        #     if keypoints_expand is None: 
        #         keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
        #         expand_mask_all = expand_mask
        #     else:
        #         keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
        #         expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

        max_num = points.shape[1]
        keypoints_expand = points.view(-1, 2)
        expand_mask_all = siftm.view(-1)
         
        # 部分按压mask
        pmask_kp = F.grid_sample(pmask,
                keypoints_expand.view(b, 1, -1, 2),
                mode='bilinear', align_corners=True) > (200 / 255)        # B X 1 X 1 X (n)
        
        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            # print(orient_batch)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        # 修正本身的A和Bpatch因限定[-90,90]导致的倒Π现象
        big_mask = torch.zeros_like(orient_batch).to(orient_batch.device)
        if trans_rot_angle is not None:
            big_mask = self.get_big_mask(orient_batch, trans_rot=trans_rot_angle)

        # orient_batch = orient_batch + big_mask * 180
        # orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
        #                 keypoints_expand.view(b, 1, -1, 2),
        #                 mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)

        # new 
        big_mask_kp = F.grid_sample(big_mask.float().view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)
        big_mask_kp = (big_mask_kp >= 0.5).int()         # {0, 1}
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (n)
        orient_batch = orient_batch + big_mask * 180
        orient_batch_kp = orient_batch_kp + big_mask_kp * 180

        # pad_zero_mask = torch.zeros_like(img_batch, device=img_batch.device)
        # pad_zero_mask[:, :, :, 2:-2] = 1        # bxcxhxw

        # 切描述patch 
        downsampe_ratio = 4
        # 随机±5度增强，兼容sift计算的角度
        rand_angle_enhance = torch.rand(1, device=orient_batch.device) * 10 - 5 # [-180, 180)

        x = (torch.linspace(0, w-1, w) * (w / (w - 1)) + 2 * self.sample_desc_patch) / downsampe_ratio      # ex: [-2, -1, 0, 1, 2] 137x41->161x65
        y = (torch.linspace(0, h-1, h) * (h / (h - 1)) + 2 * self.sample_desc_patch) / downsampe_ratio
        res_pad_size = (self.desc_patch_expand - self.sample_desc_patch) // 2          # 2
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + res_pad_size
        keypoints_expand_mid = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1])
        keypoints_expand_reszie = keypoints_expand_mid * keypoints_expand.new_tensor([w / (w - 1), h / (h - 1)])  # 137x41
        keypoints_expand = ((keypoints_expand_reszie + 2 * self.sample_desc_patch) / downsampe_ratio).to(img_batch.device) + res_pad_size

        pad_size = (res_pad_size, res_pad_size, res_pad_size, res_pad_size)
        feats_pad_batch = F.pad(feats, pad_size, "constant", 0)     # bx8x41x17

        outs = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), mesh_points.repeat(b*8, 1, 1).view(-1, 2), orient_batch.repeat(1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128) 
        valid_all_mask = (expand_mask_all==1) * (pmask_kp.view(-1))
        
        # bx8xNx4x4
        outs_kp = inv_warp_patch_batch(feats_pad_batch.contiguous().view(-1, 1, feats_pad_batch.shape[-2], feats_pad_batch.shape[-1]), keypoints_expand.view(b, 1, -1, 2).repeat(1, 8, 1, 1).view(-1, 2), orient_batch_kp.repeat(1, 1, 8, 1).view(-1) + cal_angle + rand_angle_enhance, self.desc_patch, self.desc_patch).reshape(b, 8, -1, 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 128)  
        
        # outs = self.dis_sigma_weight.repeat(outs.shape[0], 1) * outs
        # outs_kp = self.dis_sigma_weight.repeat(outs_kp.shape[0], 1) * outs_kp

        outs = L2Norm()(outs)
        outs_kp = L2Norm()(outs_kp)[valid_all_mask.bool(), :] 
        outs_map = outs.reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w) 
        # print(outs_kp.shape, outs_map.shape, valid_all_mask.shape)
        return outs_kp, outs_map.detach(), valid_all_mask, keypoints 


    def forward(self, img, img_ext=None, image_size_max=99999, sort=False, sub_pixel=False, cal_orient=0, trans_rot=None, pmask=None, sift_flag=False, siftp=None, siftm=None, h_en=None, img_crop=None, part_train_mask=None, img_ext_T=None, hT=None, bin_img_ext=None, bin_img_ext_T=None, dense_flag=False, is_teacher=False, linear_act_weight=None, is_bg_img=False):
        """
        :param img: np.array HxWx3, RGB
        :param image_size_max: maximum image size, otherwise, the image will be resized
        :param sort: sort keypoints by scores
        :param sub_pixel: whether to use sub-pixel accuracy
        :return: a dictionary with 'keypoints', 'descriptors', 'scores', and 'time'
        """
        B, C, H, W = img.shape  # [b, 1, 136, 40]

        # ==================== extract keypoints
        start = time.time()
        if self.phase == 'train':

            if linear_act_weight is not None:
                self.descriptor_net.change_act(linear_act_weight)

            if sift_flag:
                assert siftp is not None and siftm is not None
                pad_size = (2, 2, 0, 0)     # 128x52 pad->128x56
                img_point = F.pad(img_ext, pad_size, "constant", 0)
                descriptors, descriptor_map_new, valid_all_mask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_sift(img, img_ext, img_point, bin_img_ext, siftp, siftm, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                scores_new, scores_map, scoredispersitys_new = None, None, None
                # descriptors, descriptor_map_new, valid_all_mask, keypoints = self.cut_patch_from_featuremap_pad_inner_sift(img, siftp, siftm, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                # scores, scores_map, scoredispersitys = None, None, None
                
            else:
                if dense_flag:
                    # descriptor_map, scores_map = self.extract_dense_map(img) if h_en is None else self.extract_dense_map(img_crop)  # 120x40
                    # keypoints, _, scores, scoredispersitys = self.dkd(scores_map, descriptor_map,
                    #                                                     sub_pixel=sub_pixel)
                    # del descriptor_map

                    # angle_enhance = torch.rand((B, 1, 1), device=img.device) * 360 - 180    # [-180, 180)
                    # if h_en is None:
                    #     # imgB
                    #     # descriptors, descriptor_map_new, valid_all_mask, keypoints_new, kmask = self.cut_patch_from_featuremap_pad_inner_ext93(img, img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask, part_train_mask=part_train_mask)
                        
                    #     # descriptors, descriptor_map_new, valid_all_mask, keypoints_new, kmask, ori_all, ori_all_patch, ori_kp, ori_kp_patch, kmask_T = self.cut_patch_from_featuremap_pad_inner_ext93_oriR(img, img_ext, img_ext_T, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask, part_train_mask=part_train_mask, h_ATB=hT)

                    #     descriptors, descriptor_map_new, valid_all_mask, keypoints_new, kmask, ori_all, ori_all_patch, ori_kp, ori_kp_patch, kmask_T, bin_all, bin_kp, bin_T_kp = self.cut_patch_from_featuremap_pad_inner_ext93_oriR_bw(img, img_ext, img_ext_T, bin_img_ext, bin_img_ext_T, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask, part_train_mask=part_train_mask, h_ATB=hT)

                    # else:
                    #     # imgA
                    #     # descriptors, descriptor_map_new, valid_all_mask, keypoints_new, kmask = self.cut_patch_from_featuremap_pad_inner_ext_modify93(img, img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask, h_en=h_en, part_train_mask=part_train_mask)
                        
                    #     # descriptors, descriptor_map_new, valid_all_mask, keypoints_new, kmask, ori_all, ori_all_patch, ori_kp, ori_kp_patch, kmask_T = self.cut_patch_from_featuremap_pad_inner_ext_modify93_oriR(img, img_ext, img_ext_T, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask, h_en=h_en, part_train_mask=part_train_mask, h_ATB=hT)
                    
                    #     descriptors, descriptor_map_new, valid_all_mask, keypoints_new, kmask, ori_all, ori_all_patch, ori_kp, ori_kp_patch, kmask_T, bin_all, bin_kp, bin_T_kp = self.cut_patch_from_featuremap_pad_inner_ext_modify93_oriR_bw(img, img_ext, img_ext_T, bin_img_ext, bin_img_ext_T, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask, h_en=h_en, part_train_mask=part_train_mask, h_ATB=hT)
                    
                    # # scores_new = [s[sm] for s, sm in zip(scores, kmask)]
                    # # scoredispersitys_new = [sc[scm] for sc, scm in zip(scoredispersitys, kmask)]

                    # scores_new = [s[sm][smt] for s, sm, smt in zip(scores, kmask, kmask_T)]
                    # scoredispersitys_new = [sc[scm][scmt] for sc, scm, scmt in zip(scoredispersitys, kmask, kmask_T)]
                    # pass

                    pad_size = (2, 2, 0, 0)     # 128x52 pad->128x56
                    img_point = F.pad(img_ext, pad_size, "constant", 0)
                    descriptor_map, scores_map = self.extract_dense_map(img_point)    # 128 x 56
                    keypoints, _, scores, scoredispersitys = self.dkd(scores_map, descriptor_map,
                                                                        sub_pixel=sub_pixel)
                    del descriptor_map

                    # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_dense_map_interpolation_aligned_batch93_newOri(img, img_ext, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                    if img_ext_T is not None:
                        # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_dense_map_interpolation_aligned_batch93_newOri_AT_ext(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                        descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_dense_map_interpolation_aligned_batch93_siftOri_AT_ext_a45(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                        # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_dense_map_interpolation_aligned_batch93_siftOri_AT_ext_a45_amp(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)

                    else:
                        # descriptors, valid_all_mask, kmask, keypoints_new, bin_kp = self.cut_patch_unfold_dense_map_interpolation_aligned_batch93_newOri_AT_ext_NE(img, img_ext, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                        descriptors, valid_all_mask, kmask, keypoints_new, bin_kp = self.cut_patch_unfold_dense_map_interpolation_aligned_batch93_siftOri_AT_ext_NE_a45(img, img_ext, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                        # descriptors, valid_all_mask, kmask, keypoints_new, bin_kp = self.cut_patch_unfold_dense_map_interpolation_aligned_batch93_siftOri_AT_ext_NE_a45_amp(img, img_ext, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)

                        descriptor_map_new, ori_all_patch, ori_kp_patch, bin_all = None, None, None, None 

                    scores_new = [s[sm] for s, sm in zip(scores, kmask)]
                    scoredispersitys_new = [sc[scm] for sc, scm in zip(scoredispersitys, kmask)]
                else:
                    # # 122x36点图提点
                    # pad_size = (2, 2, 0, 0)     # 120x36 pad -> 120x40  ori: 128x52 pad->128x56
                    # img_point = F.pad(img_crop, pad_size, "constant", 0)

                    # 128x40图提点
                    # pad_size = (2, 2, 3, 3)     # 122x36 pad -> 128x40  ori: 128x52 pad->128x56
                    img_point = deepcopy(img_crop) # F.pad(img_crop, pad_size, "constant", 0)

                    # # 描述子图裁剪提点 128 x 52 -> 128 x 40
                    # img_point = img_ext[:, :, :, 6:-6]  
                    if self.has_point_teacher:
                        descriptor_map, scores_map, point_fea_stu = self.extract_dense_map(img_point)    # 128 x 40
                        with torch.no_grad():
                            scores_map_tea, _, point_fea_tea = self.point_tea_net(img_point)    # 128 x 40
                        point_fea_loss = self.point_fgd_loss(point_fea_stu, point_fea_tea)
                    else:
                        descriptor_map, scores_map, _ = self.extract_dense_map(img_point)    # 80 x 64
                        point_fea_loss = None
                    
                    keypoints, _, scores, scoredispersitys = self.dkd(scores_map[:, 0, :, :].unsqueeze(1), descriptor_map,
                                                                        sub_pixel=sub_pixel)
                    del descriptor_map
                    
                    # # only 22/32
                    # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri(img, img_ext, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                    
                    # # only 22/32 AT ori
                    # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_AT(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)

                    # # only 22/32 AT ori 扩边图计算角度
                    # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_AT_ext(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)

                    # # only 22/32 高斯距离衰减
                    # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_Gauss(img, img_ext, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)


                    # # 22 and 32
                    # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_ms(img, img_ext, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)

                    if img_ext_T is not None:
                        # # 22 and 32 AT ori 扩边图计算角度 256维
                        # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_ms_AT_ext(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                        # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_ms_AT_ext(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                        # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_ms_AT_ext_cat45(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)

                        if is_teacher:
                            # # 方形描述
                            # descriptors, descriptor_map_new, descriptors_tea, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_tea(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                            
                            # 方形描述 FGD
                            descriptors, descriptor_map_new, descriptors_tea, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp, fea_loss = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_tea_fgd(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)

                            # # 长条形描述
                            # descriptors, descriptor_map_new, descriptors_tea, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_rec_tea(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                            # # 长条形描述 FGD
                            # descriptors, descriptor_map_new, descriptors_tea, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp, fea_loss = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_rec_tea_fgd(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                            
                            # fea_loss = None
                        elif is_bg_img:
                            # 128 x 40
                            valid_all_mask, kmask, keypoints_new = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_onlyP_bg(img_point, keypoints, pmask=pmask)
                            descriptors, descriptor_map_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = None, None, None, None, None, None
                            descriptors_tea, fea_loss = None, None 
                        else:
                            # 方形描述
                            # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                            
                            # # 120 x 40 
                            # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_onlyP(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                            
                            # 128 x 40
                            descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_onlyP(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)

                            # # 长条形描述
                            # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_rec(img, img_ext, img_ext_T, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                            
                            descriptors_tea, fea_loss = None, None 
                    else:
                        # descriptors, valid_all_mask, kmask, keypoints_new, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_ms_AT_ext_NE(img, img_ext, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                        # descriptors, valid_all_mask, kmask, keypoints_new, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_ms_AT_ext_NE(img, img_ext, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                        # descriptors, valid_all_mask, kmask, keypoints_new, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_ms_AT_ext_NE_cat45(img, img_ext, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)

                        # 方形描述
                        descriptors, valid_all_mask, kmask, keypoints_new, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_NE(img, img_ext, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)

                        # # 长条形描述子 
                        # descriptors, valid_all_mask, kmask, keypoints_new, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_siftOri_AT_ext_rec_NE(img, img_ext, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)
                        
                        descriptors_tea, fea_loss = None, None 


                        descriptor_map_new, ori_all_patch, ori_kp_patch, bin_all = None, None, None, None     
                    # # 22 and 32 Gauss
                    # descriptors, descriptor_map_new, valid_all_mask, kmask, keypoints_new, ori_all_patch, ori_kp_patch, bin_all, bin_kp = self.cut_patch_unfold_patch_map_interpolation_aligned_batch93_newOri_ms_Gauss(img, img_ext, img_point, bin_img_ext, keypoints, True, cal_angle=cal_orient, trans_rot_angle=trans_rot, pmask=pmask)

                    scores_new = [s[sm] for s, sm in zip(scores, kmask)]
                    scoredispersitys_new = [sc[scm] for sc, scm in zip(scoredispersitys, kmask)]
                    # xy_residual_new = [xyrn[xyrnm] for xyrn, xyrnm in zip(xy_residual, kmask)]

        else:
            with torch.no_grad():
                descriptor_map, scores_map = self.extract_dense_map(img)
                keypoints, descriptors, scores, scoredispersitys = self.dkd(scores_map[:, 0, :, :].unsqueeze(1), descriptor_map,
                                                                sub_pixel=sub_pixel)
                keypoints, descriptors, scores = keypoints[0], descriptors[0], scores[0]
                keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W - 1, H - 1]])    # 归一化点坐标->尺寸坐标

        end = time.time()

        return {'keypoints': keypoints_new,
                'descriptors': descriptors,
                'descriptor_map': descriptor_map_new,
                'scores': scores_new,
                'scores_map': scores_map,
                'scoredispersitys': scoredispersitys_new,
                'partial_mask': valid_all_mask,
                # 'ori_all': ori_all, 
                'ori_all_patch': ori_all_patch,
                # 'ori_kp': ori_kp,
                'ori_kp_patch': ori_kp_patch,
                # 'kmask_T': kmask_T,
                'bin_all': bin_all,
                'bin_kp': bin_kp,
                # 'bin_T_kp': bin_T_kp
                # 'descriptors_std': descriptors_std,
                'descriptors_teacher': descriptors_tea,
                # 'orientation': orient_batch,
                # 'time': end - start, 
                'fea_loss': fea_loss,
                # 'xy_residual': xy_residual_new,
                'point_fea_loss': point_fea_loss,
                }


if __name__ == '__main__':
    import numpy as np
    from thop import profile

    net = ALikeWithHard(c1=32, c2=64, c3=128, c4=128, dim=128, single_head=False)

    image = np.random.random((640, 480, 3)).astype(np.float32)
    flops, params = profile(net, inputs=(image, 9999, False), verbose=False)
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))
