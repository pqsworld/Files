"""This is the main training interface using heatmap trick

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import numpy as np
import torch
# from torch.autograd import Variable
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
# from tqdm import tqdm
# from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging

from utils.tools import dict_update

# from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened
# from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch_soft
from utils.utils import filter_points,warp_points,homography_scaling_torch, inv_warp_image, warp_points_batch
# from utils.utils import save_checkpoint

from pathlib import Path
from Train_model_frontend import Train_model_frontend
from torch.autograd import Variable
from utils import html
import os
import math
import cv2
from torch.distributions import Categorical, Bernoulli
from timeit import default_timer as timer

import torchvision
from torchvision.transforms.functional import InterpolationMode
import random
from utils.utils import inv_warp_patch_batch, inv_warp_patch, inv_warp_patch_batch_rec
from utils.draw import draw_orientation_degree
from scipy.linalg import hadamard
from timeit import default_timer as timer
from torch.profiler import profile
# from utils.loss_functions.fastap import FastAPLoss

def thd_img(img, thd=0.015):
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()

   
def flattenDetection(semi, tensor=False):
    '''
    Flatten detection output

    :param semi:
        output from detector head
        tensor [65, Hc, Wc]
        :or
        tensor (batch_size, 65, Hc, Wc)

    :return:
        3D heatmap
        np (1, H, C)
        :or
        tensor (batch_size, 65, Hc, Wc)

    '''
    batch = False
    if len(semi.shape) == 4:
        batch = True
        batch_size = semi.shape[0]      # [batch_size,65,16,16]
    # if tensor:
    #     semi.exp_()
    #     d = semi.sum(dim=1) + 0.00001
    #     d = d.view(d.shape[0], 1, d.shape[1], d.shape[2])
    #     semi = semi / d  # how to /(64,15,20)

    #     nodust = semi[:, :-1, :, :]
    #     heatmap = flatten64to1(nodust, tensor=tensor)
    # else:
    # Convert pytorch -> numpy.
    # --- Process points.
    # dense = nn.functional.softmax(semi, dim=0) # [65, Hc, Wc]
    if batch:
        dense = nn.functional.softmax(semi, dim=1)  # [batch_size, 65, Hc, Wc] 在dim=1维度进行softmax，相当于65个分类
        # Remove dustbin.
        nodust = dense[:, :-1, :, :]                # [batch_size, 64, Hc, Wc]

        # for i in range(nodust.size(2)):
        #     for j in range(nodust.size(3)):
        #         if dense[:, -1, i, j] > (1 / 65):
        #             nodust[:, :, i, j] = 0

    else:
        dense = nn.functional.softmax(semi, dim=0) # [65, Hc, Wc]
        nodust = dense[:-1, :, :].unsqueeze(0)

        # for i in range(nodust.size(2)):
        #     for j in range(nodust.size(3)):
        #         if dense[-1, i, j] > (1 / 65):
        #             nodust[:, :, i, j] = 0
    # Reshape to get full resolution heatmap.
    # heatmap = flatten64to1(nodust, tensor=True) # [1, H, W]
    depth2space = DepthToSpace(8)   # 实例化一个DepthToSpace类
    heatmap = depth2space(nodust)                   # [batch, 1, 128, 128]
    heatmap = heatmap.squeeze(0) if not batch else heatmap      # squeeze 只会对维度为1的维度进行压缩

    return heatmap

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(0), dim=1)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(1), dim=0)
    return Z + u.unsqueeze(1) + v.unsqueeze(0)


def log_optimal_transport(descA, descB, iters=3):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    scores = torch.einsum('mc,nc->mn',descA,descB)
    m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    couplings = scores

    norm = - (ms + ns).log()
    log_mu = norm.expand(m)
    log_nu = norm.expand(n)
   
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

def fast_ap_loss(descA, descB, target, num_bins=10):
    """
    Args:
        input:     torch.Tensor(N x embed_dim), embedding matrix
        target:    torch.Tensor(N x 1), class labels
        num_bins:  int, number of bins in distance histogram
    """
    N = target.size()[0]
    M = target.size()[1]
    assert descA.size()[0] == N, "Batch size donesn't match!"
    
    # 1. get affinity matrix
    Aff = target

    I_pos = (Aff > 0).type(torch.float).cuda()
    I_neg = (Aff <= 0).type(torch.float).cuda()
    N_pos = torch.sum(I_pos, 1) + 10e-6

    # 2. compute distances from embeddings
    # squared Euclidean distance with range [0,4]
    dist2 = 2 - 2 * torch.mm(descA, descB.t())

    # 3. estimate discrete histograms
    Delta = torch.tensor(4. / num_bins).cuda()
    Z     = torch.linspace(0., 4., steps=num_bins+1).cuda()
    L     = Z.size()[0]
    h_pos = torch.zeros((N, L)).cuda()
    h_neg = torch.zeros((N, L)).cuda()

    y = 1 - torch.abs(dist2.unsqueeze(2).repeat(1,1,L)-Z.unsqueeze(0).unsqueeze(0).repeat(N,M,1))/Delta
    pulse = torch.max(torch.Tensor([0]).cuda(), y)
    h_pos = torch.sum(pulse*I_pos.unsqueeze(2).repeat(1,1,11),dim=1)
    h_neg = torch.sum(pulse*I_neg.unsqueeze(2).repeat(1,1,11),dim=1)
    # for l in range(L):
    #     pulse    = softBinning(dist2, Z[l], Delta)
    #     h_pos[:,l] = torch.sum(pulse * I_pos, 1)
    #     h_neg[:,l] = torch.sum(pulse * I_neg, 1)

    H_pos = torch.cumsum(h_pos, 1)
    h     = h_pos + h_neg
    H     = torch.cumsum(h, 1) + 10e-6
    
    # 4. compate FastAP
    FastAP = h_pos * H_pos / H
    # FastAP[torch.isnan(FastAP) | torch.isinf(FastAP)] = 0
    FastAP = torch.sum(FastAP,1)/N_pos
    # FastAP = FastAP[ ~torch.isnan(FastAP) ]
    loss   = 1 - torch.mean(FastAP)
    return loss

class Train_model_heatmap_4_RL_desc_s(Train_model_frontend):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    """
    * SuperPointFrontend_torch:
    ** note: the input, output is different from that of SuperPointFrontend
    heatmap: torch (batch_size, H, W, 1)
    dense_desc: torch (batch_size, H, W, 256)
    pts: [batch_size, np (N, 3)]
    desc: [batch_size, np(256, N)]
    """
    '''
    用于模型蒸馏
    '''
    default_config = {
        "train_iter": 170000,
        "save_interval": 2000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
        "data": {"gaussian_label": {"enable": False}},
    }

        # config
    def __init__(self, config, save_path=Path("."), device="cpu", verbose=False):
        # Update config
        print("Load Train_model_heatmap!!")

        self.config = self.default_config
        self.config = dict_update(self.config, config)
        print("check config!!", self.config)

        # init parameters
        self.device = device
        self.save_path = save_path
        self._train = True
        self._eval = True
        self.cell_size = 4
        self.subpixel = False

        self.max_iter = config["train_iter"]

        self.gaussian = False
        if self.config["data"]["gaussian_label"]["enable"]:
            self.gaussian = True

        if self.config["model"]["dense_loss"]["enable"]:
            print("use dense_loss!")
            from utils.utils import descriptor_loss
            self.desc_params = self.config["model"]["dense_loss"]["params"]
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse

            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = "sparse"

        # load model
        # self.net = self.loadModel(*config['model'])
        self.printImportantConfig()
        self.correspond = 4
        self.h_expand = 5
        self.w_expand = 6

        self.webdir = "/".join(str(self.save_path).split("/")[:-1]) + "/web"
        self.html = html.HTML(self.webdir, 'show_html')
        self.ims, self.txts, self.links = [], [], []

        self.loss_item = {}

        self.logdir = "./" + "/".join(str(self.save_path).split("/")[:-1])

        logname = self.logdir + r"/log.txt"
        self.logger = logging.getLogger("LOG")
        self.logger.setLevel(logging.DEBUG)
        # 建立一个filehandler来把日志记录在文件里，级别为debug以上
        fh = logging.FileHandler(logname)
        fh.setLevel(logging.DEBUG)
       
        # 设置日志格式
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s ")
        fh.setFormatter(formatter)
        #将相应的handler添加在logger对象中
        self.logger.addHandler(fh)

        print("Train_model_heatmap_4_RL_desc_correct_sift")

        pass
    
    def decorr_loss(self, d_a, d_b):
        reshape_da = d_a.transpose(0,1)  # [N C]
        reshape_db = d_b.transpose(0,1)  # 
        loss = self.get_r_b(reshape_da)
        loss += self.get_r_b(reshape_db)
       
        return loss

    def get_r_b(self, reshape_d):
    
        f, p = reshape_d.shape
        
        # 监督不同bit位的相关性
        x_mean = torch.mean(reshape_d, dim=1, keepdim=True)  # c p -> c 1
        x_var = torch.mean((reshape_d - x_mean) ** 2, dim=1, keepdim=True)
        x_hat = (reshape_d - x_mean) / torch.sqrt(x_var + 1e-12)
        rs = torch.mm(x_hat, x_hat.transpose(1, 0)) / p  # c p * p c -> c c
        
        # ys = (1 - torch.eye(f).cuda()) * rs
        ys = rs - torch.eye(f, device=reshape_d.device)
        loss = torch.mean(torch.pow(ys, 2))

        # 监督不同位置描数子整体相关性, -1~1 -> 0~2,
        # rs = torch.mm(reshape_d.transpose(1, 0), reshape_d) + 1
        # ys = rs - 2 * torch.eye(p, device=reshape_d.device)
       
        # loss = torch.mean(ys)

        return loss

    def get_dis(self, p_a, p_b):
        c = 2
        eps = 1e-12
        x = torch.unsqueeze(p_a[:, 0], 1) - torch.unsqueeze(p_b[:, 0], 0)  # N 2 -> NA 1 - 1 NB -> NA NB
        y = torch.unsqueeze(p_a[:, 1], 1) - torch.unsqueeze(p_b[:, 1], 0)
        dis = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + eps)
        return dis

    def select_on_last(self, values: [..., 'T'], indices: [...]) -> [...]:
        '''
        WARNING: this may be reinventing the wheel, but I don't know how to do
        it otherwise with PyTorch.

        This function uses an array of linear indices `indices` between [0, T] to
        index into `values` which has equal shape as `indices` and then one extra
        dimension of size T.
        '''
        return torch.gather(
            values,
            -1,
            indices[..., None]
        ).squeeze(-1)

    def get_position_prob(self, semi):
        '''
        不使用dustbin，利用sigmoid判断所选位置的点的置信度
        '''
        Batch_size, c, hc, wc = semi.size()
        logits = semi.permute(0,2,3,1)
        logits = logits.view(Batch_size, hc*wc, c)
       
        proposal_dist = Categorical(logits=logits)
        proposals     = proposal_dist.sample()
        proposal_logp = proposal_dist.log_prob(proposals)
        
        accept_logits = self.select_on_last(logits, proposals).squeeze(-1)

        accept_dist    = Bernoulli(logits=accept_logits)
        accept_samples = accept_dist.sample()
        accept_logp    = accept_dist.log_prob(accept_samples)
        accept_mask    = accept_samples == 1.

        x_off = proposals % self.cell_size
        y_off = proposals // self.cell_size
        xy_off = torch.stack([x_off, y_off], dim=2)

        x = torch.arange(wc, requires_grad=False, device=semi.device)
        y = torch.arange(hc, requires_grad=False, device=semi.device)
        y, x = torch.meshgrid([y, x])
        xy = torch.stack([x, y], dim=2) * self.cell_size
        xy = xy.view(hc*wc, 2).repeat(Batch_size,1,1)

        xy = xy + xy_off
        xy = xy.float()
       
        logp = proposal_logp + accept_logp

        return xy, logp, accept_mask
    
    def get_position_prob_dustbin(self, semi):
        '''
        使用dustbin，保持和superpoint对齐
        '''
        Batch_size, c, hc, wc = semi.size()
        logits = semi.permute(0,2,3,1)
        logits = logits.view(Batch_size, hc*wc, c)
        proposal_dist = Categorical(logits=logits)
        proposals     = proposal_dist.sample()
        proposal_logp = proposal_dist.log_prob(proposals)

        accept_mask    = proposals != (c - 1)

        x_off = proposals % self.cell_size
        y_off = proposals // self.cell_size
        xy_off = torch.stack([x_off, y_off], dim=2)

        x = torch.arange(wc, requires_grad=False, device=semi.device)
        y = torch.arange(hc, requires_grad=False, device=semi.device)
        y, x = torch.meshgrid([y, x])
        xy = torch.stack([x, y], dim=2) * self.cell_size
        xy = xy.view(hc*wc, 2).repeat(Batch_size,1,1)

        xy = xy + xy_off
        xy = xy.float()
       
        logp = proposal_logp

        return xy, logp, accept_mask

    def get_orientation_test(self, img, keypoints, patch_size=16):
        '''
        img:tensor
        '''
        h, w = img.shape
        offset = patch_size // 2
        device = img.device
    
        Gx=torch.zeros((h+patch_size, w+patch_size), dtype=img.dtype, device=img.device)
        Gy=torch.zeros((h+patch_size, w+patch_size), dtype=img.dtype, device=img.device)

        Gx0=torch.zeros_like(img)
        Gx2=torch.zeros_like(img)
        Gy0=torch.zeros_like(img)
        Gy2=torch.zeros_like(img)

        Gx0[:,1:-1] = img[:,:-2]
        Gx2[:,1:-1] = img[:,2:]
        Gx[offset:-offset,offset:-offset] = (Gx0 - Gx2)

        Gy0[1:-1,:] = img[:-2,:]
        Gy2[1:-1,:] = img[2:,:]
        Gy[offset:-offset,offset:-offset] = (Gy2 - Gy0)

        Gxx = Gx*Gx
        Gyy = Gy*Gy
        Gxy = Gx*Gy

        coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, patch_size-1), torch.linspace(-1, 1, patch_size-1)), dim=2)  # 产生两个网格
        coor_cells = coor_cells.transpose(0, 1)
        coor_cells = coor_cells.to(self.device)
        coor_cells = coor_cells.contiguous()
        
        keypoints_num = len(keypoints)
        keypoints_correct = torch.round(keypoints.clone())
        keypoints_correct += offset
     
        src_pixel_coords = coor_cells.unsqueeze(0).repeat(keypoints_num,1,1,1)
        src_pixel_coords = src_pixel_coords.float() * (patch_size // 2 - 1) + keypoints_correct.unsqueeze(1).unsqueeze(1).repeat(1,patch_size-1,patch_size-1,1)
      
        src_pixel_coords = src_pixel_coords.view([keypoints_num, -1, 2])
        src_pixel_coords_index = (src_pixel_coords[:,:,0] + src_pixel_coords[:,:,1]*(w+patch_size)).long()
 
        a = torch.sum(Gxx.take(src_pixel_coords_index),dim=1)
        b = torch.sum(Gyy.take(src_pixel_coords_index),dim=1)
        c = torch.sum(Gxy.take(src_pixel_coords_index),dim=1)
        
        eps = 1e-12
        degree_value = 2*c / (a - b + eps)
        angle = torch.atan(degree_value)
    
        angle = angle*57.29578049 #180/(3.1415926)
        a_small_b_mask = (a < b)
        c_mask = (c >= 0)
        angle[~a_small_b_mask] = angle[~a_small_b_mask] / 2
        angle[a_small_b_mask*c_mask] = (angle[a_small_b_mask*c_mask] + 180) /2
        angle[a_small_b_mask*(~c_mask)] = (angle[a_small_b_mask*(~c_mask)] - 180) /2
        angle_mask = (angle > 0)
        angle[angle_mask] = angle[angle_mask] - 90
        angle[~angle_mask] = angle[~angle_mask] + 90

        return angle

    def forward_patches_correct(self, img_batch, keypoints_batch, patch_size=16, sample_size=16,train_flag=True):
        # 根据关键点获得patch，并输入网络
        # 返回元组 B kpts_num desc_dim
        # 不满patchsize的patch用0补全
        assert patch_size == 16
        results = None
        Batchsize, kpt_num  = keypoints_batch.size(0), keypoints_batch.size(1)
        img_H, img_W = img_batch.size(2), img_batch.size(3)
      
        img_batch = img_batch.to(self.device)

        patch_padding = 44
        add_offset = patch_padding//2
        add_offset_x = patch_padding//2

        noise_orientation = random.randint(-20,20)
        for batch_idx in range(Batchsize):
            keypoints = keypoints_batch[batch_idx]
            img = img_batch[batch_idx]

            img_temp = torch.zeros((1, img_H + patch_padding, img_W + patch_padding),device=self.device)
            img_temp[:,add_offset:(img_H+add_offset),add_offset_x:(img_W+add_offset_x)] = img

            orientation_theta = self.get_orientation_test(img[0], keypoints, patch_size)
         
            point_correct = keypoints.clone()
            point_correct[:,0] += add_offset_x
            point_correct[:,1] += add_offset

            patch = inv_warp_patch(img_temp.unsqueeze(0), point_correct, orientation_theta, patch_size=patch_size, sample_size=sample_size)
            data = patch.unsqueeze(1)

            if results is None:
                results = data
            else:
                results = torch.cat([results,data],dim=0)
        # compute output for patch a
        results_batch = Variable(results)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)

     
        outs = outs.view(Batchsize, kpt_num, -1) #[B*N dim] -> [B dim N]
     
        return outs
    
    def get_orientation_batch(self, img, keypoints, patch_size=18):
        '''
        img:tensor
        '''
        batch, c, h, w = img.shape
        offset = patch_size // 2
        device = img.device
    
        Gx=torch.zeros((batch, c, h+patch_size, w+patch_size), dtype=img.dtype, device=img.device)
        Gy=torch.zeros((batch, c, h+patch_size, w+patch_size), dtype=img.dtype, device=img.device)

        Gx0=torch.zeros_like(img)
        Gx2=torch.zeros_like(img)
        Gy0=torch.zeros_like(img)
        Gy2=torch.zeros_like(img)

        Gx0[:,:,:,1:-1] = img[:,:,:,:-2]
        Gx2[:,:,:,1:-1] = img[:,:,:,2:]
        Gx[:,:,offset:-offset,offset:-offset] = (Gx0 - Gx2)

        Gy0[:,:,1:-1,:] = img[:,:,:-2,:]
        Gy2[:,:,1:-1,:] = img[:,:,2:,:]
        Gy[:,:,offset:-offset,offset:-offset] = (Gy2 - Gy0)

        Gxx = Gx*Gx
        Gyy = Gy*Gy
        Gxy = Gx*Gy

        coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, patch_size-1), torch.linspace(-1, 1, patch_size-1)), dim=2)  # 产生两个网格
        coor_cells = coor_cells.transpose(0, 1)
        coor_cells = coor_cells.to(self.device)
        coor_cells = coor_cells.contiguous()
        
        keypoints_num = keypoints.size(1)
        keypoints_correct = torch.round(keypoints.clone())
        keypoints_correct += offset
     
        src_pixel_coords = coor_cells.unsqueeze(0).repeat(batch, keypoints_num,1,1,1)
        src_pixel_coords = src_pixel_coords.float() * (patch_size // 2 - 1) + keypoints_correct.unsqueeze(2).unsqueeze(2).repeat(1,1,patch_size-1,patch_size-1,1)
      
        src_pixel_coords = src_pixel_coords.view([batch, keypoints_num, -1, 2])
        batch_image_coords_correct = torch.linspace(0, (batch-1)*(h+patch_size)*(w+patch_size), batch).long().to(device)
        src_pixel_coords_index = (src_pixel_coords[:,:,:,0] + src_pixel_coords[:,:,:,1]*(w+patch_size)).long()
        src_pixel_coords_index  = src_pixel_coords_index + batch_image_coords_correct[:,None,None]

        a = torch.sum(Gxx.take(src_pixel_coords_index),dim=-1)
        b = torch.sum(Gyy.take(src_pixel_coords_index),dim=-1)
        c = torch.sum(Gxy.take(src_pixel_coords_index),dim=-1)
        
        eps = 1e-12
        degree_value = 2*c / (a - b + eps)
        angle = torch.atan(degree_value)
    
        angle = angle*57.29578049 #180/(3.1415926)
        a_small_b_mask = (a < b)
        c_mask = (c >= 0)
        angle[~a_small_b_mask] = angle[~a_small_b_mask] / 2
        angle[a_small_b_mask*c_mask] = (angle[a_small_b_mask*c_mask] + 180) /2
        angle[a_small_b_mask*(~c_mask)] = (angle[a_small_b_mask*(~c_mask)] - 180) /2
        angle_mask = (angle > 0)
        angle[angle_mask] = angle[angle_mask] - 90
        angle[~angle_mask] = angle[~angle_mask] + 90

        return angle


    def get_sift_orientation_batch(self, img, keypoints, patch_size=19, bin_size=10):
        '''
        img:tensor
        '''
        patch_size=19
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

    def trans_ori_correct(self, orientation_theta_batch, trans_theta_batch):
        #对方向角进行校准
        theta = trans_theta_batch.to(self.device)
        AT_theta = orientation_theta_batch - theta[:,None]
        correct_theta = ((AT_theta > 90) + (AT_theta < -90)).float()
        correct_theta = correct_theta*180
        return correct_theta, AT_theta
        
    def forward_patches_correct_batch_expand(self, img_batch, keypoints_batch, patch_size=16, sample_size=16,correct=True,sift=False,theta=0,trans_theta=0,train_flag=True):
        # 根据关键点获得patch，并输入网络
        # 返回元组 B kpts_num desc_dim
        # theta == 0 时，默认是A图输入，180时为B图输入
        assert sample_size <= 32, "padding more!"

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
        keypoints_batch_correct[:,:,0] += (add_offset_x + self.w_expand)
        keypoints_batch_correct[:,:,1] += (add_offset + self.h_expand)

        return_theta = None
        if correct:
            keypoints_batch_ori = keypoints_batch.clone()
            keypoints_batch_ori[:,:,0] += self.w_expand
            keypoints_batch_ori[:,:,1] += self.h_expand
            orientation_theta_batch = self.get_sift_orientation_batch(img_batch, keypoints_batch_ori)

            #theta == 0,即A图时进行theta校准
            if theta == 0:
                correct_theta, AT_theta = self.trans_ori_correct(orientation_theta_batch, trans_theta)
                orientation_theta_batch += correct_theta
                return_theta = AT_theta
            else:
                return_theta = orientation_theta_batch

            if sift:
                # orientation_theta_batch = self.get_orientation_batch(img_batch, keypoints_batch_ori, 16) + correct_theta + theta
                orientation_theta_batch += theta

        else:
            orientation_theta_batch = torch.zeros(keypoints_batch_correct.size(0),keypoints_batch_correct.size(1))
        
        patch = inv_warp_patch_batch(img_batch_padding, keypoints_batch_correct, orientation_theta_batch, patch_size=patch_size, sample_size=sample_size)
        
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
        
        results_batch = Variable(results)
        
        # #扩边屏蔽
        # mask_extend = mask_batch_padding.clone()
        # mask_extend[:] = 0
        # mask_extend[:,:,28:-28,33:-33] = 1

        # patch_mask_0 = inv_warp_patch_batch(mask_extend, keypoints_batch_correct, orientation_theta_batch, patch_size=patch_size, sample_size=sample_size)
        # patch_mask_45 = inv_warp_patch_batch(mask_extend, keypoints_batch_correct, orientation_theta_batch, patch_size=patch_size, sample_size=32)
        # patch_extend_mask = torch.cat([patch_mask_0.unsqueeze(1),patch_mask_45.unsqueeze(1)],dim=1)

        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
        with torch.no_grad():
            outs_t = self.descriptor_net_t(results_batch)

        outs = outs.view(Batchsize, kpt_num, -1) #[B*N dim] -> [B dim N]
        outs_t = outs_t.view(Batchsize, kpt_num, -1) #[B*N dim] -> [B dim N]


        patch = patch.view(Batchsize, kpt_num, patch_size, patch_size)
        patch_mask = patch_mask.view(Batchsize, kpt_num, patch_size, patch_size)
        return outs, patch, patch_mask, return_theta, outs_t

    def rewards(self, pnt_A, pnt_B, Homo, th, lm_tp, lm_fp, mask=False):
        pnt_AT = warp_points(pnt_A, Homo, device=self.device)  # 利用变换矩阵变换坐标点
        key_dist_A = self.get_dis(pnt_AT, pnt_B)  # [N 2] [M 2] -> [N M]

        pnt_BT = warp_points(pnt_B, Homo.inverse(), device=self.device)  # 利用变换矩阵变换坐标点
        key_dist_B = self.get_dis(pnt_BT, pnt_A)  # [M 2] [N 2] -> [M N]

        good = (key_dist_A < th) & (key_dist_B < th).T

        ele_rewards = lm_tp * good + lm_fp * (~good)

        if mask:
            return ele_rewards.to(self.device), good.bool().to(self.device)

        return ele_rewards.to(self.device)
    
    def rewards_precise_wb(self, pnt_A, pnt_B, pnt_AT, pnt_BT, orientation_AT, orientation_B, th, lm_tp, lm_fp, wb_s, mask=False, th_ori = 5, FA_flag=0):
        # pnt_AT = warp_points(pnt_A, Homo, device=self.device)  # 利用变换矩阵变换坐标点
        key_dist_A = self.get_dis(pnt_AT, pnt_B)  # [N 2] [M 2] -> [N M]

        # pnt_BT = warp_points(pnt_B, Homo.inverse(), device=self.device)  # 利用变换矩阵变换坐标点
        key_dist_B = self.get_dis(pnt_BT, pnt_A)  # [M 2] [N 2] -> [M N]

        PI_v = 3.14159265357
        orientation_diff = abs(orientation_AT[:,None] - orientation_B[None,:])
        orientation_diff = torch.min(orientation_diff,abs(180-orientation_diff))

        orientation_diff_mask = orientation_diff < th_ori
        # orientation_diff = torch.clamp(orientation_diff, max=th_ori)
        # orientation_w = torch.sqrt(th_ori - orientation_diff) / 2.2
        orientation_w = 0.6*(torch.cos(orientation_diff*PI_v/th_ori) + 1) #cos(orientation_diff*pi/12)
        dist_w = 0.65*(torch.cos(key_dist_A*PI_v/th) + 1) #cos(orientation_diff*pi/3)

        if FA_flag:
            # good = wb_s > 1 
            good = wb_s > 0.9
            weight_tp = good*0
            ele_rewards = 0 * good + lm_fp * (~good)
        else:
            '''将黑白相似度低的当成负样本'''
            # wb_s_mask = wb_s > 0.6
            # good = (key_dist_A < th) & (key_dist_B < th).T & orientation_diff_mask & wb_s_mask
            # ele_rewards = lm_tp * good * orientation_w * dist_w  + lm_fp * (~good)
            
            '''黑白相似度低的不参与loss计算'''
            # wb_s_mask = (wb_s > -1).float() 
            wb_s_mask = (wb_s > 0.6).float()
            good = (key_dist_A < th) & (key_dist_B < th).T & orientation_diff_mask
            # weight_tp = orientation_w * dist_w * wb_s_mask
            # weight_tp = (0.4*orientation_w + 0.25 * dist_w + 0.4*wb_s)*wb_s_mask
            weight_tp = (0.75*orientation_w * dist_w + 0.25*wb_s)*wb_s_mask
            ele_rewards = lm_tp * good * weight_tp  + lm_fp * (~good)


        if mask:
            return ele_rewards.to(self.device), good.to(self.device), weight_tp.to(self.device)

        return ele_rewards.to(self.device)

    def rewards_nn(self, pnt_A, pnt_B, Homo, th, lm_tp, lm_fp):
        pnt_AT = warp_points(pnt_A.cpu(), Homo)  # 利用变换矩阵变换坐标点
        key_dist_A = self.get_dis(pnt_AT, pnt_B.cpu())  # [N 2] [M 2] -> [N M]

        n_amin = torch.argmin(key_dist_A, dim=1)
        m_amin = torch.argmin(key_dist_A, dim=0)

        # nearest neighbor's nearest neighbor
        nnnn = m_amin[n_amin]
        n_ix = torch.arange(key_dist_A.shape[0], device=key_dist_A.device)
        mask = nnnn == n_ix

        # Now `mask` is a binary mask and n_amin[mask] is an index array.
        # We use nonzero to turn `n_amin[mask]` into an index array and return
        nn_idx = torch.stack([torch.nonzero(mask, as_tuple=False)[:, 0],n_amin[mask]], dim=0).cpu()
        nn_mask = torch.zeros_like(key_dist_A, device=key_dist_A.device)
        nn_mask[nn_idx[0],nn_idx[1]] = 1
        nn_mask = nn_mask.bool()

        good = (key_dist_A < th) & nn_mask
        ele_rewards = lm_tp * good + lm_fp * (~good)
       
        return ele_rewards.to(self.device)

    def distance_matrix(self, fs1: ['N', 'F'], fs2: ['M', 'F']) -> ['N', 'M']:
        '''
        Assumes fs1 and fs2 are L2 normalized!
        '''
        #SQRT_2 = 1.414213
        return 1.414213 * (1. - fs1 @ fs2.T).clamp(min=1e-6).sqrt() #优化欧式距离与cos等效 不懂请看：https://www.zhihu.com/question/19640394

    def distance_matrix_hadamard(self, fs1: ['N', 'F'], fs2: ['M', 'F']) -> ['N', 'M']:
        '''
        Assumes fs1 and fs2 are L2 normalized!
        '''
        Hada = hadamard(128)
        Hada = torch.from_numpy(Hada).float().to(fs1.device)
        assert fs1.size(-1) == 128
        fs1_h = fs1 @ Hada / 11.3137085
        fs2_h = fs2 @ Hada / 11.3137085
        #SQRT_2 = 1.414213
        return 1.414213 * (1. - fs1_h @ fs2_h.T).clamp(min=1e-6).sqrt() #优化欧式距离与cos等效 不懂请看：https://www.zhihu.com/question/19640394
    
    def binarization_th(self, patch_batch, img_batch):
        "根据图像黑白灰度均值确定分割阈值，后对patch进行与之分割"
        Batchsize, kpt_num, c, c = patch_batch.size()
        flat = img_batch.view(Batchsize, -1)
        flat = torch.sort(flat,dim=1).values
        b_mean = torch.mean(flat[:,666:2330],dim=1)
        w_mean = torch.mean(flat[:,4326:5990],dim=1)
        mp = b_mean + (w_mean - b_mean)*0.3

        mp = mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(patch_batch)
        patch_batch = patch_batch > mp
        patch_batch = patch_batch.view(Batchsize, kpt_num, c, c)
        return patch_batch

    def AddInfo1Mask(self, mask, mask_num=4):
        #对mask进行info1处理，即限制匹配区域
        assert len(mask.size())==2
        info1_choice = random.randint(0,1)
        if not info1_choice:
            return mask
        H, W = mask.size()
        H_patch = H // 2
        W_patch = W // 2
        H_stride = H_patch // mask_num
        W_stride = W_patch // mask_num
        info1_mask = torch.zeros_like(mask)
        info1_index = random.randint(0,mask_num*mask_num-1)
        H_index = info1_index // mask_num
        W_index = info1_index % mask_num

        info1_mask[H_stride*H_index:H_stride*H_index+H_patch,W_stride*W_index:W_stride*W_index+W_patch]=1
        mask = mask*info1_mask
        return mask

    def cal_wbSim(self, A_wb, B_wb, A_mask, B_mask):
        Batch_size = A_wb.size(0)
        NA, NB = A_wb.size(1), B_wb.size(1)
        A_wb = A_wb.view(Batch_size, NA,-1).unsqueeze(2).repeat(1,1,NB,1)
        B_wb = B_wb.view(Batch_size, NB,-1).unsqueeze(1).repeat(1,NA,1,1)
        A_mask = A_mask.view(Batch_size, NA,-1).unsqueeze(2).repeat(1,1,NB,1)
        B_mask = B_mask.view(Batch_size, NB,-1).unsqueeze(1).repeat(1,NA,1,1)
        wb_mask = A_mask * B_mask
        wb_s = (A_wb == B_wb)*wb_mask
        #重叠区域的黑白相似度
        wb_s = torch.sum(wb_s.float(),dim=3) / torch.sum(wb_mask.float(),dim=3)
        overlap_s = torch.sqrt(torch.mean(wb_mask.float(),dim=3))
        return wb_s*overlap_s

    def REINFORCE_f(self, semiA, semiB, H, W, trans, trans_expand, imgA, imgB, imgATB, n_iter, trans_theta, imgA_mask, imgB_mask, FA_flag):
        '''非匹配区域的汉明距离不高于匹配对平均距离'''
        e = n_iter // 9600
        if e == 0:
            ramp = 0.
        elif e == 1:
            ramp = 0.1
        else:
            ramp = min(1., 0.1 + 0.2 * e)

        t_w = min(1.4, 0.01 + 0.2 * e)
        # t_w = 0

        lm_tp=1.0
        lm_fp=-0.25 * ramp
        th=1.5
        lm_kp=-0.0025 * ramp
        bin_use = True
        info1_use = False

        # fastaploss = FastAPLoss(10) #10个bin
      
        # this is a module which is used to perform matching. It has a single
        # parameter called θ_M in the paper and `inverse_T` here. It could be
        # learned but I instead anneal it between 15 and 50
        # inverse_T = 15 + 35 * min(1., 0.05 * e)
        inverse_T = 20 - 5 * min(1., 0.05 * e)
        # inverse_T = inverse_T*self.descriptor_net.gauss[0]
        
        pnts_A_undetached, logps_A_undetached, accept_masks_A = self.get_position_prob(semiA)  # 根据featuremap得到坐标点
        pnts_B_undetached, logps_B_undetached, accept_masks_B = self.get_position_prob(semiB)  # 根据featuremap得到坐标点

        pnts_A = pnts_A_undetached
        pnts_B = pnts_B_undetached

        logps_A = logps_A_undetached.detach().requires_grad_()
        logps_B = logps_B_undetached.detach().requires_grad_()

        pnts_AT = warp_points_batch(pnts_A, trans, device=self.device)
        pnts_BT = warp_points_batch(pnts_B, trans.inverse(), device=self.device)
        
        _,_,H,W = imgA.size()
        pnts_AT_expand = pnts_AT.clone()
        pnts_AT_expand[:,:,0] += self.w_expand
        pnts_AT_expand[:,:,1] += self.h_expand
        #多mask一点，多mask的也必不是匹配点，无需担心
        AT_mask = (pnts_AT_expand >= 0) * (pnts_AT_expand < torch.tensor([W-0.6,H-0.6]).to(self.device))
        pnts_AT_expand[~AT_mask] = 0 #起占位作用，实际此部分点不起作用

        pnts_B_expand = pnts_B.clone()
        pnts_B_expand[:,:,0] += self.w_expand
        pnts_B_expand[:,:,1] += self.h_expand

        orientations_AT = self.get_orientation_batch(imgATB.to(self.device), pnts_AT_expand)
        orientations_B = self.get_orientation_batch(imgB.to(self.device), pnts_B_expand)

        # start = timer()
        #根据坐标提取描述子
        # descAs_undetached,descAs_180_undetached = self.forward_patches_correct_180_batch(imgA, pnts_A, patch_size=16, sample_size=28, correct=True, train_flag=True)
        # descBs_undetached,descBs_180_undetached = self.forward_patches_correct_180_batch(imgB, pnts_B, patch_size=16, sample_size=28, correct=True, train_flag=True)
        
        sift_switch = random.choice([0,1])
        descAs_undetached, patchAs, patchAs_mask, _,  _ = self.forward_patches_correct_batch_expand(imgA, pnts_A, patch_size=16, sample_size=18, correct=True, sift=sift_switch, theta=0, trans_theta=trans_theta)
        descBs_undetached, patchBs, patchBs_mask, _,  descBs_t = self.forward_patches_correct_batch_expand(imgB, pnts_B, patch_size=16, sample_size=18, correct=True, sift=sift_switch, theta=180, trans_theta=0)
        
        if sift_switch:
            patchBs = torchvision.transforms.functional.rotate(patchBs,180)
            patchBs_mask = torchvision.transforms.functional.rotate(patchBs_mask,180)
        
        #patch二值化
        patchAs_wb = self.binarization_th(patchAs, imgA.to(self.device))
        patchBs_wb = self.binarization_th(patchBs, imgB.to(self.device))
        
        patches_wbSim = self.cal_wbSim(patchAs_wb,patchBs_wb,patchAs_mask,patchBs_mask)
        # from PIL import Image
        # patches_show = Image.fromarray(patchAs_wb[0,0].cpu().numpy().astype(np.uint8)*255).convert('L')
        # patches_show.save('/data/yey/work/temp/{}.bmp'.format(n_iter))
        # sift_loss_batch = self.desc_theta_loss(imgA,imgB)
        # end = timer()
        # print(end - start)
        # exit()

        descAs = descAs_undetached.detach().requires_grad_()
        descBs = descBs_undetached.detach().requires_grad_()

        # descAs_180 = descAs_180_undetached
        # descBs_180 = descBs_180_undetached
        factor_dim = descAs.size(-1) // 128
        # assert descAs.size(-1) == 128

        Batchsize = pnts_A.size(0)
        loss = 0.0
        loss_decorr = 0.0
        loss_ap = []

        pnt_A_show = None
        pnt_B_show = None
        desc_A_show = None
        desc_B_show = None
        match_show = None
        
        success_list = [] #有些图像对会被跳过，故需要记录下哪些图像对有效
        for index in range(Batchsize):
            accept_mask_A = accept_masks_A[index]
            accept_mask_B = accept_masks_B[index]
            pnt_A = pnts_A[index][accept_mask_A]
            pnt_B = pnts_B[index][accept_mask_B]
            pnt_AT = pnts_AT[index][accept_mask_A]
            pnt_BT = pnts_BT[index][accept_mask_B]
            orientation_AT = orientations_AT[index][accept_mask_A]
            orientation_B = orientations_B[index][accept_mask_B]

            patchA = patchAs[index][accept_mask_A]
            patchB = patchBs[index][accept_mask_B]
            
            wb_s = patches_wbSim[index]
            wb_s = torch.masked_select(wb_s,(accept_mask_A.unsqueeze(1) * accept_mask_B.unsqueeze(0)))
            wb_s = wb_s.reshape(patchA.size(0),-1)

            kp_logp_A = logps_A[index][accept_mask_A]
            kp_logp_B = logps_B[index][accept_mask_B]
            
            #bit位独立性loss
            # decorr_bit_loss = self.decorr_loss(descAs[index], descBs[index])

            descA = descAs[index][accept_mask_A]
            descB = descBs[index][accept_mask_B]
            descB_t = descBs_t[index][accept_mask_B]

            #有效区域mask
            pnt_A_long = pnt_A.long()
            pnt_B_long = pnt_B.long()

            if info1_use:
                imgB_mask[index,0] = self.AddInfo1Mask(imgB_mask[index,0])
            A_valid_mask = imgA_mask[index,0][pnt_A_long[:,1],pnt_A_long[:,0]].bool()
            B_valid_mask = imgB_mask[index,0][pnt_B_long[:,1],pnt_B_long[:,0]].bool()

            pnt_A = pnt_A[A_valid_mask]
            pnt_B = pnt_B[B_valid_mask]

            if (pnt_A.size(0) < 3) or (pnt_B.size(0) < 3):
                continue

            pnt_AT = pnt_AT[A_valid_mask]
            pnt_BT = pnt_BT[B_valid_mask]
            orientation_AT = orientation_AT[A_valid_mask]
            orientation_B = orientation_B[B_valid_mask]
            kp_logp_A = kp_logp_A[A_valid_mask]
            kp_logp_B = kp_logp_B[B_valid_mask]
            descA = descA[A_valid_mask]
            descB = descB[B_valid_mask]
            descB_t = descB_t[B_valid_mask]

            patchA = patchA[A_valid_mask]
            patchB = patchB[B_valid_mask]
            
            wb_s = torch.masked_select(wb_s,(A_valid_mask.unsqueeze(1) * B_valid_mask.unsqueeze(0)))
            wb_s = wb_s.reshape(patchA.size(0),-1)

            if sift_switch:
                descB = torchvision.transforms.functional.rotate(descB.view(-1,8*factor_dim,4,4),180).view(descB.size(0),-1)
                descB_t = torchvision.transforms.functional.rotate(descB_t.view(-1,8*factor_dim,4,4),180).view(descB.size(0),-1)
            
            # descA_180 = descAs_180[index][accept_masks_A[index]]
            # descB_180 = descBs_180[index][accept_masks_B[index]]
            # descA_180 = torchvision.transforms.functional.rotate(descA_180.view(-1,8,4,4),180).view(descA.size(0),-1)
            # descB_180 = torchvision.transforms.functional.rotate(descB_180.view(-1,8,4,4),180).view(descB.size(0),-1)
            
            # sift_loss = sift_loss_batch[index]
            # sift_loss_A = mse_loss(descA, descA_180)
            # sift_loss_B = mse_loss(descB, descB_180)

            # sift_loss_A = torch.sum(1 - torch.sum(descA * descA_180,dim=1))
            # sift_loss_B = torch.sum(1 - torch.sum(descB * descB_180,dim=1))
            # sift_loss = (sift_loss_A + sift_loss_B) / 2

            H_ATB = trans[index]

            #FA时禁用bin
            if FA_flag[index]:
                bin_use = False
            else:
                bin_use = True

            #根据单应变换矩阵生成reward矩阵 [N M]
            elementwise_rewards, match_mask, match_weight = self.rewards_precise_wb(pnt_A, pnt_B, pnt_AT, pnt_BT, orientation_AT, orientation_B, th, lm_tp, lm_fp, wb_s, mask=True, FA_flag=FA_flag[index])
            
            '''
            |           |mean(nomatch)|
            |   match   |mean(nomatch)|
            |           |mean(nomatch)|
            ___________________________
            |           | mean(match) |
            |  nomatch  | mean(match) |
            |           | mean(match) |
            '''
      
            match_mask_sum1 = (match_mask.sum(1) < 1).float()
            match_mask_sum0 = (match_mask.sum(0) < 1).float()

            # #空间mask
            # mask_64 = torch.randperm(16) < 8
            # mask_64 = mask_64.view(4,4).unsqueeze(0).repeat(8*factor_dim,1,1).view(-1)            

            # mask_64 = mask_64.to(self.device)

            # descA_mask = descA[:,mask_64]
            # # descA_mask = torch.cat([descA_mask,descA_mask],dim=-1)
            # descB_mask = descB[:,mask_64]
            # # descB_mask = torch.cat([descB_mask,descB_mask],dim=-1)

            # norm = torch.sqrt(torch.sum(descA_mask * descA_mask, dim = 1) + 1e-10)
            # descA_mask = descA_mask / norm.unsqueeze(-1).expand_as(descA_mask)

            # norm = torch.sqrt(torch.sum(descB_mask * descB_mask, dim = 1) + 1e-10)
            # descB_mask = descB_mask / norm.unsqueeze(-1).expand_as(descB_mask)
            # distances = self.distance_matrix(descA_mask, descB_mask) #值域为[0 2]

            # #计算AP loss
            # if torch.sum(match_mask) > 0 and not FA_flag[index]:
            #     aploss_A = fast_ap_loss(descA,descB,elementwise_rewards)
            #     # aploss_B = fastaploss(descB.detach(),descA,match_weight.t())
            #     aploss = aploss_A

            #计算欧氏距离：利用余弦相似度
            distances = self.distance_matrix(descA, descB) #值域为[0 2]
            distances_t = self.distance_matrix(descB_t, descB) #值域为[0 2]
            # sample_logp_t = log_optimal_transport(descB_t, descB)
            # distances = distances + match_gap
            if bin_use:
                nomatch_mean = torch.mean(distances[~match_mask])
                if torch.sum(match_mask) < 1:  #防止错误，没匹配对时会变成nan
                    match_mean = nomatch_mean
                else:
                    match_mean = torch.mean(distances[match_mask])
                
                #FA时对应匹配对降低权重：其匹配质量大概率较差
                FA_reverse =  1 - 2*FA_flag[index]

                # #对匹配对绝对值进行限制
                # desc_bernouli = Bernoulli(logits=(-dw * distances + db))
                # match_abs = 0.01 * FA_reverse * (desc_bernouli.logits[match_mask]).sum()

                biny = match_mask_sum1 * match_mean + (1 - match_mask_sum1) * nomatch_mean
                binx = match_mask_sum0 * match_mean + (1 - match_mask_sum0) * nomatch_mean
                biny = biny.unsqueeze(1)
                distances = torch.cat([torch.cat([distances, biny], -1),torch.cat([binx, match_mean.unsqueeze(0)], -1).unsqueeze(0)], 0)

            affinity = -inverse_T * distances #将distance转化为概率，即取值范围为[-x 0], log(p)
            desc_cat_I = Categorical(logits=affinity)
            desc_cat_T = Categorical(logits=affinity.T)

            affinity_t = -(inverse_T - 2) * distances_t #将distance转化为概率，即取值范围为[-x 0], log(p)
            desc_cat_I_t = Categorical(logits=affinity_t)
            desc_cat_T_t = Categorical(logits=affinity_t.T)
        
            if bin_use:
                desc_cat_I.probs  = desc_cat_I.probs[:-1,:-1]
                desc_cat_T.probs  = desc_cat_T.probs[:-1,:-1]
                desc_cat_I.logits = desc_cat_I.logits[:-1,:-1]
                desc_cat_T.logits = desc_cat_T.logits[:-1,:-1]

            with torch.no_grad():
                # we don't want to backpropagate through this
                sample_p = desc_cat_I.probs * desc_cat_T.probs.T #每个点对被选中的概率：根据distance，最近领选取概率大

            sample_logp = desc_cat_I.logits + desc_cat_T.logits.T #计算loss

            sample_logp_t = desc_cat_I_t.logits + desc_cat_T_t.logits.T #计算loss
       
            # [N, M]
            kps_logp = kp_logp_A.reshape(-1, 1) + kp_logp_B.reshape(1, -1)
           
            # scalar, used for introducing the lm_kp penalty, 对点数进行惩罚
            sample_lp_flat = kp_logp_A.sum() + kp_logp_B.sum()
            
            # [N, M], p * logp of sampling a pair
            sample_plogp = sample_p * (sample_logp + kps_logp)

            reinforce  = (elementwise_rewards * sample_plogp).sum()
            kp_penalty = lm_kp * sample_lp_flat
            t_loss = (torch.eye(sample_logp_t.size(0),device=self.device)*sample_logp_t).sum()


            reinforce_loss = -reinforce - kp_penalty - t_w*t_loss

            reinforce_loss.backward()
            
            loss += reinforce_loss.item()
            loss_decorr += (-t_loss).item()
            # if torch.sum(match_mask) > 0 and not FA_flag[index]:
            #     loss_ap.append(aploss.item())

            #记录成功的batch id和点用于显示
            success_list.append(index)

        pnt_A_show = pnt_A
        pnt_B_show = pnt_B
        match_show = [match_mask,patchA, patchB, match_weight, wb_s]
        desc_A_show = descA.detach() #用来画图，无需计算梯度
        desc_B_show = descB.detach() #用来画图，无需计算梯度

        leaves = []
        grads = []

        leaves.extend([logps_A_undetached,logps_B_undetached])
        grads.extend([logps_A.grad,logps_B.grad])

        leaves.extend([descAs_undetached,descBs_undetached])
        grads.extend([descAs.grad,descBs.grad])

        # finally propagate the gradients down to the network
        torch.autograd.backward(leaves, grads)

        loss /= len(success_list)
        loss_decorr /= len(success_list)
        # loss_ap = np.mean(np.array(loss_ap))

        self.loss_item.update({"loss":loss})
        self.loss_item.update({"loss_kd":loss_decorr})
        self.loss_item.update({"loss_ap":0})

        pnt_A = pnt_A_show
        pnt_B = pnt_B_show
        pnt_A[:,0] += self.w_expand
        pnt_A[:,1] += self.h_expand
        pnt_B[:,0] += self.w_expand
        pnt_B[:,1] += self.h_expand
        desc_A = desc_A_show
        desc_B = desc_B_show

        pnt_A = torch.cat([torch.ones(pnt_A.size(0)).unsqueeze(1).to(self.device),pnt_A],dim=1)
        pnt_B = torch.cat([torch.ones(pnt_B.size(0)).unsqueeze(1).to(self.device),pnt_B],dim=1)
        
        return loss, pnt_A, pnt_B, success_list, desc_A, desc_B, sift_switch, match_show

    
    def _Hamming_Hadamard_one(self, descs):
        assert descs.size(1) == 128

        Hada = hadamard(128)
        descs = (torch.round(descs*5000).long()+5000).long()
        #门限话
        norm = (torch.sqrt(torch.sum(descs * descs, dim = 1)) * 0.2).long()
        norm = norm.unsqueeze(-1).expand_as(descs)
        descs = torch.where(descs < norm, torch.sqrt(descs).long(), torch.sqrt(norm).long())

        Hada_T = descs.float() @ torch.from_numpy(Hada).float().to(descs.device)
        
        descs_Hamming = (Hada_T.long() > 0).long()
        return descs_Hamming

    def Hamming_Hadamard(self, descs):
        (descs_num, descs_dim) = descs.size()

        descA_0, descA_1, descB_0, descB_1 = None, None, None, None
        assert descs_dim in (128,256)
        if descs_dim == 128:
            descs_0, descs_1 = descs, descs
            descs_0_Hamming = self._Hamming_Hadamard_one(descs_0)
            descs_1_Hamming = descs_0_Hamming
        elif descs_dim == 256:
            descs = descs.view(-1,16,16)
            descs_0, descs_1 = descs[:,:,:8].reshape(-1,128), descs[:,:,8:].reshape(-1,128)

            descs_0_Hamming = self._Hamming_Hadamard_one(descs_0)
            descs_1_Hamming = self._Hamming_Hadamard_one(descs_1)
        
        descs_Hamming = torch.cat([descs_0_Hamming, descs_1_Hamming],dim=1)

        return descs_Hamming
        
    def Hamming_distance_Hadamard(self, descsA, descsB, dis12=False):
        #计算两组描述子的汉明距离矩阵
        #训练时采用的余弦相似度，这里需要转换
        #Hamming_distance_1计算为1部分的汉明距离
        #Hamming_distance_0计算为0部分的汉明距离
        #同反号分别计算
        same_mask = torch.tensor([1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1]).unsqueeze(1).repeat(1,8).view(-1).bool()
        # same_mask = (((torch.linspace(32,1,32)**4).unsqueeze(0) @ hadamard(32)) == ((torch.linspace(1,32,32)**4).unsqueeze(0) @ hadamard(32))).long().squeeze(0).unsqueeze(1).repeat(1,4).view(-1).bool()
        descs_dim = descsA.size(1)
        factor_dim = 1
        assert descs_dim == 256

        descsA_0, descsA_1 = descsA[:,:128], descsA[:,128:]
        descsB_0, descsB_1 = descsB[:,:128], descsB[:,128:]

        descsA_same    = torch.cat([descsA_0[:,same_mask],descsA_1[:,same_mask]],dim=1)
        descsB_same    = torch.cat([descsB_0[:,same_mask],descsB_1[:,same_mask]],dim=1)
        descsA_reverse = torch.cat([descsA_0[:,~same_mask],descsA_1[:,~same_mask]],dim=1)
        descsB_reverse = torch.cat([descsB_0[:,~same_mask],descsB_1[:,~same_mask]],dim=1)

        #计算同号位汉明距离
        Hamming_distance_same_1    = descsA_same.float() @ descsB_same.float().T
        Hamming_distance_same_0    = (descsA_same < 1).float() @ (descsB_same < 1).float().T
        Hamming_distance_same      = 128  - (Hamming_distance_same_1 + Hamming_distance_same_0)

        #计算反号位汉明距离
        Hamming_distance_reverse_1 = descsA_reverse.float() @ descsB_reverse.float().T
        Hamming_distance_reverse_0 = (descsA_reverse < 1).float() @ (descsB_reverse < 1).float().T
        Hamming_distance_reverse   = 128 - (Hamming_distance_reverse_1 + Hamming_distance_reverse_0)
        Hamming_distance_reverse   = torch.where(Hamming_distance_reverse < (128 - Hamming_distance_reverse), Hamming_distance_reverse, (128 - Hamming_distance_reverse))
        
        if dis12:
            return factor_dim *(Hamming_distance_same), factor_dim *(Hamming_distance_same + Hamming_distance_reverse)

        return factor_dim *(Hamming_distance_same + Hamming_distance_reverse)
   
      
    def show_html_dist(self, imgA, imgB, pntA, pntB, descA, descB, H, W, H_ATB, sift_flag, FA_flag, match_show):
        imgA = imgA
        imgB = imgB
        nms_dist = self.config['model']['nms']
        conf_thresh = self.config['model']['detection_threshold']

        if not os.path.isdir(self.webdir):
            os.mkdir(self.webdir)
        image_path = self.webdir + "/images"

        if not os.path.isdir(image_path):
            os.mkdir(image_path)


        from utils.var_dim import toNumpy
        from utils.utils import filter_Pts
        from utils.utils import saveImg
        from utils.draw import draw_keypoints,draw_keypoints_AB, draw_keypoints_match

        '''汉明距离'''
        descA = descA.view(descA.size(0),-1,16).permute(0,2,1).reshape(descA.size(0),-1)
        descB = descB.view(descB.size(0),-1,16).permute(0,2,1).reshape(descB.size(0),-1)
        descA_H = self.Hamming_Hadamard(descA)
        descb_H = self.Hamming_Hadamard(descB)
        dist_m = self.Hamming_distance_Hadamard(descA_H, descb_H)

        match_mask = self.rewards_nn(pntA.transpose(0,1)[:,1:], pntB.transpose(0,1)[:,1:], H_ATB, th=2, lm_tp=1, lm_fp=0)
        match_mask = match_mask.bool()
        match_dist = torch.mean(dist_m[match_mask]).item()
        unmatch_dist = torch.mean(dist_m[~match_mask]).item()
        print('sift: {:d} match_dist: {:f}, unmatch_dist: {:f}'.format(sift_flag, match_dist, unmatch_dist))

        n_amin = torch.argmin(dist_m, dim=1)
        m_amin = torch.argmin(dist_m, dim=0)

        # nearest neighbor's nearest neighbor
        nnnn = m_amin[n_amin]

        # we have a cycle consistent match for each `i` such that
        # nnnn[i] == i. We create an auxiliary array to check for that
        n_ix = torch.arange(dist_m.shape[0], device=dist_m.device)
        mask = nnnn == n_ix

        # Now `mask` is a binary mask and n_amin[mask] is an index array.
        # We use nonzero to turn `n_amin[mask]` into an index array and return
        show_pred = torch.stack([torch.nonzero(mask, as_tuple=False)[:, 0],n_amin[mask]], dim=0).cpu()

        #利用trans验证上述nnnn的正确性
        pntA_match = pntA.transpose(0,1)[:,1:][show_pred[0]]
        pntB_match = pntB.transpose(0,1)[:,1:][show_pred[1]]

        pntA_transform = warp_points(pntA.transpose(0,1)[:,1:].cpu(), H_ATB)  # 利用变换矩阵变换坐标点
        pntA_transform, mask_points = filter_points(pntA_transform, torch.tensor([W, H]), return_mask=True)
        pntA_transform = torch.cat([torch.ones(pntA_transform.size(0)).unsqueeze(1).to(pntA_transform.device),pntA_transform],dim=1)
        pntA_transform = pntA_transform.transpose(0,1)

        pntA_match_T = warp_points(pntA_match.cpu(), H_ATB)  # 利用变换矩阵变换坐标点
        key_dist_match = self.get_dis(pntA_match_T, pntB.transpose(0,1)[:,1:].cpu())  # [N 2] [M 2] -> [N M]
        kpt_min = torch.argmin(key_dist_match, dim=1)
        match_correct = kpt_min == show_pred[1]
        ch = key_dist_match[list(range(len(kpt_min))), kpt_min] < 2
        match_correct = match_correct & ch

        pntA_match = pntA_match.cpu().numpy()
        pntB_match = pntB_match.cpu().numpy()
        matches = np.hstack([pntA_match[match_correct],pntB_match[match_correct]])
        unmatches = np.hstack([pntA_match[~match_correct],pntB_match[~match_correct]])

        show_data = {}
        show_data.update({'image1':imgA.cpu().numpy().squeeze() * 255})
        show_data.update({'image2':imgB.cpu().numpy().squeeze() * 255})
        show_data.update({'keypoints1':pntA_match})
        show_data.update({'keypoints2':pntB_match})
        show_data.update({'matches':matches})
        show_data.update({'unmatches':unmatches})

        if FA_flag:
            suffix_flag = "FA"
        else:
            suffix_flag = "FR"
        img_match = draw_keypoints_match(show_data)
        match_name = "%d_match_%d_%s.bmp" % (self.n_iter, sift_flag, suffix_flag)
        saveImg(img_match, os.path.join(image_path,match_name))


       
        pntA = toNumpy(pntA)
        pts_nms_A = filter_Pts(pntA, conf_thresh, nms_dist, H, W)

        pntB = toNumpy(pntB)
        pts_nms_B = filter_Pts(pntB, conf_thresh, nms_dist, H, W)

        pntA_transform = toNumpy(pntA_transform)
        pts_nms_AT = filter_Pts(pntA_transform, conf_thresh, nms_dist, H, W)

        img_pts_A = draw_keypoints(imgA.cpu().numpy().squeeze() * 255, pts_nms_A,color=(0,0,255))
        img_pts_B = draw_keypoints(imgB.cpu().numpy().squeeze() * 255, pts_nms_B)
        img_pts_AB = draw_keypoints_AB(imgB.cpu().numpy().squeeze() * 255, pts_nms_B, pts_nms_AT)
       
        #绘制融合图
        warped_img = imgA.cpu().squeeze().unsqueeze(0).unsqueeze(0) * 255

        inv_homography = H_ATB
        warped_img = inv_warp_image(  # 利用变换矩阵变换图像
            warped_img, inv_homography.unsqueeze(0), mode="bilinear")
         
        b = np.zeros_like(imgB.cpu().numpy().squeeze() * 255)
        g = warped_img.cpu().numpy().squeeze()
        r = imgB.cpu().numpy().squeeze() * 255
        image_merge = cv2.merge([b, g, r])
        merge_name = "%d_merge_%d_%s.bmp" % (self.n_iter, sift_flag, suffix_flag)

        image_save = np.hstack([img_pts_A, img_pts_AB, img_pts_B, image_merge])
        saveImg(image_save, os.path.join(image_path,merge_name))

        if self.n_iter < 1000:
            #画patch 匹配图
            patch_path = self.webdir + "/patches"

            if not os.path.isdir(patch_path):
                os.mkdir(patch_path)

            matches_idx = torch.where(match_show[0]==1)
            matches_idx_A, matches_idx_B = matches_idx[0], matches_idx[1]
            matches_patchA = match_show[1][matches_idx_A]
            matches_patchB = match_show[2][matches_idx_B]
            matches_weight = match_show[3][matches_idx_A,matches_idx_B]
            matches_wbs = match_show[4][matches_idx_A,matches_idx_B]

            for patch_idx, patchA in enumerate(matches_patchA):
                patchB = matches_patchB[patch_idx]
                b = np.zeros_like(patchB.cpu().numpy().squeeze() * 255)
                g = patchA.cpu().numpy().squeeze() * 255
                r = patchB.cpu().numpy().squeeze() * 255
                patch_merge = cv2.merge([b, g, r])
                patch_merge_name = "%d_merge_%d_%s_w_%.4f_wbs_%.4f.bmp" % (self.n_iter, patch_idx, suffix_flag, matches_weight[patch_idx], matches_wbs[patch_idx])

                patch_save = patch_merge
                saveImg(patch_save, os.path.join(patch_path,patch_merge_name))


        self.ims, self.txts, self.links = [], [], []
        self.html.add_header(self.n_iter)

        self.ims.append(merge_name)
        self.txts.append("imgMatch")
        self.links.append(merge_name)

        
        self.ims.append(merge_name)
        self.txts.append("imgMerge")
        self.links.append(merge_name)

        self.html.add_images(self.ims, self.txts, self.links)
        self.html.save()
        return match_dist, unmatch_dist

 
    # additional subgradient descent on the sparsity-induced penalty term
    def updateBN(self,model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(0.0001*torch.sign(m.weight.data))  # L1的导数时符号函数


    def train_val_sample(self, sample, n_iter=0, train=False):
        """
        # key function
        :param sample:
        :param n_iter:
        :param train:
        :return:
        """
        to_floatTensor = lambda x: torch.tensor(x).type(torch.FloatTensor)

        task = "train" if train else "val"
        tb_interval = self.config["tensorboard_interval"]
        if_warp = self.config['data']['warped_pair']['enable']

        
        self.scalar_dict, self.images_dict, self.hist_dict = {}, {}, {}
        ## get the inputs
        # logging.info('get input img and label')
        imgA, imgB, trans_expand, mask_2D, imgAT, trans, trans_theta, imgAT_mask, imgB_mask = (
            sample["imgA"],
            sample["imgB"],
            sample["trans_expand"],
            sample["valid_mask"], # mask the pixels if bordering artifacts appear
            sample["imgAT"],
            sample["trans"],
            sample["theta"],
            sample["imgAT_mask"],
            sample["imgB_mask"]
        )
        # img, labels = img.to(self.device), labels_2D.to(self.device)
        
        # variables
        batch_size, H, W = imgA.shape[0], imgA.shape[2], imgA.shape[3]
        self.batch_size = batch_size
        det_loss_type = self.config["model"]["detector_loss"]["loss_type"]

     
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # assert self.h_expand == 0
        # forward + backward + optimize
        if train:
            # print("img: ", img.shape, ", img_warp: ", img_warp.shape)
            outsAT = self.net(imgAT[:,:,self.h_expand:-self.h_expand,self.w_expand:-self.w_expand].to(self.device))
            outsB = self.net(imgB[:,:,self.h_expand:-self.h_expand,self.w_expand:-self.w_expand].to(self.device))
            
    
        else:
            with torch.no_grad():
                outsA = self.net(imgA[:,:,self.h_expand:-self.h_expand,self.w_expand:-self.w_expand].to(self.device))
                semiA, coarse_descA = outsA["semi"], outsA["desc"]

                pass

        #对B进行扩边mask
        expand_disable = False
        if expand_disable:
            imgB[:,:,:3,:] = 0
            imgB[:,:,-3:,:] = 0
            imgB[:,:,:,:8] = 0
            imgB[:,:,:,-8:] = 0
        # with profile(activities=[torch.profiler.ProfilerActivity.CPU], 
        #     with_stack=True,record_shapes=True) as prof:
            # with torch.profiler.record_function("self.get_position_prob"):
        loss_det, pntA, pntB, success_list, descA, descB, sift_flag, match_show = self.REINFORCE_f(
            semiA=outsAT["semi"],
            semiB=outsB["semi"],
            H=H,
            W=W,
            trans=trans,
            trans_expand=trans_expand,
            imgA=imgAT,
            imgB=imgB,
            imgATB=sample['imgATB'],
            n_iter=n_iter,
            trans_theta=trans_theta,
            imgA_mask=imgAT_mask.to(self.device),
            imgB_mask=imgB_mask.to(self.device),
            FA_flag=sample["FA_flag"]
        )
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        # # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))   
        # exit()
        loss_det_warp = torch.tensor([0]).float().to(self.device)
        loss = loss_det + loss_det_warp


        self.loss = loss

        self.scalar_dict.update(
            {
                "loss": loss,
                "loss_kd": self.loss_item["loss_kd"],
            }
        )

        self.input_to_imgDict(sample, self.images_dict)

        BN_L1 = False
        if BN_L1:
            self.updateBN(self.descriptor_net)

        if train:
            self.optimizer.step()

        # H_ATB = H_AB[success_list[-1]]@H_ATA[success_list[-1]]
        #     # H_ATB = H_ATA[-1]
           
        # match_dist, unmatch_dist = self.show_html_dist(imgAT[success_list[-1]],imgB[success_list[-1]],\
        #     pntA.transpose(0,1), pntB.transpose(0,1),\
        #         descA, descB,\
        #         H, W, H_ATB,
        #         sift_flag)

        if n_iter % tb_interval == 0:
            print(self.net.desc_w,self.net.desc_b)
            self.printLosses(self.scalar_dict, task)
            H_ATB = trans_expand[success_list[-1]]
            # H_ATB = H_ATA[-1]
           
            match_dist, unmatch_dist = self.show_html_dist(imgAT[success_list[-1]],imgB[success_list[-1]],\
                pntA.transpose(0,1), pntB.transpose(0,1),\
                    descA, descB,\
                    H, W, H_ATB,
                    sift_flag,
                    sample["FA_flag"][success_list[-1]],
                    match_show)

            print("loss_kd: ", self.loss_item["loss_kd"])
            print("loss_ap: ", self.loss_item["loss_ap"])
            print("BN_L1: ",BN_L1)
            print("expand_disable: ",expand_disable)
            self.logger.debug("current iteration: %d", n_iter)
            self.logger.debug("loss: %f" , loss)
            self.logger.debug("match_dist: %f unmatch_dist: %f" , match_dist, unmatch_dist)
           
        self.tb_scalar_dict(self.scalar_dict, task)


        return loss.item()

    def heatmap_to_nms(self, images_dict, heatmap, name):
        """
        return: 
            heatmap_nms_batch: np [batch, H, W]
        """
        from utils.var_dim import toNumpy

        heatmap_np = toNumpy(heatmap)
        ## heatmap_nms

        heatmap_nms_batch = [self.heatmap_nms(self, h) for h in heatmap_np]  # [batch, H, W]
        heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)
        # images_dict.update({name + '_nms_batch': heatmap_nms_batch})
        images_dict.update({name + "_nms_batch": heatmap_nms_batch[:, np.newaxis, ...]})
        return heatmap_nms_batch

    def get_residual_loss(self, labels_2D, heatmap, labels_res, name=""):
        if abs(labels_2D).sum() == 0:
            return
        outs_res = self.pred_soft_argmax(
            labels_2D, heatmap, labels_res, patch_size=5, device=self.device
        )
        self.hist_dict[name + "_resi_loss_x"] = outs_res["loss"][:, 0]
        self.hist_dict[name + "_resi_loss_y"] = outs_res["loss"][:, 1]
        err = abs(outs_res["loss"]).mean(dim=0)
        # print("err[0]: ", err[0])
        var = abs(outs_res["loss"]).std(dim=0)
        self.scalar_dict[name + "_resi_loss_x"] = err[0]
        self.scalar_dict[name + "_resi_loss_y"] = err[1]
        self.scalar_dict[name + "_resi_var_x"] = var[0]
        self.scalar_dict[name + "_resi_var_y"] = var[1]
        self.images_dict[name + "_patches"] = outs_res["patches"]
        return outs_res

    # tb_images_dict.update({'image': sample['image'], 'valid_mask': sample['valid_mask'],
    #     'labels_2D': sample['labels_2D'], 'warped_img': sample['warped_img'],
    #     'warped_valid_mask': sample['warped_valid_mask']})
    # if self.gaussian:
    #     tb_images_dict.update({'labels_2D_gaussian': sample['labels_2D_gaussian'],
    #     'labels_2D_gaussian': sample['labels_2D_gaussian']})

    ######## static methods ########
    @staticmethod
    def batch_precision_recall(batch_pred, batch_labels):
        precision_recall_list = []
        for i in range(batch_labels.shape[0]):
            precision_recall = precisionRecall_torch_soft(batch_pred[i], batch_labels[i], soft_margin=3)
            precision_recall_list.append(precision_recall)
        precision = np.mean(
            [
                precision_recall["precision"]
                for precision_recall in precision_recall_list
            ]
        )
        recall = np.mean(
            [precision_recall["recall"] for precision_recall in precision_recall_list]
        )
        return {"precision": precision, "recall": recall}

    @staticmethod
    def pred_soft_argmax(labels_2D, heatmap, labels_res, patch_size=5, device="cuda"):
        """

        return:
            dict {'loss': mean of difference btw pred and res}
        """
        from utils.losses import norm_patches

        outs = {}
        # extract patches
        from utils.losses import extract_patches
        from utils.losses import soft_argmax_2d

        label_idx = labels_2D[...].nonzero().long()

        # patch_size = self.config['params']['patch_size']
        patches = extract_patches(
            label_idx.to(device), heatmap.to(device), patch_size=patch_size
        )
        # norm patches
        patches = norm_patches(patches)

        # predict offsets
        from utils.losses import do_log

        patches_log = do_log(patches)
        # soft_argmax
        dxdy = soft_argmax_2d(
            patches_log, normalized_coordinates=False
        )  # tensor [B, N, patch, patch]
        dxdy = dxdy.squeeze(1)  # tensor [N, 2]
        dxdy = dxdy - patch_size // 2

        # extract residual
        def ext_from_points(labels_res, points):
            """
            input:
                labels_res: tensor [batch, channel, H, W]
                points: tensor [N, 4(pos0(batch), pos1(0), pos2(H), pos3(W) )]
            return:
                tensor [N, channel]
            """
            labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
            points_res = labels_res[
                points[:, 0], points[:, 1], points[:, 2], points[:, 3], :
            ]  # tensor [N, 2]
            return points_res

        points_res = ext_from_points(labels_res, label_idx)

        # loss
        outs["pred"] = dxdy
        outs["points_res"] = points_res
        # ls = lambda x, y: dxdy.cpu() - points_res.cpu()
        # outs['loss'] = dxdy.cpu() - points_res.cpu()
        outs["loss"] = dxdy.to(device) - points_res.to(device)
        outs["patches"] = patches
        return outs

    @staticmethod
    def flatten_64to1(semi, cell_size=8):
        """
        input: 
            semi: tensor[batch, cell_size*cell_size, Hc, Wc]
            (Hc = H/8)
        outpus:
            heatmap: tensor[batch, 1, H, W]
        """
        from utils.d2s import DepthToSpace

        depth2space = DepthToSpace(cell_size)
        heatmap = depth2space(semi)
        return heatmap

    @staticmethod
    def heatmap_nms(self, heatmap, nms_dist=4, conf_thresh=0.15):
        """
        input:
            heatmap: np [(1), H, W]
        """
        from utils.utils import getPtsFromHeatmap

        nms_dist = self.config['model']['nms']
        conf_thresh = self.config['model']['detection_threshold']
        heatmap = heatmap.squeeze()
        # print("heatmap: ", heatmap.shape)
        pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)
        semi_thd_nms_sample = np.zeros_like(heatmap)
        semi_thd_nms_sample[
            pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)
        ] = 1
        return semi_thd_nms_sample


if __name__ == "__main__":
    # load config
    filename = "configs/superpoint_finger_train_heatmap.yaml"
    import yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_tensor_type(torch.FloatTensor)
    with open(filename, "r") as f:
        config = yaml.load(f)

    from utils.loader import dataLoader as dataLoader

    # data = dataLoader(config, dataset='hpatches')
    task = config["data"]["dataset"]

    data = dataLoader(config, dataset=task, warp_input=True)
    # test_set, test_loader = data['test_set'], data['test_loader']
    train_loader, val_loader = data["train_loader"], data["val_loader"]

    # model_fe = Train_model_frontend(config)
    # print('==> Successfully loaded pre-trained network.')

    train_agent = Train_model_heatmap(config, device=device)

    train_agent.train_loader = train_loader
    # train_agent.val_loader = val_loader

    train_agent.loadModel()
    train_agent.dataParallel()
    train_agent.train()

    # epoch += 1
    try:
        model_fe.train()

    # catch exception
    except KeyboardInterrupt:
        logging.info("ctrl + c is pressed. save model")
    # is_best = True
