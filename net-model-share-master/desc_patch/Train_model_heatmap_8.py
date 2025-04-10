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
from utils.utils import filter_points,warp_points,homography_scaling_torch, inv_warp_image
# from utils.utils import save_checkpoint

from pathlib import Path
from Train_model_frontend import Train_model_frontend
from torch.autograd import Variable
from utils import html
import os
import math
import cv2

def thd_img(img, thd=0.015):
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, alpha=0.25, gamma=2, reduction="None"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = None

    def forward(self, inputs, targets):
        p = inputs
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == None:
            pass
        else:
            loss = loss.sum()

        return loss

def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def unsuper_labels_score(pnts, H, W, block_size):  # 输入为标注的点标签 pnts:(N,(x,y))
            labels = torch.zeros(4, int(H/block_size), int(W/block_size))
            for item in pnts:
                labels[0, int(item[2] // block_size), int(item[1] // block_size)] += item[0]
                labels[3, int(item[2] // block_size), int(item[1] // block_size)] += 1  # 一个patch可能存在多个特征点，记录一下
                labels[1, int(item[2] // block_size), int(item[1] // block_size)] += item[2] % block_size  # 出现多个特征点，进行平均
                labels[2, int(item[2] // block_size), int(item[1] // block_size)] += item[1] % block_size  # 出现多个特征点，进行平均
            torch_ones = torch.ones_like(labels[3])
            labels[3] = torch.where(labels[3] > 0, labels[3], torch_ones)
            labels[0] = labels[0] / labels[3]
            labels[1] = labels[1] / labels[3]
            labels[2] = labels[2] / labels[3]

            labels[1] = labels[1] / block_size
            labels[2] = labels[2] / block_size

            
            return labels[:3]  # 去除用来记录一个patch内特征点数目的维度  # labels:[B,3,H/8,W/8]

def get_uni_xy(position):
    idx = torch.argsort(position)  # 返回的索引是0开始的 上面的方式loss会略大0.000x级别
    idx = idx.float()
    p = position.shape[0]
    uni_l2 = torch.mean(torch.pow(position - (idx / p), 2))
    return uni_l2

def uni_xy_loss(a_p, b_p):
        c = a_p.shape[0]
        reshape_pa = a_p.reshape((c, -1)).permute(1, 0)  # c h w -> c p -> p c where c=2
        reshape_pb = b_p.reshape((c, -1)).permute(1, 0)

        loss = (get_uni_xy(reshape_pa[:, 0]) + get_uni_xy(reshape_pa[:, 1]))
        loss += (get_uni_xy(reshape_pb[:, 0]) + get_uni_xy(reshape_pb[:, 1]))

        return loss

class Train_model_heatmap_8(Train_model_frontend):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    """
    * SuperPointFrontend_torch:
    ** note: the input, output is different from that of SuperPointFrontend
    heatmap: torch (batch_size, H, W, 1)
    dense_desc: torch (batch_size, H, W, 256)
    pts: [batch_size, np (N, 3)]
    desc: [batch_size, np(256, N)]
    """
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
        self.cell_size = 8
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
        self.correspond = 8

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

        pass

    ### loadModel inherited from Train_model_frontend
    # def loadModel(self):
    #     """
    #     load model from name and params
    #     init or load optimizer
    #     :return:
    #     """
    #     model = self.config["model"]["name"]
    #     params = self.config["model"]["params"]
    #     print("model: ", model)
    #     net = modelLoader(model=model, **params).to(self.device)
    #     logging.info("=> setting adam solver")
    #     optimizer = self.adamOptim(net, lr=self.config["model"]["learning_rate"])
    #
    #     n_iter = 0
    #     ## new model or load pretrained
    #     if self.config["retrain"] == True:
    #         logging.info("New model")
    #         pass
    #     else:
    #         path = self.config["pretrained"]
    #         mode = "" if path[:-3] == ".pth" else "full"
    #         logging.info("load pretrained model from: %s", path)
    #         net, optimizer, n_iter = pretrainedLoader(
    #             net, optimizer, n_iter, path, mode=mode, full_path=True
    #         )
    #         logging.info("successfully load pretrained model from: %s", path)
    #
    #     def setIter(n_iter):
    #         if self.config["reset_iter"]:
    #             logging.info("reset iterations to 0")
    #             n_iter = 0
    #         return n_iter
    #
    #     self.net = net
    #     self.optimizer = optimizer
    #     self.n_iter = setIter(n_iter)
    #     pass

    def detector_loss(self, input, target, mask=None, loss_type="softmax"):
        """
        # apply loss on detectors, default is softmax
        :param input: prediction
            tensor [batch_size, 65, Hc, Wc]
        :param target: constructed from labels
            tensor [batch_size, 65, Hc, Wc]
        :param mask: valid region in an image
            tensor [batch_size, 1, Hc, Wc]
        :param loss_type:
            str (l2 or softmax)
            softmax is used in original paper
        :return: normalized loss
            tensor
        """
        if loss_type == "l2":
            loss_func = nn.MSELoss(reduction="mean")
            loss = loss_func(input, target)
        elif loss_type == "softmax": 
            loss_func_BCE = nn.BCELoss(reduction='none').cuda()
            non_sigmoid = nn.Sigmoid()
            loss = loss_func_BCE(non_sigmoid(input[:,:-1,:,:]), target[:,:-1,:,:])
            # loss_position = loss_func_BCE(nn.functional.softmax(input[:,:-1,:,:], dim=1), target[:,:-1,:,:])
            
            # loss = loss_score + loss_position
            loss = (loss.sum(dim=1) * mask).sum()
            loss = loss / (mask.sum() + 1e-10)
        return loss
   
    def get_position(self, p_map,  H, W, block_size):
        x = torch.arange(W // block_size, requires_grad=False, device=p_map.device)
        y = torch.arange(H // block_size, requires_grad=False, device=p_map.device)
        y, x = torch.meshgrid([y, x])
        cell = torch.stack([y, x], dim=0)
        res = (cell + p_map) * block_size
       
        return res

    def to_pnts(self, semi):
        score = semi[:,0].view(semi.size(0),-1)
        y_coordinates = semi[:,1].view(semi.size(0),-1)
        x_coordinates = semi[:,2].view(semi.size(0),-1)

        pnts = torch.stack([score, x_coordinates,y_coordinates],dim=2)
        return pnts

    def get_dis(self, p_a, p_b):
        c = 2
        eps = 1e-12
        x = torch.unsqueeze(p_a[:, 0], 1) - torch.unsqueeze(p_b[:, 0], 0)  # N 2 -> NA 1 - 1 NB -> NA NB
        y = torch.unsqueeze(p_a[:, 1], 1) - torch.unsqueeze(p_b[:, 1], 0)
        dis = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + eps)
        return dis

    def get_point_pair(self, a_s, b_s, dis):  # 获得匹配点
        a2b_min_id = torch.argmin(dis, dim=1)
        len_p = len(a2b_min_id)
        ch = dis[list(range(len_p)), a2b_min_id] < self.correspond
        
        reshape_as = a_s
        reshape_bs = b_s

        a_s = reshape_as[ch]
        b_s = reshape_bs[a2b_min_id[ch]]
        d_k = dis[ch, a2b_min_id[ch]]
     
        return a_s, b_s, d_k
    
    def get_point_notpair(self, a_s, b_s, dis):  # 获得匹配点
        a2b_min_id = (torch.rand(a_s.size(0)) * b_s.size(0)).cuda(dis.device)
        a2b_min_id = a2b_min_id.long()
        len_p = len(a2b_min_id)
        ch = dis[list(range(len_p)), a2b_min_id] > self.correspond
        reshape_as = a_s
        reshape_bs = b_s

        a_s = reshape_as[ch]
        b_s = reshape_bs[a2b_min_id[ch]]
        d_k = dis[ch, a2b_min_id[ch]]
     
        return a_s, b_s, d_k
    
    def find_desc_tensor(self, pnts, desc, H, W, block_size):
        """
        input:
        pnts: N 3
        desc: C H W

        output:
        desc:N C
        """
        N, _ = pnts.size()
        C, _, _ = desc.size()
        desc_out = torch.zeros((N, C))
        for index in range(N):
            x_coordinate = int(pnts[index][0].round())
            y_coordinate = int(pnts[index][1].round())
            if x_coordinate > (W - 1):
                x_coordinate = W - 1
            if y_coordinate > (H - 1):
                y_coordinate = H - 1
            desc_out[index,:] = desc[:,(y_coordinate // block_size),(x_coordinate // block_size)]

        return desc_out

    def find_desc_tensor_fast(self, pnts, desc, H, W, block_size):  # 6倍加速 
        """
        input:
        pnts: N 2
        desc: H*W C

        output:
        desc:N C
        """
        N, _ = pnts.size()
        # C, Hd, Wd = desc.size()
        Wd = W // block_size
        Hd = H // block_size

        # desc = desc.view(C,-1)  # C H W -> C N
        # desc = desc.transpose(0,1)  # C N -> N C
        desc_X = (pnts[:,0] // block_size).long()
        desc_Y = (pnts[:,1] // block_size).long()

        # desc_X = pnts[:,0].long()
        # desc_Y = pnts[:,1].long()

        Wones = torch.ones_like(desc_X) * (Wd - 1)
        Hones = torch.ones_like(desc_Y) * (Hd - 1)

        desc_X = torch.where(desc_X < Wd, desc_X, Wones)  #防止数据范围溢出
        desc_Y = torch.where(desc_Y < Hd, desc_Y, Hones)

        desc_index = desc_Y * Wd + desc_X
        desc = desc[desc_index]



        return desc

    def desc_loss(self, d_a, d_b, dis):
        # 这里应该增加一个score的信息，在score低的区域，我们可以考虑不监督描述子的生成
        reshape_da, reshape_db, d_k = self.get_point_pair(d_a, d_b, dis)  # p -> k

        pos = torch.mul(reshape_da, reshape_db)  # [NA C] * [C NB] -> [NA NB]
        pos = torch.sum(pos,dim=1)

        reshape_da, reshape_db, d_k = self.get_point_notpair(d_a, d_b, dis)  # p -> k

        neg = torch.mul(reshape_da, reshape_db)  # [NA C] * [C NB] -> [NA NB]
        neg = torch.sum(neg,dim=1)
        # pos = (score >= 0.25)
        # neg = (score < 0.25)
    
        pos = (1 - pos)
        neg = (neg - 0.2)
        neg = torch.clamp(neg, min=0.0)
        loss_pn = torch.mean(pos) + torch.mean(neg)

        ab = torch.mm(d_a, d_b.permute(1, 0))  # [NA C] * [C NB] -> [NA NB]
        ab_zeros = torch.zeros_like(ab)
        ab = torch.where(ab > 0,ab,ab_zeros)
        ab = torch.sum(ab,dim=1)
        ab = (ab - 2)
        ab = torch.clamp(ab, min=0.0)
        loss_num = torch.mean(ab)
        print(ab)

        loss = loss_pn + 0.01*loss_num
        # loss = torch.mean(ab) + margin_loss * 0.05
        return loss

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

    def usp_loss(self, a_s, b_s, dis):
        alpha1 = 4
        alpha3 = 1.2
        alpha2 = alpha3 / 0.85 #期望特征点占比
        alpha4 = 0.25
        bonus = 0

        reshape_as_k, reshape_bs_k, d_k = self.get_point_pair(a_s, b_s, dis)  # p -> k

        score_k_loss = torch.mean(torch.pow(reshape_as_k - reshape_bs_k, 2))  # 监督分数一致性

        sk_ = (reshape_as_k + reshape_bs_k) / 2
        d_ = torch.mean(d_k)

        position_k_loss = torch.mean(sk_ * d_k)  # 最小化距离函数，监督offset

        usp_k_loss = torch.mean(alpha3 -  (alpha2 + bonus) * sk_) * d_  # 可重复性监督，分数高的地方->距离就小。分数低的地方->距离就大

        addition_loss = torch.mean(-torch.log(1-(1-(sk_ + (d_k/4))) * (1-(sk_ + (d_k/4)))))
        # addition_loss += 0.15 * torch.mean(-torch.log((sk_ - (d_k / 4))**2))

        # 按文章的思路，距离小的地方->分数高，距离大的地方->分数低
        # high = (d_k > d_) sk_[high] = sk_[high]
        # low = (d_k <= d_)
        # sk_[low] = 1. - sk_[low]
        # usp_k_loss = torch.mean(sk_)

        position_k_loss = position_k_loss * alpha2
        score_k_loss = score_k_loss * alpha1
        usp_k_loss = usp_k_loss * 1
        addition_loss = addition_loss * alpha4

        # 在趋于平稳后，分布是 -0.08, 1.0, 0.03
        # print(usp_k_loss, position_k_loss, score_k_loss)

        total_usp = position_k_loss + score_k_loss + usp_k_loss
        # total_usp = score_k_loss + addition_loss
        return total_usp, score_k_loss, position_k_loss, usp_k_loss, d_

    def get_orientation(self, img, keypoints, patch_size=16):
        img = np.array(img)
        img = img.astype(np.float)
     
        Gx=np.zeros_like(img)
        Gy=np.zeros_like(img)
     
        h, w = img.shape
        for i in range(1,h-1):
            Gy[i,:] = img[i-1,:] - img[i+1,:]
        for j in range(1,w-1):
            Gx[:,j] = img[:,j+1] - img[:,j-1]
        Gxx = Gx*Gx
        Gyy = Gy*Gy
        Gxy = Gx*Gy
        a = []
        b = []
        c = []
        for point in keypoints:
            x = int(point[0] + 0.5)
            y = int(point[1] + 0.5)

            #边缘检查
            if x-patch_size//2<0 or x + patch_size//2 > w:
                continue
            if y-patch_size//2<0 or y + patch_size//2 > h:
                continue

            crop_x = 0 if x-patch_size//2<0 else x-patch_size//2
            crop_y = 0 if y-patch_size//2<0 else y-patch_size//2
            
            a.append(np.sum(Gxx[crop_y:crop_y+patch_size,crop_x:crop_x+patch_size]))
            b.append(np.sum(Gyy[crop_y:crop_y+patch_size,crop_x:crop_x+patch_size]))
            c.append(np.sum(Gxy[crop_y:crop_y+patch_size,crop_x:crop_x+patch_size]))
        a = np.array(a) 
        b = np.array(b) 
        c = np.array(c)
        degree_value = 2*c / (a - b)
        angle = np.arctan(degree_value)
    
        angle = angle*57.29578049 #180/(3.1415926)
        for idx in range(len(a)):
            if a[idx]<b[idx]:
                if c[idx] >= 0:
                    angle[idx] = (angle[idx] + 180)/2
                else:
                    angle[idx] = (angle[idx] - 180)/2
            else:
                angle[idx] = angle[idx] / 2
            if angle[idx] > 5: # 正负90应该是同一个方向，无需滤除
                angle[idx] -= 90
            else:
                angle[idx] += 90
        return angle

    def unsuperpoint_loss(self, semiA, semiB, H, W, H_AB, H_ATA, imgA, imgB, mask=None, descA=None, descB=None):

        # target = torch.zeros_like(input)

        # loss_func_Focal = FocalLoss().cuda()
        # loss_func_BCE = nn.BCELoss(reduction='none').cuda()
        # loss_func_L1 = torch.nn.L1Loss(reduction='none').cuda()
        # loss_func_L2 = torch.nn.MSELoss(reduction='none').cuda()


        correct_position_A = self.get_position(semiA[:,1:,:,:], H, W, self.cell_size)  # 校准坐标值
        correct_position_B = self.get_position(semiB[:,1:,:,:], H, W, self.cell_size)

        semiA_correct = torch.cat(([semiA[:,0].unsqueeze(dim=1), correct_position_A]), dim=1)
        semiB_correct = torch.cat(([semiB[:,0].unsqueeze(dim=1), correct_position_B]), dim=1)
        
        #将featuremaps转换为坐标点 [B 3 16 16] -> [B N 3]
        pnts_A = self.to_pnts(semiA_correct)  # [B,N,(s,x,y)]
        pnts_B = self.to_pnts(semiB_correct)
       
        
        Batchsize = pnts_A.size(0)
        usp_loss = 0.0
        loss_uniform = 0.0
        usp_loss_score = 0.0
        usp_loss_position = 0.0
        usp_loss_sp = 0.0
        desc_loss = 0.0
        decorr_loss = 0.0
        d_loss = 0.0

        pnt_A_show = None
        pnt_B_show = None
        pnt_AT_show = None

        success_list = [] #有些图像对会被跳过，故需要记录下哪些图像对有效
        for index in range(Batchsize):
            pnt_A = pnts_A[index]
            pnt_B = pnts_B[index]

            if descA is not None:
                desc_A = descs_A[index]
                desc_B = descs_B[index]

            # if mask is not None:
            #     filter = (mask[index] == 1)
            #     filter = filter.view(-1)
            #     pnt_B = pnt_B[filter]

            #     if descA is not None:
            #         desc_B = desc_B[filter]

            H_ATB = H_AB[index]@H_ATA[index]
            # H_ATB = H_ATA[index]
            
          
            # warped_pnts = warp_points(pnt_A[:,1:].cpu(), homography_scaling_torch(H_ATB, H, W))  # 利用变换矩阵变换坐标点
            warped_pnts = warp_points(pnt_A[:,1:].cpu(), H_ATB)  # 利用变换矩阵变换坐标点
            warped_pnts, mask_points = filter_points(warped_pnts, torch.tensor([W, H]), return_mask=True)
            warped_score = pnt_A[:,0][mask_points].unsqueeze(dim=1)
            pnt_AT = torch.cat([warped_score,warped_pnts.to(warped_score.device)],dim=1)  #得到最终的变换坐标


            if descA is not None:
                desc_A = desc_A[mask_points]

            #对B图点进行两次变换，去除非重叠区域点
            # warped_pnts = warp_points(pnt_B[:,1:].cpu(), H_ATB.inverse())  # 利用变换矩阵变换坐标点
            # warped_pnts, mask_points = filter_points(warped_pnts, torch.tensor([W, H]), return_mask=True)
            # warped_score = pnt_B[:,0][mask_points].unsqueeze(dim=1)
            # pnt_BT = torch.cat([warped_score,warped_pnts.to(warped_score.device)],dim=1)  #得到最终的变换坐标

            # warped_pnts = warp_points(pnt_BT[:,1:].cpu(), H_ATB)  # 利用变换矩阵变换坐标点
            # warped_pnts, mask_points = filter_points(warped_pnts, torch.tensor([W, H]), return_mask=True)
            # warped_score = pnt_BT[:,0][mask_points].unsqueeze(dim=1)
            # pnt_BTB = torch.cat([warped_score,warped_pnts.to(warped_score.device)],dim=1)  #得到最终的变换坐标
            pnt_BTB = pnt_B
          
            key_dist = self.get_dis(pnt_AT[:,1:], pnt_BTB[:,1:])  # N 3 -> p p

            try:
                usp_loss_temp, usp_loss_score_temp, usp_loss_position_temp, usp_loss_sp_temp, d_loss_temp = self.usp_loss(pnt_AT[:,0], pnt_BTB[:,0], key_dist)

                desc_loss_temp = 0.0
                decorr_loss_temp = 0.0
                if descA is not None:
                    # desc_A = self.find_desc_tensor_fast(pnt_ATT[:,1:], desc_A, H, W, 8)
                    # desc_B = self.find_desc_tensor_fast(pnt_AT[:,1:], desc_B, H, W, 8)
                
                    # desc_B = self.find_desc_tensor_fast(pnt_B[:,1:], desc_B, H, W, 8)
                    desc_loss_temp = self.desc_loss(desc_A, desc_B,key_dist)
                    decorr_loss_temp = self.decorr_loss(desc_A, desc_B)

                usp_loss += usp_loss_temp
                usp_loss_score += usp_loss_score_temp
                usp_loss_position += usp_loss_position_temp
                usp_loss_sp += usp_loss_sp_temp
                desc_loss += desc_loss_temp
                decorr_loss += decorr_loss_temp
                d_loss += d_loss_temp
                
                loss_uniform += uni_xy_loss(semiA[index,1:], semiB[index,1:])

                #记录成功的batch id和点用于显示
                success_list.append(index)
                pnt_A_show = pnt_A
                pnt_B_show = pnt_B
                pnt_AT_show = pnt_AT
            except:
                pass
            

        usp_loss /= len(success_list)
        loss_uniform /= len(success_list)
        usp_loss_score /= len(success_list)
        usp_loss_position /= len(success_list)
        usp_loss_sp /= len(success_list)
        desc_loss /= len(success_list)
        decorr_loss /= len(success_list)
        d_loss /= len(success_list)

        # not_zero = max(0, -math.log((semiA[:,0,:,:].sum() / 9)))

        loss = usp_loss + 0.25*loss_uniform + 0.00*desc_loss + 0.00*decorr_loss

        self.loss_item.update({"usp_loss":usp_loss})
        self.loss_item.update({"usp_loss_score":usp_loss_score})
        self.loss_item.update({"usp_loss_position":usp_loss_position})
        self.loss_item.update({"usp_loss_sp":usp_loss_sp})
        self.loss_item.update({"loss_desc":desc_loss})
        self.loss_item.update({"loss_decorr":decorr_loss})
        self.loss_item.update({"loss_d":d_loss})

        pnt_A = pnt_A_show
        pnt_B = pnt_B_show
        pnt_AT = pnt_AT_show

        return loss, pnt_A, pnt_B, pnt_AT, success_list


    def show_html(self, imgA, imgB, pntA, pntB, pntA_transform, H, W, H_ATB):
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
        from utils.draw import draw_keypoints,draw_keypoints_AB

  
        pntA = toNumpy(pntA)
        pts_nms_A = filter_Pts(pntA, conf_thresh, nms_dist, H, W)

        pntB = toNumpy(pntB)
        pts_nms_B = filter_Pts(pntB, conf_thresh, nms_dist, H, W)

        pntA_transform = toNumpy(pntA_transform)
        pts_nms_AT = filter_Pts(pntA_transform, conf_thresh, nms_dist, H, W)

        img_pts_A = draw_keypoints(imgA.cpu().numpy().squeeze() * 255, pts_nms_A,color=(0,0,255))
        # imgA_name = "%d_imgA.bmp" % self.n_iter
        # saveImg(img_pts_A, os.path.join(image_path,imgA_name))

        img_pts_B = draw_keypoints(imgB.cpu().numpy().squeeze() * 255, pts_nms_B)
        # imgB_name = "%d_imgB.bmp" % self.n_iter
        # saveImg(img_pts_B, os.path.join(image_path,imgB_name))

        img_pts_AB = draw_keypoints_AB(imgB.cpu().numpy().squeeze() * 255, pts_nms_B, pts_nms_AT)
        # imgAB_name = "%d_imgAB.bmp" % self.n_iter
        # saveImg(img_pts_AB, os.path.join(image_path,imgAB_name))

        #绘制融合图
        warped_img = imgA.cpu().squeeze().unsqueeze(0).unsqueeze(0) * 255

        inv_homography = H_ATB
        warped_img = inv_warp_image(  # 利用变换矩阵变换图像
            warped_img, inv_homography.unsqueeze(0), mode="bilinear")
         
        b = np.zeros_like(imgB.cpu().numpy().squeeze() * 255)
        g = warped_img.cpu().numpy().squeeze()
        r = imgB.cpu().numpy().squeeze() * 255
        image_merge = cv2.merge([b, g, r])
        merge_name = "%d_merge.bmp" % self.n_iter

        image_save = np.hstack([img_pts_A, img_pts_AB, img_pts_B, image_merge])
        saveImg(image_save, os.path.join(image_path,merge_name))


        self.ims, self.txts, self.links = [], [], []
        self.html.add_header(self.n_iter)
        
        # self.ims.append(imgA_name)
        # self.txts.append("imgA")
        # self.links.append(imgA_name)

        # self.ims.append(imgB_name)
        # self.txts.append("imgB")
        # self.links.append(imgB_name)

        # self.ims.append(imgAB_name)
        # self.txts.append("imgAB")
        # self.links.append(imgAB_name)

        self.ims.append(merge_name)
        self.txts.append("imgMerge")
        self.links.append(merge_name)

        self.html.add_images(self.ims, self.txts, self.links)
        self.html.save()

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
        imgA, imgB, H_ATA, mask_2D, imgAT, H_AB = (
            sample["imgA"],
            sample["imgB"],
            sample["H_AT"],
            sample["valid_mask"], # mask the pixels if bordering artifacts appear
            sample["imgAT"],
            sample["H_AB"],
        )
        # img, labels = img.to(self.device), labels_2D.to(self.device)
        
        # variables
        batch_size, H, W = imgA.shape[0], imgA.shape[2], imgA.shape[3]
        self.batch_size = batch_size
        det_loss_type = self.config["model"]["detector_loss"]["loss_type"]

     
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        if train:
            # print("img: ", img.shape, ", img_warp: ", img_warp.shape)
            outsA = self.net(imgA.to(self.device))
            outsB = self.net(imgB.to(self.device))
            outsAT = self.net(imgAT.to(self.device))
            
    
        else:
            with torch.no_grad():
                outsA = self.net(imgA.to(self.device))
                semiA, coarse_descA = outsA["semi"], outsA["desc"]

                pass

        mask_3D_flattened = self.getMasks(mask_2D, self.cell_size, device=self.device)

        # inputA = self.semiA_transform(outsA["semi"], H, W, self.cell_size, H_AB=H_AB)

        loss_det, pntA, pntB, pntAT, success_list= self.unsuperpoint_loss(
            semiA=outsAT["semi"],
            semiB=outsB["semi"],
            H=H,
            W=W,
            H_AB=H_AB,
            H_ATA=H_ATA,
            imgA=imgA,
            imgB=imgB,
            mask=mask_3D_flattened,
            descA=None,
            descB=None
        )
       
        loss_det_warp = torch.tensor([0]).float().to(self.device)


        mask_desc = mask_3D_flattened.unsqueeze(1)
        lambda_loss = self.config["model"]["lambda_loss"]
        # print("mask_desc: ", mask_desc.shape)
        # print("mask_warp_2D: ", mask_warp_2D.shape)

        # descriptor loss
        if lambda_loss > 0:
            assert if_warp == True, "need a pair of images"
            loss_desc, mask, positive_dist, negative_dist = self.descriptor_loss(
                coarse_desc,
                coarse_desc_warp,
                mat_H,
                mask_valid=mask_desc,
                device=self.device,
                **self.desc_params
            )
        else:
            ze = torch.tensor([0]).to(self.device)
            loss_desc, positive_dist, negative_dist = ze, ze, ze

        loss = loss_det + loss_det_warp
        if lambda_loss > 0:
            loss += lambda_loss * loss_desc

        ##### try to minimize the error ######
        add_res_loss = False
        if add_res_loss and n_iter % 10 == 0:
            print("add_res_loss!!!")
            heatmap_org = self.get_heatmap(semi, det_loss_type)  # tensor []
            heatmap_org_nms_batch = self.heatmap_to_nms(
                self.images_dict, heatmap_org, name="heatmap_org"
            )
            if if_warp:
                heatmap_warp = self.get_heatmap(semi_warp, det_loss_type)
                heatmap_warp_nms_batch = self.heatmap_to_nms(
                    self.images_dict, heatmap_warp, name="heatmap_warp"
                )

            # original: pred
            ## check the loss on given labels!
            outs_res = self.get_residual_loss(
                sample["labels_2D"]
                * to_floatTensor(heatmap_org_nms_batch).unsqueeze(1),
                heatmap_org,
                sample["labels_res"],
                name="original_pred",
            )
            loss_res_ori = (outs_res["loss"] ** 2).mean()
            # warped: pred
            if if_warp:
                outs_res_warp = self.get_residual_loss(
                    sample["warped_labels"]
                    * to_floatTensor(heatmap_warp_nms_batch).unsqueeze(1),
                    heatmap_warp,
                    sample["warped_res"],
                    name="warped_pred",
                )
                loss_res_warp = (outs_res_warp["loss"] ** 2).mean()
            else:
                loss_res_warp = torch.tensor([0]).to(self.device)
            loss_res = loss_res_ori + loss_res_warp
            # print("loss_res requires_grad: ", loss_res.requires_grad)
            loss += loss_res
            self.scalar_dict.update(
                {"loss_res_ori": loss_res_ori, "loss_res_warp": loss_res_warp}
            )

        #######################################

        self.loss = loss

        self.scalar_dict.update(
            {
                "loss": loss,
                "loss_usp": self.loss_item['usp_loss'],
                "loss_score": self.loss_item['usp_loss_score'],
                "loss_position": self.loss_item['usp_loss_position'],
                "loss_sp": self.loss_item['usp_loss_sp'],
                "loss_d": self.loss_item['loss_d'],
                "loss_desc": self.loss_item['loss_desc'],
                "loss_decorr": self.loss_item['loss_decorr']
            }
        )

        self.input_to_imgDict(sample, self.images_dict)

        if train:
            loss.backward() 
            self.optimizer.step()
            
        if n_iter % tb_interval == 0:
            self.printLosses(self.scalar_dict, task)
            H_ATB = H_AB[success_list[-1]]@H_ATA[success_list[-1]]
            # H_ATB = H_ATA[-1]
           
            self.show_html(imgAT[success_list[-1]],imgB[success_list[-1]],pntA.transpose(0,1), pntB.transpose(0,1), pntAT.transpose(0,1), H, W, H_ATB)

            self.logger.debug("current iteration: %d", n_iter)
            self.logger.debug("loss: %f  loss_usp: %f  loss_desc: %f  loss_decorr: %f" , loss, self.loss_item['usp_loss'], self.loss_item['loss_desc'], self.loss_item['loss_decorr'])
            self.logger.debug("loss_score: %f  loss_position: %f  loss_sp: %f  loss_d: %f",self.loss_item['usp_loss_score'], self.loss_item['usp_loss_position'], self.loss_item['usp_loss_sp'], self.loss_item['loss_d'])
            # self.cal_RS(pntB.transpose(0,1), pntAT.transpose(0,1), H, W)

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
