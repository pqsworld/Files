"""This is the main training interface using heatmap trick
Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

from re import U
import cv2
from cv2 import distanceTransform
import numpy as np
from scipy.stats.stats import sem
from tensorflow.python.util.tf_inspect import FullArgSpec
import torch
import os
# from torch.autograd import Variable
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
# from tqdm import tqdm
# from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging
import copy

from utils.tools import dict_update

from utils.utils import labels2Dto3D, flattenDetection, flattenDetection_new, labels2Dto3D_flattened, labels2Dto3D_sort, warp_points
from utils.utils import getPtsFromHeatmap, filter_points, getPtsFromLabels2D, getPtsFromLabels2D_torch, sample_desc_from_points_torch, getPtsFromHeatmapByCoordinates
# from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch_soft
from utils.utils import compute_valid_mask
from utils import html
from utils.draw import draw_keypoints_pair_train, draw_match_pair_train, draw_orientation
from utils.utils import saveImg
from utils.utils import inv_warp_image, batch_inv_warp_image 
# from utils.utils import save_checkpoint

from pathlib import Path
from Train_model_frontend import Train_model_frontend

def remove_borders(images, borders=3):

    shape = images.shape

    if len(shape) == 4:
        for batch_id in range(shape[0]):
            images[batch_id, :, 0:borders, :] = 0
            images[batch_id, :, :, 0:borders] = 0
            images[batch_id, :, shape[2] - borders:shape[2], :] = 0
            images[batch_id, :, :, shape[3] - borders:shape[3]] = 0
    elif len(shape) == 3:
        images[:, 0:borders, :] = 0
        images[:, :, 0:borders] = 0
        images[:, shape[1] - borders:shape[1], :] = 0
        images[:, :, shape[2] - borders:shape[2]] = 0
    else:
        images[0:borders, :] = 0
        images[:, 0:borders] = 0
        images[shape[0] - borders:shape[0], :] = 0
        images[:, shape[1] - borders:shape[1]] = 0

    return images

def grid_indexes(size):
    weights = np.zeros((size, size, 1, 2), dtype=np.float32)

    columns = []
    for idx in range(1, 1+size):
        columns.append(np.ones((size))*idx)
    columns = np.asarray(columns)

    rows = []
    for idx in range(1, 1+size):
        rows.append(np.asarray(range(1, 1+size)))
    rows = np.asarray(rows)

    weights[:, :, 0, 0] = columns
    weights[:, :, 0, 1] = rows

    return weights.transpose([3, 2, 0, 1])

def ones_multiple_channels(size, num_channels):

    ones = np.ones((size, size))
    weights = np.zeros((size, size, num_channels, num_channels), dtype=np.float32)

    for i in range(num_channels):
        weights[:, :, i, i] = ones
    
    return weights.transpose([3, 2, 0, 1])

def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2

def linear_upsample_weights(half_factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with linear filter
    initialization.
    """

    filter_size = get_kernel_size(half_factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = np.ones((filter_size, filter_size))
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights.transpose([3, 2, 0, 1])

def create_kernels(MSIP_sizes, device):      
    kernels = {}
    for ksize in MSIP_sizes:
        ones_kernel = ones_multiple_channels(ksize, 1)
        indexes_kernel = grid_indexes(ksize)
        upsample_filter_np = linear_upsample_weights(ksize // 2, 1)
        
        ones_kernel_t = torch.tensor(ones_kernel, device=device)
        indexes_kernel_t = torch.tensor(indexes_kernel, device=device)
        upsample_filter_t = torch.tensor(upsample_filter_np, device=device)

        kernels['ones_kernel_'+str(ksize)] = ones_kernel_t
        kernels['indexes_kernel_'+str(ksize)] = indexes_kernel_t
        kernels['upsample_filter_'+str(ksize)] = upsample_filter_t
    return kernels

def ip_layer(scores, w_size, kernels):
    eps = 1e-6
    scores_shape = scores.shape     # [b, 1, H, W]
    # maxpool
    scores_pool = F.max_pool2d(scores.detach(), kernel_size=w_size, stride=w_size)
    scores_max_unpool = F.conv_transpose2d(scores_pool, kernels['upsample_filter_'+str(w_size)], stride=w_size)
    exp_map = torch.exp(torch.divide(scores, scores_max_unpool + eps)) - 1*(1.-eps)
    sum_exp_map = F.conv2d(exp_map, kernels['ones_kernel_' + str(w_size)], stride=w_size)
    indexes_map = F.conv2d(exp_map, kernels['indexes_kernel_' + str(w_size)], stride=w_size)
    indexes_map = torch.divide(indexes_map, sum_exp_map + eps)

    max_scores_pool = torch.max(torch.max(scores_pool, dim=3, keepdim=True).values, dim=2, keepdim=True).values
    norm_scores_pool= torch.divide(scores_pool, max_scores_pool + eps)
    return indexes_map, [scores_pool, norm_scores_pool]

def ip_softscores(scores, w_size, kernels):
    eps = 1e-6
    scores_shape = scores.shape     # [b, 1, H, W]
    # maxpool
    scores_pool = F.max_pool2d(scores, kernel_size=w_size, stride=w_size)
    scores_max_unpool = F.conv_transpose2d(scores_pool, kernels['upsample_filter_'+str(w_size)], stride=w_size)

    exp_map = torch.exp(torch.divide(scores, scores_max_unpool + eps)) - 1*(1.-eps)
    sum_exp_map = F.conv2d(exp_map, kernels['ones_kernel_' + str(w_size)], stride=w_size)
    scores_map = F.conv2d(exp_map*scores, kernels['ones_kernel_' + str(w_size)], stride=w_size)
    soft_scores = torch.divide(scores_map, sum_exp_map + eps)

    return soft_scores

def grid_indexes_nms_conv(scores, kernels, w_size):

    weights, indexes = F.max_pool2d(scores, kernel_size=w_size, stride=w_size, return_indices=True)
    weights_norm = torch.divide(weights, torch.add(weights, np.finfo(float).eps))

    score_map = F.max_unpool2d(weights_norm, indexes, kernel_size=w_size, stride=w_size)
    # score_map = unpool(weights_norm, indexes, ksize=[1, window_size, window_size, 1], scope='unpool')

    indexes_label = F.conv2d(score_map, kernels['indexes_kernel_'+str(w_size)], stride=w_size).to(scores.device)

    ind_rand = (torch.rand(indexes_label.shape) * w_size + 1).int().float().to(scores.device)

    indexes_label = torch.where(indexes_label == 0, ind_rand, indexes_label)

    return indexes_label, weights, score_map

def loss_ln_indexes_norm(pre_indexes, label_indexes, weights_indexes, window_size, n=2):

    norm_sq = torch.sum(((pre_indexes - label_indexes) / window_size)**n, dim=1, keepdim=True)
    weigthed_norm_sq = 1000 * weights_indexes * norm_sq
    loss = torch.mean(weigthed_norm_sq, dim=(0,1,2,3))
    return loss

def thd_img(img, thd=0.015):
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img

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
        loss_func_BCE = nn.BCELoss(reduction='none').cuda()
        ce_loss = loss_func_BCE(nn.functional.softmax(inputs, dim=1), targets)
        # ce_loss = F.binary_cross_entropy_with_logits(
        #     inputs, targets, reduction="none"
        # )
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

class Get_save_orientation(Train_model_frontend):
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

    def __init__(self, config, save_path=Path("."), device="cpu", verbose=False):
        # config
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
        # self.correspond = 2

        self.max_iter = config["train_iter"]
        self.conf_thresh    = config['model']['detection_threshold']
        self.nms_dist       = config['model']['nms']
        self.correspond = 2
        self.w_size = 8

        self.gaussian = False
        if self.config["data"]["gaussian_label"]["enable"]:
            self.gaussian = True

        if self.config["model"]["dense_loss"]["enable"]:
            print("use dense_loss!")
            from utils.loss_functions.sparse_loss import descriptor_loss_dense_supervised, descriptor_loss_dense_selfsupervised
            from utils.loss_functions.sparse_loss import descriptor_loss_dense_selfsupervised_new
            self.desc_params = self.config["model"]["dense_loss"]["params"]
            if self.config["model"]["dense_loss"]["self_supervised"]:
                # self.descriptor_loss = descriptor_loss_dense_selfsupervised
                self.descriptor_loss = descriptor_loss_dense_selfsupervised_new
            else:
                self.descriptor_loss = descriptor_loss_dense_supervised
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            print("use sparse_loss!") # 稀疏描述子（半稠密）
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            from utils.loss_functions.sparse_loss import descriptor_loss_sparse_supervised

            self.descriptor_loss = descriptor_loss_sparse_supervised
            self.desc_loss_type = "sparse"

        self.webdir = "/".join(str(self.save_path).split("/")[:-1]) + "/web"
        self.html = html.HTML(self.webdir, 'show_html')
        self.ims, self.txts, self.links = [], [], []

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
        # load model
        # self.net = self.loadModel(*config['model'])
        self.printImportantConfig()

        self.last_loss_1 = []    # step t-1
        self.last_loss_2 = []    # step t-2
        
        pass

    def show_html(self, imgA, imgB, pntA, pntB, show_label, show_pred, H, W, H_ATA):
        # imgA = imgA.transpose(1,2)  # transpose，WTF!
        # imgB = imgB.transpose(1,2)

        if not os.path.isdir(self.webdir):
            os.mkdir(self.webdir)
        image_path = self.webdir + "/images"

        if not os.path.isdir(image_path):
            os.mkdir(image_path)


        from utils.var_dim import toNumpy
        from utils.utils import saveImg

        pntA = toNumpy(pntA)
        pntB = toNumpy(pntB)
       
        # inv_homography = H_ATA.inverse()
        # warped_img = inv_warp_image(  # 利用变换矩阵变换图像
        #     imgB.unsqueeze(0).cpu() * 255, inv_homography.unsqueeze(0), mode="bilinear"
        # )

        warped_img = imgB.squeeze().cpu() * 255

        matches_mask = (show_label == show_pred)
        real_pred = (show_pred != -1)
        matches_mask = real_pred*matches_mask
        unmatches_mask = real_pred*(matches_mask == False)

        #将tensor转成numpy
        matches_mask = matches_mask.squeeze(0).cpu().numpy()
        unmatches_mask = unmatches_mask.squeeze(0).cpu().numpy()
        show_pred = show_pred.squeeze(0).cpu().numpy()
        show_label = show_label.squeeze(0).cpu().numpy()

        matches = np.hstack([pntA[matches_mask],pntB[show_pred[matches_mask]]])
        unmatches = np.hstack([pntA[unmatches_mask],pntB[show_pred[unmatches_mask]]])



        show_data = {}
        # show_data.update({'image1':warped_img})
        # show_data.update({'image2':imgB.cpu().numpy().squeeze() * 255})
        show_data.update({'image1':imgA.cpu().numpy().squeeze() * 255})
        show_data.update({'image2':warped_img})
        show_data.update({'keypoints1':pntA})
        show_data.update({'keypoints2':pntB})
        show_data.update({'matches':matches})
        show_data.update({'unmatches':unmatches})

        img_pts_A = draw_keypoints_match(show_data)
        imgA_name = "%d_match.bmp" % self.n_iter
        saveImg(img_pts_A, os.path.join(image_path,imgA_name))

        
        matches_mask = (show_label != -1)
        matches = np.hstack([pntA[matches_mask],pntB[show_label[matches_mask]]])
        show_data.update({'matches':matches})

        img_pts_A_label = draw_keypoints_match(show_data,show_label=True)
        imgA_label_name = "%d_match_label.bmp" % self.n_iter
        saveImg(img_pts_A_label, os.path.join(image_path,imgA_label_name))

        self.ims, self.txts, self.links = [], [], []
        self.html.add_header(self.n_iter)
        
        self.ims.append(imgA_name)
        self.txts.append("match")
        self.links.append(imgA_name)

        self.ims.append(imgA_label_name)
        self.txts.append("match_label")
        self.links.append(imgA_label_name)

        self.html.add_images(self.ims, self.txts, self.links)
        self.html.save()

    def get_point_pair(self, dis, dis_thre=-1):  # 获得匹配点
        # a2b_min_id = torch.argmin(dis, dim=1)
        # len_p = len(a2b_min_id)
        # ch = dis[list(range(len_p)), a2b_min_id] < dis_thre

        # idx_x = a2b_min_id[ch]
        # dis_pair = dis[ch, a2b_min_id[ch]]
        
        # return dis_pair
        correspond = self.correspond if dis_thre == -1 else dis_thre
            

        a2b_min_id = torch.argmin(dis, dim=1)  # M X 1
        len_p = len(a2b_min_id)
        ch = dis[list(range(len_p)), a2b_min_id] < correspond
        reshape_as = torch.tensor(list(range(len_p)), device=self.device)
        # reshape_bs = b_s

        a_s = reshape_as[ch]
        b_s = a2b_min_id[ch]
        d_k = dis[ch, a2b_min_id[ch]]
     
        return a_s, b_s, d_k

    def get_score_pair(self, scA, scB, dis, dis_thre=3):
        a2b_min_id = torch.argmin(dis, dim=1)
        len_p = len(a2b_min_id)
        ch = dis[list(range(len_p)), a2b_min_id] < dis_thre

        idx_x = a2b_min_id[ch]
        dis_pair = dis[ch, a2b_min_id[ch]]

        scA_pair = scA[ch]
        scB_pair = scB[a2b_min_id[ch]]

        return scA_pair, scB_pair, dis_pair

    def get_dis(self, p_a, p_b):
        eps = 1e-12
        x = torch.unsqueeze(p_a[:, 0], 1) - torch.unsqueeze(p_b[:, 0], 0)  # N 2 -> NA 1 - 1 NB -> NA NB
        y = torch.unsqueeze(p_a[:, 1], 1) - torch.unsqueeze(p_b[:, 1], 0)
        dis = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + eps)
        return dis

    def get_manhattan_dis(self, p_a, p_b):
        eps = 1e-12
        x = torch.abs(torch.unsqueeze(p_a[:, 0], 1) - torch.unsqueeze(p_b[:, 0], 0))  # N 2 -> NA 1 - 1 NB -> NA NB
        y = torch.abs(torch.unsqueeze(p_a[:, 1], 1) - torch.unsqueeze(p_b[:, 1], 0))
        # dis = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + eps)
        dis = x + y
        return dis

    def get_des_hanmingdis(self, des_a, des_b):
        desc_binary_a = torch.where(des_a >= 0, torch.ones_like(des_a), torch.zeros_like(des_a))
        desc_binary_b = torch.where(des_b >= 0, torch.ones_like(des_b), torch.zeros_like(des_b))
        hanming_dist = torch.ones((des_a.shape[0], des_b.shape[0]), device=self.device) * des_a.shape[1] - desc_binary_a @ desc_binary_b.t() - (1 - desc_binary_a) @ (1 - desc_binary_b.t())
        return hanming_dist

    def detector_selfsupervised_msip_loss(self, descA, descB, Homo, inv_Homo):  
        heatmapA = flattenDetection_new(descA)  #(B, 1, H, W)
        heatmapB = flattenDetection_new(descB)
        mask_borders = torch.ones_like(heatmapA)    # border_area: 0, valid_area: 1
        
        warped_heatmapA = batch_inv_warp_image(heatmapA * mask_borders, Homo, mode='bilinear').to(self.device).detach()
        warped_heatmapB = batch_inv_warp_image(heatmapB * mask_borders, inv_Homo, mode='bilinear').to(self.device).detach()
        visibleA_mask = batch_inv_warp_image(mask_borders, inv_Homo, mode='bilinear').to(self.device)
        visibleB_mask = batch_inv_warp_image(mask_borders, Homo, mode='bilinear').to(self.device)
        
        visibleA_mask = visibleA_mask * mask_borders
        visibleB_mask = visibleB_mask * mask_borders

        heatmapA *= visibleA_mask
        heatmapB *= visibleB_mask
        warped_heatmapA *= visibleB_mask
        warped_heatmapB *= visibleA_mask

        H, W = heatmapA.shape[2], heatmapA.shape[3]
        MSIP_sizes = [8]
        MSIP_factor_loss = [2]   # [256.0, 64.0, 16.0, 4.0, 1.0]
        loss_indexes = 0
        ip_layer_kernels = create_kernels(MSIP_sizes, device=self.device)

        torch.manual_seed(12345)
        # torch.cuda.manual_seed(12345)

        for idx in range(len(MSIP_sizes)):
            win = MSIP_sizes[idx]
            _, weights_visibleA, map_nms = grid_indexes_nms_conv(visibleA_mask, ip_layer_kernels, win)
            _, weights_visibleB, _ = grid_indexes_nms_conv(visibleB_mask, ip_layer_kernels, win)
            imgA_indexes_nms_warped, _, _ = grid_indexes_nms_conv(warped_heatmapA, ip_layer_kernels, win)
            imgB_indexes_nms_warped, _, _ = grid_indexes_nms_conv(warped_heatmapB, ip_layer_kernels, win)       

            imgA_indexes, _ = ip_layer(heatmapA, win, ip_layer_kernels)     # [2, H/win, W/win]
            imgB_indexes, _ = ip_layer(heatmapB, win, ip_layer_kernels)
            weightA = ip_softscores(heatmapA, win, ip_layer_kernels).detach()        # [H/win, W/win]
            weightB = ip_softscores(heatmapB, win, ip_layer_kernels).detach()
            
            coordinate_weighting = True
            if coordinate_weighting:
                shape = weightA.shape

                weightA = torch.flatten(weightA)
                weightB = torch.flatten(weightB)

                weightA = F.softmax(weightA, dim=-1)
                weightB = F.softmax(weightB, dim=-1)

                weightA = 100 * weights_visibleA * torch.reshape(weightA, shape)
                weightB = 100 * weights_visibleB * torch.reshape(weightB, shape)
            else:
                weightA = weights_visibleA
                weightB = weights_visibleB
        
            lossA = loss_ln_indexes_norm(imgA_indexes, imgB_indexes_nms_warped, weightA, win, n=2)
            lossB = loss_ln_indexes_norm(imgB_indexes, imgA_indexes_nms_warped, weightB, win, n=2)

            loss_indexes += (lossA + lossB) / 2. * MSIP_factor_loss[idx]
       
        return loss_indexes

    def detector_selfsupervised_loss(self, descA, descB, maskA, maskB, Homo):
        """
        # apply loss on detectors, default is softmax
        :param descA, descB: prediction
            tensor [batch_size, 65, Hc, Wc]
        :param maskA, maskB: valid region in an image
            tensor [batch_size, 1, Hc, Wc]
        :param loss_type:
            str (l2 or softmax)
            softmax is used in original paper
        :return: normalized loss
            tensor
        """
        from utils.utils import warp_points
        batch_size = descA.shape[0]
        
        heatmapA = flattenDetection(descA)  #(B, C, h, w)
        heatmapB = flattenDetection(descB)
        H, W = heatmapA.shape[2], heatmapA.shape[3]
        # ptsA = getPtsFromHeatmap(heatmapA.to('cpu'), self.conf_thresh, self.nms_dist)   # (x,y, prob) 3行n列
        # ptsB = getPtsFromHeatmap(heatmapB.to('cpu'), self.conf_thresh, self.nms_dist)
        pntsA = [getPtsFromHeatmap(heatmapA[i,:,:,:].squeeze().cpu().detach().numpy(), self.conf_thresh, self.nms_dist) for i in range(batch_size)]
        pntsB = [getPtsFromHeatmap(heatmapB[i,:,:,:].squeeze().cpu().detach().numpy(), self.conf_thresh, self.nms_dist) for i in range(batch_size)]
        # ptsA: is list
        
        dis_mean_total = 0
        for idx in range(batch_size):
            pntA, pntB = pntsA[idx], pntsB[idx]
            pntA = torch.tensor(pntA.transpose()).type(torch.FloatTensor)
            pntB = torch.tensor(pntB.transpose()).type(torch.FloatTensor)

            pntA_H = warp_points(pntA[:2, :].transpose(0,1), Homo[idx].squeeze())  # 利用变换矩阵变换坐标点
            pntA_H, mask_points = filter_points(pntA_H, torch.tensor([W, H]), return_mask=True)


            key_dis = self.get_dis(pntA_H[:, :2], pntB[:, :2])
            dis_pair = self.get_point_pair(key_dis, dis_thre=3)  # p -> k
            dis_pair_mean = torch.mean(dis_pair)

            dis_mean_total += dis_pair_mean


            self.get_score_pair()

        loss_dis_pair = dis_mean_total / batch_size


        return loss_dis_pair

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
            loss_func_BCE = nn.BCELoss(reduction='none').cuda()     # BCELoss二分类交叉熵； reduction = ‘none’，直接返回向量形式的 loss
            loss = loss_func_BCE(nn.functional.softmax(input, dim=1), target)
            loss = (loss.sum(dim=1) * mask).sum()
            loss = loss / (mask.sum() + 1e-10)
        elif loss_type == "focalloss":
            loss_func_Focal = FocalLoss().cuda()
            loss = loss_func_Focal(input, target)
            loss = (loss.sum(dim=1) * mask).sum()
            loss = loss / (mask.sum() + 1e-10)
        return loss

    def detector_selfsupervised_loss_dkd(self, kpA, kpB, descA, descB, desc_mapA, desc_mapB, scoreddisA, scoreddisB, scoresA, scoresB, scores_mapA, scores_mapB, Homo, inv_Homo, H, W, wrp=1, wpl=1, wdl=1, wrl=1, t_des=0.02, t_rel=1):
        batch_size = len(kpA)

        dis_mean_total = 0
        dis_peak_scored = 0
        dis_desc = 0
        dis_rel = 0
        eps = 1e-12
        valid_batch_size = batch_size
        for idx in range(batch_size):
            pntA, pntB = kpA[idx], kpB[idx]     # M X 2 (x,y)
            pntA = pntA.float()
            pntB = pntB.float()
            # pntA = torch.tensor(pntA, device=self.device).type(torch.FloatTensor)
            # pntB = torch.tensor(pntB, device=self.device).type(torch.FloatTensor)
            pntA = (pntA + 1) / 2 * pntA.new_tensor([[W - 1, H - 1]]).to(pntA.device)
            pntB = (pntB + 1) / 2 * pntB.new_tensor([[W - 1, H - 1]]).to(pntB.device)
            pntA_H = warp_points(pntA[:, :2], Homo[idx].squeeze(), device=self.device)  # 利用变换矩阵变换坐标点
            pntA_H, mask_pointA = filter_points(pntA_H, torch.tensor([W, H], device=self.device), return_mask=True)
            pntB_invH = warp_points(pntB[:, :2], inv_Homo[idx].squeeze(), device=self.device)
            pntB_invH, mask_pointB = filter_points(pntB_invH, torch.tensor([W, H], device=self.device), return_mask=True)
            
            try:
                assert pntA_H.shape[0] > 0 and pntB_invH.shape[0] > 0 
                # print(pntA_H.shape, pntB.shape, pntB_invH.shape, pntA.shape)
                key_disAB = self.get_dis(pntA_H[:, :2], pntB[:, :2])
                key_disBA = self.get_dis(pntB_invH[:, :2], pntA[:, :2])
                pairAB_A, pairAB_B, dis_pairAB = self.get_point_pair(key_disAB, dis_thre=self.correspond)  # p -> k
                pairBA_B, pairBA_A, dis_pairBA = self.get_point_pair(key_disBA, dis_thre=self.correspond)  # p -> k
                key_manhattandisAB = self.get_manhattan_dis(pntA_H[:, :2], pntB[:, :2])
                key_manhattandisBA = self.get_manhattan_dis(pntB_invH[:, :2], pntA[:, :2])
                assert dis_pairAB.shape[0] > 0 and dis_pairBA.shape[0] > 0     
                # dis_pair_meanAB = torch.mean(dis_pairAB)
                # dis_pair_meanBA = torch.mean(dis_pairBA)
                dis_pair_meanAB = torch.mean(key_manhattandisAB[pairAB_A, pairAB_B])
                dis_pair_meanBA = torch.mean(key_manhattandisBA[pairBA_B, pairBA_A])

                dis_peak_mean = torch.mean(torch.cat((scoreddisA[idx], scoreddisB[idx]), dim=0))
                # dis_peak_meanB = torch.mean(scoreddisB[idx])

                dis_mean_total += dis_pair_meanAB + dis_pair_meanBA
                dis_peak_scored += dis_peak_mean

                pntA_H_normalized = pntA_H / pntA_H.new_tensor([W - 1, H - 1]).to(pntA_H.device) * 2 - 1  # (w,h) -> (-1~1,-1~1)
                pntB_invH_normalized = pntB_invH / pntB_invH.new_tensor([W - 1, H - 1]).to(pntB_invH.device) * 2 - 1  # (w,h) -> (-1~1,-1~1)

                cross_sim_AB = descA[idx] @ desc_mapB[idx, :, :, :].view(-1, H * W)
                cross_sim_BA = descB[idx] @ desc_mapA[idx, :, :, :].view(-1, H * W)   # M x H x W
                valid_cross_sim_AB = cross_sim_AB[mask_pointA, :]   # repeat_nA x (H X W)
                valid_cross_sim_BA = cross_sim_BA[mask_pointB, :]   # repeat_nB x (H X W)
                soft_valid_cross_sim_AtoB = F.softmax((valid_cross_sim_AB - 1) / t_des, dim=-1).view(-1, H, W)
                soft_valid_cross_sim_BtoA = F.softmax((valid_cross_sim_BA - 1) / t_des, dim=-1).view(-1, H, W)

                soft_valid_cross_sim_AB_all = F.grid_sample(soft_valid_cross_sim_AtoB.unsqueeze(0),
                                                            pntA_H_normalized.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True).squeeze(2).squeeze(0)  # repeat_nA x repeat_nA
                
                soft_valid_cross_sim_BA_all = F.grid_sample(soft_valid_cross_sim_BtoA.unsqueeze(0),
                                                            pntB_invH_normalized.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True).squeeze(2).squeeze(0)  # repeat_nA x repeat_nA

                valid_mask_AB = torch.eye(valid_cross_sim_AB.shape[0]) == 1
                valid_mask_BA = torch.eye(valid_cross_sim_BA.shape[0]) == 1

                # 取对角线
                soft_valid_cross_sim_AB = soft_valid_cross_sim_AB_all[valid_mask_AB]
                soft_valid_cross_sim_BA = soft_valid_cross_sim_BA_all[valid_mask_BA]
                log_soft_valid_cross_sim_AB = -torch.log(soft_valid_cross_sim_AB + eps)
                log_soft_valid_cross_sim_BA = -torch.log(soft_valid_cross_sim_BA + eps)
                dis_cross_sim_mean = torch.sum(torch.cat((log_soft_valid_cross_sim_AB, log_soft_valid_cross_sim_BA), dim=-1), dim=-1) / (pntA.shape[0] + pntB.shape[0])
                dis_desc += dis_cross_sim_mean
                
                exp_valid_cross_sim_AtoB = torch.exp((valid_cross_sim_AB - 1) / t_rel).view(-1, H, W)
                exp_valid_cross_sim_BtoA = torch.exp((valid_cross_sim_BA - 1) / t_rel).view(-1, H, W)
                exp_valid_cross_sim_AB_all = F.grid_sample(exp_valid_cross_sim_AtoB.unsqueeze(0),
                                                        pntA_H_normalized.view(1, 1, -1, 2),
                                                        mode='bilinear', align_corners=True).squeeze(2).squeeze(0)  # repeat_nA x repeat_nA
                
                exp_valid_cross_sim_BA_all = F.grid_sample(exp_valid_cross_sim_BtoA.unsqueeze(0),
                                                            pntB_invH_normalized.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True).squeeze(2).squeeze(0)  # repeat_nA x repeat_nA
                rel_valid_cross_sim_AB = exp_valid_cross_sim_AB_all[valid_mask_AB]
                rel_valid_cross_sim_BA = exp_valid_cross_sim_BA_all[valid_mask_BA]

                scoresAB_all = F.grid_sample(scores_mapB[idx, :, :, :].unsqueeze(0),
                                            pntA_H_normalized.view(1, 1, -1, 2),
                                            mode='bilinear', align_corners=True).squeeze(2).squeeze(0)      # 1 x repeat_nA

                scoresBA_all = F.grid_sample(scores_mapA[idx, :, :, :].unsqueeze(0),
                                            pntB_invH_normalized.view(1, 1, -1, 2),
                                            mode='bilinear', align_corners=True).squeeze(2).squeeze(0)      # 1 x repeat_nB

                scores_AAB = scoresA[idx][mask_pointA] * scoresAB_all
                scores_BBA = scoresB[idx][mask_pointB] * scoresBA_all
                scores_sum_AAB = torch.sum(scores_AAB, dim=-1)
                scores_sum_BBA = torch.sum(scores_BBA, dim=-1)
                scores_normalize_AAB = scores_AAB / (scores_sum_AAB + eps)
                scores_normalize_BBA = scores_BBA / (scores_sum_BBA + eps)
                rel_meanA = torch.sum(scores_normalize_AAB * (1 - rel_valid_cross_sim_AB), dim=-1) / pntA.shape[0]
                rel_meanB = torch.sum(scores_normalize_BBA * (1 - rel_valid_cross_sim_BA), dim=-1) / pntB.shape[0]
                dis_rel += rel_meanA + rel_meanB

            except:
                # dis_mean_total += 0
                # dis_peak_scored += 0
                # dis_desc += 0
                # dis_rel += 0
                print('Overlapping too small or Repeatable point pairs distance is greater than correspond(4):', pntA.shape[0], pntA_H.shape[0], pntB.shape[0], pntB_invH.shape[0])
                valid_batch_size -= 1

        valid_batch_size = valid_batch_size if valid_batch_size > 0 else eps
        # Reprojection Loss 
        reprojection_loss = dis_mean_total / (valid_batch_size * 2)
        # Dispersity Peak Loss
        dispersity_peak_loss = dis_peak_scored / valid_batch_size
        # NRE Loss
        descriptor_loss = dis_desc / valid_batch_size
        # Reliability Loss
        reliability_loss = dis_rel / (valid_batch_size * 2)
        
        loss_total = wrp * reprojection_loss + wpl * dispersity_peak_loss + wdl * descriptor_loss + wrl * reliability_loss
        loss_group = {
            'reprojection_loss': reprojection_loss, 
            'dispersity_peak_loss': dispersity_peak_loss,
            'descriptor_loss': descriptor_loss,
            'reliability_loss': reliability_loss}
        return loss_total, loss_group

    def detector_selfsupervised_loss_dkd_dense(self, kpA, kpB, descA, descB, desc_mapA, desc_mapB, scoreddisA, scoreddisB, scoresA, scoresB, scores_mapA, scores_mapB, Homo, inv_Homo, H, W, wrp=1, wpl=1, wdl=1, wrl=1, t_des=0.02, t_rel=1):
        batch_size = len(kpA)

        dis_mean_total = 0
        dis_peak_scored = 0
        dis_desc = 0
        dis_rel = 0
        eps = 1e-12
        valid_batch_size = batch_size
        countA, countB = 0, 0
        for idx in range(batch_size):
            pntA, pntB = kpA[idx], kpB[idx]     # M X 2 (x,y)
            pntA = pntA.float()
            pntB = pntB.float()
            # pntA = torch.tensor(pntA, device=self.device).type(torch.FloatTensor)
            # pntB = torch.tensor(pntB, device=self.device).type(torch.FloatTensor)
            numA, numB = pntA.shape[0], pntB.shape[0]
            pntA = (pntA + 1) / 2 * pntA.new_tensor([[W - 1, H - 1]]).to(pntA.device)
            pntB = (pntB + 1) / 2 * pntB.new_tensor([[W - 1, H - 1]]).to(pntB.device)
            pntA_H = warp_points(pntA[:, :2], Homo[idx].squeeze(), device=self.device)  # 利用变换矩阵变换坐标点
            pntA_H, mask_pointA = filter_points(pntA_H, torch.tensor([W, H], device=self.device), return_mask=True)
            pntB_invH = warp_points(pntB[:, :2], inv_Homo[idx].squeeze(), device=self.device)
            pntB_invH, mask_pointB = filter_points(pntB_invH, torch.tensor([W, H], device=self.device), return_mask=True)
            
            try:
                assert pntA_H.shape[0] > 0 and pntB_invH.shape[0] > 0 
                # print(pntA_H.shape, pntB.shape, pntB_invH.shape, pntA.shape)
                key_disAB = self.get_dis(pntA_H[:, :2], pntB[:, :2])
                key_disBA = self.get_dis(pntB_invH[:, :2], pntA[:, :2])
                pairAB_A, pairAB_B, dis_pairAB = self.get_point_pair(key_disAB, dis_thre=self.correspond)  # p -> k
                pairBA_B, pairBA_A, dis_pairBA = self.get_point_pair(key_disBA, dis_thre=self.correspond)  # p -> k
                key_manhattandisAB = self.get_manhattan_dis(pntA_H[:, :2], pntB[:, :2])
                key_manhattandisBA = self.get_manhattan_dis(pntB_invH[:, :2], pntA[:, :2])
                assert dis_pairAB.shape[0] > 0 and dis_pairBA.shape[0] > 0     
                # dis_pair_meanAB = torch.mean(dis_pairAB)
                # dis_pair_meanBA = torch.mean(dis_pairBA)
                dis_pair_meanAB = torch.mean(key_manhattandisAB[pairAB_A, pairAB_B])
                dis_pair_meanBA = torch.mean(key_manhattandisBA[pairBA_B, pairBA_A])
                dis_peak_mean = torch.mean(torch.cat((scoreddisA[idx], scoreddisB[idx]), dim=0))
                # dis_peak_meanB = torch.mean(scoreddisB[idx])
                
                dis_mean_total += dis_pair_meanAB + dis_pair_meanBA
                dis_peak_scored += dis_peak_mean
                pntA_H_normalized = pntA_H / pntA_H.new_tensor([W - 1, H - 1]).to(pntA_H.device) * 2 - 1  # (w,h) -> (-1~1,-1~1)
                pntB_invH_normalized = pntB_invH / pntB_invH.new_tensor([W - 1, H - 1]).to(pntB_invH.device) * 2 - 1  # (w,h) -> (-1~1,-1~1)

                current_descA, current_descB = descA[countA:countA+numA], descB[countB:countB+numB]
                
                cross_sim_AB = current_descA @ desc_mapB[idx, :, :, :].view(-1, H * W)
                cross_sim_BA = current_descB @ desc_mapA[idx, :, :, :].view(-1, H * W)   # M x H x W
                cross_sim_AB_out = torch.cat((cross_sim_AB, torch.ones((numA, 1), device=cross_sim_AB.device)), dim=-1)      # NA X (H X W + 1)
                cross_sim_BA_out = torch.cat((cross_sim_BA, torch.ones((numB, 1), device=cross_sim_BA.device)), dim=-1)      # NB X (H X W + 1)
                cross_sim_AB_out[mask_pointA, -1] -= 1
                cross_sim_BA_out[mask_pointB, -1] -= 1
                soft_cross_sim_AtoB = F.softmax((cross_sim_AB_out - 1) / t_des, dim=-1)         # NA X (H X W + 1)
                soft_cross_sim_BtoA = F.softmax((cross_sim_BA_out - 1) / t_des, dim=-1)         # NB X (H X W + 1)
                soft_cross_sim_AB_all = F.grid_sample(soft_cross_sim_AtoB[:, :-1].view(-1, H, W).unsqueeze(0),
                                                    pntA_H_normalized.detach().view(1, 1, -1, 2),
                                                    mode='bilinear', align_corners=True).squeeze(2).squeeze(0)  # NA x repeat_nA
                
                soft_cross_sim_BA_all = F.grid_sample(soft_cross_sim_BtoA[:, :-1].view(-1, H, W).unsqueeze(0),
                                                    pntB_invH_normalized.detach().view(1, 1, -1, 2),
                                                    mode='bilinear', align_corners=True).squeeze(2).squeeze(0)  # NB x repeat_nB
                soft_valid_cross_sim_AB_all = soft_cross_sim_AB_all[mask_pointA, :]             # repeat_nA x repeat_nA
                soft_valid_cross_sim_BA_all = soft_cross_sim_BA_all[mask_pointB, :]             # repeat_nB x repeat_nB

                valid_mask_AB = torch.eye(soft_valid_cross_sim_AB_all.shape[0]) == 1
                valid_mask_BA = torch.eye(soft_valid_cross_sim_BA_all.shape[0]) == 1


                # 取对角线
                soft_valid_cross_sim_AB = soft_valid_cross_sim_AB_all[valid_mask_AB]
                soft_valid_cross_sim_BA = soft_valid_cross_sim_BA_all[valid_mask_BA]
                soft_novalid_cross_sim_AB = soft_cross_sim_AtoB[mask_pointA==False, -1]              # (NA-repeat_nA) x 1
                soft_novalid_cross_sim_BA = soft_cross_sim_BtoA[mask_pointB==False, -1]
                log_soft_valid_cross_sim_AB = -torch.log(soft_valid_cross_sim_AB + eps)
                log_soft_valid_cross_sim_BA = -torch.log(soft_valid_cross_sim_BA + eps)
                log_soft_novalid_cross_sim_AB = -torch.log(soft_novalid_cross_sim_AB + eps)
                log_soft_novalid_cross_sim_BA = -torch.log(soft_novalid_cross_sim_BA + eps)

                dis_cross_sim_mean = torch.sum(torch.cat((log_soft_valid_cross_sim_AB, log_soft_valid_cross_sim_BA, log_soft_novalid_cross_sim_AB, log_soft_novalid_cross_sim_BA), dim=-1), dim=-1) / (pntA.shape[0] + pntB.shape[0])
                dis_desc += dis_cross_sim_mean
                
                valid_cross_sim_AB = cross_sim_AB[mask_pointA, :]   # repeat_nA x (H X W)
                valid_cross_sim_BA = cross_sim_BA[mask_pointB, :]   # repeat_nB x (H X W)
                exp_valid_cross_sim_AtoB = torch.exp((valid_cross_sim_AB - 1) / t_rel).view(-1, H, W)
                exp_valid_cross_sim_BtoA = torch.exp((valid_cross_sim_BA - 1) / t_rel).view(-1, H, W)
                exp_valid_cross_sim_AB_all = F.grid_sample(exp_valid_cross_sim_AtoB.unsqueeze(0),
                                                        pntA_H_normalized.detach().view(1, 1, -1, 2),
                                                        mode='bilinear', align_corners=True).squeeze(2).squeeze(0)  # repeat_nA x repeat_nA
                
                exp_valid_cross_sim_BA_all = F.grid_sample(exp_valid_cross_sim_BtoA.unsqueeze(0),
                                                        pntB_invH_normalized.detach().view(1, 1, -1, 2),
                                                        mode='bilinear', align_corners=True).squeeze(2).squeeze(0)  # repeat_nB x repeat_nB
                rel_valid_cross_sim_AB = exp_valid_cross_sim_AB_all[valid_mask_AB]
                rel_valid_cross_sim_BA = exp_valid_cross_sim_BA_all[valid_mask_BA]

                scoresAB_all = F.grid_sample(scores_mapB[idx, :, :, :].unsqueeze(0),
                                            pntA_H_normalized.detach().view(1, 1, -1, 2),
                                            mode='bilinear', align_corners=True).squeeze(2).squeeze(0)      # 1 x repeat_nA

                scoresBA_all = F.grid_sample(scores_mapA[idx, :, :, :].unsqueeze(0),
                                            pntB_invH_normalized.detach().view(1, 1, -1, 2),
                                            mode='bilinear', align_corners=True).squeeze(2).squeeze(0)      # 1 x repeat_nB
                scores_AAB = scoresA[idx][mask_pointA] * scoresAB_all
                scores_BBA = scoresB[idx][mask_pointB] * scoresBA_all
                scores_sum_AAB = torch.sum(scores_AAB, dim=-1)
                scores_sum_BBA = torch.sum(scores_BBA, dim=-1)
                scores_normalize_AAB = scores_AAB / (scores_sum_AAB + eps)
                scores_normalize_BBA = scores_BBA / (scores_sum_BBA + eps)
                rel_meanA = torch.sum(scores_normalize_AAB * (1 - rel_valid_cross_sim_AB), dim=-1) / pntA.shape[0]
                rel_meanB = torch.sum(scores_normalize_BBA * (1 - rel_valid_cross_sim_BA), dim=-1) / pntB.shape[0]
                dis_rel += rel_meanA + rel_meanB
                countA += numA
                countB += numB

                # # Penalty
                # scores_mean += (-torch.log(torch.mean(scoresA[idx], dim=-1) + eps) - torch.log(torch.mean(scoresB[idx], dim=-1) + eps)) 
                # print(scores_mean)
            except:
                # dis_mean_total += 0
                # dis_peak_scored += 0
                # dis_desc += 0
                # dis_rel += 0
                print('Overlapping too small or Repeatable point pairs distance is greater than correspond(4):', pntA.shape[0], pntA_H.shape[0], pntB.shape[0], pntB_invH.shape[0])
                valid_batch_size -= 1
                countA += numA
                countB += numB

        valid_batch_size = valid_batch_size if valid_batch_size > 0 else eps
        # Reprojection Loss 
        reprojection_loss = dis_mean_total / (valid_batch_size * 2)
        # Dispersity Peak Loss
        dispersity_peak_loss = dis_peak_scored / valid_batch_size
        # NRE Loss
        descriptor_loss = dis_desc / valid_batch_size
        # Reliability Loss
        reliability_loss = dis_rel / (valid_batch_size * 2)
        # # Score Penalty
        # score_penalty = scores_mean / (valid_batch_size * 2)
        loss_total = wrp * reprojection_loss + wpl * dispersity_peak_loss + wdl * descriptor_loss + wrl * reliability_loss
        loss_group = {
            'reprojection_loss': reprojection_loss, 
            'dispersity_peak_loss': dispersity_peak_loss,
            'descriptor_loss': descriptor_loss,
            'reliability_loss': reliability_loss}
        return loss_total, loss_group

    def detector_selfsupervised_loss_dkd_sparse(self, kpA, kpB, descA, descB, desc_mapA, desc_mapB, scoreddisA, scoreddisB, scoresA, scoresB, scores_mapA, scores_mapB, Homo, inv_Homo, H, W, wrp=1, wpl=1, wdl=1, wrl=1, t_des=0.02, t_rel=1):
        batch_size = len(kpA)

        dis_mean_total = 0
        dis_peak_scored = 0
        dis_desc = 0
        dis_rel = 0
        eps = 1e-12
        valid_batch_size = batch_size
        countA, countB = 0, 0
        
        for idx in range(batch_size):
            pntA, pntB = kpA[idx], kpB[idx]     # M X 2 (x,y)
            pntA = pntA.float()
            pntB = pntB.float()
            # pntA = torch.tensor(pntA, device=self.device).type(torch.FloatTensor)
            # pntB = torch.tensor(pntB, device=self.device).type(torch.FloatTensor)
            # print(pntA.shape, pntA, W, H, pntA.device)
            # pntA = (pntA + 1) / 2 * torch.tensor([[W - 1, H - 1]], device=pntA.device)
            # pntB = (pntB + 1) / 2 * torch.tensor([[W - 1, H - 1]], device=pntB.device)         
            pntA = (pntA + 1) / 2 * pntA.new_tensor([[W - 1, H - 1]])
            pntB = (pntB + 1) / 2 * pntB.new_tensor([[W - 1, H - 1]])
            numA, numB = pntA.shape[0], pntB.shape[0]
            pntA_H = warp_points(pntA[:, :2], Homo[idx].squeeze(), device=self.device)  # 利用变换矩阵变换坐标点
            pntA_H, mask_pointA = filter_points(pntA_H, torch.tensor([W, H], device=self.device), return_mask=True)
            pntB_invH = warp_points(pntB[:, :2], inv_Homo[idx].squeeze(), device=self.device)
            pntB_invH, mask_pointB = filter_points(pntB_invH, torch.tensor([W, H], device=self.device), return_mask=True)
            try:
                assert pntA_H.shape[0] > 0 and pntB_invH.shape[0] > 0 
                # print(pntA_H.shape, pntB.shape, pntB_invH.shape, pntA.shape)
                key_disAB = self.get_dis(pntA_H[:, :2], pntB[:, :2])
                key_disBA = self.get_dis(pntB_invH[:, :2], pntA[:, :2])
                pairAB_A, pairAB_B, dis_pairAB = self.get_point_pair(key_disAB, dis_thre=self.correspond)  # p -> k
                pairBA_B, pairBA_A, dis_pairBA = self.get_point_pair(key_disBA, dis_thre=self.correspond)  # p -> k
                key_manhattandisAB = self.get_manhattan_dis(pntA_H[:, :2], pntB[:, :2])
                key_manhattandisBA = self.get_manhattan_dis(pntB_invH[:, :2], pntA[:, :2])

                assert dis_pairAB.shape[0] > 0 and dis_pairBA.shape[0] > 0     
                # dis_pair_meanAB = torch.mean(dis_pairAB)
                # dis_pair_meanBA = torch.mean(dis_pairBA)
                dis_pair_meanAB = torch.mean(key_manhattandisAB[pairAB_A, pairAB_B])
                dis_pair_meanBA = torch.mean(key_manhattandisBA[pairBA_B, pairBA_A])

                dis_peak_mean = torch.mean(torch.cat((scoreddisA[idx], scoreddisB[idx]), dim=0))
                # dis_peak_meanB = torch.mean(scoreddisB[idx])

                dis_mean_total += dis_pair_meanAB + dis_pair_meanBA
                dis_peak_scored += dis_peak_mean

                # pntA_H_normalized = pntA_H / pntA_H.new_tensor([W - 1, H - 1]).to(pntA_H.device) * 2 - 1  # (w,h) -> (-1~1,-1~1)
                # pntB_invH_normalized = pntB_invH / pntB_invH.new_tensor([W - 1, H - 1]).to(pntB_invH.device) * 2 - 1  # (w,h) -> (-1~1,-1~1)
                current_descA, current_descB = descA[countA:countA+numA], descB[countB:countB+numB]

                cross_sim_AB = current_descA @ current_descB.transpose(0, 1)
                cross_sim_BA = current_descB @ current_descA.transpose(0, 1)
                cross_sim_AB_out = torch.cat((cross_sim_AB, torch.ones((numA, 1), device=cross_sim_AB.device)), dim=-1)      # NA X (NB + 1)
                cross_sim_BA_out = torch.cat((cross_sim_BA, torch.ones((numB, 1), device=cross_sim_BA.device)), dim=-1)      # NB X (NA + 1)
                valid_indexesA = torch.tensor(list(range(numA)), device=self.device)[mask_pointA][pairAB_A]
                valid_indexesB = torch.tensor(list(range(numB)), device=self.device)[mask_pointB][pairBA_B]
                cross_sim_AB_out[valid_indexesA, -1] -= 1
                cross_sim_BA_out[valid_indexesB, -1] -= 1
                # print(cross_sim_AB_out[:, -1])
                valid_cross_sim_AB = cross_sim_AB[mask_pointA, :]   # repeat_nA x nB
                valid_cross_sim_BA = cross_sim_BA[mask_pointB, :]   # repeat_nB x nA

                soft_cross_sim_AtoB = F.softmax((cross_sim_AB_out - 1) / t_des, dim=-1)
                soft_cross_sim_BtoA = F.softmax((cross_sim_BA_out - 1) / t_des, dim=-1)

                # soft_valid_cross_sim_AB_all = F.grid_sample(soft_valid_cross_sim_AtoB.unsqueeze(0),
                #                                             pntA_H_normalized.view(1, 1, -1, 2),
                #                                             mode='bilinear', align_corners=True).squeeze(2).squeeze(0)  # repeat_nA x repeat_nA
                
                # soft_valid_cross_sim_BA_all = F.grid_sample(soft_valid_cross_sim_BtoA.unsqueeze(0),
                #                                             pntB_invH_normalized.view(1, 1, -1, 2),
                #                                             mode='bilinear', align_corners=True).squeeze(2).squeeze(0)  # repeat_nA x repeat_nA

                # valid_mask_AB = torch.eye(valid_cross_sim_AB.shape[0]) == 1
                # valid_mask_BA = torch.eye(valid_cross_sim_BA.shape[0]) == 1

                # 取对角线
                soft_valid_cross_sim_AB = soft_cross_sim_AtoB[mask_pointA, :-1][pairAB_A, pairAB_B]
                soft_valid_cross_sim_BA = soft_cross_sim_BtoA[mask_pointB, :-1][pairBA_B, pairBA_A]
                soft_novalid_cross_sim_AB = soft_cross_sim_AtoB[cross_sim_AB_out[:, -1]==1, -1]
                soft_novalid_cross_sim_BA = soft_cross_sim_BtoA[cross_sim_BA_out[:, -1]==1, -1]
                log_soft_valid_cross_sim_AB = -torch.log(soft_valid_cross_sim_AB + eps)
                log_soft_valid_cross_sim_BA = -torch.log(soft_valid_cross_sim_BA + eps)
                log_soft_novalid_cross_sim_AB = -torch.log(soft_novalid_cross_sim_AB + eps)
                log_soft_novalid_cross_sim_BA = -torch.log(soft_novalid_cross_sim_BA + eps)
                dis_cross_sim_mean = torch.sum(torch.cat((log_soft_valid_cross_sim_AB, log_soft_valid_cross_sim_BA, log_soft_novalid_cross_sim_AB, log_soft_novalid_cross_sim_BA), dim=-1), dim=-1) / (numA + numB)
                dis_desc += dis_cross_sim_mean
                
                exp_valid_cross_sim_AtoB = torch.exp((valid_cross_sim_AB - 1) / t_rel)  # repeat_nA x nB
                exp_valid_cross_sim_BtoA = torch.exp((valid_cross_sim_BA - 1) / t_rel)  # repeat_nB x nA
                # exp_valid_cross_sim_AB_all = F.grid_sample(exp_valid_cross_sim_AtoB.unsqueeze(0),
                #                                         pntA_H_normalized.view(1, 1, -1, 2),
                #                                         mode='bilinear', align_corners=True).squeeze(2).squeeze(0)  # repeat_nA x repeat_nA
                
                # exp_valid_cross_sim_BA_all = F.grid_sample(exp_valid_cross_sim_BtoA.unsqueeze(0),
                #                                             pntB_invH_normalized.view(1, 1, -1, 2),
                #                                             mode='bilinear', align_corners=True).squeeze(2).squeeze(0)  # repeat_nA x repeat_nA
                # rel_valid_cross_sim_AB = exp_valid_cross_sim_AB_all[valid_mask_AB]
                # rel_valid_cross_sim_BA = exp_valid_cross_sim_BA_all[valid_mask_BA]
                rel_valid_cross_sim_AB = exp_valid_cross_sim_AtoB[pairAB_A, pairAB_B]
                rel_valid_cross_sim_BA = exp_valid_cross_sim_BtoA[pairBA_B, pairBA_A]

                # scoresAB_all = F.grid_sample(scores_mapB[idx, :, :, :].unsqueeze(0),
                #                             pntA_H_normalized.view(1, 1, -1, 2),
                #                             mode='bilinear', align_corners=True).squeeze(2).squeeze(0)      # 1 x repeat_nA

                # scoresBA_all = F.grid_sample(scores_mapA[idx, :, :, :].unsqueeze(0),
                #                             pntB_invH_normalized.view(1, 1, -1, 2),
                #                             mode='bilinear', align_corners=True).squeeze(2).squeeze(0)      # 1 x repeat_nB
                scores_AAB = scoresA[idx][mask_pointA][pairAB_A] * scoresB[idx][pairAB_B]
                scores_BBA = scoresB[idx][mask_pointB][pairBA_B] * scoresA[idx][pairBA_A]
                scores_sum_AAB = torch.sum(scores_AAB, dim=-1)
                scores_sum_BBA = torch.sum(scores_BBA, dim=-1)
                scores_normalize_AAB = scores_AAB / (scores_sum_AAB + eps)
                scores_normalize_BBA = scores_BBA / (scores_sum_BBA + eps)
                rel_meanA = torch.sum(scores_normalize_AAB * (1 - rel_valid_cross_sim_AB), dim=-1) / numA
                rel_meanB = torch.sum(scores_normalize_BBA * (1 - rel_valid_cross_sim_BA), dim=-1) / numB
                dis_rel += rel_meanA + rel_meanB
                countA += numA
                countB += numB

            except:
                # dis_mean_total += 0
                # dis_peak_scored += 0
                # dis_desc += 0
                # dis_rel += 0
                print('Overlapping too small or Repeatable point pairs distance is greater than correspond(4):', pntA.shape[0], pntA_H.shape[0], pntB.shape[0], pntB_invH.shape[0])
                valid_batch_size -= 1
                countA += numA
                countB += numB
            

        valid_batch_size = valid_batch_size if valid_batch_size > 0 else eps
        # Reprojection Loss 
        reprojection_loss = dis_mean_total / (valid_batch_size * 2)
        # Dispersity Peak Loss
        dispersity_peak_loss = dis_peak_scored / valid_batch_size
        # NRE Loss
        descriptor_loss = dis_desc / valid_batch_size
        # Reliability Loss
        reliability_loss = dis_rel / (valid_batch_size * 2)
        
        loss_total = wrp * reprojection_loss + wpl * dispersity_peak_loss + wdl * descriptor_loss + wrl * reliability_loss
        loss_group = {
            'reprojection_loss': reprojection_loss, 
            'dispersity_peak_loss': dispersity_peak_loss,
            'descriptor_loss': descriptor_loss,
            'reliability_loss': reliability_loss}
        return loss_total, loss_group

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
        det_loss_type = self.config["model"]["detector_loss"]["loss_type"]

        add_dustbin = False
        if det_loss_type == "l2":
            add_dustbin = False
        elif det_loss_type == "softmax":    # 如果是softmax，打开add_dustbin，即64维增加1维=65维
            add_dustbin = True
        elif det_loss_type == "focalloss":
            add_dustbin = True

        # if_warp = self.config['data']['warped_pair']['enable']

        self.scalar_dict, self.images_dict, self.hist_dict = {}, {}, {}

        ## get the inputs
        imgA, _, labelA_sort, _, imgB, _, labelB_sort, valid_mask_H, name, homography, inv_homography, homography_o= (
            sample['image'], 
            sample['labels'].to(self.device),
            sample['labels_sort'].to(self.device),
            sample['valid_mask'].to(self.device),
            sample['image_H'],
            sample['labels_H'].to(self.device),
            sample['labels_H_sort'].to(self.device),
            sample['valid_mask_H'].to(self.device),
            sample['name'],
            sample['homography'],
            sample['inv_homography'],
            sample['homography_o']
            )
        desc_labelA, desc_labelB = sample['desc'].to(self.device), sample['warped_desc'].to(self.device)
        batch_size, H, W = imgA.shape[0], imgA.shape[2], imgA.shape[3]
        self.batch_size = batch_size
        
        orientA_batch = self.net(imgA.to(self.device))['orientation']        # bx1x(hxw)
        orientB_batch = self.net(imgB.to(self.device))['orientation']

        # 保存方向为csv
        orientation_path = '/hdd/file-input/linwc/Descriptor/data/orientation32/'
        for idx in range(batch_size):
            imgA_name = orientation_path + str(name[idx]).split('_to_')[0] + '.csv'
            imgB_name = orientation_path + str(name[idx]).split('_to_')[1] + '.csv'

            with open(imgA_name, "w") as f:
                np.savetxt(f, orientA_batch[idx,:,:].squeeze().view(H, W).detach().cpu().numpy(), delimiter=',',fmt="%f")
                f.close()

            with open(imgB_name, "w") as f:
                np.savetxt(f, orientB_batch[idx,:,:].squeeze().view(H, W).detach().cpu().numpy(), delimiter=',',fmt="%f")
                f.close()


        # # 画点和方向场
        # x = torch.linspace(0, W-1, W)     # ex: [-2, -1, 0, 1, 2]
        # y = torch.linspace(0, H-1, H) 
        # mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(imgA.device)
        # randp = torch.randint(0, W*H, (100,)).long()
        # pred, img_pair = {}, {}
        # pred.update({
        #     "pts": mesh_points[randp].detach().cpu().numpy(), 
        #     "angles": (orientA_batch[0].squeeze(0))[randp].detach().cpu().numpy(),
        #     })
        # img_pair.update({
        #     "img": imgA[0].cpu().numpy().squeeze(),
        #     })
        # img_pts = draw_orientation(img_pair, pred, radius=1, s=1)
        # cv2.imwrite('demo/orientation_grid' + str(0) + '.bmp', img_pts)
        # exit()

        # train 
        # self.optimizer.zero_grad()
        # outA = self.net(imgA.to(self.device), sub_pixel=True)
        semiA, coarse_descA, desc_mapA, scoresA, scores_mapA, scoredispersitysA = 0, 0, 0, 0, 0, 0
        # outB = self.net(imgB.to(self.device), sub_pixel=True)
        semiB, coarse_descB, desc_mapB, scoresB, scores_mapB, scoredispersitysB = 0, 0, 0, 0, 0, 0
        # ===detecAB===
        mat_H, mat_invH = homography.to(self.device), inv_homography.to(self.device)
        mat_Ho = homography_o.to(self.device)
        # coarse_descA = coarse_descA.squeeze(-1)
        # coarse_descB = coarse_descB.squeeze(-1)
        # print(coarse_descA.shape)
        # loss_supervised = self.detector_selfsupervised_loss(semiA, semiB, mask_3D_flattened, mask_3D_flattened, mat_H)
        # loss_det_selfsupervised = self.detector_selfsupervised_msip_loss(semiA, semiB, mat_H, mat_invH)

        loss_det_selfsupervised, loss_group = 0, {
            'reprojection_loss': 0, 
            'dispersity_peak_loss': 0,
            'descriptor_loss': 0,
            'reliability_loss': 0}
 
        # loss_det_A = 1 * loss_det_A
        # loss_det_B = 1 * loss_det_B
        loss_det_A = torch.tensor([0]).to(self.device)
        loss_det_B = torch.tensor([0]).to(self.device)

        # ===descriptor loss===
        mask_desc = valid_mask_H.squeeze()
        lambda_loss = self.config["model"]["lambda_loss"]
        self_supervised = self.config["model"]["dense_loss"]["self_supervised"]

        if lambda_loss > 0:
            ze = torch.tensor([0]).to(self.device)
            from utils.utils import descriptor_loss_fix
            if not self_supervised:
                loss_desc, loss_descA, loss_descB = self.descriptor_loss(
                    desc_labelA,
                    desc_labelB,
                    coarse_descA,
                    coarse_descB,
                    labelA_sort,
                    labelB_sort,
                    device=self.device
                )
                positive_dist, negative_dist = ze, ze
            else:
                loss_desc, mask, positive_dist, negative_dist = self.descriptor_loss(
                    coarse_descA,
                    coarse_descB,
                    homography,
                    mask_valid=mask_desc,
                    device=self.device,
                )
                loss_descA, loss_descB = ze, ze
        else:
            ze = torch.tensor([0]).to(self.device)
            loss_desc, loss_descA, loss_descB, positive_dist, negative_dist = ze, ze, ze, ze, ze
        loss_desc *= lambda_loss
        
        # loss = loss_det_A + loss_det_B + loss_desc
        loss = loss_det_selfsupervised  + loss_desc

        self.loss = loss
        self.scalar_dict.update(
            {
                "loss": loss,
                "loss_detA": loss_det_A,
                "loss_detB": loss_det_B,
                # "loss_res": loss_res,
                "loss_selfsupervised": loss_det_selfsupervised,
                "loss_desc_total": loss_desc,
                "loss_descA": loss_descA,
                "loss_descB": loss_descB,
                "positive_dist": positive_dist,
                "negative_dist": negative_dist,
            }
        )
        self.input_to_imgDict(sample, self.images_dict)

        # if train:
        #     loss.backward()
        #     # loss.backward(torch.ones_like(loss))             
        #     self.optimizer.step()       # 更新网络参数
        
        # if n_iter % tb_interval == 0 or task == "val":
        if 0:
            logging.info(
                "current iteration: %d, tensorboard_interval: %d", n_iter, tb_interval
            )
            logging.debug(
                "current iteration: %d, tensorboard_interval: %d", n_iter, tb_interval
            )

            topk = 150
            pts_A_lable = getPtsFromLabels2D(toNumpy(labelA_sort[0, 0, :, :])).transpose(1, 0)   # (3, N)
            pts_B_lable = getPtsFromLabels2D(toNumpy(labelB_sort[0, 0, :, :])).transpose(1, 0)
            try: 
                # === get A's heatmap & pts ===
                # heatmapA = flattenDetection_new(semiA, tensor=True)
                # pts_A = getPtsFromHeatmap(toNumpy(heatmapA)[0, 0, :, :], self.conf_thresh, self.nms_dist)  # only touch one
                # pts_A = getPtsFromHeatmapByCoordinates(heatmapA[0, 0, :, :].unsqueeze(0).unsqueeze(0), self.conf_thresh, self.w_size, bord=0)
                pts_A = semiA[0].to(self.device)
                pts_A = (pts_A + 1) / 2 * pts_A.new_tensor([[W - 1, H - 1]]).to(pts_A.device)
                numA = pts_A.shape[0]
                print(numA)
                topk_indices_A = torch.topk(scoresA[0], topk).indices if pts_A.shape[0] >= topk else scoresA[0] >= 0
                pts_A = pts_A[topk_indices_A, :]
                # pts_A = pts_A.transpose(0, 1)
                # pts_A = pts_A.transpose()[:, [0, 1]]
                # pts_A = pts_A.transpose()[:, [0, 1]]
                
                warped_pts_A = warp_points(pts_A, mat_H[0].squeeze(), device=pts_A.device)  # 利用变换矩阵变换坐标点
                warped_pts_A, mask_points = filter_points(warped_pts_A, torch.tensor([W, H], device=pts_A.device), return_mask=True)
                print(warped_pts_A.shape[0])
                # === get B's heatmap & pts ===
                # heatmapB = flattenDetection_new(semiB, tensor=True)

                # heatmapB = heatmapB * valid_mask_H
                pts_B = semiB[0].to(self.device)
                pts_B = (pts_B + 1) / 2 * pts_B.new_tensor([[W - 1, H - 1]]).to(pts_B.device)
                numB = pts_B.shape[0]
                print(pts_B.shape[0])
                topk_indices_B = torch.topk(scoresB[0], topk).indices if pts_B.shape[0] >= topk else scoresB[0] >= 0
                pts_B = pts_B[topk_indices_B, :]
                # pts_B = getPtsFromHeatmap(toNumpy(heatmapB)[0, 0, :, :], self.conf_thresh, self.nms_dist)  # only touch one
                # pts_B = getPtsFromHeatmapByCoordinates(heatmapB[0, 0, :, :].unsqueeze(0).unsqueeze(0), self.conf_thresh, self.w_size, bord=0)
                # pts_B = pts_B.transpose(0, 1)
                # pts_B = pts_B.transpose()[:, [0, 1]]

                key_disAtoB = self.get_dis(warped_pts_A[:, :2], pts_B[:, :2])
                pos_repeatA_mask, pos_repeatB_mask, _ = self.get_point_pair(key_disAtoB, dis_thre=self.correspond)  # p -> k
                assert pos_repeatA_mask.shape[0] > 0 
                pos_repeatA = pts_A[mask_points, :][pos_repeatA_mask, :]
                pos_repeatB = pts_B[pos_repeatB_mask, :]

                # Hanming Distance Nearest Neighborhood
                pred_descA = coarse_descA[:numA][topk_indices_A, :]
                pred_descB = coarse_descB[:numB][topk_indices_B, :]
                hanmingdist_AtoB = self.get_des_hanmingdis(pred_descA, pred_descB)

                pos_nncandA_mask, pos_nncandB_mask, _ = self.get_point_pair(hanmingdist_AtoB, dis_thre=pred_descA.shape[1] + 1)
                pos_nncandA = pts_A[pos_nncandA_mask, :]
                pos_nncandB = pts_B[pos_nncandB_mask, :]

                hanming_thr_ratio = 0.2
                pos_nnA_mask, pos_nnB_mask, _ = self.get_point_pair(hanmingdist_AtoB, dis_thre=int(pred_descA.shape[1] * hanming_thr_ratio))
                pos_nnA = pts_A[pos_nnA_mask, :]
                pos_nnB = pts_B[pos_nnB_mask, :]
                print('pos_nnA:', pos_nnA.shape, 'pos_nnB:', pos_nnB.shape)
                # desc_repeatA = desc_mapA[0, :, torch.tensor(pos_repeatA[:, 1]).long(), torch.tensor(pos_repeatA[:, 0]).long()].transpose(1, 0)
                # desc_repeatB = desc_mapB[0, :, torch.tensor(pos_repeatB[:, 1]).long(), torch.tensor(pos_repeatB[:, 0]).long()].transpose(1, 0)
            except:
                print("The number of points is not enough or overlapping too small!")
            else:
                pred, img_pair = {}, {}
                pred.update({
                    "pts": pts_A.detach().cpu().numpy(), 
                    "pts_H": pts_B.detach().cpu().numpy(),
                    "lab": pts_A_lable,
                    "lab_H": pts_B_lable,
                    "pts_TH": warped_pts_A.detach().cpu().numpy(),
                    "pts_repeatA": pos_repeatA.detach().cpu().numpy(),
                    "pts_repeatB": pos_repeatB.detach().cpu().numpy(),
                    "pts_nncandA": pos_nncandA.detach().cpu().numpy(),
                    "pts_nncandB": pos_nncandB.detach().cpu().numpy(),
                    "pts_nnA": pos_nnA.detach().cpu().numpy(),
                    "pts_nnB": pos_nnB.detach().cpu().numpy(),
                    })
                img_pair.update({
                    "img": imgA[0].cpu().numpy().squeeze(),
                    "img_H": imgB[0].cpu().numpy().squeeze()
                    })
                img_pts = draw_keypoints_pair_train(img_pair, pred, None, radius=1, s=1, H=mat_H[0].detach().cpu().squeeze())
                f = Path(self.webdir) / (str(n_iter) + '_' + str(name[0]) + ".bmp")
                saveImg(img_pts, str(f))

                img_nn_match = draw_match_pair_train(img_pair, pred, radius=1, s=1)
                f1 = Path(self.webdir) / (str(n_iter) + '_' + str(name[0]) + "_nn_match.bmp")
                saveImg(img_nn_match, str(f1))
                # draw_match_picture(imgA[0].cpu().squeeze(), imgB[0].cpu().squeeze(), str(save_match_patho), mat_Ho[0].detach().cpu().squeeze())

            #计算第一张图卡阈值后的hanming距离,从置信度最高的地方开始
            # desc_A = sample_desc_from_points_torch(coarse_descA[0, :, :, :].detach().cpu().unsqueeze(0), torch.tensor(pts_A_lable).transpose(1, 0)).transpose(1, 0)#根据点的置信度获取描述子
            desc_A = desc_mapA[0, :, torch.tensor(pts_A_lable[:120, 1]).long(), torch.tensor(pts_A_lable[:120, 0]).long()].transpose(1, 0)
            for i in range(15):
                desc_predict = desc_A[i,:]
                desc_predict = torch.sigmoid(desc_predict)
                desc_A_thres = torch.where(desc_predict > 0.5, torch.ones_like(desc_predict), torch.zeros_like(desc_predict))
                desc_tmp = desc_labelA[0,i,:]#根据置信度排序
                hanmingdist = sum([e1!=e2 for (e1, e2) in zip(desc_tmp, desc_A_thres)])
                print("A  -- top 15 of %d dist : %d"%(i, hanmingdist))
            # desc_A_single = torch.sigmoid(desc_A)
            desc_A_single = desc_A
            desc_A_single_thres = torch.where(desc_A_single > 0.5, torch.ones_like(desc_A_single), torch.zeros_like(desc_A_single))
            dist_list = []
            for (e1, e2) in zip(desc_A_single_thres, desc_labelA[0, :, :]):
                dist_list.append(sum(e1!=e2))
            dist_meanA = sum(dist_list) / len(dist_list)
            print("A -- dist mean: %d"%(dist_meanA))

            desc_B = desc_mapB[0, :, torch.tensor(pts_B_lable[:120, 1]).long(), torch.tensor(pts_B_lable[:120, 0]).long()].transpose(1, 0)
            for i in range(15):
                desc_predict = desc_B[i,:]
                desc_predict = torch.sigmoid(desc_predict)
                desc_B_thres = torch.where(desc_predict > 0.5, torch.ones_like(desc_predict), torch.zeros_like(desc_predict))
                desc_tmp = desc_labelB[0,i,:]#根据置信度排序
                hanmingdist = sum([e1!=e2 for (e1, e2) in zip(desc_tmp, desc_B_thres)])
                print("B -- top 15 of %d dist : %d"%(i, hanmingdist))

            # desc_B_single = torch.sigmoid(desc_B)
            desc_B_single = desc_B
            desc_B_single_thres = torch.where(desc_B_single > 0.5, torch.ones_like(desc_B_single), torch.zeros_like(desc_B_single))
            dist_list = []
            for (e1, e2) in zip(desc_B_single_thres, desc_labelB[0, :, :]):
                dist_list.append(sum(e1!=e2))
            dist_meanB = sum(dist_list) / len(dist_list)
            print("B -- dist mean: %d"%(dist_meanB))


            self.printLosses(self.scalar_dict, task)
            self.logger.debug("current iteration: %d", n_iter)
            self.logger.debug(
                "loss: %f, loss_detA: %f, loss_detB: %f, loss_det_selfsupervised: %f, loss_desctotal: %f, loss_descA: %f, loss_descB: %f",\
                loss, loss_det_A, loss_det_B, loss_det_selfsupervised, loss_desc, loss_descA, loss_descB
            )
            self.logger.debug(
                "Hamming disA: %f, Hamming disB: %f",\
                dist_meanA, dist_meanB
            )
            # self.tb_images_dict(task, self.images_dict, max_img=2)
            # self.tb_hist_dict(task, self.hist_dict)

        self.tb_scalar_dict(self.scalar_dict, task)

        return loss


    def heatmap_to_nms(self, images_dict, heatmap, name):
        """
        return: 
            heatmap_nms_batch: np [batch, H, W]
        """
        from utils.var_dim import toNumpy

        heatmap_np = toNumpy(heatmap)
        ## heatmap_nms
        # nms_dist = self.config['model']['nms']                      # Andy 添加，直接采用yaml中的定义
        # conf_thresh = self.config['model']['detection_threshold']   # Andy 添加，直接采用yaml中的定义
        heatmap_nms_batch = [self.heatmap_nms(self, h) for h in heatmap_np]  # [batch, H, W] 这里需要修改
        heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)
        # images_dict.update({name + '_nms_batch': heatmap_nms_batch})
        images_dict.update({name + "_nms_batch": heatmap_nms_batch[:, np.newaxis, ...]})
        return heatmap_nms_batch

    def get_residual_loss(self, labels_2D, heatmap, labels_res, name=""):
        # labels_2D：非极大值移植后的heatmap， heatmap：网络输出的未NMS的heatmap
        if abs(labels_2D).sum() == 0:
            return
        outs_res = self.pred_soft_argmax(
            labels_2D, heatmap, labels_res, patch_size=5, device=self.device
        )# 标签的点，和网络预测的heatmap做NMS之后的点，之间的loss
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

        if 0:
            patches_log = do_log(patches)
            # with torch.no_grad():
            dxdy = soft_argmax_2d(
                patches_log, normalized_coordinates=False
            )  # tensor [B, N, patch, patch]
        else:
            dxdy = soft_argmax_2d(
            patches, normalized_coordinates=False
        )  # tensor [B, N, patch, patch], soft_argmax作为非极大值抑制NMS的可微分版本
        
       
        
        dxdy = dxdy.squeeze(1)  # tensor [N, 2]
        dxdy = dxdy - patch_size // 2       # 网络预测的点（NMS后的极值点）对标签位置的偏移

        # extract residual
        def ext_from_points(labels_res, points):
            """
            input:
                labels_res: tensor [batch, channel, H, W]
                points: tensor [N, 4(pos0(batch), pos1(0), pos2(H), pos3(W) )]
            return:
                tensor [N, channel]
            """
            # labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
            labels_res = labels_res.unsqueeze(1)
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
    def heatmap_nms(self, heatmap, nms_dist=2, conf_thresh=0.01):
        """
        input:
            heatmap: np [(1), H, W]
        """
        from utils.utils import getPtsFromHeatmap
        
        nms_dist = self.config['model']['nms']
        conf_thresh = self.config['model']['detection_threshold']
        heatmap = heatmap.squeeze()     # [1,128,128] --> [128,128]
        # print("heatmap: ", heatmap.shape)
        pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)
        semi_thd_nms_sample = np.zeros_like(heatmap)
        semi_thd_nms_sample[
            pts_nms[1, :].astype(int), pts_nms[0, :].astype(int)
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
