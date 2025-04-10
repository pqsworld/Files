"""This is the frontend interface for training
base class: inherited by other Train_model_*.py

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
from tqdm import tqdm
from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging

from utils.tools import dict_update

from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened
from utils.utils import filter_points,warp_points,homography_scaling_torch, inv_warp_image, warp_points_batch

from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch
from utils.utils import save_checkpoint

from pathlib import Path
from timeit import default_timer as timer
import itertools

import os
import math
import cv2



def thd_img(img, thd=0.015):
    """
    thresholding the image.
    :param img:
    :param thd:
    :return:
    """
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


class Train_model_frontend_fjy(object):
    """
    # This is the base class for training classes. Wrap pytorch net to help training process.
    
    """

    default_config = {
        "train_iter": 170000,
        "save_interval": 2000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
    }

    def __init__(self, config, save_path=Path("."), device="cpu", verbose=False):
        """
        ## default dimension:
            heatmap: torch (batch_size, H, W, 1)
            dense_desc: torch (batch_size, H, W, 256)
            pts: [batch_size, np (N, 3)]
            desc: [batch_size, np(256, N)]
        
        :param config:
            dense_loss, sparse_loss (default)
            
        :param save_path:
        :param device:
        :param verbose:
        """
        # config
        print("Load Train_model_frontend!!")
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
        self.loss = 0
        self.correspond = 4
        self.h_expand = 0
        self.w_expand = 6

        self.max_iter = config["train_iter"]

        if self.config["model"]["dense_loss"]["enable"]:
            ## original superpoint paper uses dense loss
            print("use dense_loss!")
            from utils.utils import descriptor_loss

            self.desc_params = self.config["model"]["dense_loss"]["params"]
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            ## our sparse loss has similar performace, more efficient
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse

            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = "sparse"

        if self.config["model"]["subpixel"]["enable"]:
            ## deprecated: only for testing subpixel prediction
            self.subpixel = True

            def get_func(path, name):
                logging.info("=> from %s import %s", path, name)
                mod = __import__("{}".format(path), fromlist=[""])
                return getattr(mod, name)

            self.subpixel_loss_func = get_func(
                "utils.losses", self.config["model"]["subpixel"]["loss_func"]
            )

        # load model

        self.printImportantConfig()

        pass

    def printImportantConfig(self):
        """
        # print important configs
        :return:
        """
        print("=" * 10, " check!!! ", "=" * 10)

        print("learning_rate: ", self.config["model"]["learning_rate"])
        print("lambda_loss: ", self.config["model"]["lambda_loss"])
        print("detection_threshold: ", self.config["model"]["detection_threshold"])
        print("batch_size: ", self.config["model"]["batch_size"])

        print("=" * 10, " descriptor: ", self.desc_loss_type, "=" * 10)
        for item in list(self.desc_params):
            print(item, ": ", self.desc_params[item])

        print("=" * 32)
        pass

    def dataParallel(self):
        """
        put network and optimizer to multiple gpus
        :return:
        """
        # print("=== Let's use", torch.cuda.device_count(), "GPUs!")
        # self.net = nn.DataParallel(self.net, device_ids=[self.device.index], output_device=[self.device.index])

        # self.descripter_net = nn.DataParallel(self.descripter_net, device_ids=[self.device.index], output_device=[self.device.index])
        self.net = self.net.to(self.device)
        self.featurePts_net_s = self.featurePts_net_s.to(self.device)
        self.net_params = [self.net.parameters()]
        # self.net_params.append(self.detector_net.parameters())
        self.net_params.append(self.featurePts_net_s.parameters())

        self.optimizer = self.adamOptim(
            itertools.chain(*self.net_params), lr=self.config["model"]["learning_rate"]
        )
        pass

    def adamOptim(self, net, lr):
        """
        initiate adam optimizer
        :param net: network structure
        :param lr: learning rate
        :return:
        """
        print("adam optimizer")
        import torch.optim as optim

        optimizer = optim.Adam(itertools.chain(*self.net_params), lr=lr, betas=(0.9, 0.999))
        return optimizer

    def loadModel(self):
        """
        load model from name and params
        init or load optimizer
        :return:
        """
        model = self.config["model"]["name"]
        params = self.config["model"]["params"]
        print("model: ", model)
        net = modelLoader(model=model, **params).to(self.device)
        logging.info("=> setting adam solver")

        #load detector and descriptor
        from desc import HardNet, HardNet_small, HardNet_tiny, HardNet_fast, HardNet_Linear, HardNet_dense, HardNet_dense_Re
        from desc import HardNet_dense_test, HardNet_fast_ghost, HardNet_fast_subpixel, HardNet_fast_GNN, HardNet_fast_PCA
        from desc import HardNet_dense_rot, HardNet_dense_sym, HardNet_dense_ds3_us2, HardNet_dense_sym_ghost, HardNet_fast_cat
        from desc import UnSuperPointNet_small_8_theta_04_04_teacher, UnSuperPointNet_small_8_theta_04_04_student
        
        self.featurePts_net_s = UnSuperPointNet_small_8_theta_04_04_student().to(self.device)  #学生网络


        self.featurePts_net_t = UnSuperPointNet_small_8_theta_04_04_teacher().to(self.device)
        # pthpath = r'/home/wanghl/work/des_net/0411/desc_patch/logs/0517_6192_sampsz18_nocat_teacher/checkpoints/superPointNet_495600_desc.pth.tar'
        # pthpath = r'/home/fengjy/Work/Work/6193_self_supervised/unsuperpoint_pts_distillation_resize_160*48_v1_6_v1_1/logs/06_12/unsuperpoint_160_48_06_12/checkpoints/superPointNet_81300_checkpoint.pth.tar'
        pthpath = r'/home/fengjy/Work/Work/6193_train_9800/unsuperpoint_pts_resize_160*48_0821_v1_2/logs/08_22/unsuperpoint_160_48_08_22/checkpoints/superPointNet_435000_checkpoint.pth.tar'
        
        # pthpath = '/home/wanghl/work/des_net/0411/desc_patch/logs/0526_6192_netDes_rect32x16_128_teacher/checkpoints/superPointNet_500000_desc.pth.tar'
        # pthpath = '/home/wanghl/work/des_net/0411/desc_patch/logs/0523_6192_netDes_sampsz18_GCNV2/checkpoints/superPointNet_500000_desc.pth.tar'
        
        # checkpoint = torch.load(pthpath, map_location=lambda storage, loc: storage.cuda(5))    # model_state_dict
        checkpoint = torch.load(pthpath, map_location=lambda storage, loc: storage)
        
        self.featurePts_net_t.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.featurePts_net_t.eval()


        self.net_params = [net.parameters()]
        # self.net_params.append(self.detector_net.parameters())
        self.net_params.append(self.featurePts_net_s.parameters())
    
        optimizer = self.adamOptim(itertools.chain(*self.net_params), lr=self.config["model"]["learning_rate"])

        n_iter = 0
        ## new model or load pretrained
        if self.config["retrain"] == True:
            logging.info("New model")
            pass
        else:
            print("frontend error cant load model")
            path = self.config["pretrained"]
            mode = "" if path[-4:] == ".pth" else "full" # the suffix is '.pth' or 'tar.gz'
            logging.info("load pretrained model from: %s", path)
            net, optimizer, n_iter = pretrainedLoader(
                net, optimizer, n_iter, path, mode="", full_path=True
            )
            logging.info("successfully load pretrained model from: %s", path)

        def setIter(n_iter):
            if self.config["reset_iter"]:
                logging.info("reset iterations to 0")
                n_iter = 0
            return n_iter

        self.net = net
        self.optimizer = optimizer
        self.n_iter = setIter(n_iter)

       
       


        pass


    @property
    def writer(self):
        """
        # writer for tensorboard
        :return:
        """
        # print("get writer")
        return self._writer

    @writer.setter
    def writer(self, writer):
        print("set writer")
        self._writer = writer

    @property
    def train_loader(self):
        """
        loader for dataset, set from outside
        :return:
        """
        print("get dataloader")
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader):
        print("set train loader")
        self._train_loader = loader

    @property
    def val_loader(self):
        print("get dataloader")
        return self._val_loader

    @val_loader.setter
    def val_loader(self, loader):
        print("set train loader")
        self._val_loader = loader

    def train(self, **options):
        """
        # outer loop for training
        # control training and validation pace
        # stop when reaching max iterations
        :param options:
        :return:
        """
        # training info
        logging.info("n_iter: %d", self.n_iter)
        logging.info("max_iter: %d", self.max_iter)
        running_losses = []
        epoch = 0
        # Train one epoch
        from torch.optim import lr_scheduler
        # def lambda_rule(epoch):
        #     lr_l = 1.0 - max(0, epoch - self.n_epochs) / float(opt.n_epochs_decay + 1)
        #     return lr_l
        # self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, 5, gamma=0.8, last_epoch=-1)

        while self.n_iter < self.max_iter:
            print("epoch: ", epoch)
            epoch += 1
            start = timer()
            for i, sample_train in tqdm(enumerate(self.train_loader)):
                # train one sample
                loss_out = self.train_val_sample(sample_train, self.n_iter, True)
                self.n_iter += 1
                running_losses.append(loss_out)

                # run validation
                # if self._eval and self.n_iter % self.config["validation_interval"] == 0:
                #     logging.info("====== Validating...")
                    
                #     for j, sample_val in enumerate(self.val_loader):
                #         self.train_val_sample(sample_val, self.n_iter + j, False)
                #         if j > self.config.get("validation_size", 3):
                #             break
                # save model
                if self.n_iter % self.config["save_interval"] == 0:
                    logging.info(
                        "save model: every %d interval, current iteration: %d",
                        self.config["save_interval"],
                        self.n_iter,
                    )
                    self.saveModel()
                # ending condition
                if self.n_iter > self.max_iter:
                    # end training
                    logging.info("End training: %d", self.n_iter)
                    break
            end = timer()
            print("epoch consuming:" ,end - start)
            self.scheduler.step()

        pass

    def getLabels(self, labels_2D, cell_size, device="cpu"):
        """
        # transform 2D labels to 3D shape for training
        :param labels_2D:
        :param cell_size:
        :param device:
        :return:
        """
        labels3D_flattened = labels2Dto3D_flattened(
            labels_2D.to(device), cell_size=cell_size
        )
        labels3D_in_loss = labels3D_flattened
        return labels3D_in_loss

    def getMasks(self, mask_2D, cell_size, device="cpu"):
        """
        # 2D mask is constructed into 3D (Hc, Wc) space for training
        :param mask_2D:
            tensor [batch, 1, H, W]
        :param cell_size:
            8 (default)
        :param device:
        :return:
            flattened 3D mask for training
        """
        mask_3D = labels2Dto3D(
            mask_2D.to(device), cell_size=cell_size, add_dustbin=False
        ).float()
        mask_3D_flattened = torch.prod(mask_3D, 1)
        return mask_3D_flattened

    

    def get_loss(self, semi, labels3D_in_loss, mask_3D_flattened, device="cpu"):
        """
        ## deprecated: loss function
        :param semi:
        :param labels3D_in_loss:
        :param mask_3D_flattened:
        :param device:
        :return:
        """
        loss_func = nn.CrossEntropyLoss(reduce=False).to(device)
        # if self.config['data']['gaussian_label']['enable']:
        #     loss = loss_func_BCE(nn.functional.softmax(semi, dim=1), labels3D_in_loss)
        #     loss = (loss.sum(dim=1) * mask_3D_flattened).sum()
        # else:
        loss = loss_func(semi, labels3D_in_loss)
        loss = (loss * mask_3D_flattened).sum()
        loss = loss / (mask_3D_flattened.sum() + 1e-10)
        return loss


    def usp_loss(self, a_s, b_s, dis):
        # a_s: a的分数 score
        # b_s: b的分数 score
        # dis: 距离矩阵
        alpha1 = 4
        alpha3 = 1.2
        
        # alpha2 = alpha3 / 0.85 #期望特征点占比
        alpha2 = 2.0 * alpha3 #期望特征点占比
        # alpha2 = 0.0 * alpha3 #期望特征点占比
        
        alpha4 = 0.25
        bonus =  0

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


    def unsuperpoint_loss(self, semiA, semiB, H, W, H_AB, H_ATA, imgA, imgB, mask=None, descA=None, descB=None):

        # target = torch.zeros_like(input)

        # loss_func_Focal = FocalLoss().cuda()
        # loss_func_BCE = nn.BCELoss(reduction='none').cuda()
        # loss_func_L1 = torch.nn.L1Loss(reduction='none').cuda()
        # loss_func_L2 = torch.nn.MSELoss(reduction='none').cuda()


        correct_position_A = self.get_position(semiA[:,1:3,:,:], H, W, self.cell_size)  # 校准坐标值
        correct_position_B = self.get_position(semiB[:,1:3,:,:], H, W, self.cell_size)

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
            pnt_A = pnts_A[index] # pnt_A : 对应尺寸： 160*48
            pnt_B = pnts_B[index] # pnt_B : 对应尺寸： 160*48

            # pnt_A, pnt_B resize 回原来尺寸[120 * 36], 比例为 3/4
            pnt_A[:,1:] = pnt_A[:,1:] * 0.75
            pnt_B[:,1:] = pnt_B[:,1:] * 0.75

            ori_H, ori_W = 120,36
            if descA is not None:
                desc_A = descs_A[index]
                desc_B = descs_B[index]

            H_ATB = H_AB[index]@H_ATA[index]  # H_ATB 对应尺寸 ： 120 * 36
            # H_ATB = H_ATA[index]
            
          
            # warped_pnts = warp_points(pnt_A[:,1:].cpu(), homography_scaling_torch(H_ATB, H, W))  # 利用变换矩阵变换坐标点
            warped_pnts = warp_points(pnt_A[:,1:].cpu(), H_ATB)  # 利用变换矩阵变换坐标点, pts_AT --> pts_ATB
            warped_pnts, mask_points = filter_points(warped_pnts, torch.tensor([ori_W, ori_H]), return_mask=True)
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
                
                # uni_xy 是正则化， 用于保证每个 8*8 区域内的点对的均匀分布
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
        # imgA, imgB, trans_expand, imgAT, trans, trans_theta, imgAT_mask, imgB_mask = (
        #     sample["imgA"],
        #     sample["imgB"],
        #     sample["trans_expand"],
        #     sample["imgAT"],
        #     sample["trans"],
        #     sample["theta"],
        #     sample["imgAT_mask"],
        #     sample["imgB_mask"]
        # )
        
        # img: 对应尺寸 ： 160 * 48
        # H ： 对应尺寸 ： 120 * 36
        imgA, imgB, H_ATA, imgAT, H_AB, mask_2D = (
            sample["imgA"],
            sample["imgB"],
            sample["H_ATA"],
            sample["imgAT"],
            sample["H_AB"],
            sample["imgA_mask"],
        )        
        
        imgA_Ne, imgB_Ne, imgAT_Ne = (
            sample["imgA_Ne"],
            sample["imgB_Ne"],
            sample["imgAT_Ne"],
        )
                
        # variables
        batch_size, H, W = imgA.shape[0], imgA.shape[2], imgA.shape[3]
        self.batch_size = batch_size
        det_loss_type = self.config["model"]["detector_loss"]["loss_type"]

     
        # zero the parameter gradients
        self.optimizer.zero_grad()
        assert self.h_expand == 0
        # forward + backward + optimize
        if train:
            outsA = self.featurePts_net_s(imgA.to(self.device))
            outsB = self.featurePts_net_s(imgB.to(self.device))
            outsAT = self.featurePts_net_s(imgAT.to(self.device))
                       
    
        else:
            with torch.no_grad():
                outsA = self.featurePts_net_s(imgA.to(self.device))
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


        self.loss = loss_det

        loss = loss_det


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

    def saveModel(self):
        """
        # save checkpoint for resuming training
        :return:
        """
        # model_state_dict = self.net.module.state_dict()
        model_save = self.net
        model_state_dict = model_save.state_dict()
        save_checkpoint(
            self.save_path,
            {
                "n_iter": self.n_iter + 1,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            self.n_iter,
        )

        model_state_dict = self.descriptor_net.state_dict()
        save_checkpoint(
            self.save_path,
            {
                "n_iter": self.n_iter + 1,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            self.n_iter,
            filename='desc.pth.tar'
        )

        pass

    def add_single_image_to_tb(self, task, img_tensor, n_iter, name="img"):
        """
        # add image to tensorboard for visualization
        :param task:
        :param img_tensor:
        :param n_iter:
        :param name:
        :return:
        """
        if img_tensor.dim() == 4:
            for i in range(min(img_tensor.shape[0], 5)):
                self.writer.add_image(
                    task + "-" + name + "/%d" % i, img_tensor[i, :, :, :], n_iter
                )
        else:
            self.writer.add_image(task + "-" + name, img_tensor[:, :, :], n_iter)

    # tensorboard
    def addImg2tensorboard(
        self,
        img,
        labels_2D,
        semi,
        img_warp=None,
        labels_warp_2D=None,
        mask_warp_2D=None,
        semi_warp=None,
        mask_3D_flattened=None,
        task="train",
    ):
        """
        # deprecated: add images to tensorboard
        :param img:
        :param labels_2D:
        :param semi:
        :param img_warp:
        :param labels_warp_2D:
        :param mask_warp_2D:
        :param semi_warp:
        :param mask_3D_flattened:
        :param task:
        :return:
        """
        # print("add images to tensorboard")

        n_iter = self.n_iter
        semi_flat = flattenDetection(semi[0, :, :, :])
        semi_warp_flat = flattenDetection(semi_warp[0, :, :, :])

        thd = self.config["model"]["detection_threshold"]
        semi_thd = thd_img(semi_flat, thd=thd)
        semi_warp_thd = thd_img(semi_warp_flat, thd=thd)

        result_overlap = img_overlap(
            toNumpy(labels_2D[0, :, :, :]), toNumpy(semi_thd), toNumpy(img[0, :, :, :])
        )

        self.writer.add_image(
            task + "-detector_output_thd_overlay", result_overlap, n_iter
        )
        saveImg(
            result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, "test_0.png"
        )  # rgb to bgr * 255

        result_overlap = img_overlap(
            toNumpy(labels_warp_2D[0, :, :, :]),
            toNumpy(semi_warp_thd),
            toNumpy(img_warp[0, :, :, :]),
        )
        self.writer.add_image(
            task + "-warp_detector_output_thd_overlay", result_overlap, n_iter
        )
        saveImg(
            result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, "test_1.png"
        )  # rgb to bgr * 255

        mask_overlap = img_overlap(
            toNumpy(1 - mask_warp_2D[0, :, :, :]) / 2,
            np.zeros_like(toNumpy(img_warp[0, :, :, :])),
            toNumpy(img_warp[0, :, :, :]),
        )

        # writer.add_image(task + '_mask_valid_first_layer', mask_warp[0, :, :, :], n_iter)
        # writer.add_image(task + '_mask_valid_last_layer', mask_warp[-1, :, :, :], n_iter)
        ##### print to check
        # print("mask_2D shape: ", mask_warp_2D.shape)
        # print("mask_3D_flattened shape: ", mask_3D_flattened.shape)
        for i in range(self.batch_size):
            if i < 5:
                self.writer.add_image(
                    task + "-mask_warp_origin", mask_warp_2D[i, :, :, :], n_iter
                )
                self.writer.add_image(
                    task + "-mask_warp_3D_flattened", mask_3D_flattened[i, :, :], n_iter
                )
        # self.writer.add_image(task + '-mask_warp_origin-1', mask_warp_2D[1, :, :, :], n_iter)
        # self.writer.add_image(task + '-mask_warp_3D_flattened-1', mask_3D_flattened[1, :, :], n_iter)
        self.writer.add_image(task + "-mask_warp_overlay", mask_overlap, n_iter)

    def tb_scalar_dict(self, losses, task="training"):
        """
        # add scalar dictionary to tensorboard
        :param losses:
        :param task:
        :return:
        """
        for element in list(losses):
            self.writer.add_scalar(task + "-" + element, losses[element], self.n_iter)
            # print (task, '-', element, ": ", losses[element].item())

    def tb_images_dict(self, task, tb_imgs, max_img=5):
        """
        # add image dictionary to tensorboard
        :param task:
            str (train, val)
        :param tb_imgs:
        :param max_img:
            int - number of images
        :return:
        """
        for element in list(tb_imgs):
            for idx in range(tb_imgs[element].shape[0]):
                if idx >= max_img:
                    break
                # print(f"element: {element}")
                self.writer.add_image(
                    task + "-" + element + "/%d" % idx,
                    tb_imgs[element][idx, ...],
                    self.n_iter,
                )


    def tb_hist_dict(self, task, tb_dict):
        for element in list(tb_dict):
            self.writer.add_histogram(
                task + "-" + element, tb_dict[element], self.n_iter
            )
        pass

    def printLosses(self, losses, task="train"):
        """
        # print loss for tracking training
        :param losses:
        :param task:
        :return:
        """
        for element in list(losses):
            # print ('add to tb: ', element)
            try:
                print(task, "-", element, ": ", losses[element].item())
            except AttributeError:
                pass

    def add2tensorboard_nms(self, img, labels_2D, semi, task="train", batch_size=1):
        """
        # deprecated:
        :param img:
        :param labels_2D:
        :param semi:
        :param task:
        :param batch_size:
        :return:
        """
        from utils.utils import getPtsFromHeatmap
        from utils.utils import box_nms

        boxNms = False
        n_iter = self.n_iter

        nms_dist = self.config["model"]["nms"]
        conf_thresh = self.config["model"]["detection_threshold"]
        # print("nms_dist: ", nms_dist)
        precision_recall_list = []
        precision_recall_boxnms_list = []
        for idx in range(batch_size):
            semi_flat_tensor = flattenDetection(semi[idx, :, :, :]).detach()
            semi_flat = toNumpy(semi_flat_tensor)
            semi_thd = np.squeeze(semi_flat, 0)
            pts_nms = getPtsFromHeatmap(semi_thd, conf_thresh, nms_dist)
            semi_thd_nms_sample = np.zeros_like(semi_thd)
            semi_thd_nms_sample[
                pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)
            ] = 1

            label_sample = torch.squeeze(labels_2D[idx, :, :, :])
            # pts_nms = getPtsFromHeatmap(label_sample.numpy(), conf_thresh, nms_dist)
            # label_sample_rms_sample = np.zeros_like(label_sample.numpy())
            # label_sample_rms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1
            label_sample_nms_sample = label_sample

            if idx < 5:
                result_overlap = img_overlap(
                    np.expand_dims(label_sample_nms_sample, 0),
                    np.expand_dims(semi_thd_nms_sample, 0),
                    toNumpy(img[idx, :, :, :]),
                )
                self.writer.add_image(
                    task + "-detector_output_thd_overlay-NMS" + "/%d" % idx,
                    result_overlap,
                    n_iter,
                )
            assert semi_thd_nms_sample.shape == label_sample_nms_sample.size()
            precision_recall = precisionRecall_torch(
                torch.from_numpy(semi_thd_nms_sample), label_sample_nms_sample
            )
            precision_recall_list.append(precision_recall)

            if boxNms:
                semi_flat_tensor_nms = box_nms(
                    semi_flat_tensor.squeeze(), nms_dist, min_prob=conf_thresh
                ).cpu()
                semi_flat_tensor_nms = (semi_flat_tensor_nms >= conf_thresh).float()

                if idx < 5:
                    result_overlap = img_overlap(
                        np.expand_dims(label_sample_nms_sample, 0),
                        semi_flat_tensor_nms.numpy()[np.newaxis, :, :],
                        toNumpy(img[idx, :, :, :]),
                    )
                    self.writer.add_image(
                        task + "-detector_output_thd_overlay-boxNMS" + "/%d" % idx,
                        result_overlap,
                        n_iter,
                    )
                precision_recall_boxnms = precisionRecall_torch(
                    semi_flat_tensor_nms, label_sample_nms_sample
                )
                precision_recall_boxnms_list.append(precision_recall_boxnms)

        precision = np.mean(
            [
                precision_recall["precision"]
                for precision_recall in precision_recall_list
            ]
        )
        recall = np.mean(
            [precision_recall["recall"] for precision_recall in precision_recall_list]
        )
        self.writer.add_scalar(task + "-precision_nms", precision, n_iter)
        self.writer.add_scalar(task + "-recall_nms", recall, n_iter)
        print(
            "-- [%s-%d-fast NMS] precision: %.4f, recall: %.4f"
            % (task, n_iter, precision, recall)
        )
        if boxNms:
            precision = np.mean(
                [
                    precision_recall["precision"]
                    for precision_recall in precision_recall_boxnms_list
                ]
            )
            recall = np.mean(
                [
                    precision_recall["recall"]
                    for precision_recall in precision_recall_boxnms_list
                ]
            )
            self.writer.add_scalar(task + "-precision_boxnms", precision, n_iter)
            self.writer.add_scalar(task + "-recall_boxnms", recall, n_iter)
            print(
                "-- [%s-%d-boxNMS] precision: %.4f, recall: %.4f"
                % (task, n_iter, precision, recall)
            )

    def get_heatmap(self, semi, det_loss_type="softmax"):
        if det_loss_type == "l2":
            heatmap = self.flatten_64to1(semi)
        else:
            heatmap = flattenDetection(semi)
        return heatmap
    
    def get_heatmap_unsuperpoint(self, semi, block_size=8):
        heatmap = torch.zeros(semi.size(0), 1, semi.size(2)*block_size, semi.size(3)*block_size)
        for index_batch in range(semi.size(0)):
            for index_i in range(semi.size(2)):
                for index_j in range(semi.size(3)):
                    y_label = (index_i*block_size+semi[index_batch, 1,index_i,index_j]*block_size).round().long()
                    x_label = (index_j*block_size+semi[index_batch, 2,index_i,index_j]*block_size).round().long()
                    if (y_label < semi.size(2) * block_size) and  (x_label < semi.size(3) * block_size):
                        heatmap[index_batch, 0, y_label, x_label] = semi[index_batch, 0,index_i,index_j]
        return heatmap

    ######## static methods ########
    @staticmethod
    def input_to_imgDict(sample, tb_images_dict):
        # for e in list(sample):
        #     print("sample[e]", sample[e].shape)
        #     if (sample[e]).dim() == 4:
        #         tb_images_dict[e] = sample[e]
        for e in list(sample):
            element = sample[e]
            if type(element) is torch.Tensor:
                if element.dim() == 4:
                    tb_images_dict[e] = element
                # print("shape of ", i, " ", element.shape)
        return tb_images_dict

    @staticmethod
    def interpolate_to_dense(coarse_desc, cell_size=8):
        dense_desc = nn.functional.interpolate(
            coarse_desc, scale_factor=(cell_size, cell_size), mode="bilinear"
        )
        # norm the descriptor
        def norm_desc(desc):
            dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
            desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
            return desc

        dense_desc = norm_desc(dense_desc)
        return dense_desc


if __name__ == "__main__":
    # load config
    filename = "configs/superpoint_coco_test.yaml"
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

    train_agent = Train_model_frontend_fjy(config, device=device)

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
