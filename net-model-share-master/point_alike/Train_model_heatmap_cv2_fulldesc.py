"""This is the main training interface using heatmap trick
Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

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

from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened
from utils.utils import getPtsFromHeatmap, filter_points, getPtsFromLabels2D
# from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch_soft
from utils.utils import compute_valid_mask
from utils import html
from utils.draw import draw_keypoints_pair_train
from utils.utils import saveImg
# from utils.utils import save_checkpoint

from pathlib import Path
from Train_model_frontend import Train_model_frontend


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

class Train_model_heatmap_cv2_fulldesc(Train_model_frontend):
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

        self.max_iter = config["train_iter"]
        self.conf_thresh    = config['model']['detection_threshold']
        self.nms_dist       = config['model']['nms']
        self.correspond = 4

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

    def get_point_pair(self, dis, dis_thre=3):  # 获得匹配点
        a2b_min_id = torch.argmin(dis, dim=1)
        len_p = len(a2b_min_id)
        ch = dis[list(range(len_p)), a2b_min_id] < dis_thre

        idx_x = a2b_min_id[ch]
        dis_pair = dis[ch, a2b_min_id[ch]]
        
        return dis_pair

        a2b_min_id = torch.argmin(dis, dim=1)
        len_p = len(a2b_min_id)
        ch = dis[list(range(len_p)), a2b_min_id] < self.correspond
        
        reshape_as = a_s
        reshape_bs = b_s

        a_s = reshape_as[ch]
        b_s = reshape_bs[a2b_min_id[ch]]
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
        c = 2
        eps = 1e-12
        x = torch.unsqueeze(p_a[:, 0], 1) - torch.unsqueeze(p_b[:, 0], 0)  # N 2 -> NA 1 - 1 NB -> NA NB
        y = torch.unsqueeze(p_a[:, 1], 1) - torch.unsqueeze(p_b[:, 1], 0)
        dis = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + eps)
        return dis

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
        imgA, labelA, valid_mask, imgB, labelB, valid_mask_H, name, homography = (
            sample['image'], 
            sample['labels'].to(self.device),
            sample['valid_mask'].to(self.device),
            sample['image_H'],
            sample['labels_H'].to(self.device),
            sample['valid_mask_H'].to(self.device),
            sample['name'],
            sample['homography']
            )
        desc_labelA, desc_labelB = sample['desc'].to(self.device), sample['warped_desc'].to(self.device)
        batch_size, H, W = imgA.shape[0], imgA.shape[2], imgA.shape[3]
        self.batch_size = batch_size
        
        # train 
        self.optimizer.zero_grad()
        outA = self.net(imgA.to(self.device))
        semiA, coarse_descA = outA["semi"], outA["desc"]
        outB = self.net(imgB.to(self.device))
        semiB, coarse_descB = outB["semi"], outB["desc"]
        
        # ===  ===
        # ---detecA---
        labelA_3D = labels2Dto3D(
            labelA.to(self.device), cell_size=self.cell_size, add_dustbin=add_dustbin
        ).float()
        mask_3D_flattened = self.getMasks(valid_mask, self.cell_size, device=self.device)

        loss_det_A = self.detector_loss(
            input=semiA,
            target=labelA_3D.to(self.device),
            mask=mask_3D_flattened,
            loss_type=det_loss_type,
        )
        
        # ---detecB---
        labelB_3D = labels2Dto3D(
            labelB.to(self.device), cell_size=self.cell_size, add_dustbin=add_dustbin
        ).float()
        mask_3D_flattened = self.getMasks(valid_mask_H, self.cell_size, device=self.device)

        loss_det_B = self.detector_loss(
            input=semiB,
            target=labelB_3D.to(self.device),
            mask=mask_3D_flattened,
            loss_type=det_loss_type,
        )
        # ===detecAB===
        # loss_supervised = self.detector_selfsupervised_loss(semiA, semiB, mask_3D_flattened, mask_3D_flattened, mat_H)

        loss_det_A = 1 * loss_det_A
        loss_det_B = 1 * loss_det_B
        # ===descriptor loss===
        mask_desc = mask_3D_flattened.unsqueeze(1)
        lambda_loss = self.config["model"]["lambda_loss"]

        if lambda_loss > 0:
            from utils.utils import descriptor_loss_fix
            loss_desc, loss_descA, loss_descB = self.descriptor_loss(
                desc_labelA,
                desc_labelB,
                coarse_descA,
                coarse_descB,
                labelA,
                labelB,
                device=self.device
            )
        else:
            ze = torch.tensor([0]).to(self.device)
            loss_desc, positive_dist, negative_dist = ze, ze, ze
        loss_desc *= lambda_loss
        
        loss = loss_det_A + loss_det_B + loss_desc
        
        self.loss = loss
        self.scalar_dict.update(
            {
                "loss": loss,
                "loss_detA": loss_det_A,
                "loss_detB": loss_det_B,
                # "loss_res": loss_res,
                "loss_desc_total": loss_desc,
                "loss_descA": loss_descA,
                "loss_descB": loss_descB,
                # "positive_dist": positive_dist,
                # "negative_dist": negative_dist,
                # "w1": torch.tensor(w[0], dtype=torch.float32).detach(),
                # "w2": torch.tensor(w[1], dtype=torch.float32).detach()
            }
        )
        self.input_to_imgDict(sample, self.images_dict)

        if train:
            loss.backward()             
            self.optimizer.step()       # 更新网络参数
        
        if n_iter % tb_interval == 0 or task == "val":
            logging.info(
                "current iteration: %d, tensorboard_interval: %d", n_iter, tb_interval
            )
            logging.debug(
                "current iteration: %d, tensorboard_interval: %d", n_iter, tb_interval
            )

            # === get A's heatmap & pts ===
            heatmapA = flattenDetection(semiA, tensor=True)
            pts_A = getPtsFromHeatmap(toNumpy(heatmapA)[0, 0, :, :], self.conf_thresh, self.nms_dist)  # only touch one
            pts_A = pts_A.transpose()[:, [0, 1]]
            pts_A_lable = getPtsFromLabels2D(toNumpy(labelA[0, 0, :, :])).transpose(1, 0)   # (3, N)
            # === get B's heatmap & pts ===
            heatmapB = flattenDetection(semiB, tensor=True)
            heatmapB = heatmapB * valid_mask_H
            pts_B = getPtsFromHeatmap(toNumpy(heatmapB)[0, 0, :, :], self.conf_thresh, self.nms_dist)  # only touch one
            pts_B = pts_B.transpose()[:, [0, 1]]
            pts_B_lable = getPtsFromLabels2D(toNumpy(labelB[0, 0, :, :])).transpose(1, 0)

            pred, img_pair = {}, {}
            pred.update({
                "pts": pts_A, 
                "pts_H": pts_B,
                "lab": pts_A_lable,
                "lab_H": pts_B_lable
                })
            img_pair.update({
                "img": imgA[0].cpu().numpy().squeeze(),
                "img_H": imgB[0].cpu().numpy().squeeze()
                })
            img_pts = draw_keypoints_pair_train(img_pair, pred, None, radius=1, s=1)
            f = Path(self.webdir) / (str(n_iter) + '_' + str(name[0]) + ".bmp")
            saveImg(img_pts, str(f))

            self.printLosses(self.scalar_dict, task)
            self.logger.debug("current iteration: %d", n_iter)
            self.logger.debug("loss: %f, loss_detA: %f", loss, loss_det_A)
            # self.tb_images_dict(task, self.images_dict, max_img=2)
            # self.tb_hist_dict(task, self.hist_dict)

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
