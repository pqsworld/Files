"""
Adapted from https://github.com/rpautrat/SuperPoint/blob/master/superpoint/datasets/synthetic_dataset.py

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import torch.utils.data as data
import torch
import numpy as np
from imageio import imread

# from os import path as Path
import tensorflow as tf
from pathlib import Path
import tarfile

# import os
import random
import logging

from utils.tools import dict_update


# from models.homographies import sample_homography

from tqdm import tqdm
import cv2
import shutil

# DATA_PATH = '.'
import multiprocessing
import os
import random
from PIL import Image
import pandas as pd
import re
from utils.utils import filter_points,warp_points, inv_warp_image, saveImg
from skimage.transform import AffineTransform

'''
91模拟93，将144*52裁成128*52
'''
IMAGE_ROOT = '/hdd/file-input/guest/descshare/dataset/92/DesNet_Train/DesNet_Train'
IMAGE_MASK_ROOT = '/hdd/file-input/guest/descshare/dataset/92/DesNet_Train/DesNet_Train'
IMAGE_EXPAND_ROOT = '/hdd/file-input/guest/descshare/dataset/92/DesNet_Train/DesNet_Train'

# DATA_PATH = '/home/yey/work/superglue/datasets'
def homography_centorcrop(homography, Hdev_top, Wdev_left):
    #裁剪相当于平移，修正H矩阵
    trans_homo = torch.tensor([[1, 0, Wdev_left], [0, 1, Hdev_top], [0, 0, 1]], dtype=torch.float32)
    homography = trans_homo@homography@trans_homo.inverse()
    return homography

def load_as_float(path): 
    image_PIL = Image.open(path).convert('L')
    image_np = np.array(image_PIL)
    #resize可能会对描述子有影响
    # imgA = Image.fromarray(imgA).resize((128,128))
    # imgB = Image.fromarray(imgB).resize((128,128))
    h, w = image_np.shape

    if h == 180 and w == 36:
        image_temp = np.zeros((176,40)) #长宽均控制位8的倍数
        image_temp[:,2:-2] = image_np[1:-1,:]
        image_np = image_temp


    image_np = np.array(image_np).astype(np.float32) / 255
    return image_np


def get_orientation(img, keypoints, patch_size=8):
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

class Read_txt2data_92_mask_expand_clean(data.Dataset):
    """
    """

    default_config = {
        "primitives": "all",
        "truncate": {},
        "validation_size": -1,
        "test_size": -1,
        "on-the-fly": False,
        "cache_in_memory": False,
        "suffix": None,
        "add_augmentation_to_test_set": False,
        "num_parallel_calls": 10,
        "generation": {
            "split_sizes": {"train": 100, "val": 2, "test": 5},
            "image_size": [960, 1280],
            "random_seed": 0,
            "params": {
                "generate_background": {
                    "min_kernel_size": 150,
                    "max_kernel_size": 500,
                    "min_rad_ratio": 0.02,
                    "max_rad_ratio": 0.031,
                },
                "draw_stripes": {"transform_params": (0.1, 0.1)},
                "draw_multiple_polygons": {"kernel_boundaries": (50, 100)},
            },
        },
        "preprocessing": {"resize": [240, 320], "blur_size": 11,},
        "augmentation": {
            "photometric": {
                "enable": False,
                "primitives": "all",
                "params": {},
                "random_order": True,
            },
            "homographic": {"enable": False, "params": {}, "valid_border_margin": 0,},
        },
    }


    def __init__(
        self,
        seed=None,
        task="train",
        sequence_length=3,
        transform=None,
        target_transform=None,
        getPts=False,
        warp_input=False,
        **config,
    ):
        from utils.homographies import sample_homography_cv as sample_homography
        from utils.photometric import ImgAugTransform, customizedTransform
        from utils.utils import compute_valid_mask
        from utils.utils import inv_warp_image, warp_points

        torch.set_default_tensor_type(torch.FloatTensor)
        np.random.seed(seed)
        random.seed(seed)

        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, dict(config))

        self.transform = transform
        self.sample_homography = sample_homography
        self.compute_valid_mask = compute_valid_mask
        self.inv_warp_image = inv_warp_image
        self.warp_points = warp_points
        self.ImgAugTransform = ImgAugTransform
        self.customizedTransform = customizedTransform

        ######
        self.enable_photo_train = self.config["augmentation"]["photometric"]["enable"]
        self.enable_homo_train = self.config["augmentation"]["homographic"]["enable"]
        self.enable_homo_val = False
        self.enable_photo_val = False
        ######

        self.action = "train" if task == "train" else "val"
        # self.warp_input = warp_input

        self.getPts = getPts
        self.pool = multiprocessing.Pool(6)
        
        datasets_91 = [
            # "6192_DK7_140_mul8_normal_p20s200_test",
            # "6192_DK7_140_mul8_partpress_p20s200_test",
            # "6192_DK7_140_mul8_rot_p20s200_test",
            # "6192_DK7_140_mul8_sanfeng_p20s200_test",
            # "6192_DK7_140_mul8_wet_p20s200_test",
            # "6192_normal140_mul4_p30s199",
            # "6192_partialpress140_mul4_p23s200",
            # '6192_rot140_mul4_p26s200',
            # '6192_wet140_mul4_p11s199',
            'FA_6192_normal',
            'FA_6192_partial',
            'FA_6192_rot'
        ]
        # ------ matched images read ------
        sequence_set = []
        print("### Clean datesets: Please wait ####")

        for dataset_name in datasets_91:
            FA_flag = 0

            if "FA" in dataset_name:
                _net_path = os.path.join(IMAGE_ROOT,dataset_name)
                for parent, _, filenames in os.walk(_net_path,  followlinks=True):
                    filenames = sorted(filenames)
                    for filename in filenames:
                        file_path = os.path.join(parent, filename)
                        if file_path.endswith("trans.txt"):
                            file_path_split = file_path.split('/')
                            dataset_name = file_path_split[-2]
                            MATCH_PATH = file_path
                            if "FA" in file_path_split[-3]:
                                FA_flag = 1
            else:
                MATCH_PATH = os.path.join(IMAGE_ROOT,dataset_name + "/trans.txt")
                
            IMAGE_PATH = os.path.join(IMAGE_ROOT,dataset_name)
            with open(MATCH_PATH,"r") as f:   
                data_all = f.readlines()

            trans_all = []
            sample_count = 0
            for idx_line, item in enumerate(data_all):
                item_split = item.split(':')
                assert len(item_split) == 6
                trans_item = item_split[1]
                enroll_item = item_split[3]
                verify_item = item_split[5]

                trans_item = ",".join(trans_item.split(",")[:6])
                enroll_item = enroll_item.split('\\')[-1][:-7]
                enroll_item = "/".join(enroll_item.split('/')[1:])
                verify_item = verify_item.split('\\')[-1][:-2] #FA -2,FR -1

                enroll_order = int(enroll_item.split('/')[-1][:-4])
                verify_order = int(verify_item.split('/')[-1][:-4])

                #对录入即解锁order进行限制，减少匹配对
                if enroll_order < 55 and verify_order >= 55:
                    one_trans_prefix = "ENROLL: {0}, verify: {1}, path: {2},Trans:".format(enroll_item, int(verify_order), verify_item)
                    one_trans_suffix = ",score={0},up={1}".format(0,1)
                    one_trans = one_trans_prefix + trans_item + one_trans_suffix
                    if sample_count % 2 == 0: #每3个存一下
                        trans_all.append(one_trans)
                    sample_count += 1
                else:
                    pass
            print(dataset_name,len(trans_all))
            trans_save_path = '/'.join(MATCH_PATH.split('/')[:-1]) +  "/trans_choose.txt"
            np.savetxt(trans_save_path,trans_all,fmt='%s')

        self.samples = sequence_set
        print("samples: ", len(self.samples))

    def __getitem__(self, index):
        """
        :param index:
        :return:
            labels_2D: tensor(1, H, W)
            image: tensor(1, H, W)
        """

        def checkSat(img, name=""):
            if img.max() > 1:
                print(name, img.max())
            elif img.min() < 0:
                print(name, img.min())

        def imgPhotometric(img):
            """

            :param img:
                numpy (H, W)
            :return:
            """
            augmentation = self.ImgAugTransform(**self.config["augmentation"])
            img = img[:, :, np.newaxis]
            img = augmentation(img)
            cusAug = self.customizedTransform()
            img = cusAug(img, **self.config["augmentation"])
            return img


        from datasets.data_tools import np_to_tensor
        from utils.utils import filter_points
        from utils.var_dim import squeezeToNumpy

        sample_one = self.samples[index]
        #后面默认imgA为sample，imgB为temple，trans为imgA->imgB
        imgA = load_as_float(sample_one["imgA"])
        imgB = load_as_float(sample_one["imgB"])
        # ptsA = np.load(sample_one["pntA"])
        # ptsA = torch.from_numpy(ptsA)[:,:2]
        # ptsB = np.load(sample_one["pntB"])
        # ptsB = torch.from_numpy(ptsB)[:,:2]

        imgA_mask = load_as_float(sample_one["imgA_mask"])
        imgB_mask = load_as_float(sample_one["imgB_mask"])

        border_mask = np.zeros_like(imgA_mask)
        border_mask[1:-1,5:-5] = 1 #图像176*40 ，实际174*30

        imgA_mask *= border_mask
        imgB_mask *= border_mask

        #trans处理，直接使用trans有问题，详情问qint
        label_H = np.array(sample_one['trans'])

        H, W = imgA.shape[0], imgA.shape[1]
        self.H = H
        self.W = W
      
        sample = {}

        ABFLAG = random.choice([0,1]) #选择变换方向
        if ABFLAG:
            img_temp = imgA
            imgA = imgB
            imgB = img_temp

            img_mask_temp = imgA_mask
            imgA_mask = imgB_mask
            imgB_mask = img_mask_temp

            label_H = np.linalg.inv(label_H)


        imgA_ori = imgA
        imgA_mask_ori = imgA_mask
        if self.transform is not None:
            imgA_ori = self.transform(imgA_ori)
            imgB = self.transform(imgB)
            imgA_mask_ori = self.transform(imgA_mask_ori)
            imgB_mask = self.transform(imgB_mask)

        sample["imgA"] = imgA_ori
        sample["imgB"] = imgB
        sample["imgA_mask"] = imgA_mask_ori
        sample["imgB_mask"] = imgB_mask
        sample["FA_flag"] = sample_one["FA_flag"]
       

        # assert Hc == round(Hc) and Wc == round(Wc), "Input image size not fit in the block size"
        if (
            self.config["augmentation"]["photometric"]["enable_train"]
            and self.action == "train"
        ) or (
            self.config["augmentation"]["photometric"]["enable_val"]
            and self.action == "val"
        ):
            imgA = imgPhotometric(imgA)
        else:
            # print('>>> Photometric aug disabled for %s.'%self.action)
            pass

        from utils.utils import homography_scaling_torch as homography_scaling
        from numpy.linalg import inv

        flip_flag = self.config["augmentation"]["homographic"]['params']["flip"]
        config_max_angle = self.config["augmentation"]["homographic"]['params']["max_angle"]
        homography = self.sample_homography(H, W, max_angle=config_max_angle, flip=flip_flag)
     
        # homography = self.sample_homography(  # 得到变换矩阵
        #     np.array([2, 2]),  # 这个矩阵是干嘛的？
        #     shift=-1,
        #     **self.config["augmentation"]["homographic"]["params"],
        # )

        ##### use inverse from the sample homography
        homography = inv(homography)
        ######

        homography = torch.tensor(homography).float()
       
        label_H = torch.tensor(label_H).float()

        inv_homography = homography.inverse()
        imgA = torch.from_numpy(imgA)
        warped_img = self.inv_warp_image(imgA.squeeze(), homography, mode='bilinear').unsqueeze(0)
        imgA_mask = torch.from_numpy(imgA_mask)
        warped_img_mask = self.inv_warp_image(imgA_mask.squeeze(), homography_centorcrop(homography, -5, -6), mode='bilinear').unsqueeze(0)
        warped_img_mask = (warped_img_mask > 0.5).float()
        # warped_img = self.inv_warp_image(  # 利用变换矩阵变换图像
        #     imgA.squeeze(), inv_homography, mode="bilinear"
        # )
        # warped_img = warped_img.squeeze().numpy()
        # warped_img = warped_img[:, :, np.newaxis]

        
        # warped_pnts = self.warp_points(pnts, homography_scaling(homography, H, W))  # 利用变换矩阵变换坐标点
        # warped_pnts = filter_points(warped_pnts, torch.tensor([W, H]))
        
        # sample = {'image': warped_img, 'labels_2D': warped_labels}
        sample["imgAT"] = warped_img
        sample["imgAT_mask"] = warped_img_mask

        sample.update({"valid_mask": warped_img_mask})
        
        #裁剪后需要修正变换矩阵
        # label_H = homography_centorcrop(label_H, 3, 8) #(128 - 122)/2, (52 - 36)/2
        label_H = homography_centorcrop(label_H, 6, 11) #(186 - 174)/2, (52 - 30)/2


        #计算trans的旋转角度
        trans_expand = label_H@inv_homography
        trans_obj = AffineTransform(matrix=trans_expand)
        theta = trans_obj.rotation * 180 / 3.1415926 # [-180, 180]

        #
        imgATB = self.inv_warp_image(sample["imgAT"].squeeze(), trans_expand, mode='bilinear').unsqueeze(0)
        sample.update({"imgATB": imgATB})

        #裁剪后需要修正变换矩阵
        # trans = homography_centorcrop(trans_expand, 0, -6) #(128 - 128)/2, (40 - 52)/2
        trans = homography_centorcrop(trans_expand, -5, -6) #(176 - 186)/2, (40 - 52)/2


        sample.update({"trans_expand": trans_expand}) # A=homography@AT B=label_H@A B=label_H@homography@AT
        sample.update({"trans": trans}) # A=homography@AT B=label_H@A B=label_H@homography@AT
        sample.update({"theta": theta})
        # sample.update({"pntA": ptsA})
        # sample.update({"pntB": ptsB})
        return sample

    def __len__(self):
        return len(self.samples)

