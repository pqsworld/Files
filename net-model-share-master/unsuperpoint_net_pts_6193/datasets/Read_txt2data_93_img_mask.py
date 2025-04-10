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
IMAGE_ROOT = './datasets/6193'
IMAGE_MASK_ROOT = './datasets/6193'
IMAGE_EXPAND_ROOT = './datasets/6193'


# DATA_PATH = '/home/yey/work/superglue/datasets'
def homography_centorcrop(homography, Hdev_top, Wdev_left):
    #裁剪相当于平移，修正H矩阵
    '''
    裁剪: -
    扩边: +
    '''
    trans_homo = torch.tensor([[1, 0, Wdev_left], [0, 1, Hdev_top], [0, 0, 1]], dtype=torch.float32)
    homography = trans_homo@homography@trans_homo.inverse()
    return homography

def load_as_float_to_120_36(path, flag): 
    image_PIL = Image.open(path).convert('L')
    image_np = np.array(image_PIL)
    
    if flag == -1:
        image_np = 255 - image_np
    
    
    #resize可能会对描述子有影响
    # imgA = Image.fromarray(imgA).resize((128,128))
    # imgB = Image.fromarray(imgB).resize((128,128))
    h, w = image_np.shape

    if h == 122 and w == 36:
        image_np = image_np[1:-1,:]
    if h == 144 and w == 52:
        image_temp = np.zeros((128,52))
        image_temp = image_np[8:-8,:] 
        image_np = image_temp

    image_np = np.array(image_np).astype(np.float32) / 255
    return image_np

def load_as_float_to_120_40(path, flag): 
    image_PIL = Image.open(path).convert('L')
    image_np = np.array(image_PIL)
    
    if flag == -1:
        image_np = 255 - image_np
    
    
    #resize可能会对描述子有影响
    # imgA = Image.fromarray(imgA).resize((128,128))
    # imgB = Image.fromarray(imgB).resize((128,128))
    h, w = image_np.shape

    if h == 122 and w == 36:
        image_np = image_np[1:-1,:]
        image_temp = np.zeros((120,40))
        image_temp[:,2:-2] = image_np[:,:]
        image_np = image_temp 
        
    if h == 144 and w == 52:
        image_temp = np.zeros((128,52))
        image_temp = image_np[8:-8,:] 
        image_np = image_temp

    image_np = np.array(image_np).astype(np.float32) / 255
    return image_np

def load_as_float_to_122_36(path, flag): 
    image_PIL = Image.open(path).convert('L')
    image_np = np.array(image_PIL)
    
    if flag == -1:
        image_np = 255 - image_np
    
    
    #resize可能会对描述子有影响
    # imgA = Image.fromarray(imgA).resize((128,128))
    # imgB = Image.fromarray(imgB).resize((128,128))
    h, w = image_np.shape

    if h == 122 and w == 36:
        image_np = image_np 
        
    if h == 144 and w == 52:
        image_temp = np.zeros((128,52))
        image_temp = image_np[8:-8,:] 
        image_np = image_temp

    image_np = np.array(image_np).astype(np.float32) / 255
    return image_np


def resize_img_float(img, resize_h, resize_w):
    if img.shape[0] == 1:
        img = img[0]
        
    np.array(img)
    resize_img = cv2.resize(np.array(img), (resize_w, resize_h),interpolation=cv2.INTER_LINEAR)
    resize_img = torch.tensor(resize_img).unsqueeze(0)
    return resize_img

def resize_mask_float(img, resize_h, resize_w):
    if img.shape[0] == 1:
        img = img[0]
        
    np.array(img)
    resize_img = (cv2.resize(np.array(img), (resize_w, resize_h),interpolation=cv2.INTER_LINEAR) + 0.5).astype(np.int)    
    resize_img = torch.tensor(resize_img.astype(np.float)).unsqueeze(0)
    return resize_img

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

class Read_txt2data_93_img_mask(data.Dataset):
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
        
        datasets_93 = [
            "6193_DK7_normal_8mul_merge",
        ]
        
        # ------ matched images read ------
        sequence_set = []
        print("### Clean datesets: Please wait ####")

        for dataset_name in datasets_93:
            FA_flag = 0

            if "net" in dataset_name:
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
                # MATCH_PATH = os.path.join(IMAGE_ROOT,dataset_name + "/trans.txt")
                MATCH_PATH = os.path.join(IMAGE_ROOT,dataset_name + "/trans_info_choose.txt")
                
            IMAGE_PATH = os.path.join(IMAGE_ROOT,dataset_name)
            df = pd.read_csv(
                        MATCH_PATH,
                        header=None,
                        encoding = "gb2312",
                        names=['enroll', 'verify', 'path', 
                            'H0', 'H1', 'H2', 'H3', 'H4', 'H5',
                            'score', 'up', 'rotation', 'shear', 'sacle', 'translation', 'Rarea', 'Bsim']
                        )    # 'gb2312'
            image1_list = df['enroll'].to_list()
            image1_list = [i[8:].rjust(4, '0') for i in image1_list]    # 获得名称并补齐 '1'-> '001'
            if not FA_flag:
                image1_list = [i + '_extend.bmp' for i in image1_list]
            image2_list = df['path'].to_list()

            image2_list = [i[7:] for i in image2_list]
            H0 = df['H0'].to_list()
            H0 = [(int)(h[6:]) for h in H0]
            H1 = df['H1'].to_list()
            H2 = df['H2'].to_list()
            H3 = df['H3'].to_list()
            H4 = df['H4'].to_list()
            H5 = df['H5'].to_list()

            if FA_flag:
                for i in range(len(image2_list)):
                    path_share = '/'.join(image2_list[i].split('/')[:-3])
                    image1_list[i] = path_share + '/' + image1_list[i]
            else:
                for i in range(len(image2_list)):
                    path_share = '/'.join(image2_list[i].split('/')[:-1])
                    image1_list[i] = path_share + '/' + image1_list[i]

            trans = [[H0[i], H1[i], H2[i], H3[i], H4[i], H5[i]] for i in range(len(H0))]    # combined into list
            # trans = np.array(trans) # (N, 6)

            image1_name = ['_'.join(k.split('/')[-3:])[:-4] for k in image1_list]
            image2_name = ['_'.join(k.split('/')[-3:])[:-4] for k in image2_list]

            points_save_dir = os.path.join(IMAGE_ROOT,dataset_name + "/points")

            NEED_PNT = False
            '''生成pts npy'''
            if not os.path.exists(points_save_dir) and NEED_PNT:
                os.makedirs(points_save_dir)
                print("Generate Points:",points_save_dir)

                # ------ images points read ------
                point_path = os.path.join(IMAGE_ROOT,dataset_name + "/point.txt")
                coor = None
                name_list = []
                coor_list = []
                df_p = pd.read_csv(point_path, header=None, encoding="gb2312", sep='\t')    # 'gb2312'

                for line in zip(df_p[0]):  # line: [index, .]
                    if line[0][0:4] == '/hdd':
                        '''换成npy格式(DOG数据)'''
                        pts_name = "_".join(line[0].split('/')[-3:])[:-4]
                        name_list.append(pts_name)

                        #将每张图的点数据打包
                        if coor is not None:
                            coor_list.append(np.array(coor))
                            pts_save_path = os.path.join(points_save_dir, name_list[-2] + ".npy")
                            np.save(pts_save_path, np.array(coor))
                            coor = None

                    if line[0][0] == 'x':
                        ls = re.split('[,:]', line[0])
                        
                        #跳过错误数据
                        try:
                            coor_one = np.array([int(ls[1]),int(ls[3]),int(ls[5]) - 90])
                            if coor is None:
                                coor = coor_one
                            else:
                                coor = np.vstack((coor, coor_one))
                        except:
                            print(line[0])
                            pass

                #将每张图的点数据打包
                if coor is not None:
                    coor_list.append(np.array(coor))
                    pts_save_path = os.path.join(points_save_dir, name_list[-1] + ".npy")
                    np.save(pts_save_path, np.array(coor))
                    coor = []

            if NEED_PNT:
                ptsA = [os.path.join(points_save_dir, name + ".npy") for name in image1_name]
                ptsB = [os.path.join(points_save_dir, name + ".npy") for name in image2_name]
            else:
                ptsA = image1_name
                ptsB = image2_name

            # ------ dic ------
            files1 = {
                'imgA': image2_list,
                'imgB': image1_list,       # 模板
                'imgA_name': image2_name,
                'imgB_name': image1_name,
                'trans': trans,
                'ptsA': ptsA,
                'ptsB': ptsB
            }
        
            orientation_quality_poor_np = np.zeros(len(files1['imgA']))
            orientation_poor_npy_exist = 0
            try:
                orientation_quality_poor_np = np.load(os.path.join(IMAGE_PATH, "orientation_poor.npy"))
                
                #只读取1000对，仅供调试
                orientation_quality_poor_np[1000:] = 1
                
                orientation_poor_npy_exist = 1
            except:
                orientation_poor_npy_exist = 0
                pass
            
            count = 0
            for sample_num, (A, N_A, B, N_B, M_H, P_A, P_B) in enumerate(zip(
                files1['imgA'], files1['imgA_name'],
                files1['imgB'], files1['imgB_name'],files1['trans'],
                files1['ptsA'], files1['ptsB'])):   # zip返回元组

                #图库path需要修改
                A = A.split("/")[-3:]
                A = IMAGE_PATH + "/" + "/".join(A)
                B = B.split("/")[-3:]
                B = IMAGE_PATH + "/" + "/".join(B)
            
                # '''小位移'''
                # if (int(N_A[8:11]) >= 200) or (int(N_B[8:11]) >= 200):
                #     continue

                #trans处理，直接使用trans有问题，详情问qint
                M_H = np.array(M_H)
                M_H = torch.tensor(M_H.reshape(2, 3), dtype=torch.float32)
                M_H = 2 * M_H / 512.  # csv中的数据需要乘上2再除以512
                vec_one = torch.tensor([[0, 0, 1]], dtype=torch.float32)
                M_H = torch.cat((M_H, vec_one), 0)

                # M_H = homography_centorcrop(M_H, -4, 0)

                if orientation_poor_npy_exist:
                    if sample_num < orientation_quality_poor_np.shape[0]:
                        if orientation_quality_poor_np[sample_num]:  #跳过异常数据
                            continue
                else:
                    #清除异常数据
                    image_A_PIL = Image.open(A).convert("L")
                    image_A_np = np.array(image_A_PIL)
                    image_B_PIL = Image.open(B).convert("L")
                    image_B_np = np.array(image_B_PIL)
                    (image_H, image_W) = image_A_np.shape
                    cell_size = 4
                    x = torch.arange(image_W // cell_size + 1, requires_grad=False)
                    y = torch.arange(image_H // cell_size + 1, requires_grad=False)
                    y, x = torch.meshgrid([y, x])
                    pnt_A = torch.stack([x, y], dim=2) * cell_size
                    pnt_A = pnt_A.view(-1, 2)

                    warped_pnts = warp_points(pnt_A, M_H)  # 利用变换矩阵变换坐标点
                    warped_pnts, mask_points = filter_points(warped_pnts, torch.tensor([image_W, image_H]), return_mask=True)

                    if len(warped_pnts) < 5:
                        orientation_quality_poor_np[sample_num] = 1
                        continue
                    #利用方向一致性检查trans正确性，trans错误则忽略该样本
                    warped_img = inv_warp_image(torch.from_numpy(image_A_np).unsqueeze(0).unsqueeze(0), M_H.unsqueeze(0), mode="bilinear")
                
                    match_A_warp_orientation = get_orientation(warped_img.cpu().numpy(), warped_pnts)
                    match_B_orientation = get_orientation(image_B_np, warped_pnts)

                    match_orientation_mask = abs(match_A_warp_orientation - match_B_orientation) < 10 #获得方向一致的mask
                    orientation_quality_poor = np.mean(match_orientation_mask) < 0.75

                    if orientation_quality_poor: #方向一致性较差，图像应该有问题
                        orientation_quality_poor_np[sample_num] = 1
                        # b = np.zeros_like(image_B_np)
                        # g = warped_img.cpu().numpy().squeeze().astype(image_B_np.dtype)
                        # r = image_B_np
                        # image_merge = cv2.merge([b, g, r])
                        # saveImg(image_merge, os.path.join("/data/yey/work/temp",str(sample_num)+"_"+str(np.mean(match_orientation_mask))+".bmp"))
                        continue
                
                A_mask = IMAGE_MASK_ROOT + A[len(IMAGE_ROOT):]
                B_mask = IMAGE_MASK_ROOT + B[len(IMAGE_ROOT):]
                A_mask = A_mask.replace('_extend.bmp','__msk1.bmp')
                B_mask = B_mask.replace('_extend.bmp','__msk1.bmp')
                A = IMAGE_EXPAND_ROOT + A[len(IMAGE_ROOT):]
                B = IMAGE_EXPAND_ROOT + B[len(IMAGE_ROOT):]
                # A = A.replace('_enhance.bmp','_extend.bmp')
                # B = B.replace('_enhance.bmp','_extend.bmp')
                A = A.replace('_extend.bmp','__enhanceMergeOld.bmp')
                B = B.replace('_extend.bmp','__enhanceMergeOld.bmp')
                sample = {
                    'imgA': A,
                    'imgA_mask': A_mask,
                    'imgA_name': N_A, 
                    'imgB': B,
                    'imgB_mask': B_mask,
                    'imgB_name': N_B,
                    'trans': M_H,
                    'pntA': P_A,
                    'pntB': P_B,
                    'FA_flag': FA_flag
                    }   # 无标签时，只有img数据
                sequence_set.append(sample)
                count += 1
            print(dataset_name, count)
            #保留异常数据id，再次运行直接提出相关id数据
            if os.path.exists(os.path.join(IMAGE_PATH,"orientation_poor.npy")):
                pass
            else:
                np.save(os.path.join(IMAGE_PATH,"orientation_poor.npy"),orientation_quality_poor_np)
        print("CLean over!")

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

        def load_img_120_36(sample):
            
            #后面默认imgA为sample，imgB为temple，trans为imgA->imgB
            imgA = load_as_float_to_120_36(sample["imgA"], 1) # 增强图：图像 size 由 122*36 ---> 128 * 40
            imgB = load_as_float_to_120_36(sample["imgB"], 1) # 增强图：图像 size 由 122*36 ---> 128 * 40

            imgA_Ne = load_as_float_to_120_36(sample["imgA"], -1) # 增强图：图像 size 由 122*36 ---> 128 * 40
            imgB_Ne = load_as_float_to_120_36(sample["imgB"], -1) # 增强图：图像 size 由 122*36 ---> 128 * 40

            imgA_mask = load_as_float_to_120_36(sample["imgA_mask"], 1)
            imgB_mask = load_as_float_to_120_36(sample["imgB_mask"], 1)            
            
            return imgA, imgB, imgA_Ne, imgB_Ne,imgA_mask, imgB_mask
        
        def load_img_120_40(sample):
            
            #后面默认imgA为sample，imgB为temple，trans为imgA->imgB
            imgA = load_as_float_to_120_40(sample["imgA"], 1) # 增强图：图像 size 由 122*36 ---> 128 * 40
            imgB = load_as_float_to_120_40(sample["imgB"], 1) # 增强图：图像 size 由 122*36 ---> 128 * 40

            imgA_Ne = load_as_float_to_120_40(sample["imgA"], -1) # 增强图：图像 size 由 122*36 ---> 128 * 40
            imgB_Ne = load_as_float_to_120_40(sample["imgB"], -1) # 增强图：图像 size 由 122*36 ---> 128 * 40

            imgA_mask = load_as_float_to_120_40(sample["imgA_mask"], 1)
            imgB_mask = load_as_float_to_120_40(sample["imgB_mask"], 1)            
            
            return imgA, imgB, imgA_Ne, imgB_Ne,imgA_mask, imgB_mask
        
        def load_img_122_36(sample):
            
            #后面默认imgA为sample，imgB为temple，trans为imgA->imgB
            imgA = load_as_float_to_122_36(sample["imgA"], 1) # 增强图：图像 size 由 122*36 ---> 128 * 40
            imgB = load_as_float_to_122_36(sample["imgB"], 1) # 增强图：图像 size 由 122*36 ---> 128 * 40

            imgA_Ne = load_as_float_to_122_36(sample["imgA"], -1) # 增强图：图像 size 由 122*36 ---> 128 * 40
            imgB_Ne = load_as_float_to_122_36(sample["imgB"], -1) # 增强图：图像 size 由 122*36 ---> 128 * 40

            imgA_mask = load_as_float_to_122_36(sample["imgA_mask"], 1)
            imgB_mask = load_as_float_to_122_36(sample["imgB_mask"], 1)            
            
            return imgA, imgB, imgA_Ne, imgB_Ne,imgA_mask, imgB_mask        
        
        def random_transform_img_and_H(imgA, imgB, imgA_Ne, imgB_Ne,imgA_mask, imgB_mask, H):
            
            ABFLAG = random.choice([0,1]) #选择变换方向
            if ABFLAG:
                img_temp = imgA
                imgA = imgB
                imgB = img_temp
            
                img_temp = imgA_Ne
                imgA_Ne = imgB_Ne
                imgB_Ne = img_temp

                img_mask_temp = imgA_mask
                imgA_mask = imgB_mask
                imgB_mask = img_mask_temp

                H = np.linalg.inv(H)
            return imgA, imgB, imgA_Ne, imgB_Ne,imgA_mask, imgB_mask, H        

        def getImgName(img_pth):
            split = img_pth.rsplit('/', 6)
            str_1 = split[4]
            str_2 = split[5]
            str_3 = split[6][:4]
            
            name = str_1 + '_' + str_2 + '_' + str_3

            return name

        def resize_imgs(imgA, imgB, imgA_Ne, imgB_Ne, imgA_mask, imgB_mask, resize_h=160, resize_w=48):
            
            #后面默认imgA为sample，imgB为temple，trans为imgA->imgB
            imgA_resize = resize_img_float(imgA, resize_h, resize_w)
            imgB_resize = resize_img_float(imgB, resize_h, resize_w)
        
            imgA_Ne_resize = resize_img_float(imgA_Ne, resize_h, resize_w)
            imgB_Ne_resize = resize_img_float(imgB_Ne, resize_h, resize_w)
        
            imgA_mask_resize = resize_mask_float(imgA_mask, resize_h, resize_w)
            imgB_mask_resize = resize_mask_float(imgB_mask, resize_h, resize_w)        
            
            return imgA_resize, imgB_resize, imgA_Ne_resize, imgB_Ne_resize,imgA_mask_resize, imgB_mask_resize

        def resize_warp_imgs(img, img_ne, img_mask, resize_h=160, resize_w=48):
            
            #后面默认imgA为sample，imgB为temple，trans为imgA->imgB
            img_resize = resize_img_float(img, resize_h, resize_w)
            img_ne_resize = resize_img_float(img_ne, resize_h, resize_w)        
            img_mask_resize = resize_img_float(img_mask, resize_h, resize_w)
            
            return img_resize, img_ne_resize, img_mask_resize

        def get_homo(h ,w):
            from numpy.linalg import inv
            flip_flag = self.config["augmentation"]["homographic"]['params']["flip"]
            config_max_angle = self.config["augmentation"]["homographic"]['params']["max_angle"]
            homography = self.sample_homography(h, w, max_angle=config_max_angle, flip=flip_flag)
            homography = inv(homography)          
            
            return homography

        def get_warpImgs_120_36(homo, img, img_Ne, img_mask):
            img = torch.from_numpy(img)
            img_ne = torch.from_numpy(img_Ne)
            img_mask = torch.from_numpy(img_mask)
            warped_img = self.inv_warp_image(img.squeeze(), homo, mode='bilinear')
            warped_img_Ne = self.inv_warp_image(img_ne.squeeze(), homo, mode='bilinear')
            warped_img_mask = self.inv_warp_image(img_mask.squeeze(), homo, mode='bilinear')
            warped_img_mask = (warped_img_mask > 0.5).float()     
            
            return warped_img.numpy(), warped_img_Ne.numpy(), warped_img_mask.numpy()

        def get_warpImgs(homo, img, img_Ne, img_mask):
            img = torch.from_numpy(img)
            img_ne = torch.from_numpy(img_Ne)
            img_mask = torch.from_numpy(img_mask)
            warped_img = self.inv_warp_image(img.squeeze(), homo, mode='bilinear')
            warped_img_Ne = self.inv_warp_image(img_ne.squeeze(), homo, mode='bilinear')
            warped_img_mask = self.inv_warp_image(img_mask.squeeze(), homo, mode='bilinear')
            warped_img_mask = (warped_img_mask > 0.5).float()     
            
            return warped_img.numpy(), warped_img_Ne.numpy(), warped_img_mask.numpy()
                

        from datasets.data_tools import np_to_tensor
        from utils.utils import filter_points, homograghy_transform
        from utils.var_dim import squeezeToNumpy

        sample_one = self.samples[index]
        
        ################### 大模型由 120*36  ---->  160*48, 对应 trans 尺寸为 120 * 36
        #0. 获取 name
        imgA_name = getImgName(sample_one['imgA'])
        imgB_name = getImgName(sample_one['imgB'])
        
        #1. 加载图片
        imgA, imgB, imgA_Ne, imgB_Ne,imgA_mask, imgB_mask = load_img_120_36(sample_one)

        #2. 加载 trans( 加载trans 尺寸为 122*36，代码需要的 trans 尺寸为 120*36)
        trans_h, trans_w = 120, 36 
        label_H = np.array(sample_one['trans'])  # label_H 对应的图像 size: 122*36
        label_H = np.array(homograghy_transform(label_H, Hdev_top=-1, Wdev_left=0)) # 对应的图像 size: 120*36
        
        #3. 对加载图像进行随机转换
        # imgA, imgB, imgA_Ne, imgB_Ne,imgA_mask, imgB_mask, label_H = random_transform_img_and_H(imgA, imgB, imgA_Ne, imgB_Ne,imgA_mask, imgB_mask, label_H)
        
        #4. 对加载图像进行resize
        resize_h, resize_w = 160, 48
        imgA_resize, imgB_resize, imgA_Ne_resize, imgB_Ne_resize,imgA_mask_resize, imgB_mask_resize =  resize_imgs(imgA, imgB, imgA_Ne, imgB_Ne, imgA_mask, imgB_mask, resize_h, resize_w)

        #5. 将部分处理数据，放进容器中( A --> trans --> B)
        sample = {}
        sample["imgA_name"] = imgA_name
        sample["imgB_name"] = imgB_name        
        sample["imgA_t"] = imgA_resize
        sample["imgB_t"] = imgB_resize
        sample["imgA_Ne_t"] = imgA_Ne_resize
        sample["imgB_Ne_t"] = imgB_Ne_resize
        sample["imgA_mask_t"] = imgA_mask_resize
        sample["imgB_mask_t"] = imgB_mask_resize
        sample["FA_flag_t"] = sample_one["FA_flag"]
        sample["H_AB_t"] = label_H #对应尺寸120*36
      
        #6. 根据参数设置，计算一个 H 矩阵
        homography = get_homo(trans_h, trans_w)
      
        #7. 对各H矩阵进行处理，将其转为torch.tensor\
        homography = torch.tensor(homography).float() #120*36
        label_H = torch.tensor(label_H).float()       #120*36
        inv_homography = homography.inverse()
      
        #8. 根据 6中获取的 H ，计算 warp_img, warp_img_ne, warp_img_mask, 
        warped_imgA, warped_imgA_Ne, warped_imgA_mask = get_warpImgs_120_36(homography, imgA, imgA_Ne, imgA_mask)

        #9. 对warp_img 进行 resize
        warped_imgA_resize, warped_imgA_Ne_resize, warped_imgA_mask_resize = resize_warp_imgs(warped_imgA, warped_imgA_Ne, warped_imgA_mask, resize_h, resize_w)

        #9. 将 warp_img 放进容器中( A --> trans --> A_warp)
        sample["imgAT_t"] = warped_imgA_resize
        sample["imgAT_Ne_t"] = warped_imgA_Ne_resize
        sample["imgAT_mask_t"] = warped_imgA_mask_resize
        sample["H_AT_t"] = homography
        sample["H_ATA_t"] = inv_homography
        










        
        ################### 小模型由 122*36  ---->  122*36, 对应 trans 尺寸为 122 * 36
        #0. 获取 name
        imgA_name = getImgName(sample_one['imgA'])
        imgB_name = getImgName(sample_one['imgB'])
        
        #1. 加载图片
        imgA, imgB, imgA_Ne, imgB_Ne,imgA_mask, imgB_mask = load_img_122_36(sample_one)

        #2. 加载 trans( 加载trans 尺寸为 122*36，代码需要的 trans 尺寸为 128*40)
        trans_h, trans_w = 122, 36 
        label_H = np.array(sample_one['trans'])  # label_H 对应的图像 size: 122*36
        label_H = np.array(homograghy_transform(label_H, Hdev_top=0, Wdev_left=0)) # 对应的图像 size: 122*36
        
        #3. 对加载图像进行随机转换
        # imgA, imgB, imgA_Ne, imgB_Ne,imgA_mask, imgB_mask, label_H = random_transform_img_and_H(imgA, imgB, imgA_Ne, imgB_Ne,imgA_mask, imgB_mask, label_H)
        
        #4. 对加载图像进行resize
        resize_h, resize_w = 96, 28
        imgA_resize, imgB_resize, imgA_Ne_resize, imgB_Ne_resize,imgA_mask_resize, imgB_mask_resize =  resize_imgs(imgA, imgB, imgA_Ne, imgB_Ne, imgA_mask, imgB_mask, resize_h, resize_w)

        #5. 将部分处理数据，放进容器中( A --> trans --> B)
        sample["imgA_name"] = imgA_name
        sample["imgB_name"] = imgB_name        
        sample["imgA_s"] = imgA_resize
        sample["imgB_s"] = imgB_resize
        sample["imgA_Ne_s"] = imgA_Ne_resize
        sample["imgB_Ne_s"] = imgB_Ne_resize
        sample["imgA_mask_s"] = imgA_mask_resize
        sample["imgB_mask_s"] = imgB_mask_resize
        sample["FA_flag_s"] = sample_one["FA_flag"]
        sample["H_AB_s"] = label_H #对应尺寸120*36
      
        #6. 根据参数设置，计算一个 H 矩阵
        homography = get_homo(trans_h, trans_w)
      
        #7. 对各H矩阵进行处理，将其转为torch.tensor\
        homography = torch.tensor(homography).float() #120*36
        label_H = torch.tensor(label_H).float()       #120*36
        inv_homography = homography.inverse()
      
        #8. 根据 6中获取的 H ，计算 warp_img, warp_img_ne, warp_img_mask, 
        warped_imgA, warped_imgA_Ne, warped_imgA_mask = get_warpImgs(homography, imgA, imgA_Ne, imgA_mask) #122*36

        #9. 对warp_img 进行 resize
        warped_imgA_resize, warped_imgA_Ne_resize, warped_imgA_mask_resize = resize_warp_imgs(warped_imgA, warped_imgA_Ne, warped_imgA_mask, resize_h, resize_w)

        #9. 将 warp_img 放进容器中( A --> trans --> A_warp)
        sample["imgAT_s"] = warped_imgA_resize
        sample["imgAT_Ne_s"] = warped_imgA_Ne_resize
        sample["imgAT_mask_s"] = warped_imgA_mask_resize
        sample["H_AT_s"] = homography
        sample["H_ATA_s"] = inv_homography        
        
        return sample

    def __len__(self):
        return len(self.samples)

