"""
Adapted from https://github.com/rpautrat/SuperPoint/blob/master/superpoint/datasets/synthetic_dataset.py

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import os
import re
from unicodedata import name
from matplotlib.pyplot import axis
from scipy.misc.doccer import indentcount_lines
import math
from skimage.transform import AffineTransform
from torch._C import Size
import torch.utils.data as data
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from imageio import imread

# from os import path as Path
import tensorflow as tf
from pathlib import Path
import tarfile

# import os
import random
import logging
from Train_model_frontend import toNumpy
from utils.homographies import sample_homography_cv
# from utils.draw import draw_keypoints_pair_test

from utils.tools import dict_update

from datasets import synthetic_dataset

# from models.homographies import sample_homography

from tqdm import tqdm
import cv2
import shutil
from settings import DEBUG as debug
from settings import DATA_PATH
from settings import SYN_TMPDIR

# DATA_PATH = '.'
import multiprocessing

TMPDIR = SYN_TMPDIR  # './datasets/' # you can define your tmp dir


def load_as_float(path):
    return imread(path).astype(np.float32) / 255

quan = lambda x: x.round().long() 

class SyntheticDataset_gaussian_cv2_enhance(data.Dataset):
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
            "split_sizes": {"train": 5000, "val": 1000, "test": 0},
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

    # debug = True

    """
    def dump_primitive_data(self, primitive, tar_path, config):
        pass
    """

    def dump_primitive_data(self, primitive, tar_path, config):
        # temp_dir = Path(os.environ['TMPDIR'], primitive)
        temp_dir = Path(TMPDIR, primitive)

        tf.compat.v1.logging.info("Generating tarfile for primitive {}.".format(primitive))
        synthetic_dataset.set_random_state(
            np.random.RandomState(config["generation"]["random_seed"])
        )
        for split, size in self.config["generation"]["split_sizes"].items():
            im_dir, pts_dir = [Path(temp_dir, i, split) for i in ["images", "points"]]
            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)  # points dir

            for i in tqdm(range(size), desc=split, leave=False):
                image = synthetic_dataset.generate_background(
                    config["generation"]["image_size"],
                    **config["generation"]["params"]["generate_background"],
                )

                points = np.array(
                    getattr(synthetic_dataset, primitive)(
                        image, **config["generation"]["params"].get(primitive, {})  # **分配字典参数，getattr就是获取synthetic_dataset中的draw_XXX函数
                    )
                )

                points = np.flip(points, 1)  # reverse convention with opencv

                b = config["preprocessing"]["blur_size"]
                image = cv2.GaussianBlur(image, (b, b), 0)
                points = (
                    points
                    * np.array(config["preprocessing"]["resize"], np.float)
                    / np.array(config["generation"]["image_size"], np.float)
                )
                image = cv2.resize(
                    image,
                    tuple(config["preprocessing"]["resize"][::-1]),
                    interpolation=cv2.INTER_LINEAR,
                )

                cv2.imwrite(str(Path(im_dir, "{}.bmp".format(i))), image)
                np.save(Path(pts_dir, "{}.npy".format(i)), points)

        # Pack into a tar file
        tar = tarfile.open(tar_path, mode="w:gz")
        tar.add(temp_dir, arcname=primitive)
        tar.close()
        shutil.rmtree(temp_dir)
        tf.compat.v1.logging.info("Tarfile dumped to {}.".format(tar_path))

    def parse_primitives(self, names, all_primitives):
        p = (
            all_primitives
            if (names == "all")
            else (names if isinstance(names, list) else [names])
        )
        assert set(p) <= set(all_primitives)
        return p

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

        self.double_descriptor = True

        self.action = "train" if task == "train" else "val"
        # self.warp_input = warp_input

        self.cell_size = 8
        self.getPts = getPts
        if self.config['preprocessing']['resize']:
            self.sizer = self.config['preprocessing']['resize']

        self.gaussian_label = False
        if self.config["gaussian_label"]["enable"]:
            # self.params_transform = {'crop_size_y': 120, 'crop_size_x': 160, 'stride': 1, 'sigma': self.config['gaussian_label']['sigma']}
            self.gaussian_label = True

        self.pool = multiprocessing.Pool(6)

        basepath = Path(
            DATA_PATH,
            "pre_finger"
            + ("_{}".format(config["suffix"]) if config["suffix"] is not None else ""),
        )
        basepath.mkdir(parents=True, exist_ok=True)

        # splits = {s: {"images": [], "points": [], "names": []} for s in [self.action]}       # self.action = 'train'
        splits = {s: {"images": [], "points": [], "names": [], "desc": [], "trans": []} for s in [self.action]}       # self.action = 'train'

        # image_paths = []
        # names = []
        # path = Path('/hdd/file-input/qint/data/6159_cd_p11s400_enhance_image/')
        # for s in splits:
        #     for p in Path(path).iterdir():
        #         if str(p).split('/')[-1] == 'train':
        #             continue
        #         for i in p.rglob('*.bmp'):  
        #             filename = '_'.join(str(i).split('/')[-3:])
        #             image_paths.append(str(i))
        #             names.append(filename[:-4])
        #     splits[s]["images"].extend(image_paths)
        #     splits[s]["names"].extend(names)

        self.newLabels = True
        '''文件夹下包含多个子文件夹,数据放在qin/data/'''
        self.flag6159 = False
        if self.flag6159:   # 6159图库
            lib_gallery = [
                '6159_cd_p11s400_enhance_image'
            ]
            root_path = '/hdd/file-input/linwc/Descriptor/data/6159_cd_p11s400_enhance_image/'
        else:
            lib_gallery = [
                    '6191_DK4_halfpress_p10s150',       # 5100
                    '6191_DK4_powder_p10s150',          # 7041
                    '6191_DK4_rot90_p10s150',           # 7465
                    '6191_DK4_SH_p10s150',              # 7163
                    '6191_DK4_SH_rot_p9s150',           # 7797
                    '6191_DK4_wet_p10s150',             # 7022
                    '6191_DK7-140-normal-X888-SH',      # 1800
                    '6191_DK7_halfpress_p10s150',       # 7031
                    '6191_DK7_p10s200',                 # 2399
                    '6191_DK7_powder_p9s150',           # 6862
                    '6191_DK7_SH_rot_p9s150',           # 7743
                    '6191_DK7_wet_p10s150',             # 7771
                    '6191_DK4_SH_SZ_p20s150'            # 14628  
            ]                                           # total: 89822
            root_path = '/hdd/file-input/linwc/Descriptor/data/6191/process/'

        image1_list = []
        image2_list = []
        image1_name = []
        image2_name = []
        trans = []
        coor_group_all = []
        img_path_all = []
        ban_orientation_all = None
        for lib_name in lib_gallery:
            match_path = Path(root_path + lib_name + '/trans.txt')
            point_path = Path(root_path + lib_name + '/point.txt')
            orientation_poor_path = Path(root_path + lib_name + '/orientation_poor.npy')

        # if 1:   # 6159图库
        #     match_path = Path('/hdd/file-input/linwc/Descriptor/data/6159_cd_p11s400_enhance_image/6159_cd_p11s400_enhance_image/log.1217.pick.txt')
        #     point_path = Path('/hdd/file-input/linwc/Descriptor/data/6159_cd_p11s400_enhance_image/6159_cd_p11s400_enhance_image/log2.feature.point.txt')
        #     orientation_poor_path = Path('/hdd/file-input/linwc/Descriptor/data/6159_cd_p11s400_enhance_image/6159_cd_p11s400_enhance_image/orientation_poor.npy')
        # else:   # 6191
        #     match_path = Path('/hdd/file-input/linwc/Descriptor/data/6159_cd_p11s400_enhance_image/6159_cd_p11s400_enhance_image/log.1217.pick.txt')
        #     point_path = Path('/hdd/file-input/linwc/Descriptor/data/6159_cd_p11s400_enhance_image/6159_cd_p11s400_enhance_image/log2.feature.point.txt')
        #     orientation_poor_path = Path('/hdd/file-input/linwc/Descriptor/data/6159_cd_p11s400_enhance_image/6159_cd_p11s400_enhance_image/orientation_poor.npy')
            
            # path_root = Path('/hdd/file-input/linwc/Descriptor/data/6191/6191_DK4-110_cd_p12s200/')
            # # path_root = Path('/hdd/file-input/qint/data/toQt_Jy/newLabelLog/')
            # gallery = [
            #     # "6159_cd_p11s400",
            #     # "6159_rot_p20s80",
            #     # "6159_S1_p20s80",
            #     # "03_大数据库",
            #     "6191_DK4-110_cd_p12s200",
            #     "6191_DK4-110_p20s200",
            #     "6191_DK4-110_p34s200",
            #     "6191_DK4-110_rot_p10s200",
            # ]
        # rep2 = '/hdd/file-input/qint/6159_parallel/Test_sys/logs/ori_siftlabels_out/output'
            image_paths = []
            npy_path = []
            truncate = 1
            df = pd.read_csv(
                        match_path,
                        header=None,
                        encoding = "gb2312",
                        names=['enroll', 'verify', 'path', 
                            'H0', 'H1', 'H2', 'H3', 'H4', 'H5',
                            'score', 'up']           
                        )
            im1_list = df['enroll'].to_list()
            if self.flag6159:    
                im1_list = [i[8:].rjust(3, '0') + '_enhance' for i in im1_list]    # 获得名称并补齐 '1'-> '001'
            else:
                im1_list = [i[8:].rjust(4, '0') + '_enhance' for i in im1_list]
            # image1_list.extend([i + '_enhance' for i in im1_list])
            im2_list = df['path'].to_list()
            im2_list = [i[7:] for i in im2_list]
            # print(im2_list[0])
            image2_list.extend(im2_list)

            H0 = df['H0'].to_list()
            H0 = [(int)(h[6:]) for h in H0]
            H1 = df['H1'].to_list()
            H2 = df['H2'].to_list()
            H3 = df['H3'].to_list()
            H4 = df['H4'].to_list()
            H5 = df['H5'].to_list()
            # print(im1_list[0])
            for i in range(len(im2_list)):
                suffix = '.bmp'
                path_share = '/'.join(im2_list[i].split('/')[:-1])
                im1_list[i] = path_share + '/' + im1_list[i] + suffix
            # print(im1_list[0])
            image1_list.extend(im1_list)
            trans.extend([[H0[i], H1[i], H2[i], H3[i], H4[i], H5[i]] for i in range(len(H0))])    # combined into list
            # trans = np.array(trans) # (N, 6)

            image1_name.extend(['_'.join(k.split('/')[-3:])[:-4] for k in im1_list])
            # print(image1_name[0])
            image2_name.extend(['_'.join(k.split('/')[-3:])[:-4] for k in im2_list])

            # ------ images points read ------
            coor = []
            coor_group = []
            img_6159_path = []
            df_p = pd.read_csv(point_path, header=None, encoding="gb2312", sep='\t')    # 'gb2312'

            for line in zip(df_p[0]):  # line: [index, .]
                if line[0][0:4] == '/hdd':
                    '''换成npy格式(DOG数据)'''
                    # tmp = line[0]
                    # tmp = tmp.replace('bmp', 'npy')
                    # tmp = tmp.replace('6159_cd_p11s400', '6159_cd_p11s400_DOG')
                    # img_6159_path.append(tmp[1:])

                    img_6159_path.append(line[0][0:])

                if line[0][0] == 'x':
                    ls = re.split('[,:]', line[0])
                    coor.append(ls[1])    # 提取x,y, ori
                    coor.append(ls[3])
                    # coor.append(ls[5])

                if (line[0][0] == '/') and (len(coor) != 0):
                    coor_group.append(coor)
                    coor = []
            # 方向一致性
            ban_orientation = np.load(orientation_poor_path, allow_pickle=True)
            if ban_orientation_all is None:
                ban_orientation_all = ban_orientation
            else:
                ban_orientation_all = np.concatenate((ban_orientation_all, ban_orientation), axis=-1)

            coor_group_all.extend(coor_group)
            img_path_all.extend(img_6159_path)
        # print(len(coor_group_all), len(coor_group), len(img_path_all))

        # ------ dic ------
        files1 = {
            'imgA': image2_list,    
            'imgB': image1_list,       # 模板
            'imgA_name': image2_name,
            'imgB_name': image1_name,
            'trans': trans
        }
        self.files2 = {
            'img_6159': img_path_all,
            'pts': coor_group_all
        }
    

        sequence_set1 = []
        sequence_set2 = []

        count = -1
        for (A, N_A, B, N_B, M_H) in zip(
            files1['imgA'], files1['imgA_name'],
            files1['imgB'], files1['imgB_name'],files1['trans']):   # zip返回元组

            '''换成npy格式(DOG数据)'''
            # A = A.replace('bmp', 'npy')
            # A = A.replace('6159_cd_p11s400', '6159_cd_p11s400_DOG')
            # B = B.replace('bmp', 'npy')
            # B = B.replace('6159_cd_p11s400', '6159_cd_p11s400_DOG')

            '''方向一致性'''
            count += 1
            if ban_orientation_all[count] == 1:
                continue

            if self.flag6159:
                '''小位移'''
                if (int(N_A[8:11]) >= 200) or (int(N_B[8:11]) >= 200):
                    continue

            '''旋转'''
            # if (int(N_A[8:11]) > 200 and int(N_B[8:11]) > 200) or (int(N_A[8:11]) < 200 and int(N_B[8:11]) < 200):
            #     continue

            sample = {
                'imgA': A, 
                'imgA_name': N_A, 
                'imgB': B, 
                'imgB_name': N_B,
                'm_all': M_H
                }   # 无标签时，只有img数据
            sequence_set1.append(sample)

        
        # for select in gallery:
        #     logging.info("add image {}.".format(select))
        #     path = Path(path_root, select)
        for s in splits:
            e = [[sam['imgA'], sam['imgB']] for sam in sequence_set1]
            f = ['' for _ in sequence_set1]
            d = ['' for _ in sequence_set1]    # descriptor
            n = [[sam['imgA_name'], sam['imgB_name']] for sam in sequence_set1]
            h = [sam['m_all'] for sam in sequence_set1]
            splits[s]["images"].extend(e)
            splits[s]["points"].extend(f)
            splits[s]["desc"].extend(d)
            splits[s]["names"].extend(n)
            splits[s]["trans"].extend(h)
            # for s in splits:
            #     e = [str(p) for p in Path(path, s, "pic").iterdir() if str(p)[-4:] == '.bmp']
            #     f = [p.replace("pic", "csv") for p in e]
            #     # f = [p.replace(str(path), str(rep2)) for p in e]
            #     if not self.newLabels:
            #         f = [p.replace(".bmp", ".csv") for p in f]
            #     else:
            #         f = [p.replace("enhance", "point") for p in f]
            #     n = [(p.split('/')[-1])[:-4] for p in f]
            #     splits[s]["images"].extend(e[: int(truncate * len(e))])
            #     if Path(f[0]).exists():
            #         splits[s]["points"].extend(f[: int(truncate * len(f))])
            #     splits[s]["names"].extend(n[: int(truncate * len(n))])

        # Shuffle
        for s in splits:
            perm = np.random.RandomState(0).permutation(len(splits[s]["images"]))
            if len(splits[s]['desc']) != 0:
                for obj in ["images", "points", "names", "desc", "trans"]:
                    splits[s][obj] = np.array(splits[s][obj])[perm].tolist()
            else:
                for obj in ["images", "points", "names", "trans"]:
                    splits[s][obj] = np.array(splits[s][obj])[perm].tolist()

        self.crawl_folders(splits)

    def crawl_folders(self, splits):
        sequence_set = []
        if len(splits[self.action]['desc']) != 0:
            for (img, pnt, name, desc, trans) in zip(
                splits[self.action]["images"], splits[self.action]["points"], splits[self.action]["names"], splits[self.action]["desc"], splits[self.action]["trans"]
            ):
                sample = {"image": img, "point": pnt, "name": name, "desc": desc, "trans": trans}
                sequence_set.append(sample)
        else:
            for (img, name, pnt, trans) in zip(
                splits[self.action]["images"], splits[self.action]["names"], splits[self.action]["points"], splits[self.action]["trans"]
            ):
                # if names[-7:] == "enhance":
                sample = {"image": img, "point": pnt, "name": name, "trans": trans}
                sequence_set.append(sample)
        self.samples = sequence_set     # 成对（.bmp&.npy）的数据列表
        print(len(self.samples))

    def putGaussianMaps(self, center, accumulate_confid_map):
        crop_size_y = self.params_transform["crop_size_y"]
        crop_size_x = self.params_transform["crop_size_x"]
        stride = self.params_transform["stride"]
        sigma = self.params_transform["sigma"]

        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        start = stride / 2.0 - 0.5
        xx, yy = np.meshgrid(range(int(grid_x)), range(int(grid_y)))
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        mask = exponent <= sigma
        cofid_map = np.exp(-exponent)
        cofid_map = np.multiply(mask, cofid_map)
        accumulate_confid_map += cofid_map
        accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
        return accumulate_confid_map

    def __getitem__(self, index):
        """
        :param index:
        :return:
            labels_2D: tensor(1, H, W)
            image: tensor(1, H, W)
        """
        def _read_image(path):
            cell = 8
            input_image = cv2.imread(path)
            # print(f"path: {path}, image: {image}")
            # print(f"path: {path}, image: {input_image.shape}")
            input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                                     interpolation=cv2.INTER_AREA)
            H, W = input_image.shape[0], input_image.shape[1]
            # H = H//cell*cell
            # W = W//cell*cell
            # input_image = input_image[:H,:W,:]
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

            input_image = input_image.astype('float32') / 255.0
            return input_image

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

        def get_labels(pnts, H, W):
            labels = torch.zeros(H, W)
            # print('--2', pnts, pnts.size())
            # pnts_int = torch.min(pnts.round().long(), torch.tensor([[H-1, W-1]]).long())
            pnts_int = torch.min(
                pnts.round().long(), torch.tensor([[W - 1, H - 1]]).long()
            )
            # print('--3', pnts_int, pnts_int.size())
            labels[pnts_int[:, 1], pnts_int[:, 0]] = 1
            return labels
        
        def get_labels_sort(pnts, H, W):
            labels = torch.zeros(H, W)
            # print('--2', pnts, pnts.size())
            # pnts_int = torch.min(pnts.round().long(), torch.tensor([[H-1, W-1]]).long())
            pnts_int = torch.min(
                pnts.round().long(), torch.tensor([[W - 1, H - 1]]).long()
            )
            # print('--3', pnts_int, pnts_int.size())
            for i in range(len(pnts)):
                labels[pnts_int[i, 1], pnts_int[i, 0]] = i + 1
            return labels

        def get_label_res(H, W, pnts):
            quan = lambda x: x.round().long()
            labels_res = torch.zeros(H, W, 2)
            # pnts_int = torch.min(pnts.round().long(), torch.tensor([[H-1, W-1]]).long())

            labels_res[quan(pnts)[:, 1], quan(pnts)[:, 0], :] = pnts - pnts.round()
            # print("pnts max: ", quan(pnts).max(dim=0))
            # print("labels_res: ", labels_res.shape)
            labels_res = labels_res.transpose(1, 2).transpose(0, 1)
            return labels_res

        from datasets.data_tools import np_to_tensor
        from utils.utils import filter_points
        from utils.var_dim import squeezeToNumpy

        sample_i = self.samples[index]
        if self.newLabels:
            imgA = load_as_float(sample_i['image'][0])
            imgB = load_as_float(sample_i['image'][1])
            imgA_part_mask = load_as_float(sample_i['image'][0].replace('_enhance', '_mask'))  
            imgB_part_mask = load_as_float(sample_i['image'][1].replace('_enhance', '_mask'))
        else:   # 新数据(newLabelLog)(.csv格式)
            img = pd.read_csv(sample_i['image'], header=None).to_numpy()
            img = img.astype(np.float32) / 255
        # if (img.shape[0] != self.sizer[0]) or (img.shape[1] != self.sizer[1]):
        #     print("error！！！this img is %s", sample["image"])
        if self.transform is not None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
            imgA_part_mask = self.transform(imgA_part_mask)
            imgB_part_mask = self.transform(imgB_part_mask)

        '''传统增强融合扩边处理-逆:36->32'''
        if self.flag6159:
            Hdev_top = 12
            Wdev_left = 2
        else:
            Hdev_top = 0
            Wdev_left = -2
        # # [1, 136, 40]
        # imgA = F.pad(imgA, pad=(Wdev_left, Wdev_left), mode="constant", value=0)
        # imgB = F.pad(imgB, pad=(Wdev_left, Wdev_left), mode="constant", value=0)
        # imgA = imgA[0, Hdev_top:-Hdev_top, :].unsqueeze(0)   # [1, 136, 40]
        # imgB = imgB[0, Hdev_top:-Hdev_top, :].unsqueeze(0)
        
        # [1, 136, 32]
        if Wdev_left < 0:
            imgA = F.pad(imgA, pad=(-Wdev_left, -Wdev_left), mode="constant", value=0)      # pad -> [136, 40]
            imgB = F.pad(imgB, pad=(-Wdev_left, -Wdev_left), mode="constant", value=0)
            imgA = imgA[0, Hdev_top:imgA.shape[1]-Hdev_top, :].unsqueeze(0)   
            imgB = imgB[0, Hdev_top:imgB.shape[1]-Hdev_top, :].unsqueeze(0)
            imgA_part_mask = F.pad(imgA_part_mask, pad=(-Wdev_left, -Wdev_left), mode="constant", value=0)  # pad -> [136, 40]
            imgB_part_mask = F.pad(imgB_part_mask, pad=(-Wdev_left, -Wdev_left), mode="constant", value=0)
            imgA_point_mask = torch.zeros_like(imgA_part_mask, device=imgA_part_mask.device).float()
            imgB_point_mask = torch.zeros_like(imgB_part_mask, device=imgB_part_mask.device).float()
            imgA_point_mask[:, 2:-2, 4:-4] = 1
            imgB_point_mask[:, 2:-2, 4:-4] = 1
            imgA_part_mask *= imgA_point_mask       # 136x40
            imgB_part_mask *= imgB_point_mask
        else:    
            imgA = imgA[0, Hdev_top:imgA.shape[1]-Hdev_top, Wdev_left:imgA.shape[2]-Wdev_left].unsqueeze(0)   # [1, 136, 32]
            imgB = imgB[0, Hdev_top:imgB.shape[1]-Hdev_top, Wdev_left:imgB.shape[2]-Wdev_left].unsqueeze(0)
            imgA_part_mask = imgA_part_mask[0, Hdev_top:imgA_part_mask.shape[1]-Hdev_top, Wdev_left:imgA.shape[2]-Wdev_left].unsqueeze(0)
            imgB_part_mask = imgB_part_mask[0, Hdev_top:imgB_part_mask.shape[1]-Hdev_top, Wdev_left:imgB.shape[2]-Wdev_left].unsqueeze(0)

        H, W = imgA.shape[1], imgA.shape[2]
        # print(H, W)
        # print(H, W)
        self.H = H
        self.W = W

        sample = {}

        valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=torch.eye(3))
        sample.update({"valid_mask": valid_mask})
        # assert Hc == round(Hc) and Wc == round(Wc), "Input image size not fit in the block size"

        from utils.utils import homography_centorcrop, homography_transform 
        homography = np.array(sample_i["trans"])
        homography = torch.tensor(homography.reshape(2, 3), dtype=torch.float32)
        homography = 2 * homography / 512
        vec_one = torch.tensor([[0, 0, 1]], dtype=torch.float32)
        homography = torch.cat((homography, vec_one), 0)
        
        # After CentorCrop Trans
        # # [1, 136, 40]
        # homography = homography_centorcrop(homography, Hdev_top, -Wdev_left)
        
        # [1, 136, 32]
        homography = homography_centorcrop(homography, Hdev_top, Wdev_left)
        # homography = homography_transform(homography, -Hdev_top, -Wdev_left)
        
        homography_en = self.sample_homography(H, W)
        homography_en = torch.tensor(homography_en, dtype=torch.float32)
        inv_homography_en = torch.inverse(homography_en)
        
        homography = homography @ inv_homography_en
        inv_homography = torch.inverse(homography)
        
        trans_obj = AffineTransform(matrix=homography.cpu().numpy())
        sample['rotation'] = trans_obj.rotation * 180 / math.pi # [-180, 180]
        sample['shear'] = trans_obj.shear * 180 / math.pi
        homography_o = homography

        # images
        imgA = imgPhotometric(imgA.squeeze().numpy())   # imgA: no Photometric imgB: Photometric
        imgA = torch.tensor(imgA, dtype=torch.float32)
        imgA = self.inv_warp_image(imgA.squeeze(), homography_en, mode='bilinear').unsqueeze(0)
        imgA_part_mask = self.inv_warp_image(imgA_part_mask.squeeze(), homography_en, mode='bilinear').unsqueeze(0)
        # sample.update({'warped_img': warped_img})
        # masks
        # valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=homography)
        # valid_mask_inv = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=inv_homography)
        # valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=inv_homographies,
        #                                      erosion_radius=self.config['augmentation']['homographic'][
        #                                          'valid_border_margin'])

        sample["image"] = imgA
        sample["image_H"] = imgB
        sample["image_pmask"] = imgA_part_mask
        sample["image_H_pmask"] = imgB_part_mask
        sample["name"] = sample_i["name"][0] + '_to_' + sample_i["name"][1]

        if 'point' in self.samples[index]:
            # pnts = np.load(self.samples[index]["point"], allow_pickle=True)  # (y, x)读取label
            if 0:
                df = pd.read_csv(self.samples[index]["point"], encoding = "gb2312", names=['sort', 'x', 'y', 'desc1', 'desc2', 'desc3', 'desc4'])
                pnts_x = df['x'].to_list()
                pnts_y = df['y'].to_list()
                pnts_x = np.array(pnts_x[1:]).astype(np.float32)
                pnts_y = np.array(pnts_y[1:]).astype(np.float32)
                desc1 = np.array(df['desc1'].to_list()[1:]).astype(np.int32)
                desc2 = np.array(df['desc2'].to_list()[1:]).astype(np.int32)
                desc3 = np.array(df['desc3'].to_list()[1:]).astype(np.int32)
                desc4 = np.array(df['desc4'].to_list()[1:]).astype(np.int32)

                # 32位对齐
                desc1_list = ['{0:32b}'.format(p, 'b').replace(' ', '0').replace('-', '1')[:32] for p in desc1]
                desc2_list = ['{0:32b}'.format(p, 'b').replace(' ', '0').replace('-', '1')[:32] for p in desc2]
                desc3_list = ['{0:32b}'.format(p, 'b').replace(' ', '0').replace('-', '1')[:32] for p in desc3]
                desc4_list = ['{0:32b}'.format(p, 'b').replace(' ', '0').replace('-', '1')[:32] for p in desc4]

                # str -> array
                desc1 = np.array([np.fromiter(p, dtype=np.int) for p in desc1_list]).astype(np.float32)
                desc2 = np.array([np.fromiter(p, dtype=np.int) for p in desc2_list]).astype(np.float32)
                desc3 = np.array([np.fromiter(p, dtype=np.int) for p in desc3_list]).astype(np.float32)
                desc4 = np.array([np.fromiter(p, dtype=np.int) for p in desc4_list]).astype(np.float32)

                # concate
                desc = np.concatenate((desc1, desc2, desc3, desc4), axis=-1)
            elif not self.newLabels:
                df = pd.read_csv(self.samples[index]["point"], encoding = "gb2312", names=['sort', 'x', 'y', 'desc1', 'desc2'])
                pnts_x = np.array(df['x'].to_list()[1:]).astype(np.int32)
                pnts_y = np.array(df['y'].to_list()[1:]).astype(np.int32)
                desc1_list = df['desc1'].to_list()[1:]
                desc2_list = df['desc2'].to_list()[1:]
                if self.double_descriptor:
                    desc1 = np.array([np.fromiter(p, dtype=np.int) for p in desc1_list], dtype="float32")
                    desc2 = np.array([np.fromiter(p[:-1], dtype=np.int) for p in desc2_list], dtype="float32")
                    desc = np.concatenate((desc1, desc2), axis=1)
                else:
                    desc = np.array([np.fromiter(p, dtype=np.int) for p in desc1_list], dtype="float32")
                    
                # desc2_list = [p[:-1] for p in desc2_list if p[-1] == ' ']
                # desc2 = np.array([np.fromiter(p, dtype=np.int) for p in desc2_list], dtype="float32")
                
            else:
                imgA_path = sample_i['image'][0]
                imgB_path = sample_i['image'][1]
                idxA = self.files2['img_6159'].index(imgA_path) # 获取同手指图在点坐标列表中的index
                idxB = self.files2['img_6159'].index(imgB_path)
                ptsA = np.array(self.files2['pts'][idxA], dtype=np.float32).reshape(-1, 2)    #(x, y)
                ptsB = np.array(self.files2['pts'][idxB], dtype=np.float32).reshape(-1, 2)
                ptsA = torch.tensor(ptsA).type(torch.FloatTensor)
                ptsB = torch.tensor(ptsB).type(torch.FloatTensor)

                pntsA_x = ptsA[:, 0]
                pntsA_y = ptsA[:, 1]
                pntsA_x = np.array(pntsA_x).astype(np.float32)
                pntsA_y = np.array(pntsA_y).astype(np.float32)
                pntsB_x = ptsB[:, 0]
                pntsB_y = ptsB[:, 1]
                pntsB_x = np.array(pntsB_x).astype(np.float32)
                pntsB_y = np.array(pntsB_y).astype(np.float32)
            pntsA = np.stack((pntsA_x, pntsA_y), axis=1)   #(x, y)
            pntsB = np.stack((pntsB_x, pntsB_y), axis=1)   #(x, y)
            # pnts_after_resize = pnts.copy()
            mask_ptsA = (pntsA[:, 0] >= Wdev_left) * (pntsA[:, 0] <= Wdev_left+W-1) * (pntsA[:, 1] >= Hdev_top) * (pntsA[:, 1] <= Hdev_top+H-1)
            mask_ptsB = (pntsB[:, 0] >= Wdev_left) * (pntsB[:, 0] <= Wdev_left+W-1) * (pntsB[:, 1] >= Hdev_top) * (pntsB[:, 1] <= Hdev_top+H-1)
            pntsA = pntsA[mask_ptsA]
            pntsB = pntsB[mask_ptsB]
            # desc = desc[mask_pts]
            pntsA[:, 0] -= Wdev_left    # 36->32  cut 2 pixel
            pntsB[:, 0] -= Wdev_left 
            pntsA[:, 1] -= Hdev_top    # 36->32  cut 2 pixel
            pntsB[:, 1] -= Hdev_top 
            pntsA = torch.tensor(pntsA).float()
            pntsB = torch.tensor(pntsB).float()

            # desc = torch.tensor(desc).float()
            # pnts = torch.stack((pnts[:, 0], pnts[:, 1]), dim=1)  # (x, y)
            # pnts, mask_filter = filter_points(pnts, torch.tensor([W, H]), return_mask=True)
            # desc = desc[mask_filter]
            
            # 有监督
            # '''截断120'''
            # cut = 68
            # pntsA = pntsA[:cut, :]
            # pntsB = pntsB[:cut, :]

            # # Pesudo Descriptor
            # des_dim = 128
            # desc = torch.zeros((cut, des_dim), dtype=torch.float32)
            # sample.update({"desc": desc})

            warped_pnts = self.warp_points(pntsA, homography_en)
            warped_pnts, _ = filter_points(warped_pnts, torch.tensor([W, H]), return_mask=True)
            warped_pnts_norm = warped_pnts / warped_pnts.new_tensor([W - 1, H - 1]).to(warped_pnts.device) * 2 - 1
            pntsB_norm = pntsB / pntsB.new_tensor([W - 1, H - 1]).to(pntsB.device) * 2 - 1
            cut = 150
            sift_maskA = torch.ones(cut, device=warped_pnts_norm.device)
            sift_maskB = torch.ones(cut, device=pntsB_norm.device)

            if warped_pnts_norm.shape[0] < cut:
                expandA = torch.zeros((cut - warped_pnts_norm.shape[0], 2), device=warped_pnts_norm.device)
                sift_maskA[warped_pnts_norm.shape[0]:] = 0
                warped_pnts_norm = torch.cat((warped_pnts_norm, expandA), dim=0)
            if pntsB_norm.shape[0] < cut:
                expandB = torch.zeros((cut - pntsB_norm.shape[0], 2), device=pntsB_norm.device)
                sift_maskB[pntsB_norm.shape[0]:] = 0
                pntsB_norm = torch.cat((pntsB_norm, expandB), dim=0)

            sample.update({"sift_pntA": warped_pnts_norm})    # 150 x 2
            sample.update({"sift_pntB": pntsB_norm})
            sample.update({"sift_maskA": sift_maskA})    # 150
            sample.update({"sift_maskB": sift_maskB})


            # warped_desc = desc[mask_warped]
            # if warped_desc.shape[0] < cut:
            #     zeros_mat = torch.zeros([cut - warped_desc.shape[0], warped_desc.shape[1]], dtype=torch.float)
            #     warped_desc = torch.cat((warped_desc, zeros_mat), dim=0)  # 统一desc的维度
            # sample.update({"warped_desc": warped_desc})

            # desc = desc[:cut, :]
            # if desc.shape[0] < cut:
            #     # 不足描述子补零
            #     zeros_mat = torch.zeros([cut - desc.shape[0], desc.shape[1]], dtype=torch.float)
            #     desc_fill = torch.cat((desc, zeros_mat), dim=0)  # 统一desc的维度
            #     sample.update({"desc": desc_fill})
            # else:
            #     sample.update({"desc": desc})

            labels_2D = get_labels(pntsA, H, W)
            labels_2D_sort = get_labels_sort(pntsA, H, W)  # 记录点的排序顺序
            sample.update({"labels": labels_2D.unsqueeze(0)})
            sample.update({"labels_sort": labels_2D_sort.unsqueeze(0)})


            # '''截断120'''
            # warped_pnts = warped_pnts[:120, :]
            labels_2D_H = get_labels(pntsB, H, W)     # 注意：存在精度损失
            labels_2D_H_sort = get_labels_sort(pntsB, H, W)     # 注意：存在精度损失
            sample.update({"labels_H": labels_2D_H.unsqueeze(0)})
            sample.update({"labels_H_sort": labels_2D_H_sort.unsqueeze(0)})

            warped_labels_res = torch.zeros(H, W, 2)
            warped_labels_res[quan(pntsB)[:, 1], quan(pntsB)[:, 0], :] = pntsB - pntsB.round()
            sample.update({"warped_res": warped_labels_res})
            
            '''CV2 SIFT points'''
            if 0:
                img = imread(self.samples[index]['image'])
                img = img[:, 2:-2]
                img = np.uint8(img)
                sift = cv2.SIFT_create(contrastThreshold=1e-5)
                # find the keypoints and descriptors with SIFT
                kp, des = sift.detectAndCompute(img, None)
                pts_sift = np.array([p.pt for p in kp])
                pts_sift = torch.tensor(pts_sift).float()
                if pts_sift.size() == torch.Size() or pts_sift.size() == torch.Size([0]):
                    labels_2D = torch.zeros(H, W)
                else:
                    labels_2D = get_labels(pts_sift, H, W)
                sample.update({"labels": labels_2D.unsqueeze(0)})

                if pts_sift.size() == torch.Size() or pts_sift.size() == torch.Size([0]):
                    labels_2D_H = torch.zeros(H, W)
                    
                else:
                    pts_warped_sift = self.warp_points(pts_sift, homography)
                    pts_warped_sift = filter_points(pts_warped_sift, torch.tensor([W, H]))
                    labels_2D_H = get_labels(pts_warped_sift, H, W)
                sample.update({"labels_H": labels_2D_H.unsqueeze(0)})


        # sample.update({'image_H': warped_img, 'valid_mask_H': valid_mask})
        sample.update({'valid_mask_H': valid_mask})
        # sample.update({'homography': homography, 'inv_homography': inv_homography})
        sample.update({'homography': homography, 'inv_homography': inv_homography, 'homography_o': homography_o})
        
        return sample

    def __len__(self):
        return len(self.samples)

    ## util functions
    def gaussian_blur(self, image):
        """
        image: np [H, W]
        return:
            blurred_image: np [H, W]
        """
        aug_par = {"photometric": {}}
        aug_par["photometric"]["enable"] = True
        aug_par["photometric"]["params"] = self.config["gaussian_label"]["params"]
        augmentation = self.ImgAugTransform(**aug_par)
        # get label_2D
        # labels = points_to_2D(pnts, H, W)
        image = image[:, :, np.newaxis]
        heatmaps = augmentation(image)
        return heatmaps.squeeze()

