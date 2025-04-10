import os
import re
import cv2
import csv
import math
from tqdm import tqdm
import yaml
from kornia import tensor_to_image
import torch
import copy
import glob
# import torch.nn as nn
import logging
import importlib
import numpy as np
import pandas as pd
import torch.nn.functional as F

from pathlib import Path
from imageio import imread
from PIL import Image, ImageDraw
from skimage.measure import ransac
from skimage.transform import AffineTransform
import torchvision.transforms as transforms


from Model_component import inv_warp_image, compute_valid_mask, warp_points
from Model_component import filter_points, draw_keypoints_pair, draw_keypoints_pair_tradition
from Model_component import get_point_pair_repeat, getPtsFromHeatmap, flattenDetection
from Model_component import homograghy_transform
from Model_component import thresholding_desc

def get_module(name):  
    mod = importlib.import_module('Test_component')
    print(name)
    return getattr(mod, name)

def load_as_float(path):
    return imread(path).astype(np.float32) / 255

def Split_list(list_in, num, new_list=[]):
    if len(list_in) <= num:
        new_list.append(list_in)
        return new_list
    else:
        new_list.append(list_in[:num])
        return Split_list(list_in[num:], num)


'''真实重复率测试V3(自监督网络：增强数据6193--SIFT_transSucc)'''
class Test_Real_Repeat_Enhance_V3_self_supervised_student(object):
    def __init__(self, img_path=None, info_path=None, device="cpu", **config):

        self.device         = device
        self.top_k          = config['top_k']
        # self.sizer          = config['resize']
        self.output_ratio   = config['output_ratio']
        self.output_images  = config['output_images']
        self.conf_thresh    = config['detec_thre']
        self.nms            = config['nms']
        self.dis_thr        = config['dis_thr']     # 点对距离满足<self.dis_thr认为是匹配对

        self.isDilation     = config['isDilation']  # True: 扩充36->40  False: 裁剪36->32
        self.use_128x52     = config['use_128x52']  # 使用extend图像
        self.use_160x48     = config['use_160x48']  # 使用extend图像

        self.output_dir     = Path(config['output_dir']) / 'val'
        os.makedirs(self.output_dir, exist_ok=True)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((self.sizer[0], self.sizer[1])),
        ])

        choice_6193 = [
                # "6193_DK7_merge_test1_9800",
                # "6193_DK7_merge_test2_9800",
                # "6193_DK7_XA_rot_test_9800",
                "6193-DK7-140-8-wet2_clear_test_9800",
                # "6193_DK7_partialpress_test",
                # "6193-DK7-140-8-normal_test",
                # "6193-DK7-140-8-powder_test",
                # "6193-DK7-140-8-rotate_test",
                # "6193-DK7-140-8-suppress_test",
                # "6193_DK7_CDXA_normal_merge_test",
            ]

        # print("load dataset_files from: ", img_path)
        # ------ matched images ------
        self.sift_succ_dict = []
        for chs in choice_6193:
            sift_succ_path = Path('/hdd/file-input/qint/data/6193Test/' + chs + '/SIFT_Trans/SIFT_transSucc.csv')
            sift_total_path = Path('/hdd/file-input/qint/data/6193Test/' + chs + '/')
            dict_ = self.Decode_Succ_Fail_Csv_Other(sift_succ_path, 'pnt_desc_data_X')
            self.sift_succ_dict.extend(dict_)

        pass
    
    def Decode_Succ_Fail_Csv_Other(self, path, pts_desc_dir, image_file=''):
        # pts_desc_dir = 'NET_Point_6159_p21s600' if (file_name == 'Net_Succ' or file_name == 'Net_Fail') else 'SIFT_Point_6159_p21s600'
        if self.use_128x52:
            img_path = Path('/'.join(str(path).split('/')[:-2])) / 'img_extend_data' / str(image_file)
            pts_path = Path('/'.join(str(path).split('/')[:-2])) / ('pnt_desc_data/6193_DK7_merge_test1_9800' + '/' + str(image_file))
        else:
            img_path = Path('/'.join(str(path).split('/')[:-2])) / 'img_pnt_data' / str(image_file)
            pts_path = Path('/'.join(str(path).split('/')[:-2])) / (str(pts_desc_dir) + '/' + str(image_file))

        namelist = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'score','deformation0', 'deformation1', 'enrollverify', 'path', 'score_flag', 'up']
        df = pd.read_csv(path, names=namelist)

        H0, H1, H2 = df['h0'].to_list(), df['h1'].to_list(), df['h2'].to_list()
        H3, H4, H5 = df['h3'].to_list(), df['h4'].to_list(), df['h5'].to_list()
        homos = [[H0[i], H1[i], H2[i], H3[i], H4[i], H5[i]] for i in range(len(H0))]

        enr_vefy = df['enrollverify'].to_list()
        enr_vefy = [[e.split(' ')[1].rjust(4, '0'), e.split(' ')[3].rjust(4, '0')] for e in enr_vefy]   # e.g. '3'->'0003'  ['0003', '0015']
        # enroll, verify = enr_vefy[:, 0], enr_vefy[:, 1]

        tmp_path = df['path'].to_list()
        tmp_verify_name = ['_'.join(e.split('/')[-3:]) for e in tmp_path]  # 0002/L1/0013.bmp -> 0002_L1_0013.csv
        tmp_enroll_name = [vef_n.replace('_' + str(e_v[1]), '_' + str(e_v[0])) for (e_v, vef_n) in zip(enr_vefy, tmp_verify_name)]

        tmp_verify_name = [e.replace('_extend.bmp', '.bmp') for e in tmp_verify_name]
        tmp_enroll_name = [e.replace('_extend.bmp', '.bmp') for e in tmp_enroll_name]
        # tmp_img_name = []
        if self.use_128x52:
            img_verify_file = [Path(img_path, '/'.join(e.split('_')).replace('.bmp', '_extend.bmp')) for e in tmp_verify_name]
            img_enroll_file = [Path(img_path, '/'.join(e.split('_')).replace('.bmp', '_extend.bmp')) for e in tmp_enroll_name]
        else:
            img_verify_file = [Path(img_path, '/'.join(e.split('_')).replace('.bmp', '_kpt.bmp')) for e in tmp_verify_name]
            img_enroll_file = [Path(img_path, '/'.join(e.split('_')).replace('.bmp', '_kpt.bmp')) for e in tmp_enroll_name]
        pts_verify_file = [Path(pts_path, e.replace('.bmp', '.csv')) for e in tmp_verify_name]
        pts_enroll_file = [Path(pts_path, e.replace('.bmp', '.csv')) for e in tmp_enroll_name]

        tmp_n = re.split('[/.]', str(path))[-2]     # re
        NorS, SorF = [tmp_n.split('_')[0]], [tmp_n.split('_')[-1][-4:]]
        NorS, SorF = list(NorS) * len(homos), list(SorF) * len(homos)       # flag: Net/Sift, Succ/Fail 
        files = {
            'img_verify': img_verify_file,
            'img_enroll': img_enroll_file,
            'pts_verify': pts_verify_file,
            'pts_enroll': pts_enroll_file,
            'verify_name': tmp_verify_name,
            'enroll_name': tmp_enroll_name,
            'homos': homos,
            'NorS': NorS,
            'SorF': SorF
        }

        sequence_set = []
        for (img_vf, img_en, pts_vf, pts_en, name_vf, name_en, homo, NorS, SorF) in zip(
            files['img_verify'],
            files['img_enroll'],
            files['pts_verify'],    # including (x, y) & desc in the file
            files['pts_enroll'],
            files['verify_name'],
            files['enroll_name'],
            files['homos'],         # size (2, 3)
            files['NorS'],
            files['SorF']
            ):
            sample = {'img_vf': img_vf, 'img_en': img_en, 'pts_vf': pts_vf, 'pts_en': pts_en, 'name_vf': name_vf, 'name_en': name_en, 'homo': homo, 'NorS': NorS, 'SorF': SorF}
            sequence_set.append(sample)

        return sequence_set

    def pts_desc_decode_Other(self, path):
        ''' 6193: pnt_desc_data_X 文件中只有点坐标数据'''
        from scipy.linalg import hadamard
        # name = ['sort', 'x', 'y', 'desc1', 'desc2', 'desc3', 'desc4', 'desc5', 'desc6', 'desc7', 'desc8']
        # df = pd.read_csv(path, names=name, usecols=[0,1,2,3,4,5,6,7,8,9,10])
        df = pd.read_csv(path, header=None)
        mat = df.to_numpy()
        pnts_x = torch.tensor(df[1].to_list()).type(torch.FloatTensor)
        pnts_y = torch.tensor(df[2].to_list()).type(torch.FloatTensor)
        pnts_ang = torch.tensor(mat[:, -1]).type(torch.FloatTensor) - 90.   # [0,180] -> [-90, 90]
        pts_ori = torch.stack((pnts_x, pnts_y, pnts_ang), dim=1)   # (x, y)
        pts = copy.deepcopy(pts_ori)


        mask_pts = (pts[:, 0] >= 0) * (pts[:, 0] < 36) * (pts[:, 1] >= 0) * (pts[:, 1] < 122)
        # mask_pts = (pts[:, 0] >= 2) * (pts[:, 0] < 34) * (pts[:, 1] >= 2) * (pts[:, 1] < 120)
        pts = pts[mask_pts]
        pts_ori = pts_ori[mask_pts]

        if 1:
            desc = mat[:, 259:267]
            desc1, desc5 = desc[:,0].astype(np.int64), desc[:, 4].astype(np.int64)
            desc2, desc6 = desc[:,1].astype(np.int64), desc[:, 5].astype(np.int64)
            desc3, desc7 = desc[:,2].astype(np.int64), desc[:, 6].astype(np.int64)
            desc4, desc8 = desc[:,3].astype(np.int64), desc[:, 7].astype(np.int64)
            desc1 = torch.tensor([[(e >> n) & 0x1 for n in range(32)] for e in desc1]).type(torch.FloatTensor)  # 不用reversed
            desc2 = torch.tensor([[(e >> n) & 0x1 for n in range(32)] for e in desc2]).type(torch.FloatTensor)
            desc3 = torch.tensor([[(e >> n) & 0x1 for n in range(32)] for e in desc3]).type(torch.FloatTensor)
            desc4 = torch.tensor([[(e >> n) & 0x1 for n in range(32)] for e in desc4]).type(torch.FloatTensor)
            desc5 = torch.tensor([[(e >> n) & 0x1 for n in range(32)] for e in desc5]).type(torch.FloatTensor)
            desc6 = torch.tensor([[(e >> n) & 0x1 for n in range(32)] for e in desc6]).type(torch.FloatTensor)
            desc7 = torch.tensor([[(e >> n) & 0x1 for n in range(32)] for e in desc7]).type(torch.FloatTensor)
            desc8 = torch.tensor([[(e >> n) & 0x1 for n in range(32)] for e in desc8]).type(torch.FloatTensor)
            desc_front  = torch.cat((desc1, desc2, desc3, desc4), dim=-1)
            desc_back   = torch.cat((desc5, desc6, desc7, desc8), dim=-1)
        else:
            '''256dim'''
            mat = df.to_numpy()
            desc_tmp = mat[:, 3:259]
            desc_front_o = torch.tensor(desc_tmp[:, :128]).int()
            desc_back_o = torch.tensor(desc_tmp[:, 128:]).int()
            desc_front_thre = thresholding_desc(desc_front_o)
            desc_back_thre = thresholding_desc(desc_back_o)
            Hada = hadamard(128)
            Hada_front_T = desc_front_thre.long() @ torch.from_numpy(Hada)
            Hada_back_T = desc_back_thre.long() @ torch.from_numpy(Hada)
            desc_front = (Hada_front_T > 0).long()
            desc_back = (Hada_back_T > 0).long()

        desc_front = desc_front[mask_pts, :]
        desc_back = desc_back[mask_pts, :]

        if self.top_k:
            pts = pts[:self.top_k, :]
            pts_ori = pts_ori[:self.top_k, :]
            desc_front = desc_front[:self.top_k, :]
            desc_back = desc_back[:self.top_k, :]

        return pts_ori, pts, desc_front.type(torch.FloatTensor), desc_back.type(torch.FloatTensor)
    
    def Extract_Succ_Csv_Other_enroll2verify(self, index, samples):
        ''' enroll to verify , 和识别对齐'''

        '''#### img A(verify) & B(enroll) ####'''
        # img = pd.read_csv(samples[index]['img_vf'], header=None).to_numpy()
        
        img_vf_pth = str(samples[index]['img_vf']).replace('img_extend_data', 'img_pnt_data')
        img_vf_pth = str(img_vf_pth).replace('_extend.bmp', '_kpt.bmp')
        img = imread(img_vf_pth)
        img_o_A = img.astype(np.float32) / 255
        img_o_A_ne = 1 - img.astype(np.float32) / 255
        # img = pd.read_csv(samples[index]['img_en'], header=None).to_numpy()
        
        img_en_pth = str(samples[index]['img_en']).replace('img_extend_data', 'img_pnt_data')
        img_en_pth = str(img_en_pth).replace('_extend.bmp', '_kpt.bmp')        
        img = imread(img_en_pth)
        img_o_B = img.astype(np.float32) / 255
        img_o_B_ne = 1 - img.astype(np.float32) / 255
        # img_o_A, img_o_B = img_o_A[:, :36], img_o_B[:, :36]

        img_o_A = self.transforms(img_o_A)
        img_o_B = self.transforms(img_o_B)
        
        img_o_A_ne = self.transforms(img_o_A_ne)
        img_o_B_ne = self.transforms(img_o_B_ne)
        
        # 修改尺寸后对应的trans要变
        
        if self.use_160x48:
            
            imgA_np = img_o_A[:,1:-1,:][0].numpy()
            imgB_np = img_o_B[:,1:-1,:][0].numpy()
            
            imgA_resize = cv2.resize(np.array(imgA_np), (48, 160),interpolation=cv2.INTER_LINEAR)
            imgA = torch.tensor(imgA_resize.astype(np.float32)).unsqueeze(0)

            imgB_resize = cv2.resize(np.array(imgB_np), (48, 160),interpolation=cv2.INTER_LINEAR)
            imgB = torch.tensor(imgB_resize.astype(np.float32)).unsqueeze(0)



            imgA_np = img_o_A_ne[:,1:-1,:][0].numpy()
            imgB_np = img_o_B_ne[:,1:-1,:][0].numpy()
            imgA_ne_resize = cv2.resize(np.array(imgA_np), (48, 160),interpolation=cv2.INTER_LINEAR)
            imgA_ne = torch.tensor(imgA_ne_resize.astype(np.float32)).unsqueeze(0)

            imgB_ne_resize = cv2.resize(np.array(imgB_np), (48, 160),interpolation=cv2.INTER_LINEAR)
            imgB_ne = torch.tensor(imgB_ne_resize.astype(np.float32)).unsqueeze(0)


        else:
            # imgA = img_o_A[:, :, 2:-2]      # [1, 136, 32] 传统增强扩边处理-逆：36->32
            # imgB = img_o_B[:, :, 2:-2]      # [1, 136, 32]
            imgA = img_o_A[:, 1:-1, 2:-2]     # 122x36->120x32
            imgB = img_o_B[:, 1:-1, 2:-2]
        H, W = imgA.shape[1], imgA.shape[2]

        '''#### verify/enroll pts & desc ####'''
        pts_ori_vf, pts_vf, desc_front_vf, desc_back_vf = self.pts_desc_decode_Other(samples[index]['pts_vf'])     # sift点基于122x36
        pts_ori_en, pts_en, desc_front_en, desc_back_en = self.pts_desc_decode_Other(samples[index]['pts_en'])
        
        '''#### homo ####'''
        homo = torch.tensor(samples[index]['homo']).type(torch.FloatTensor)
        homo = 2 * homo / 512.  # csv中的数据需要乘上2再除以512
        vec_one = torch.tensor([[0, 0, 1]], dtype=torch.float32)
        homo = torch.cat((homo.reshape(2, 3), vec_one), dim=0)  # base on 122x36
        if self.use_160x48:
            # homo_fixed = homograghy_transform(homo, 3, 2) # 尺寸修改后对应的trans也要变 122x36->128x52
            homo_fixed = homo
        else:
            homo_fixed = homograghy_transform(homo, -1, -2)   # 6193基于122x36:->120x32

        # if self.set_144x52:
        #     w_off = (52 - 36) // 2      # 8
        #     h_off = (144 - 136) // 2    # 4
        #     if H == 136 and W == 36:
        #         imgA = F.pad(imgA, (w_off, w_off, h_off, h_off), 'constant')
        #         imgB = F.pad(imgB, (w_off, w_off, h_off, h_off), 'constant')
        #     pts_en = pts_ori_en + torch.tensor([w_off, h_off, 0.], dtype=torch.float32)
        #     pts_vf = pts_ori_vf + torch.tensor([w_off, h_off, 0.], dtype=torch.float32)
        #     homo_fixed = homograghy_transform(homo, h_off, w_off)   # 144x52的扩边图，不需要pad，但trans要变换
        
        '''#### imgA warped ####'''
        # imgA_warped = inv_warp_image(imgA.squeeze(), homo)
        H ,W = img_o_B.shape[1], img_o_B.shape[2]
        homo_ori_img = homo_fixed if self.use_160x48 else homo
        imgB_o_warped = inv_warp_image(img_o_B.squeeze(), torch.inverse(homo_ori_img))
        # imgB_warped_mask = compute_valid_mask(torch.tensor([H, W]), homography=homo)

        input  = {}
        input.update({'imgA': imgB, 'imgB': imgA, 'img_o_A': img_o_B, 'img_o_B': img_o_A})  # imgA: 非原图尺寸  img_o_A: 原图尺寸122*36
        input.update({'imgA_ne': imgB_ne, 'imgB_ne': imgA_ne })  # imgA: 非原图尺寸  img_o_A: 原图尺寸122*36
        input.update({'pts_vf': pts_en, 'pts_en': pts_vf})
        input.update({'pts_ori_vf': pts_ori_en, 'pts_ori_en': pts_ori_vf})  # 原始坐标
        input.update({'desc_front_vf': desc_front_en, 'desc_back_vf': desc_back_en})
        input.update({'desc_front_en': desc_front_vf, 'desc_back_en': desc_back_vf})
        input.update({'homo': torch.inverse(homo), 'homo_fixed': torch.inverse(homo_fixed)})
        input.update({'imgA_o_warped': imgB_o_warped})        # 非模板按照csv中trans进行的旋转
        input.update({'name_vf': samples[index]['name_en'], 'name_en': samples[index]['name_vf']})

        return input

    def Extract_Csv(self, index, samples):
        # from Model_component import sample_homography_cv
        # from Model_component import imgPhotometric
        def pts_desc_decode(path):
            name = ['sort', 'x', 'y', 'desc1', 'desc2', 'desc3', 'desc4', 'desc5', 'desc6', 'desc7', 'desc8']
            df = pd.read_csv(path, names=name)
            pnts_x = torch.tensor(df['x'].to_list()).type(torch.FloatTensor)
            pnts_y = torch.tensor(df['y'].to_list()).type(torch.FloatTensor)
            pts = torch.stack((pnts_x, pnts_y), dim=1)   # (x, y)
            mask_pts = (pts[:, 0] >= 2) * (pts[:, 0] <= 33)
            pts = pts[mask_pts]     # 点在136*36范围
            # if self.isDilation:
            #     pts[:, 0] += 2    # 36->40  add 2 pixel
            # else:
            #     pts[:, 0] -= 2    # 36->32  cut 2 pixel
            desc1, desc5 = df['desc1'].to_list(), df['desc5'].to_list()
            desc2, desc6 = df['desc2'].to_list(), df['desc6'].to_list()
            desc3, desc7 = df['desc3'].to_list(), df['desc7'].to_list()
            desc4, desc8 = df['desc4'].to_list(), df['desc8'].to_list()
            desc1 = torch.tensor([[(e >> n) & 0x1 for n in reversed(range(32))] for e in desc1]).type(torch.FloatTensor)
            desc2 = torch.tensor([[(e >> n) & 0x1 for n in reversed(range(32))] for e in desc2]).type(torch.FloatTensor)
            desc3 = torch.tensor([[(e >> n) & 0x1 for n in reversed(range(32))] for e in desc3]).type(torch.FloatTensor)
            desc4 = torch.tensor([[(e >> n) & 0x1 for n in reversed(range(32))] for e in desc4]).type(torch.FloatTensor)
            desc5 = torch.tensor([[(e >> n) & 0x1 for n in reversed(range(32))] for e in desc5]).type(torch.FloatTensor)
            desc6 = torch.tensor([[(e >> n) & 0x1 for n in reversed(range(32))] for e in desc6]).type(torch.FloatTensor)
            desc7 = torch.tensor([[(e >> n) & 0x1 for n in reversed(range(32))] for e in desc7]).type(torch.FloatTensor)
            desc8 = torch.tensor([[(e >> n) & 0x1 for n in reversed(range(32))] for e in desc8]).type(torch.FloatTensor)

            desc_front  = torch.cat((desc1, desc2, desc3, desc4), dim=-1)
            desc_back   = torch.cat((desc5, desc6, desc7, desc8), dim=-1)

            return pts, desc_front[mask_pts], desc_back[mask_pts]

        '''#### img A(verify) & B(enroll) ####'''
        img = pd.read_csv(samples[index]['img_vf'], header=None).to_numpy()
        img_o_A = img.astype(np.float32) / 255
        img = pd.read_csv(samples[index]['img_en'], header=None).to_numpy()
        img_o_B = img.astype(np.float32) / 255
        img_o_A, img_o_B = img_o_A[:, :36], img_o_B[:, :36]

        img_o_A = self.transforms(img_o_A)
        img_o_B = self.transforms(img_o_B)
        if self.isDilation:
            import torch.nn.functional as F
            imgA = F.pad(img_o_A, (2, 2), 'constant')     # 36 -> 40
            imgB = F.pad(img_o_B, (2, 2), 'constant')     # 36 -> 40
        else:
            imgA = img_o_A[:, :, 2:-2]      # [1, 136, 32] 传统增强扩边处理-逆：36->32
            imgB = img_o_B[:, :, 2:-2]      # [1, 136, 32]
        H, W = imgA.shape[1], imgA.shape[2]

        '''#### verify/enroll pts & desc ####'''
        pts_vf, desc_front_vf, desc_back_vf = pts_desc_decode(samples[index]['pts_vf'])     # (x, y), [n, 128], [n, 128]
        pts_en, desc_front_en, desc_back_en = pts_desc_decode(samples[index]['pts_en'])
        
        '''#### homo ####'''
        homo = torch.tensor(samples[index]['homo']).type(torch.FloatTensor)
        homo = 2 * homo / 512.  # csv中的数据需要乘上2再除以512
        vec_one = torch.tensor([[0, 0, 1]], dtype=torch.float32)
        homo = torch.cat((homo.reshape(2, 3), vec_one), dim=0)

        '''#### imgA warped ####'''
        imgA_warped = inv_warp_image(imgA.squeeze(), homo)
        imgA_o_warped = inv_warp_image(img_o_A.squeeze(), homo)
        imgA_warped_mask = compute_valid_mask(torch.tensor([H, W]), homography=homo)

        input  = {}
        input.update({'imgA': imgA, 'imgB': imgB, 'img_o_A': img_o_A, 'img_o_B': img_o_B})
        input.update({'pts_vf': pts_vf, 'pts_en': pts_en})
        input.update({'desc_front_vf': desc_front_vf, 'desc_back_vf': desc_back_vf})
        input.update({'desc_front_en': desc_front_en, 'desc_back_en': desc_back_en})
        input.update({'homo': homo})
        input.update({'imgA_warped': imgA_warped, 'imgA_warped_mask': imgA_warped_mask, 'imgA_o_warped': imgA_o_warped})        # 非模板按照csv中trans进行的旋转
        input.update({'name_vf': samples[index]['name_vf'], 'name_en': samples[index]['name_en']})

        return input

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def sort_pts_for_score(self, pnts, H, W, nms_dist = 2):
        
        mask_pts = (pnts[:,1] <= W - 1) * (pnts[:,2] <= H - 1)
        pnts = pnts[mask_pts] #.cpu().numpy()  
        
        pnts = pnts.transpose(1,0)
        pnts_t = pnts.clone()
        pnts_t[0,:] = pnts[1,:]
        pnts_t[1,:] = pnts[2,:]
        pnts_t[2,:] = pnts[0,:]
        
        # pnts_t = pnts_t.cpu().numpy()
        pnts_t, _ = self.nms_fast(np.array(pnts_t.cpu()), H, W, dist_thresh = nms_dist)  # Apply NMS.
        
        inds = np.argsort(pnts_t[2, :])
        pnts_t = pnts_t[:, inds[::-1]]  # Sort by confidence.        
        
        # 
        pnts = torch.tensor(pnts_t.transpose(1,0))
        temp = pnts[:,2].clone()
        pnts[:,2] = pnts[:,1]
        pnts[:,1] = pnts[:,0]
        pnts[:,0] = temp
        return pnts

    def pnts_resize_122_36_no_resetScore(self, pnts, oir_H, oir_W, resize_H, resize_W, nms_dist, bord, top_k) :

        top_k = 130
        nms_dist = 2
        bord = 3
        
        pnts = self.sort_pts_for_score(pnts, resize_H, resize_W, nms_dist) # pnts : 160 * 48
        
        # pnts : 尺寸还原到 120 * 36
        pnts[:,1:] = pnts[:,1:] * 0.75 # pnts : 120 * 36
        
        # pnts : 尺寸还原到 122 * 36
        pnts[:,2] = pnts[:,2] + 1      # pnts : 122 * 36
        
        # pts ： 剔除 [122,36] 边缘 2个 pixel 的点
        mask_pts = (pnts[:,1] >=bord) * (pnts[:,1] <= oir_W - 1 - bord) * (pnts[:,2] >=bord) * (pnts[:,2] <= oir_H - 1 - bord)
        pnts = pnts[mask_pts]
                
        # pts, 取 前 top_k 个点
        if top_k:
            if pnts.shape[0] > top_k:
                pnts = pnts[:top_k, :]
            pass 
        pass             
    
        return pnts

    def test_process(self, FPDT):

        count = 0
        repeat_out = 0.
        repeat_out_dis2 = 0.
        sift_totalrepeat_out = 0.
        sift_totalrepeat_out_dis2 = 0.
        content = []
        content_dis2 = []

        total_count = len(self.sift_succ_dict)
        for idx in tqdm(range(total_count)):
            count += 1
            # print("progress:{0}%".format(round((count + 1) * 100 / total_count)), end="\r")
            # print("it: ", count)
            sample = self.Extract_Succ_Csv_Other_enroll2verify(idx, self.sift_succ_dict)    # 点会做限制 注意修改
            imgA, imgB, mat_H, mat_H_fix = sample["imgA"].unsqueeze(0), sample["imgB"].unsqueeze(0), sample['homo'], sample['homo_fixed']   # mat_H: base on 122x36
            imgA_ne, imgB_ne = sample["imgA_ne"].unsqueeze(0), sample["imgB_ne"].unsqueeze(0)
            # imgA_warped_mask = sample['imgA_warped_mask']

            nameA = sample['name_vf'][:-4]
            nameB = sample['name_en'][:-4]
            nameAB = str(nameA) + '_' + str(nameB)
            # logging.info(f"nameA: {nameA} <-> nameB: {nameB}")

            '''NET'''
            # pass through network
            pts_A = FPDT.forward_self_supervised(imgA.to(self.device))
            pts_B = FPDT.forward_self_supervised(imgB.to(self.device))
            
            pts_A_ne = FPDT.forward_self_supervised(imgA_ne.to(self.device))
            pts_B_ne = FPDT.forward_self_supervised(imgB_ne.to(self.device))

            pts_A = torch.cat((pts_A, pts_A_ne),0)      # size: 160 * 48
            pts_B = torch.cat((pts_B, pts_B_ne),0)      # size: 160 * 48
            
            pts_A = self.pnts_resize_122_36_no_resetScore(pts_A, 122, 36, 160, 48, 2, 2, 130) #pnts : 128 * 40
            pts_B = self.pnts_resize_122_36_no_resetScore(pts_B, 122, 36, 160, 48, 2, 2, 130) #pnts : 128 * 40
            
            # heatmap_B = heatmap_B * imgA_warped_mask    # A-->B 后进行mask，只显示匹配区域
            # pts_A = getPtsFromHeatmap(heatmap.squeeze().detach().cpu().numpy(), self.conf_thresh, self.nms, soft_nms=False)
            # pts_B = getPtsFromHeatmap(heatmap_B.squeeze().detach().cpu().numpy(), self.conf_thresh, self.nms, soft_nms=False)  # pts_B 只包含匹配区域的预测点

            '''后续计算均还原到原图'''
            imgA, imgB = sample['img_o_A'].unsqueeze(0), sample['img_o_B'].unsqueeze(0)
            # if self.use_128x52:
            #     pts_A[:, 0] += 2    # 128x48 -> 128x52
            #     pts_B[:, 0] += 2
            # else:
            #     pts_A[:, 0] += 2    # 120x32 -> 122x36
            #     pts_A[:, 1] += 1
            #     pts_B[:, 0] += 2    # 120x32 -> 122x36
            #     pts_B[:, 1] += 1

            '''截断'''
            if self.top_k:
                if pts_A.shape[0] > self.top_k:
                    pts_A = pts_A[:self.top_k, :]
                if pts_B.shape[0] > self.top_k:
                    pts_B = pts_B[:self.top_k, :]

            # A,B点集
            H, W = imgA.shape[2], imgA.shape[3]   # img:[1,1,136,36]
            pts_A = pts_A.cpu()
            pts_B = pts_B.cpu()
            if self.use_128x52:
                mask_ptsA = (pts_A[:, 0] >= 0) * (pts_A[:, 0] < 36) * (pts_A[:, 1] >= 0) * (pts_A[:, 1] < 122)
                mask_ptsB = (pts_B[:, 0] >= 0) * (pts_B[:, 0] < 36) * (pts_B[:, 1] >= 0) * (pts_B[:, 1] < 122)
                pts_A = pts_A[mask_ptsA, :]
                pts_B = pts_B[mask_ptsB, :]
                warped_pnts = warp_points(pts_A, mat_H_fix)  # 利用变换矩阵变换坐标点
            else:
                warped_pnts = warp_points(pts_A[:,1:], mat_H)  # 122x36
            warped_pnts, _ = filter_points(warped_pnts.squeeze(), torch.tensor([W, H]), return_mask=True)
            # B_pnts_mask = points_to_2D(pts_B.cpu().squeeze().numpy().transpose(1, 0), H, W)
            # B_label = imgA_warped_mask.cpu().squeeze().numpy() * B_pnts_mask
            # pts_B = getPtsFromLabels2D(B_label)

            pred = {}
            pred.update({"pts": pts_A})   # 图像坐标系
            pred.update({'pts_B': pts_B})

            '''Tradition'''
            sift_ptsA, sift_ptsB = sample['pts_vf'].squeeze(), sample['pts_en'].squeeze()     # tensor 限制后的点 122x36
            if self.use_128x52:
                sift_ptsA[:, :2] += torch.tensor([8, 3])    # 122x36->128x52
                sift_ptsB[:, :2] += torch.tensor([8, 3]) 
            if self.top_k:
                if sift_ptsA.shape[0] > self.top_k:
                    sift_ptsA = sift_ptsA[:self.top_k, :]
                if sift_ptsB.shape[0] > self.top_k:
                    sift_ptsB = sift_ptsB[:self.top_k, :]
            if self.use_128x52:
                sift_warped_pts = warp_points(sift_ptsA[:, :2], mat_H_fix)
            else:
                sift_warped_pts = warp_points(sift_ptsA[:, :2], mat_H)
            sift_warped_pts, _ = filter_points(sift_warped_pts.squeeze(), torch.tensor([W, H]), return_mask=True)  # imgB points

            # tra_heatmapB = getLabel2DFromPts(tra_ptsB, H, W)  # to mask B's points
            # tra_heatmapB = tra_heatmapB * imgA_warped_mask.squeeze()
            # tra_warped_pts_masked = getPtsFromLabels2D(tra_heatmapB).transpose(1, 0) # heatmap -> imgB points with mask

            pred.update({"tra_pts": sift_ptsA})
            pred.update({'tra_pts_B': sift_ptsB})   # (x, y, 1)

            ## output images for visualization labels
            if self.output_images and count % 200 == 0:
                img_pair = {}
                '''NET'''
                img_2D_A = imgA.numpy().squeeze()
                img_2D_B = imgB.numpy().squeeze()
                img_pair.update({'img_1': img_2D_A, 'img_2': img_2D_B})

                img_pts = draw_keypoints_pair(img_pair, pred, warped_pnts.numpy(), radius=3, s=3)
                # label_pts = fe.getPtsFromLabels2D(label_2d.cpu().squeeze())
                f = self.output_dir / (nameAB + ".bmp")
                cv2.imwrite(str(f), img_pts)

                '''Tradition'''
                tra_img_pts = draw_keypoints_pair_tradition(img_pair, pred, sift_warped_pts.numpy(), radius=3, s=3)
                f = self.output_dir / (nameAB + '_tradition.bmp')
                cv2.imwrite(str(f), tra_img_pts)

                require_image = True
                if require_image:
                    img_2D_warped = sample['imgA_o_warped'].numpy().squeeze()
                    b = np.zeros_like(img_2D_A)
                    g = img_2D_warped * 255  #  旋转后的模板
                    r = img_2D_B * 255    
                    image_merge = cv2.merge([b, g, r])
                    image_merge = cv2.resize(image_merge, None, fx=1, fy=1)
                    cv2.imwrite(os.path.join(self.output_dir, nameAB + '_match.bmp'), image_merge)

            # =========================================dis1=========================================
            if self.output_ratio:   
                # to_floatTensor = lambda x: torch.tensor(x).type(torch.FloatTensor)
                eps = 1e-6
                '''NET'''
                # pts_B = torch.tensor(pts_B, dtype=torch.float32)    # 预测点
                match_idx, mask_idx = get_point_pair_repeat(warped_pnts.squeeze(), pts_B[:,1:], correspond=1)
                repeat_ratio = len(set(match_idx.numpy().tolist())) / (len(warped_pnts.squeeze()) + eps)
                # recall_ratio = len(set(match_idx)) / len(pts_B.squeeze())
                repeat_out += repeat_ratio
                # print("repetitive rate:{:f}".format(repeat_ratio))

                '''Tradition'''
                # tra_ptsB = torch.tensor(tra_ptsB).type(torch.FloatTensor)   # sift点
                tra_match_idx, mask_idx = get_point_pair_repeat(sift_warped_pts.squeeze(), sift_ptsB[:, :2], correspond=1)
                sift_repeat_ratio = len(set(tra_match_idx.numpy().tolist())) / (len(sift_warped_pts.squeeze()) + eps)

                sift_totalrepeat_out += sift_repeat_ratio
                # print("traditional repetitive rate:{:f}".format(repeat_ratio_tra))

            '''results .csv'''
            numA, numB = pred['pts'].shape[0], pred['pts_B'].shape[0]
            sift_numA, sift_numB = pred['tra_pts'].shape[0], pred['tra_pts_B'].shape[0]
            numerator, denominator = len(set(match_idx.numpy().tolist())), len(warped_pnts.squeeze())    # 分子,分母
            sift_numerator, sift_denominator = len(set(tra_match_idx.numpy().tolist())), len(sift_warped_pts.squeeze())
            content.append([nameA, nameB, repeat_ratio, numerator, denominator, numA, numB, sift_repeat_ratio, sift_numerator, sift_denominator, sift_numA, sift_numB])

            # ==========================================dis2========================================
            eps = 1e-6
            '''NET'''
            # pts_B = torch.tensor(pts_B, dtype=torch.float32)    # 预测点
            match_idx, mask_idx = get_point_pair_repeat(warped_pnts.squeeze(), pts_B[:,1:], correspond=2)
            repeat_ratio = len(set(match_idx.numpy().tolist())) / (len(warped_pnts.squeeze()) + eps)
            # recall_ratio = len(set(match_idx)) / len(pts_B.squeeze())
            repeat_out_dis2 += repeat_ratio
            # print("repetitive rate:{:f}".format(repeat_ratio))

            '''Tradition'''
            # tra_ptsB = torch.tensor(tra_ptsB).type(torch.FloatTensor)   # sift点
            tra_match_idx, mask_idx = get_point_pair_repeat(sift_warped_pts.squeeze(), sift_ptsB[:, :2], correspond=2)
            sift_repeat_ratio = len(set(tra_match_idx.numpy().tolist())) / (len(sift_warped_pts.squeeze()) + eps)

            sift_totalrepeat_out_dis2 += sift_repeat_ratio

            '''results .csv'''
            numA, numB = pred['pts'].shape[0], pred['pts_B'].shape[0]
            sift_numA, sift_numB = pred['tra_pts'].shape[0], pred['tra_pts_B'].shape[0]
            numerator, denominator = len(set(match_idx.numpy().tolist())), len(warped_pnts.squeeze())    # 分子,分母
            sift_numerator, sift_denominator = len(set(tra_match_idx.numpy().tolist())), len(sift_warped_pts.squeeze())
            content_dis2.append([nameA, nameB, repeat_ratio, numerator, denominator, numA, numB, sift_repeat_ratio, sift_numerator, sift_denominator, sift_numA, sift_numB])


        df = pd.DataFrame(content, 
                columns=[
                    'nameA',
                    'nameB',
                    'net repeat ratio',
                    'net_numerator',
                    'net_denominator',
                    'net_numA',
                    'net_numB',
                    'sift repeat ratio',
                    'sift_numerator',
                    'sift_denominator',
                    'sift_numA',
                    'sift_numB'
                    ])
        df.to_csv(os.path.join(self.output_dir, 'results_dis1.csv'))

        df_dis2 = pd.DataFrame(content_dis2, 
                columns=[
                    'nameA',
                    'nameB',
                    'net repeat ratio',
                    'net_numerator',
                    'net_denominator',
                    'net_numA',
                    'net_numB',
                    'sift repeat ratio',
                    'sift_numerator',
                    'sift_denominator',
                    'sift_numA',
                    'sift_numB'
                    ])
        df_dis2.to_csv(os.path.join(self.output_dir, 'results_dis2.csv'))
        
        if count:   # 防止count为0
            print("output pseudo ground truth: ", count)
            # print("repetitive rate:{:f} ".format(repeat_out / count))      # 均值
            print("net repetitive ratio(dis<1):{:f}  traditional repetitive ratio(dis<1):{:f}".format(repeat_out / count, sift_totalrepeat_out / count))      # 均值
            print("net repetitive ratio(dis<2):{:f}  traditional repetitive ratio(dis<2):{:f}".format(repeat_out_dis2 / count, sift_totalrepeat_out_dis2 / count))      # 均值
        else:
            print("the count is 0, please check the path of the labels!!!")

        pass

    def __init__(self, img_path=None, info_path=None, device="cpu", **config):

        self.device         = device
        self.top_k          = config['top_k']
        self.output_images  = config['output_images']
        self.conf_thresh    = config['detec_thre']
        self.nms            = config['nms']

        self.isDilation     = config['isDilation']  # True: 扩充36->40  False: 裁剪36->32

        self.is256          = config['is256']
        self.output_dir     = Path(config['output_dir']) / 'val'
        os.makedirs(self.output_dir, exist_ok=True)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((self.sizer[0], self.sizer[1])),
        ])

        if img_path is not None:
            print("load dataset_files from: ", img_path)
            # ------ matched images ------
            names = []
            image_paths = []

            e = [str(p) for p in Path(img_path).iterdir()]
            n = [(p.split('/')[-1])[:-4] for p in e]
            image_paths.extend(e)
            names.extend(n)

            # ------ dic ------
            files = {'image_paths': image_paths, 'names': names}
        
            sequence_set = []
            for (img, name) in zip(files['image_paths'], files['names']):   # zip返回元组 （成对的bmp&npz）
                sample = {'image': img, 'name': name}   # 无标签时，只有img数据
                sequence_set.append(sample)
            self.samples = sequence_set

        self.IMG = torch.tensor([
            125,  122,  113,   97,   83,   75,   82,   94,  111,  128,  132,  136,  130,  122,  117,  115,  122,  134,  152,  163,  156,  143,  116,   91,   81,   80,   90,  100,  107,  111,  110,  110,  115,  124,  134,  134,  119,  102,   93,   94,  118,  148,  170,  183,  169,  154,  140,  125,  117,  108,  104,   99,
            137,  142,  134,  113,   93,   77,   74,   78,   90,  102,  108,  112,  112,  106,  104,  100,  108,  122,  146,  161,  159,  146,  120,   91,   79,   77,   86,   97,  105,  107,  109,  109,  116,  128,  141,  140,  122,   97,   81,   76,   92,  119,  144,  159,  156,  150,  154,  149,  144,  132,  117,  107,
            139,  148,  145,  130,  109,   88,   74,   70,   72,   78,   80,   83,   83,   81,   80,   78,   84,   99,  123,  145,  148,  138,  112,   84,   69,   65,   69,   79,   87,   92,   96,  100,  107,  122,  133,  136,  120,   94,   72,   59,   61,   75,   91,  107,  116,  128,  151,  162,  160,  149,  126,  109,
            132,  146,  149,  143,  133,  112,   93,   81,   81,   76,   75,   73,   64,   57,   54,   54,   59,   71,   92,  117,  132,  127,  105,   78,   62,   57,   63,   73,   88,   94,   95,   95,   95,  103,  111,  118,  118,  113,  104,   86,   60,   55,   60,   73,   90,  109,  142,  163,  166,  157,  134,  114,
            120,  134,  142,  147,  151,  141,  120,  103,   98,   89,   84,   77,   67,   56,   52,   53,   61,   76,  100,  127,  141,  133,  105,   75,   58,   54,   63,   82,  105,  116,  115,  105,   94,   89,   92,   99,  106,  112,  113,  101,   71,   58,   55,   60,   70,   88,  121,  153,  169,  168,  147,  125,
            105,  114,  123,  137,  151,  153,  144,  131,  128,  117,  107,   94,   77,   60,   53,   55,   65,   84,  115,  144,  156,  143,  108,   74,   55,   53,   66,   93,  128,  145,  143,  125,   98,   79,   74,   79,   91,  108,  124,  122,  100,   77,   63,   57,   63,   73,   97,  132,  158,  164,  151,  127,
            96,   95,   99,  109,  124,  141,  149,  150,  152,  148,  136,  116,   93,   67,   56,   57,   70,   94,  132,  162,  173,  155,  114,   75,   55,   53,   69,  104,  149,  172,  171,  148,  109,   76,   63,   64,   77,  103,  134,  144,  130,  102,   77,   62,   63,   67,   80,  111,  141,  154,  148,  128,
            92,   82,   79,   80,   92,  112,  132,  148,  163,  171,  164,  142,  110,   76,   58,   58,   72,  101,  147,  179,  188,  167,  120,   76,   54,   52,   69,  110,  163,  191,  193,  170,  121,   77,   55,   52,   64,   96,  140,  162,  158,  132,   98,   72,   64,   61,   70,   94,  124,  142,  146,  129,
            101,   87,   77,   70,   73,   86,  104,  132,  161,  180,  180,  161,  123,   82,   59,   57,   71,  103,  154,  189,  198,  177,  126,   78,   53,   50,   66,  108,  166,  199,  205,  184,  132,   79,   52,   45,   56,   90,  141,  172,  174,  155,  120,   87,   69,   63,   69,   88,  111,  130,  140,  131,
            118,  103,   87,   67,   64,   69,   84,  114,  146,  174,  182,  165,  128,   83,   57,   53,   67,  100,  156,  194,  203,  183,  131,   80,   53,   47,   61,   99,  158,  198,  208,  189,  137,   82,   51,   43,   52,   86,  139,  173,  180,  169,  142,  109,   85,   72,   74,   86,  104,  122,  139,  133,
            131,  119,  101,   77,   68,   67,   74,   99,  126,  160,  176,  164,  125,   79,   53,   48,   62,   96,  154,  194,  206,  187,  136,   84,   53,   46,   56,   89,  146,  191,  205,  188,  137,   82,   50,   42,   51,   83,  134,  169,  180,  178,  160,  134,  106,   84,   77,   82,   96,  113,  131,  130,
            139,  135,  118,   91,   77,   67,   70,   88,  114,  150,  170,  158,  120,   74,   49,   44,   57,   92,  151,  194,  206,  189,  141,   88,   55,   45,   52,   81,  135,  183,  200,  183,  133,   79,   49,   42,   50,   81,  127,  162,  176,  181,  171,  152,  125,   98,   85,   83,   94,  109,  126,  126,
            140,  139,  127,  104,   87,   75,   75,   91,  116,  149,  167,  153,  115,   70,   46,   42,   55,   89,  148,  191,  205,  190,  145,   92,   57,   45,   50,   76,  128,  176,  194,  177,  127,   76,   48,   42,   50,   79,  121,  153,  170,  178,  174,  161,  138,  110,   92,   84,   93,  106,  120,  121,
            136,  138,  130,  110,   94,   81,   82,   98,  127,  157,  169,  152,  114,   70,   46,   42,   54,   87,  144,  187,  202,  188,  146,   94,   59,   47,   50,   74,  124,  171,  189,  170,  122,   73,   48,   42,   50,   79,  118,  147,  162,  168,  163,  153,  134,  111,   95,   87,   94,  102,  113,  114,
            134,  136,  129,  110,   96,   86,   87,  108,  141,  168,  173,  153,  115,   72,   48,   43,   53,   85,  138,  180,  196,  182,  143,   95,   61,   48,   50,   73,  121,  166,  184,  166,  119,   72,   48,   42,   51,   80,  118,  145,  155,  153,  141,  130,  120,  107,   98,   92,   96,  100,  108,  110,
            131,  135,  128,  108,   95,   85,   89,  113,  150,  174,  176,  154,  117,   75,   51,   45,   54,   83,  131,  172,  187,  175,  139,   93,   61,   48,   50,   72,  118,  162,  180,  164,  118,   73,   48,   43,   52,   81,  122,  148,  152,  138,  116,  102,   99,   97,   98,   98,  102,  100,  104,  104,
            131,  136,  131,  112,   95,   84,   86,  113,  149,  173,  175,  155,  117,   77,   53,   46,   54,   80,  125,  163,  178,  166,  133,   91,   61,   48,   50,   70,  113,  158,  178,  164,  120,   75,   50,   43,   52,   82,  125,  153,  154,  130,  101,   83,   81,   88,   97,  106,  110,  105,  103,  102,
            131,  139,  134,  113,   94,   78,   81,  108,  142,  166,  170,  150,  114,   76,   53,   46,   53,   79,  121,  157,  171,  159,  127,   88,   60,   48,   49,   67,  109,  155,  177,  164,  122,   77,   51,   43,   52,   83,  128,  157,  157,  128,   91,   71,   71,   82,   99,  113,  120,  111,  105,  101,
            132,  140,  136,  114,   93,   75,   75,   99,  130,  155,  161,  144,  109,   74,   52,   46,   53,   79,  120,  156,  168,  156,  123,   85,   59,   47,   48,   66,  106,  152,  175,  163,  122,   78,   51,   44,   52,   84,  130,  160,  157,  127,   90,   68,   69,   82,   99,  119,  130,  122,  112,  106,
            130,  139,  132,  109,   89,   69,   70,   89,  118,  142,  152,  135,  105,   71,   50,   45,   53,   80,  122,  158,  170,  155,  120,   82,   57,   46,   47,   65,  104,  149,  171,  158,  118,   76,   51,   45,   55,   87,  132,  159,  153,  122,   87,   67,   67,   81,  101,  125,  142,  133,  119,  109,
            129,  135,  126,  103,   85,   68,   66,   84,  111,  136,  146,  130,  101,   69,   50,   45,   53,   81,  126,  163,  174,  158,  120,   80,   55,   45,   47,   66,  105,  148,  167,  151,  110,   71,   49,   46,   59,   93,  136,  157,  146,  114,   83,   66,   65,   81,  103,  131,  151,  143,  125,  114,
            124,  128,  116,   93,   81,   67,   67,   84,  113,  137,  146,  128,  100,   68,   50,   45,   54,   83,  130,  169,  180,  162,  121,   79,   54,   45,   47,   67,  107,  148,  162,  143,  101,   66,   48,   48,   65,  102,  141,  155,  139,  104,   76,   64,   64,   81,  104,  134,  155,  145,  126,  112,
            119,  119,  107,   85,   76,   69,   70,   92,  123,  145,  148,  128,  100,   68,   50,   46,   56,   86,  136,  175,  186,  167,  122,   79,   54,   45,   48,   70,  111,  149,  159,  136,   95,   62,   47,   51,   73,  113,  149,  154,  133,   95,   70,   61,   63,   84,  109,  138,  155,  144,  122,  110,
            109,  105,   95,   77,   73,   73,   81,  107,  140,  157,  154,  130,   99,   67,   50,   48,   59,   91,  143,  182,  191,  169,  122,   78,   53,   45,   50,   72,  114,  150,  158,  133,   91,   60,   48,   54,   81,  124,  155,  153,  126,   86,   65,   57,   63,   87,  114,  140,  150,  134,  113,  103,
            101,   92,   85,   73,   71,   80,   95,  126,  155,  168,  158,  129,   97,   66,   50,   49,   63,   98,  152,  187,  193,  167,  118,   75,   53,   46,   52,   76,  117,  151,  157,  131,   90,   60,   49,   59,   90,  134,  160,  151,  119,   79,   61,   56,   67,   97,  127,  148,  149,  127,  104,   95,
            95,   83,   80,   72,   72,   87,  114,  148,  169,  173,  157,  123,   90,   64,   50,   51,   67,  106,  160,  191,  190,  159,  110,   70,   51,   47,   56,   82,  122,  153,  153,  127,   89,   61,   51,   64,   99,  141,  161,  146,  109,   71,   58,   57,   72,  105,  138,  149,  141,  113,   94,   89,
            95,   80,   79,   75,   79,  100,  133,  166,  176,  172,  148,  113,   83,   61,   50,   52,   72,  115,  167,  191,  183,  147,   97,   64,   50,   49,   62,   91,  128,  154,  152,  122,   83,   59,   53,   69,  107,  147,  160,  139,  100,   67,   59,   64,   81,  118,  144,  147,  131,  103,   87,   87,
            97,   82,   82,   82,   90,  116,  149,  174,  173,  161,  133,   98,   75,   58,   51,   55,   80,  126,  173,  190,  174,  130,   84,   58,   49,   54,   71,  104,  140,  157,  146,  111,   74,   54,   54,   74,  114,  150,  155,  129,   91,   64,   64,   78,   99,  134,  152,  144,  119,   91,   84,   87,
            101,   85,   87,   91,  104,  132,  159,  174,  163,  144,  114,   83,   67,   56,   52,   60,   90,  138,  178,  186,  161,  113,   72,   54,   50,   60,   85,  120,  150,  157,  136,   96,   64,   51,   55,   80,  122,  152,  150,  120,   85,   67,   72,   97,  123,  154,  159,  139,  105,   82,   82,   90,
            106,   89,   91,   95,  109,  134,  156,  163,  146,  122,   93,   71,   61,   56,   55,   67,  100,  148,  182,  181,  147,   98,   64,   51,   53,   69,  100,  134,  155,  151,  120,   81,   55,   48,   58,   90,  132,  155,  146,  112,   81,   69,   82,  115,  143,  163,  157,  127,   93,   76,   84,   94,
            108,   92,   93,   97,  107,  129,  146,  148,  128,  104,   79,   63,   57,   55,   57,   73,  109,  155,  182,  174,  134,   87,   58,   50,   55,   78,  112,  143,  155,  139,  102,   67,   49,   49,   66,  105,  146,  161,  144,  106,   78,   71,   89,  124,  151,  161,  147,  115,   85,   73,   85,   98,
            109,   92,   91,   91,  100,  120,  135,  133,  115,   91,   70,   59,   56,   56,   59,   77,  114,  158,  180,  165,  122,   78,   54,   49,   59,   84,  120,  146,  149,  124,   85,   56,   46,   53,   80,  124,  161,  167,  140,   99,   73,   69,   89,  124,  154,  156,  135,  101,   78,   72,   89,  102,
            107,   89,   88,   88,   96,  117,  130,  127,  110,   86,   67,   57,   56,   56,   61,   79,  116,  157,  174,  155,  110,   70,   51,   49,   62,   91,  126,  148,  142,  109,   71,   50,   46,   61,   97,  144,  172,  167,  132,   90,   67,   68,   90,  123,  154,  148,  121,   90,   74,   74,   92,  105,
            102,   85,   84,   84,   95,  116,  130,  126,  110,   85,   66,   56,   56,   56,   62,   82,  119,  157,  167,  142,   97,   62,   48,   50,   66,   99,  135,  150,  136,   99,   62,   47,   48,   71,  114,  156,  173,  157,  116,   79,   63,   70,   96,  127,  154,  141,  110,   80,   73,   76,   96,  110,
            99,   82,   81,   83,   95,  118,  133,  131,  112,   87,   66,   56,   55,   57,   66,   89,  128,  158,  160,  127,   83,   55,   46,   53,   74,  111,  146,  154,  133,   91,   57,   46,   53,   82,  127,  160,  162,  135,   95,   68,   63,   78,  107,  135,  154,  138,  105,   77,   72,   80,   99,  113,
            95,   78,   79,   82,   96,  122,  139,  135,  114,   86,   65,   53,   54,   59,   73,  103,  141,  161,  151,  110,   69,   49,   46,   58,   87,  129,  158,  158,  128,   83,   54,   47,   61,   94,  136,  155,  143,  109,   76,   61,   66,   92,  123,  147,  156,  135,  102,   74,   73,   82,  105,  120,
            93,   76,   77,   83,   99,  128,  145,  139,  112,   84,   62,   51,   53,   63,   85,  122,  156,  164,  140,   93,   57,   44,   46,   65,  104,  147,  168,  157,  118,   75,   51,   51,   70,  107,  141,  146,  122,   86,   62,   58,   71,  104,  135,  153,  156,  131,   98,   72,   73,   87,  111,  127,
            91,   74,   77,   84,  102,  134,  150,  139,  109,   79,   58,   49,   54,   69,  101,  141,  167,  162,  124,   76,   48,   41,   49,   77,  124,  162,  172,  150,  104,   65,   49,   56,   82,  120,  145,  138,  105,   72,   55,   58,   76,  113,  143,  154,  152,  120,   92,   71,   74,   90,  118,  134,
            92,   75,   78,   87,  107,  140,  154,  139,  104,   74,   55,   48,   56,   77,  116,  157,  173,  155,  107,   62,   41,   40,   56,   93,  142,  171,  169,  135,   88,   57,   48,   61,   94,  132,  148,  131,   95,   65,   53,   61,   81,  120,  148,  153,  146,  110,   84,   68,   76,   98,  127,  141,
            91,   75,   78,   89,  112,  144,  155,  134,   98,   68,   52,   48,   60,   87,  132,  168,  174,  144,   91,   52,   38,   43,   66,  111,  156,  174,  159,  116,   73,   50,   49,   68,  106,  143,  151,  128,   89,   61,   53,   65,   90,  130,  154,  152,  134,   92,   74,   67,   83,  108,  138,  149,
            92,   77,   81,   93,  119,  149,  155,  131,   91,   62,   48,   49,   65,   99,  145,  175,  171,  132,   79,   46,   37,   47,   80,  129,  165,  169,  142,   97,   61,   48,   53,   79,  120,  152,  154,  124,   85,   60,   55,   72,  104,  144,  161,  151,  121,   78,   67,   71,   94,  123,  149,  156,
            91,   77,   82,   97,  125,  151,  152,  123,   82,   56,   46,   50,   71,  111,  158,  180,  167,  121,   70,   43,   38,   55,   95,  143,  167,  158,  121,   79,   54,   50,   62,   95,  136,  160,  153,  118,   79,   58,   59,   83,  120,  154,  164,  147,  106,   66,   62,   77,  109,  137,  157,  160,
            92,   78,   84,  102,  130,  152,  149,  118,   77,   52,   44,   53,   79,  125,  168,  182,  161,  111,   63,   42,   42,   65,  110,  152,  163,  141,  100,   67,   53,   57,   78,  115,  151,  163,  145,  107,   72,   58,   67,   99,  140,  165,  159,  130,   92,   61,   62,   89,  126,  151,  164,  163,
            89,   78,   86,  109,  137,  154,  144,  109,   72,   50,   44,   57,   90,  139,  176,  181,  152,  100,   58,   43,   48,   78,  125,  156,  155,  123,   83,   61,   57,   71,  100,  135,  160,  157,  127,   90,   65,   61,   81,  121,  158,  166,  145,  105,   77,   58,   66,  100,  142,  161,  165,  160,
            89,   80,   91,  118,  143,  154,  139,  104,   69,   51,   48,   66,  104,  152,  180,  175,  138,   88,   54,   46,   58,   94,  138,  156,  142,  106,   74,   62,   69,   92,  124,  149,  158,  138,  104,   74,   61,   70,  102,  145,  169,  160,  125,   83,   67,   61,   74,  115,  156,  169,  163,  155,
            87,   83,   98,  127,  149,  153,  130,   93,   67,   53,   56,   79,  122,  163,  178,  162,  118,   75,   53,   52,   73,  112,  145,  149,  125,   93,   73,   73,   90,  117,  146,  155,  141,  111,   81,   63,   65,   88,  130,  166,  172,  147,  104,   67,   62,   65,   86,  126,  163,  168,  156,  145,
            90,   91,  112,  141,  155,  148,  117,   84,   65,   59,   69,   98,  141,  171,  172,  142,   97,   64,   53,   63,   92,  128,  146,  135,  107,   84,   79,   92,  115,  137,  153,  140,  113,   83,   65,   63,   80,  117,  158,  178,  166,  129,   87,   60,   60,   74,   99,  138,  166,  165,  148,  135,
            92,  102,  128,  153,  157,  138,  104,   75,   64,   64,   85,  121,  160,  175,  160,  120,   78,   57,   56,   75,  108,  135,  139,  119,   91,   80,   90,  114,  141,  152,  140,  113,   83,   64,   61,   74,  106,  149,  177,  179,  152,  108,   73,   56,   61,   81,  113,  148,  167,  157,  134,  119,
            102,  119,  146,  164,  159,  130,   92,   67,   62,   72,  102,  143,  175,  176,  146,  100,   64,   53,   59,   85,  118,  135,  128,  107,   89,   93,  113,  138,  152,  143,  115,   83,   63,   57,   67,   96,  138,  173,  185,  171,  134,   91,   67,   58,   65,   92,  127,  159,  169,  152,  121,  104,
            113,  135,  160,  169,  152,  114,   81,   62,   63,   80,  119,  162,  184,  174,  133,   84,   55,   50,   66,   91,  115,  124,  116,  105,  104,  117,  138,  152,  145,  118,   84,   60,   53,   59,   84,  125,  165,  185,  181,  155,  115,   79,   65,   63,   70,  101,  139,  164,  166,  139,  107,   91,
            128,  150,  166,  164,  137,   99,   72,   60,   65,   91,  135,  174,  188,  169,  121,   74,   50,   47,   65,   89,  109,  121,  123,  124,  132,  146,  157,  151,  125,   88,   60,   49,   51,   70,  108,  153,  181,  185,  168,  133,   98,   72,   66,   70,   79,  113,  147,  163,  156,  126,   95,   82,
            144,  160,  165,  151,  119,   82,   64,   60,   71,  103,  151,  183,  186,  159,  108,   66,   50,   50,   63,   85,  109,  131,  146,  156,  165,  170,  162,  136,   97,   64,   47,   44,   57,   90,  137,  174,  186,  175,  144,  107,   81,   65,   67,   74,   91,  126,  154,  157,  141,  109,   86,   78,
            157,  168,  164,  139,  100,   68,   58,   65,   82,  120,  164,  187,  179,  144,   95,   59,   47,   48,   62,   88,  122,  156,  178,  188,  190,  180,  154,  112,   73,   50,   42,   48,   72,  117,  162,  185,  181,  154,  114,   82,   65,   61,   68,   82,  109,  142,  159,  148,  122,   94,   80,   78,
            166,  168,  153,  120,   86,   61,   59,   74,  100,  141,  176,  186,  167,  125,   82,   54,   46,   49,   66,  100,  146,  184,  203,  207,  199,  175,  133,   88,   57,   44,   43,   59,   96,  146,  180,  186,  165,  125,   85,   64,   55,   60,   74,   96,  131,  153,  158,  135,  105,   83,   81,   83,
            163,  154,  129,   97,   72,   58,   62,   88,  124,  163,  185,  182,  151,  106,   70,   51,   46,   54,   77,  122,  173,  205,  216,  212,  193,  155,  107,   69,   49,   44,   51,   79,  126,  169,  187,  176,  140,   94,   65,   54,   53,   67,   88,  117,  150,  159,  150,  121,   92,   78,   83,   90,
            157,  140,  108,   77,   62,   58,   70,  107,  151,  183,  191,  175,  133,   91,   62,   49,   49,   62,   95,  148,  193,  215,  218,  205,  173,  125,   82,   58,   49,   51,   69,  107,  154,  183,  183,  156,  110,   72,   54,   52,   60,   82,  108,  136,  161,  156,  138,  106,   84,   78,   92,  102,
            144,  122,   90,   64,   56,   62,   83,  131,  174,  196,  189,  160,  114,   77,   57,   49,   54,   75,  118,  169,  204,  215,  208,  183,  139,   95,   67,   55,   55,   68,   97,  139,  174,  183,  167,  127,   85,   59,   53,   59,   77,  101,  123,  142,  157,  144,  122,   96,   82,   83,  101,  113,
            127,  101,   76,   55,   55,   75,  108,  157,  189,  197,  177,  138,   95,   67,   53,   51,   62,   92,  141,  185,  206,  206,  187,  148,  105,   75,   63,   63,   74,   98,  132,  165,  178,  171,  140,  100,   70,   58,   61,   75,   93,  110,  124,  132,  146,  133,  114,   94,   85,   89,  110,  122,
            116,   90,   71,   56,   61,   97,  144,  182,  194,  189,  158,  115,   77,   59,   51,   55,   74,  113,  160,  192,  200,  186,  152,  111,   80,   68,   69,   82,  106,  135,  161,  173,  168,  145,  111,   83,   68,   68,   78,   91,  100,  107,  113,  118,  133,  127,  115,  102,   95,   99,  115,  127,
            107,   82,   73,   65,   75,  121,  173,  199,  192,  171,  133,   90,   64,   53,   51,   62,   89,  133,  173,  190,  183,  154,  114,   83,   70,   72,   89,  116,  144,  165,  171,  162,  141,  114,   91,   80,   80,   88,   97,  102,   97,   95,   99,  110,  130,  130,  122,  109,  104,  104,  118,  127,
            101,   78,   76,   84,  104,  149,  186,  202,  183,  151,  107,   72,   56,   50,   54,   72,  108,  152,  180,  180,  155,  116,   83,   69,   71,   89,  121,  153,  172,  174,  159,  134,  108,   91,   86,   93,  104,  111,  110,  100,   87,   84,   94,  117,  139,  143,  132,  115,  105,  104,  116,  124,
            96,   77,   81,   99,  131,  167,  187,  189,  163,  123,   84,   58,   51,   50,   60,   87,  129,  167,  178,  161,  123,   85,   66,   65,   82,  117,  154,  177,  178,  160,  128,   99,   84,   83,   98,  118,  131,  127,  110,   90,   76,   79,   99,  129,  150,  148,  133,  113,  101,   99,  112,  120,
            95,   79,   86,  111,  147,  175,  180,  167,  135,   98,   68,   51,   49,   53,   71,  107,  150,  175,  170,  136,   93,   65,   58,   70,  102,  145,  176,  182,  164,  128,   92,   74,   74,   93,  124,  147,  149,  130,  101,   76,   69,   80,  107,  137,  154,  143,  124,  104,   94,   96,  110,  119,
            91,   80,   91,  118,  148,  164,  160,  136,  105,   77,   59,   50,   50,   59,   86,  128,  164,  175,  153,  110,   71,   54,   58,   82,  126,  167,  181,  169,  132,   91,   67,   63,   80,  116,  152,  166,  153,  120,   85,   66,   63,   81,  110,  133,  149,  129,  107,   90,   86,   91,  110,  120,
            89,   80,   94,  119,  139,  145,  136,  113,   89,   70,   57,   51,   53,   69,  103,  146,  172,  167,  132,   86,   57,   51,   65,  102,  149,  176,  172,  140,   95,   65,   55,   66,   99,  143,  171,  170,  141,  101,   71,   58,   61,   81,  107,  124,  136,  116,   94,   83,   84,   93,  110,  121,
            85,   78,   91,  113,  132,  134,  120,  100,   83,   69,   58,   53,   57,   80,  120,  158,  171,  152,  107,   67,   50,   54,   80,  127,  165,  174,  149,  104,   68,   52,   56,   81,  125,  164,  175,  157,  117,   79,   59,   55,   63,   85,  107,  117,  126,  103,   85,   77,   86,   97,  114,  126,
            85,   76,   87,  108,  126,  132,  119,  102,   87,   72,   59,   56,   63,   93,  134,  163,  162,  129,   84,   55,   48,   64,  102,  149,  171,  159,  119,   77,   54,   52,   68,  106,  149,  171,  164,  130,   89,   63,   54,   58,   71,   95,  112,  118,  121,   95,   78,   73,   86,  103,  121,  133,
            83,   76,   89,  112,  131,  138,  125,  104,   91,   70,   58,   57,   69,  105,  144,  161,  146,  105,   67,   49,   53,   80,  126,  162,  165,  135,   92,   61,   53,   63,   92,  134,  164,  167,  140,   99,   68,   54,   55,   69,   90,  112,  122,  118,  113,   84,   74,   75,   93,  112,  131,  140,
            87,   82,   98,  126,  146,  151,  132,  105,   88,   65,   55,   59,   77,  115,  146,  150,  124,   84,   57,   50,   65,  103,  146,  164,  149,  109,   73,   58,   63,   86,  124,  157,  168,  148,  110,   74,   56,   54,   66,   91,  116,  131,  128,  114,   99,   71,   67,   78,  102,  126,  144,  148,
            92,   94,  116,  146,  157,  152,  128,   96,   78,   58,   53,   62,   85,  119,  139,  131,   99,   68,   54,   58,   84,  125,  155,  155,  124,   87,   66,   65,   84,  120,  154,  170,  158,  124,   84,   60,   54,   62,   87,  119,  140,  141,  125,  100,   85,   64,   66,   87,  119,  141,  153,  152,
            108,  119,  144,  165,  164,  145,  111,   79,   65,   54,   53,   68,   91,  117,  124,  106,   79,   61,   59,   75,  107,  140,  151,  132,   99,   74,   68,   83,  116,  153,  173,  168,  138,   97,   66,   55,   60,   80,  115,  144,  152,  138,  112,   83,   73,   64,   73,  106,  140,  155,  157,  152,
            130,  145,  163,  169,  156,  124,   91,   67,   56,   53,   60,   77,   96,  110,  105,   86,   68,   64,   76,  102,  130,  143,  131,  104,   80,   70,   81,  112,  150,  174,  175,  152,  112,   76,   58,   57,   73,  106,  142,  159,  153,  125,   95,   70,   66,   70,   88,  125,  158,  164,  153,  143,
            153,  165,  169,  158,  128,   94,   73,   60,   55,   60,   72,   89,  101,  102,   91,   77,   71,   80,  100,  124,  137,  127,  105,   81,   71,   79,  106,  144,  173,  179,  162,  125,   86,   62,   55,   65,   93,  133,  160,  162,  141,  106,   80,   65,   67,   87,  115,  150,  170,  163,  140,  126,
            165,  168,  158,  132,  101,   73,   62,   60,   61,   73,   89,  104,  107,  100,   89,   83,   88,  104,  121,  130,  121,  101,   79,   68,   75,   99,  136,  167,  178,  166,  133,   93,   65,   54,   58,   80,  118,  153,  166,  153,  122,   86,   69,   63,   70,  101,  140,  166,  169,  146,  119,  106,
            159,  150,  127,   99,   77,   63,   60,   68,   75,   94,  113,  125,  121,  110,  102,  101,  111,  122,  126,  115,   94,   74,   64,   69,   91,  127,  160,  175,  167,  137,   97,   66,   52,   52,   68,  101,  140,  163,  160,  132,  100,   71,   63,   65,   79,  117,  153,  169,  160,  130,  101,   91,
            138,  121,   94,   71,   62,   61,   69,   85,  102,  126,  143,  148,  138,  128,  122,  124,  127,  124,  110,   88,   68,   59,   64,   83,  117,  152,  171,  167,  139,  100,   67,   51,   49,   59,   86,  126,  156,  162,  141,  105,   77,   59,   61,   72,   99,  137,  162,  161,  139,  105,   85,   80,
            109,   90,   73,   60,   59,   69,   86,  114,  141,  161,  168,  168,  157,  147,  140,  135,  125,  107,   84,   64,   55,   59,   76,  109,  144,  166,  167,  143,  105,   69,   50,   46,   53,   76,  114,  149,  162,  148,  114,   79,   61,   56,   66,   90,  130,  159,  167,  147,  112,   84,   76,   77,
            87,   70,   65,   62,   68,   89,  116,  150,  176,  186,  183,  177,  166,  155,  143,  128,  106,   82,   63,   53,   57,   72,  103,  138,  162,  166,  147,  111,   75,   52,   45,   49,   69,  104,  142,  162,  155,  123,   86,   62,   53,   61,   82,  115,  155,  168,  159,  125,   92,   75,   77,   81,
            83,   65,   66,   74,   90,  123,  156,  183,  195,  195,  187,  174,  160,  145,  128,  105,   83,   64,   54,   57,   72,  100,  133,  156,  164,  149,  119,   82,   58,   47,   48,   63,   95,  136,  162,  162,  136,   96,   66,   56,   56,   77,  108,  142,  169,  163,  138,  101,   77,   71,   81,   90,
            84,   71,   77,   96,  123,  157,  183,  197,  195,  187,  172,  153,  137,  119,  100,   81,   66,   58,   61,   75,  101,  130,  152,  160,  148,  122,   89,   64,   50,   49,   60,   88,  127,  157,  164,  146,  109,   75,   59,   60,   73,  106,  141,  162,  170,  145,  113,   80,   68,   74,   96,  107,
            95,   88,   99,  129,  157,  180,  192,  195,  181,  162,  141,  119,  104,   89,   76,   66,   62,   67,   81,  105,  132,  150,  156,  144,  121,   91,   68,   55,   51,   59,   81,  116,  149,  162,  153,  120,   84,   64,   60,   75,  105,  143,  165,  166,  155,  117,   90,   67,   65,   84,  113,  126,
            104,  105,  126,  155,  175,  184,  182,  170,  147,  122,  102,   84,   76,   68,   64,   65,   74,   90,  114,  136,  150,  152,  139,  117,   90,   69,   58,   55,   62,   80,  109,  138,  154,  151,  125,   92,   68,   61,   72,  100,  137,  164,  170,  154,  130,   89,   73,   65,   75,  101,  133,  143,
            119,  127,  148,  166,  175,  172,  155,  132,  107,   84,   71,   62,   61,   62,   67,   81,  101,  125,  144,  153,  149,  133,  111,   86,   68,   60,   59,   68,   86,  111,  135,  146,  144,  122,   94,   71,   61,   68,   92,  128,  158,  168,  155,  127,  104,   72,   66,   72,   92,  123,  149,  154,
            139,  147,  160,  167,  161,  141,  118,   91,   73,   58,   53,   53,   58,   68,   87,  112,  138,  155,  158,  148,  127,  102,   80,   66,   60,   63,   75,   96,  122,  142,  148,  140,  117,   91,   70,   60,   65,   85,  118,  151,  164,  158,  133,  100,   83,   63,   65,   81,  113,  139,  156,  156,
            152,  159,  165,  162,  143,  113,   87,   65,   55,   48,   48,   55,   67,   91,  122,  150,  166,  167,  152,  124,   94,   72,   60,   58,   64,   81,  107,  135,  155,  158,  145,  117,   88,   67,   58,   62,   79,  110,  144,  163,  162,  142,  113,   85,   73,   65,   70,   96,  131,  149,  155,  153,
            160,  164,  161,  146,  121,   89,   68,   55,   51,   49,   54,   69,   92,  128,  160,  177,  176,  158,  125,   90,   65,   54,   53,   62,   82,  112,  144,  165,  168,  154,  123,   89,   65,   56,   59,   75,  104,  139,  164,  170,  155,  129,  100,   79,   70,   68,   79,  108,  141,  150,  146,  142,
            162,  160,  149,  126,   96,   71,   57,   52,   50,   56,   69,   96,  131,  166,  185,  185,  167,  131,   91,   62,   49,   48,   57,   79,  112,  146,  168,  173,  160,  128,   91,   64,   53,   56,   71,  100,  137,  165,  177,  168,  147,  119,   96,   79,   69,   71,   86,  117,  146,  150,  140,  134,
            156,  150,  131,  103,   77,   59,   55,   57,   59,   73,   97,  135,  167,  189,  192,  176,  139,   96,   64,   48,   45,   52,   74,  108,  143,  166,  171,  160,  130,   93,   64,   50,   51,   64,   93,  131,  164,  181,  178,  158,  132,  105,   88,   75,   68,   74,   93,  124,  149,  146,  132,  125,
            142,  129,  106,   82,   66,   56,   57,   67,   76,  100,  133,  168,  189,  194,  182,  149,  105,   69,   50,   45,   51,   71,  104,  140,  164,  169,  158,  129,   93,   64,   49,   47,   56,   82,  120,  158,  180,  182,  166,  136,  108,   85,   74,   65,   66,   81,  104,  137,  155,  147,  128,  119,
            120,  104,   84,   68,   63,   64,   72,   90,  108,  137,  165,  185,  190,  182,  155,  114,   77,   56,   49,   53,   72,  104,  140,  164,  169,  158,  130,   96,   67,   50,   45,   51,   71,  107,  147,  174,  180,  167,  137,  104,   80,   65,   60,   59,   71,   94,  123,  149,  158,  142,  121,  111,
            99,   83,   72,   63,   66,   81,  100,  129,  150,  170,  181,  184,  173,  152,  119,   85,   63,   55,   58,   76,  107,  142,  166,  171,  160,  134,  101,   72,   54,   47,   49,   65,   96,  136,  166,  174,  163,  133,  100,   74,   58,   53,   56,   66,   91,  123,  149,  160,  156,  132,  110,  102,
            85,   71,   69,   72,   82,  108,  136,  167,  176,  181,  175,  161,  140,  115,   89,   70,   63,   66,   83,  112,  145,  168,  172,  161,  136,  106,   78,   61,   53,   53,   65,   91,  128,  158,  169,  158,  129,   94,   69,   56,   50,   55,   68,   91,  126,  152,  166,  160,  142,  115,   99,   96,
            82,   68,   75,   94,  116,  147,  171,  190,  182,  171,  149,  123,  101,   84,   72,   70,   75,   93,  119,  148,  168,  171,  160,  135,  106,   81,   66,   60,   61,   72,   94,  125,  153,  163,  155,  126,   92,   66,   54,   54,   58,   76,  102,  130,  160,  169,  165,  142,  113,   92,   88,   91,
            83,   76,   87,  116,  150,  177,  187,  188,  166,  140,  110,   85,   74,   68,   71,   82,  103,  130,  154,  168,  167,  153,  128,  101,   80,   68,   66,   70,   83,  103,  128,  149,  157,  149,  123,   92,   68,   57,   58,   70,   88,  117,  142,  162,  173,  162,  145,  111,   87,   78,   84,   92,
            95,   98,  116,  147,  173,  189,  185,  169,  133,  100,   77,   62,   61,   67,   82,  109,  140,  163,  172,  165,  145,  116,   91,   73,   65,   67,   77,   95,  117,  137,  151,  152,  141,  117,   91,   71,   63,   67,   82,  106,  133,  155,  165,  165,  159,  131,  105,   80,   73,   79,   92,  101,
            107,  118,  149,  175,  183,  179,  162,  128,   94,   69,   56,   52,   60,   77,  108,  145,  172,  181,  170,  143,  108,   80,   64,   59,   65,   80,  104,  130,  149,  157,  151,  134,  110,   88,   74,   71,   79,   98,  124,  149,  164,  166,  155,  135,  123,   92,   76,   67,   71,   87,  108,  117,
            126,  141,  167,  185,  181,  159,  125,   88,   67,   53,   48,   53,   69,  101,  144,  177,  190,  181,  151,  109,   75,   57,   53,   60,   78,  108,  139,  159,  164,  153,  130,  103,   83,   75,   78,   93,  116,  142,  163,  170,  164,  144,  117,   90,   82,   69,   68,   77,   91,  112,  130,  135,
            152,  163,  174,  176,  162,  125,   92,   65,   53,   47,   51,   65,   92,  137,  176,  195,  191,  165,  120,   79,   55,   48,   54,   73,  106,  142,  165,  171,  158,  130,   99,   78,   74,   82,  104,  132,  157,  173,  175,  162,  137,  103,   76,   60,   65,   69,   80,  103,  128,  146,  153,  147,
            163,  173,  173,  159,  129,   92,   70,   56,   49,   51,   61,   88,  126,  170,  196,  198,  178,  136,   90,   59,   46,   49,   66,   99,  139,  166,  176,  164,  134,   99,   76,   72,   83,  111,  145,  170,  182,  177,  159,  127,   95,   68,   57,   57,   71,   92,  114,  140,  158,  163,  158,  146,
            168,  172,  162,  133,   98,   70,   58,   54,   53,   62,   82,  120,  160,  192,  200,  188,  152,  105,   68,   50,   48,   61,   90,  130,  161,  175,  166,  140,  104,   77,   71,   82,  112,  152,  179,  189,  180,  156,  120,   87,   63,   53,   58,   75,   98,  128,  155,  168,  167,  155,  144,  132,
            163,  157,  137,  106,   77,   59,   55,   58,   62,   80,  111,  152,  182,  195,  189,  160,  117,   79,   58,   53,   61,   84,  119,  151,  168,  163,  141,  108,   81,   72,   80,  110,  152,  183,  194,  184,  156,  116,   81,   61,   52,   60,   80,  115,  143,  165,  176,  172,  155,  131,  118,  110,
            148,  135,  109,   81,   65,   57,   60,   71,   81,  109,  144,  173,  185,  183,  161,  123,   89,   68,   63,   69,   86,  113,  140,  156,  153,  136,  108,   84,   74,   79,  105,  145,  179,  194,  186,  159,  116,   79,   59,   53,   60,   84,  119,  153,  171,  173,  167,  147,  126,  103,   94,   93,
            124,  107,   85,   67,   62,   63,   71,   95,  114,  144,  168,  178,  172,  152,  121,   92,   76,   75,   82,   97,  116,  135,  145,  140,  125,  103,   83,   75,   79,  101,  136,  168,  186,  182,  158,  118,   80,   59,   52,   60,   81,  119,  152,  167,  170,  153,  132,  106,   93,   85,   87,   91,
            101,   85,   74,   66,   70,   82,   99,  132,  150,  169,  172,  164,  142,  114,   90,   79,   82,   96,  115,  129,  137,  136,  126,  111,   95,   81,   76,   81,   99,  127,  155,  171,  169,  150,  115,   80,   59,   52,   60,   80,  111,  144,  162,  158,  148,  114,   95,   78,   74,   78,   91,  101,
            90,   75,   74,   77,   90,  120,  146,  173,  174,  174,  158,  132,  105,   85,   76,   84,  104,  129,  146,  148,  136,  116,   97,   83,   75,   75,   83,  100,  122,  142,  154,  150,  134,  106,   78,   61,   55,   61,   78,  107,  137,  153,  152,  133,  111,   79,   70,   69,   74,   89,  109,  122,
            85,   77,   83,   98,  121,  156,  177,  187,  171,  153,  123,   93,   78,   72,   81,  105,  137,  159,  162,  145,  115,   88,   71,   66,   71,   84,  102,  121,  134,  138,  130,  115,   93,   74,   63,   61,   66,   85,  113,  139,  152,  149,  131,  102,   82,   62,   63,   75,   94,  115,  134,  140,
            96,   94,  109,  135,  157,  179,  185,  176,  142,  111,   85,   68,   66,   75,  100,  137,  167,  174,  159,  124,   87,   64,   57,   63,   79,  102,  122,  131,  129,  115,   99,   81,   69,   65,   69,   81,  100,  127,  150,  160,  153,  134,  107,   79,   67,   63,   72,  100,  127,  147,  153,  149,
            110,  117,  140,  162,  171,  173,  166,  139,  100,   74,   61,   57,   68,   92,  132,  168,  183,  173,  139,   96,   63,   50,   53,   69,   95,  119,  129,  125,  108,   89,   74,   66,   67,   77,   96,  118,  143,  160,  165,  156,  135,  107,   83,   67,   65,   76,   95,  129,  157,  163,  157,  146,
            136,  147,  162,  170,  165,  147,  124,   95,   69,   56,   53,   61,   84,  124,  165,  187,  185,  157,  111,   71,   50,   46,   57,   81,  109,  124,  122,  106,   87,   72,   67,   74,   89,  113,  134,  152,  165,  162,  148,  125,  101,   79,   69,   68,   77,  104,  132,  158,  170,  160,  141,  127,
            156,  163,  162,  156,  140,  111,   87,   67,   55,   53,   59,   80,  116,  159,  188,  192,  171,  129,   83,   54,   44,   48,   64,   91,  112,  118,  106,   89,   76,   71,   79,   99,  126,  150,  162,  163,  152,  131,  106,   84,   69,   63,   68,   82,  104,  134,  160,  168,  162,  137,  116,  109,
            151,  153,  144,  125,  102,   79,   66,   59,   55,   63,   78,  112,  152,  184,  194,  181,  145,   98,   63,   46,   43,   51,   70,   92,  106,  105,   95,   84,   81,   88,  108,  133,  155,  163,  156,  136,  110,   84,   66,   56,   54,   64,   84,  113,  144,  161,  169,  155,  130,  104,   94,   96,
            133,  130,  117,   95,   76,   63,   60,   63,   68,   83,  108,  144,  176,  191,  185,  156,  111,   72,   50,   42,   43,   53,   69,   87,   98,  100,   95,   95,  104,  119,  140,  156,  159,  146,  121,   92,   68,   53,   47,   50,   61,   85,  115,  146,  166,  163,  153,  121,   96,   81,   80,   88,
            106,   97,   89,   78,   71,   68,   68,   80,   90,  114,  141,  167,  182,  181,  159,  119,   81,   56,   44,   41,   44,   54,   68,   84,  100,  112,  117,  126,  137,  150,  157,  152,  135,  108,   80,   59,   47,   45,   50,   65,   89,  120,  145,  159,  161,  138,  112,   83,   72,   75,   84,   95,
            88,   79,   77,   79,   83,   89,   94,  112,  122,  143,  162,  172,  170,  154,  122,   88,   63,   48,   42,   41,   45,   54,   70,   92,  118,  140,  155,  163,  166,  162,  148,  125,   97,   73,   56,   47,   47,   56,   73,  100,  124,  144,  152,  145,  136,  101,   80,   64,   63,   78,  102,  114,
            91,   85,   88,   99,  109,  125,  134,  152,  149,  159,  164,  160,  143,  116,   87,   64,   51,   45,   44,   46,   53,   69,   95,  128,  160,  179,  187,  184,  171,  148,  118,   88,   68,   55,   51,   54,   66,   87,  114,  137,  144,  142,  131,  109,   97,   73,   65,   66,   73,   97,  127,  137,
            97,  100,  112,  130,  143,  157,  163,  168,  150,  150,  144,  129,  106,   83,   65,   54,   50,   51,   56,   64,   81,  108,  144,  177,  196,  202,  196,  178,  149,  113,   83,   64,   56,   56,   64,   81,  105,  132,  150,  153,  139,  116,   95,   74,   71,   61,   64,   79,  101,  131,  156,  155,
            113,  129,  145,  160,  167,  173,  172,  164,  131,  120,  109,   94,   78,   65,   57,   56,   60,   70,   85,  104,  130,  160,  188,  204,  209,  201,  181,  148,  110,   79,   62,   58,   62,   76,   98,  125,  151,  164,  162,  143,  115,   85,   69,   58,   62,   66,   77,  106,  137,  161,  172,  163,
            132,  152,  166,  174,  170,  164,  156,  135,   99,   86,   77,   70,   67,   63,   63,   71,   88,  109,  131,  153,  174,  193,  203,  205,  196,  175,  142,  105,   77,   62,   59,   68,   87,  115,  144,  166,  176,  172,  152,  118,   88,   64,   58,   60,   69,   87,  109,  140,  167,  177,  173,  158,
            146,  162,  172,  171,  158,  139,  121,   97,   70,   61,   60,   62,   69,   76,   86,  106,  131,  155,  171,  182,  189,  191,  188,  177,  157,  127,   97,   75,   63,   63,   75,  100,  131,  159,  178,  184,  179,  160,  128,   93,   69,   58,   61,   76,   90,  121,  149,  169,  179,  175,  159,  142,
            152,  162,  163,  154,  133,  105,   85,   67,   54,   53,   60,   72,   91,  108,  128,  152,  171,  182,  184,  179,  169,  157,  144,  125,  105,   85,   72,   66,   70,   86,  115,  148,  173,  186,  188,  180,  162,  133,  100,   74,   60,   60,   74,   99,  119,  152,  175,  182,  177,  160,  139,  126,
            147,  150,  144,  130,  104,   79,   64,   54,   50,   59,   76,   99,  129,  151,  172,  185,  190,  183,  166,  143,  122,  105,   91,   80,   72,   68,   70,   81,  103,  134,  166,  186,  194,  191,  179,  158,  132,  103,   78,   64,   59,   68,   90,  120,  147,  169,  181,  178,  167,  146,  127,  116,
            143,  142,  127,  105,   82,   61,   54,   52,   59,   81,  109,  142,  173,  191,  199,  198,  185,  158,  124,   94,   75,   65,   61,   62,   68,   80,   98,  125,  155,  182,  197,  201,  193,  177,  152,  125,  100,   79,   64,   60,   59,   75,  102,  133,  160,  175,  177,  168,  154,  131,  114,  107,
            138,  137,  122,   99,   77,   57,   54,   60,   83,  118,  155,  183,  202,  208,  205,  190,  160,  119,   83,   61,   52,   52,   60,   74,   96,  123,  151,  176,  194,  203,  203,  192,  172,  144,  116,   93,   75,   63,   56,   57,   60,   82,  113,  144,  167,  177,  174,  163,  149,  128,  111,  106,
            133,  136,  125,  106,   92,   80,   81,   95,  126,  161,  188,  204,  211,  209,  195,  167,  124,   83,   58,   48,   51,   64,   87,  118,  150,  177,  193,  203,  204,  198,  183,  160,  133,  106,   85,   70,   59,   52,   52,   60,   75,  104,  139,  167,  177,  182,  173,  158,  144,  123,  109,  107,
            125,  127,  120,  111,  110,  116,  129,  149,  168,  191,  202,  206,  209,  196,  170,  126,   82,   56,   47,   53,   71,  102,  138,  170,  193,  204,  207,  202,  190,  170,  146,  120,   97,   80,   65,   54,   48,   48,   59,   85,  111,  146,  171,  185,  183,  180,  167,  147,  130,  109,   99,  101,
            120,  125,  121,  120,  122,  137,  157,  179,  196,  203,  200,  195,  176,  140,   97,   65,   56,   59,   64,   85,  113,  145,  177,  191,  194,  192,  184,  169,  143,  108,   81,   64,   60,   61,   62,   68,   75,   83,   92,  111,  140,  163,  181,  185,  178,  170,  154,  135,  118,  101,   92,   98,
            120,  122,  116,  116,  119,  135,  158,  180,  195,  198,  190,  177,  151,  110,   76,   57,   54,   64,   78,  112,  152,  174,  186,  186,  177,  164,  146,  126,  104,   79,   65,   58,   57,   62,   67,   82,  100,  119,  134,  151,  171,  179,  184,  179,  165,  152,  140,  123,  108,   97,   94,  103,
            119,  116,  108,  109,  112,  126,  144,  161,  174,  172,  163,  148,  128,  101,   83,   74,   73,   90,  111,  146,  175,  184,  178,  164,  148,  129,  114,  100,   90,   80,   76,   76,   76,   85,   94,  115,  134,  153,  164,  174,  181,  178,  171,  156,  140,  126,  120,  108,   97,   93,   97,  109
        ])
        
        self.nX = torch.tensor([
            17,   15,   22,   11,   17,   17,   17,   18,   20,   19,   17,   22,   14,   21,   14,   17,   16,   15,   15,   17,   15,   21,   17,   14,   12,   21,   12,   25,   13,   12,   16,   13,   22,   21,   23,   20,   22,   22,   27,   15,   21,   12,   22,   25,   22,   10,   25,   23,   12,   24,   21,   26,   17,   12,   16,   11,   22,   18,   18,   20,   11,   25,   10,   27,   10,   25,   11,   13,   10,   17,   28,   12,   23,   23,    7,    7,   12,    8,    6,    7,    8,    6,   29,   28,   25,   27,   27,    7,   26,    9,   29,   25,    5,   28,   27,   29,   29,   26,    7,   29,   28,   30,   19,   27,   30,   30,    4,   30,   27,   30,    5,   22,    5,   30,    6,   31,   29,   30,    5,   31,   31,    5,    3,    4,    3,   32,    6,   32,    3,    3
        ])
        self.nY = torch.tensor([
            9,  114,    5,  109,   96,   36,   17,   34,   82,   59,  117,  114,   51,   69,   31,   74,   63,   28,  118,    7,   65,   62,   55,   88,    5,   24,    9,   98,   34,   21,   24,   60,   15,   51,  118,   26,   96,    7,    8,   41,   37,   95,   86,   77,   74,   28,   53,   48,   37,   28,   44,   62,   22,  106,   45,   64,   22,   15,   84,  107,   40,   39,   86,   10,  111,  107,   23,  104,  116,  105,   30,   84,  104,   31,   11,   15,   74,   62,   96,   40,   35,  107,   47,  115,   26,  110,    5,   38,  116,   49,   85,   82,   85,  105,   43,  100,   90,   20,    9,  109,   66,   36,   70,   81,  104,   45,   47,   94,   16,   70,   44,   35,   26,   22,  113,    5,   78,   52,  102,   15,   62,   58,   92,   33,   80,  117,    3,   30,   40,   63
        ])
        self.ori = torch.tensor([
            266, 8620,  249, 9984, 9778, 11358, 12728, 11422, 9713, 10025, 8584, 8862, 10614, 10079, 11501, 9917, 10494, 11597, 8472,  379, 10445, 10192, 10039, 9508,  228, 11674,  143, 9522, 11461, 12188, 11723, 10561, 12782, 10392, 8799, 11515, 9725,  205,   47, 11252, 11251, 9940, 9879, 9967, 10011, 11767, 10305, 10406, 11526, 11452, 10008, 10351, 12040, 10516, 10646, 10647, 12074, 12830, 9794, 9042, 11512, 10836, 9606, 12866, 8823, 8869, 12020, 10410, 8533, 9525, 11371, 9664, 8910, 11506,   11, 12743, 9588, 10865, 9969, 11488, 11509, 10032, 10354, 9130, 11568, 8813,  156, 11569, 8965, 11365, 9790, 9886, 9568, 8760, 10230, 9377, 9137, 12196,    5, 8789, 10273, 11369, 10019, 9923, 8870, 10406, 11341, 9153, 12471, 10305, 11245, 11420, 11918, 12026, 8765,  286, 10021, 10372, 9691, 12474, 10290, 10950, 9687, 11590, 9091, 9900,   13, 11421, 11415, 11428
        ])


        pass

    # def __len__(self):
    #     return len(self.samples)

    def get_data(self, index):

        img_o_A = load_as_float(self.samples[index]['image'])

        '''传统增强扩边处理-逆：36->32'''  
        img_o_A = self.transforms(img_o_A)
        if self.isDilation:
            import torch.nn.functional as F
            img_aug = F.pad(img_o_A, (2, 2), 'constant')     # 36 -> 40
        else:
            img_aug = img_o_A[:, :, 2:-2]      # [1, 136, 32]

        H, W = img_aug.shape[1], img_aug.shape[2]
        img_aug = torch.tensor(img_aug, dtype=torch.float32).view(-1, H, W)

        input  = {}
        input.update({'image': img_aug})
        input.update({'name': self.samples[index]['name']})

        return input

    def test_process(self, FPDT):

        img = self.IMG.float() / 255
        img = img.to(self.device).reshape(1, 1, 128, 52)
        pts = torch.cat((self.nX.unsqueeze(1), self.nY.unsqueeze(1)), dim=-1).to(self.device).unsqueeze(0)
        angle = self.ori.to(self.device) / 4096 - 1.570796327
        angle = angle / (2 * np.pi) * 360.
        desc_out, _, desc_hadama, _ = FPDT.output_pts_desc_batch_netV2(img, pts, angle, 16, 22, quantize=True, is256=self.is256)

        pass

    '''
    分类器训练的linux版本(from Matlab)
    合并fafr跑库log, 规整训练数据并进行训练
    '''
    def __init__(self, img_path=None, info_path=None, device="cpu", **config):
        # 不接收外部config
        self.device         = device

        self.path_log = Path(info_path) / 'LRclassifier_train_log'
        # 存放fafr log的文件夹(*Far*.csv)
        mychoice = [
            # "sift_6193_DK7_merge_test1_info_20_55_200",
            "Net0106_6193_DK7_merge_test1_info_20_55_200_ONE",
            # "6193_DK7_merge_test2",
        ]

        self.fa_csv_list = []
        self.fr_csv_list = []
        for lis in mychoice:
            path_csv = self.path_log / lis
            self.fa_csv_list.extend(path_csv.rglob('*Far*.csv'))
            self.fr_csv_list.extend(path_csv.rglob('*Frr*.csv'))


        pass
    
    def process_pool(self, path_item):
        idx = path_item[0]
        path = path_item[1]

        content = pd.read_csv(path)
        data_temp = torch.tensor(content.values[:, 2:].tolist())    # 去掉前2列

        data_info0 = data_temp[(data_temp[:, 40] == 0) * (data_temp[:, 43] <= 0), :]     # & data_temp[:, -1] == 0
        data_info1 = data_temp[(data_temp[:, 40] == 1) * (data_temp[:, 43] <= 0), :]
        data_info2 = data_temp[(data_temp[:, 40] == 2) * (data_temp[:, 43] <= 0), :]
        print('{} is done! '.format(idx))

        return data_info0, data_info1, data_info2

    def merge_csv_to_mat(self, path_list):

        data_info0 = [[]] * len(path_list)
        data_info1 = [[]] * len(path_list)
        data_info2 = [[]] * len(path_list)

        '''并行处理'''
        # from multiprocessing import Pool
        # path_items = [[id, path] for id, path in enumerate(path_list)]
        # with Pool(5) as p:
        #     result_list = p.map(self.process_pool, path_items)

        # for i in range(len(result_list)):
        #     data_info0[i] = result_list[i][0]
        #     data_info1[i] = result_list[i][1]
        #     data_info2[i] = result_list[i][2]
        
        '''非并行'''
        for i, path in enumerate(path_list):
            content = pd.read_csv(path)
            data_temp = torch.tensor(content.values[:, 2:].tolist())    # 去掉前2列

            data_temp0_info0_Notrecog = data_temp[(data_temp[:, 40] == 0) * (data_temp[:, 43] <= 0), :]     # & data_temp[:, -1] == 0
            data_info0[i] = data_temp0_info0_Notrecog

            data_temp0_info1_Notrecog = data_temp[(data_temp[:, 40] == 1) * (data_temp[:, 43] <= 0), :]
            data_info1[i] = data_temp0_info1_Notrecog

            data_temp0_info2_Notrecog = data_temp[(data_temp[:, 40] == 2) * (data_temp[:, 43] <= 0), :]
            data_info2[i] = data_temp0_info2_Notrecog

            print('{} is done! '.format(i))
        
        data_info0 = torch.cat((data_info0), dim=0) if len(data_info0) != 0 else None
        data_info1 = torch.cat((data_info1), dim=0) if len(data_info1) != 0 else None
        data_info2 = torch.cat((data_info2), dim=0) if len(data_info2) != 0 else None
        return data_info0, data_info1, data_info2

    def LRtrain(self, farAll, frrAll, feature_pick):
        '''
        fx_mat: N x 539
        '''
        import datetime
        from sklearn.linear_model import LogisticRegression

        newFeatFar = torch.cat((farAll[:, 391: 400], farAll[:, 176: 185], farAll[:, 400:]), dim=-1)
        newFeatFrr = torch.cat((frrAll[:, 391: 400], frrAll[:, 176: 185], frrAll[:, 400:]), dim=-1)
        farAll = torch.cat((farAll[:, :391], farAll[:, 74: 76], (farAll[:, 74] + farAll[:, 75]).unsqueeze(1), farAll[:, 78].unsqueeze(1), newFeatFar), dim=-1)
        frrAll = torch.cat((frrAll[:, :391], frrAll[:, 74: 76], (frrAll[:, 74] + frrAll[:, 75]).unsqueeze(1), frrAll[:, 78].unsqueeze(1), newFeatFrr), dim=-1)

        fa = farAll[:, feature_pick]
        fr = frrAll[:, feature_pick]

        fafr_input = torch.cat((fa, fr), dim=0)
        label = torch.cat((torch.zeros((fa.shape[0])), torch.ones((fr.shape[0]))), dim=0)
        clf = LogisticRegression(max_iter=100)      # 迭代停止条件如何与matlab对齐？ 

        
        print("start time is {}".format(datetime.datetime.now()))

        lr_l2 = clf.fit(fafr_input, label)
        print(lr_l2.coef_)
        print("end time is {}".format(datetime.datetime.now()))



        pass

    def test_process(self, FPDT):
        '''注意区分matlab和python索引起始'''
        

        # cols_len33 = [1,2,3,4,5,6,7,8,10,11,22,23,24,25,26,27,28,30,31,32,33,34,35,52,53,54,55,56,57,58,59,60,67] # matlab index
        cols_len33   = [0,1,2,3,4,5,6,7,9,10,21,22,23,24,25,26,27,29,30,31,32,33,34,51,52,53,54,55,56,57,58,59,66]
        len539_feature_pick_cols = cols_len33 + list(range(83, 176)) + list(range(186, 195)) + list(range(196, 404)) + list(range(413, 609))

        fr_info0_mat, fr_info1_mat, fr_info2_mat = self.merge_csv_to_mat(self.fr_csv_list)
        fa_info0_mat, fa_info1_mat, fa_info2_mat = self.merge_csv_to_mat(self.fa_csv_list)

        '''Info0'''
        self.LRtrain(fa_info0_mat, fr_info0_mat, len539_feature_pick_cols)

        pass