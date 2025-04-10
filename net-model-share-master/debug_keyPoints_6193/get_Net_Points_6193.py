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
from Model_component import filter_points, draw_keypoints, draw_keypoints_pair, draw_keypoints_pair_tradition
from Model_component import get_point_pair_repeat, getPtsFromHeatmap, flattenDetection
from Model_component import homograghy_transform
from Model_component import thresholding_desc

def get_module(name):  
    mod = importlib.import_module('get_Net_Points_6193')
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

'''计算6193网络提点结果'''
class get_net_pts_6193(object):
    def __init__(self, img_path=None, device="cpu", **config):

        self.device         = device
        self.top_k          = config['top_k']
        self.img_h          = config['img_h']
        self.img_w          = config['img_w']
        self.img_resize_h   = config['img_resize_h']
        self.img_resize_w   = config['img_resize_w']
        self.border_remove  = config['border_remove']     # 点对距离满足<self.dis_thr认为是匹配对
        self.nms            = config['nms']  # True: 扩充36->40  False: 裁剪36->32
        self.image_pth      = config['image_path']
        self.output_dir     = Path(config['output_dir'])
        
        os.makedirs(self.output_dir, exist_ok=True)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((self.sizer[0], self.sizer[1])),
        ])

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

    def load_image(self):
        
        image_pth, img_resize_h, img_resize_w = self.image_pth, self.img_resize_h,self.img_resize_w
        
        img = imread(image_pth)
        img_p = img.astype(np.float32) / 255
        img_n = 1 - img.astype(np.float32) / 255

        img_p_resize = cv2.resize(img_p, (img_resize_w, img_resize_h),interpolation=cv2.INTER_LINEAR)
        img_n_resize = cv2.resize(img_n, (img_resize_w, img_resize_h),interpolation=cv2.INTER_LINEAR)

        img_p_resize = self.transforms(img_p_resize).unsqueeze(0)
        img_n_resize = self.transforms(img_n_resize).unsqueeze(0)
        
        name = image_pth.rsplit("/", 1)
        name = name[-1][:-4]
        
        sample  = {}
        sample.update({'name': name, 'img': img})
        sample.update({'img_p_resize': img_p_resize, 'img_n_resize': img_n_resize})
        
        
        return sample


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

    def process_net_points(self, pnts) :

        top_k = self.top_k
        nms_dist = self.nms
        bord = self.border_remove
        resize_h, resize_w = self.img_resize_h, self.img_resize_w
        img_h, img_w = self.img_h, self.img_w
        
        ratio_h = img_h / resize_h
        ratio_w = img_w / resize_w
        
        # pnts : 尺寸还原到 122 * 36
        pnts[:,1] = pnts[:,1] * ratio_w
        pnts[:,2] = pnts[:,2] * ratio_h
        
        # pts ： 剔除 [122,36] 边缘 点
        mask_pts = (pnts[:,1] >=bord) * (pnts[:,1] <= img_w - 1 - bord) * (pnts[:,2] >=bord) * (pnts[:,2] <= img_h - 1 - bord)
        pnts = pnts[mask_pts]        
        
        # nms
        pnts = self.sort_pts_for_score(pnts, img_h, img_w, nms_dist) # pnts : 160 * 48
                
        # pts, 取 前 top_k 个点
        if top_k:
            if pnts.shape[0] > top_k:
                pnts = pnts[:top_k, :]
            pass 
        pass             
    
        return pnts

    def test_process(self, FPDT):

        sample = self.load_image()    # 点会做限制 注意修改

        pts_p = FPDT.forward_self_supervised_6193(sample['img_p_resize'].to(self.device))
        pts_n = FPDT.forward_self_supervised_6193(sample['img_n_resize'].to(self.device))
        
        pts = torch.cat((pts_p, pts_n),0)      # size: 160 * 48
        pts = self.process_net_points(pts) #pnts : 128 * 40

        img_pts = draw_keypoints(sample['img'], pts,  radius=3, s=3)
        
        out = {}
        out.update({'name': sample['name']})
        out.update({'img': img_pts, 'pts': np.array(pts)})
        
        return out

        pass


   
    
    
    
    