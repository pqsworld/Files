# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image as Img
import cv2

def ori_filter(img, ori_map, gabor_delta_xy):
    # if ori_map == None:
    #     img_tensor = torch.from_numpy(np.float32(img)).unsqueeze(0).unsqueeze(0)
    #     kernel_x = torch.tensor([[[[-1,0,1],[-2, 0, 2], [-1, 0, 1]]]], dtype = torch.float32)
    #     kernel_y = torch.tensor([[[[-1,-2,-1],[0, 0, 0], [1, 2, 1]]]], dtype = torch.float32)

    #     sobel_x = F.conv2d(img_tensor, kernel_x, padding = 1).squeeze(0).squeeze(0).numpy()
    #     sobel_y = F.conv2d(img_tensor, kernel_y, padding = 1).squeeze(0).squeeze(0).numpy()
    #     img_ori = np.arctan2(sobel_x, sobel_y)   #考虑用提点网络输出的方向图代替
    # else:
    img_ori = ori_map
    img_ori[img_ori < 0] = img_ori[img_ori < 0] + np.pi
    img_ori_12 = np.int32(np.round(img_ori / (np.pi/12)))
    img_ori_12[img_ori_12 == 12] = 0

    h,w = img.shape
    img_coordinate = np.indices((w, h)).reshape(2, w*h).transpose()    #w,h
    img_gabor_xy = gabor_delta_xy[img_ori_12[img_coordinate[:,1], img_coordinate[:,0]]]  #[img.size, 7, 2]
    img_gabor_xy = img_gabor_xy + img_coordinate.reshape(w*h, 1, 2)
    mask = (img_gabor_xy[:, :, 0] >= 0) * (img_gabor_xy[:, :, 0] < w) * (img_gabor_xy[:, :, 1] >= 0) * (img_gabor_xy[:, :, 1] < h)
    img_gabor_xy[~mask] = [0,0]
    img_gabor_neibo = img[img_gabor_xy[:,:,1], img_gabor_xy[:, :, 0]]  #[img.size, 7]
    img_gabor_neibo[~mask] = 0
    kernel = np.array([1,2,4,8,4,2,1])
    #kernel = np.array([0,0,0,8,0,0,0])
    img_gabor = img_gabor_neibo @ kernel
    kernel_sum = mask @ kernel
    img_gabor = np.uint8(img_gabor / kernel_sum + 0.5)
    img_gabor = img_gabor.reshape(w,h).T

    # # debug
    # wa,ha = 24,74
    # neibo_xy = img_gabor_xy[ha + wa * h]
    # img_gabor[neibo_xy[:,1], neibo_xy[:,0]] = 255

    return img_gabor

read_data_from_file = False

def calculate_Dbvalue(sc: np.array):
    '''
    目前按照工程中网络点分数*-3500作为dbvalue
    '''
    dbval = -3500 * sc
    # return np.trunc(dbval)
    return np.int32(dbval)

def get_feature(img, ma, name):
    'img: numpy, ma: model_api'

    quality = ma.quality(img)

    partial_mask,wet_score = ma.mask(img)

    enhance_img = ma.enhance(img)

    point_img = cv2.GaussianBlur(enhance_img, (5,5), 0.5)  #bordertype 默认reflect

    if read_data_from_file :
        partial_mask_path = name.replace('img_ori_data', 'img_mask_data')[:-4] + '_mask.bmp'
        point_img_path = name.replace('img_ori_data', 'img_pnt_data')[:-4] + '_kpt.bmp'
        if os.path.isfile(partial_mask_path) and os.path.isfile(point_img_path):
            partial_mask = np.array(Img.open(partial_mask_path))
            partial_mask = partial_mask / 255
            point_img = np.array(Img.open(point_img_path))
            
    points = ma.point(point_img, partial_mask)
    if points['pts'].shape[0] < 3:   #避免空图
        return None 

    ori_filter_img = ori_filter(enhance_img, points['ori_map'], ma.gabor_delta_xy) 

    #获取16级灰度图
    def downsample2x2(img):

        h,w = img.shape
        x = range(0,w,2)
        y = range(0,h,2)
        img_downsample = img[y,:][:,x]
        return img_downsample

    level_16_img = ori_filter(ori_filter_img, points['ori_map'], ma.gabor_delta_xy)
    level_16_img = cv2.GaussianBlur(level_16_img, (5,5), 0.8)  #bordertype 默认reflect
    level_16_img = downsample2x2(level_16_img)   #0,1
    level_16_img = (level_16_img // 16)*16 + 8

    desc_img = ma.expand(ori_filter_img)

    if read_data_from_file :
        desc_img_path = name.replace('img_ori_data', 'img_extend_data')[:-4] + '_extend.bmp'
        if os.path.isfile(desc_img_path):
            desc_img = np.array(Img.open(desc_img_path))

    desc, wb_mask = ma.decriptor(desc_img, points)

    index_black = np.argwhere(wb_mask == 0).squeeze(-1)      # 有序黑点索引
    index_white = np.argwhere(wb_mask == 1).squeeze(-1)

    index_wb = np.concatenate((index_black, index_white), axis=0)
    # index_wb = np.argsort(wb_mask.cpu().detach().numpy())     # 无序
    black_num = index_wb.shape[0] - np.sum(wb_mask)

    if black_num < 2 or points['pts'].shape[0] - black_num < 2:
        return None 
    #print(black_num)

    dbval = calculate_Dbvalue(points['prob'])

    return {
            'h':enhance_img.shape[0],
            'w':enhance_img.shape[1],
            'name':name,
            'quality':quality,
            'mask':partial_mask,
            'wet_score':wet_score,
            'pts':points['pts'][index_wb],       #numpy Nx2 (float)
            'prob':points['prob'][index_wb],     #numpy Nx1  (float)
            'angles':points['angles'][index_wb], #numpy Nx1 (float)
            'dbvalue': dbval[index_wb],          #numpy Nx1 (int32)
            'desc':desc[index_wb],               #torch Nx(256bit)
            'black_num':black_num,         #numpy int64
            'enhance_img':enhance_img, #numpy h*w
            #'expand_img':expand_img,   #numpy h*w
            'level_16_img':level_16_img, #numpy h*w
           }
