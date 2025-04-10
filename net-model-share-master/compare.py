# -*- coding: utf-8 -*-

from desc_patch.utils.desc import Hamming_distance_Hadamard,get_candmask_wb
from desc_patch.utils.draw import draw_keypoints_match
from point_alike.utils.draw import draw_pair_trans
from point_alike.utils.utils import inv_warp_image
from SLransac import slransac
from skimage.transform import AffineTransform
#from skimage.measure import ransac
from model import model_api
from get_img_feature import get_feature
from get_match_feature import get_match_feat
from feature_info import feature_info
from PIL import Image as Img
import numpy as np
import torch
import copy
import cv2
from recognize import recognize
from point_alike.utils.draw import inv_warp_image_bilinear

def crop_img_batch(img,center):
    center_n = ((center + 200) // 2).int()
    # print(center_n.shape)
    crop_l = center_n - 61
    crop_r = center_n + 61

    img_re = torch.zeros((img.shape[0],img.shape[1],122,122),device=img.device)

    for k in range(img.shape[0]):
        img_re[k] = img[k,:,crop_l[k][1]:crop_r[k][1],crop_l[k][0]:crop_r[k][0]]

    return img_re

def compare_batch(feat1,feat2,rec_feat,samples,fm):
    temp = (feat1['enhance_img']*feat1['mask'])
    samp = (feat2['enhance_img']*feat2['mask'])[samples]
    Htrans = rec_feat['model'][samples]

    temp = temp.repeat(samp.shape[0],1,1,1)
    pad = torch.nn.ZeroPad2d(padding=(182, 182, 139, 139))
    temp_p = pad(temp)
    samp_m = torch.zeros_like(samp)
    samp_m[:] = 255
    samp_p = pad(samp)
    samp_m_p = pad(samp_m)

    Htrans[:,0, 2] = Htrans[:,0, 2] + 182 - 182 * Htrans[:,0, 0] - 139 * Htrans[:,0, 1]
    Htrans[:,1, 2] = Htrans[:,1, 2] + 139 - 182 * Htrans[:,1, 0] - 139 * Htrans[:,1, 1]

    center = torch.sum(Htrans * torch.tensor([200, 200, 1],device=Htrans.device),-1) + 0.5

    samp_p_H = inv_warp_image_bilinear(samp_p, Htrans)
    samp_m_p_H = inv_warp_image_bilinear(samp_m_p, Htrans)
    #samp_p_H = samp_p
    #samp_m_p_H = samp_m_p
    samp_m_p_H[samp_m_p_H <= 254] = 0
    samp_m_p_H[samp_m_p_H >= 254] = 1
    image_warp = samp_p_H * samp_m_p_H

    b = torch.zeros_like(temp_p)
    g = image_warp
    r = temp_p
    img_match = torch.cat((r, g, b), axis=1)
    img_match = crop_img_batch(img_match, center)
    score = fm.compare_batch(img_match)

    img_match[:,[0,2],:,:] = img_match[:,[2,0],:,:]   #rgb->bgr
 
    return score, img_match

def crop_img(img,center):

    #trans=trans.reshape(-1)

    #Htrans = trans.reshape(-1).copy()


    #Htrans[2] = Htrans[2] + 182 - 182 * Htrans[0] - 139 * Htrans[1]

    #Htrans[5] = Htrans[5] + 139 - 182 * Htrans[3] - 139 * Htrans[4]

    center_c_x = int(center[0] + 0.5)

    center_c_y = int(center[1] + 0.5)


    # center_c_x = int(name.split("_")[-3])

    # center_c_y = int(name.split("_")[-2])

    center_t_x = 200

    center_t_y = 200

    center_x = (center_c_x + center_t_x) // 2

    center_y = (center_c_y + center_t_y) // 2

    crop_x_l = center_x - 61

    crop_x_r = center_x + 61

    crop_y_l = center_y - 61

    crop_y_r = center_y + 61

    #img = np.array(img)

    #print(img.shape)

    img = img[crop_y_l:crop_y_r,crop_x_l:crop_x_r, :]

    #print(img.shape)

    return img

def compare(feat1, feat2, rec_feat, fm):

    if rec_feat['model'] == None or np.linalg.matrix_rank(rec_feat['model'].params) < 3:

        return 0, None


    sx, sy = rec_feat['model'].scale

    shear = rec_feat['model'].shear

    if sx < 0.4 or sx > 2 or sy < 0.6 or sy > 1.5 or shear > 1 or shear < -1:

        return 0, None


    h, w = feat1['h'], feat1['w']

    #_, overlap_mask = cal_overlap_temple(w, h, rec_feat['model'], feat2['mask'], feat1['mask'])

    level_256_A = feat1['enhance_img'] * feat1['mask']

    level_256_A = np.pad(level_256_A, pad_width=[[139, 139], [182, 182]], mode="constant")

    level_256_B = feat2['enhance_img'] * feat2['mask']

    level_256_B_M = np.ones_like(level_256_B)

    level_256_B_M[level_256_B_M > 0] = 255

    level_256_B = np.pad(level_256_B, pad_width=[[139, 139], [182, 182]], mode="constant")

    level_256_B_M = np.pad(level_256_B_M, pad_width=[[139, 139], [182, 182]], mode="constant")


    input_img = {'img': level_256_A, 'img_H': level_256_B}


    Htrans = torch.from_numpy(rec_feat['model'].params).clone()

    Htrans[0, 2] = Htrans[0, 2] + 182 - 182 * Htrans[0, 0] - 139 * Htrans[0, 1]

    Htrans[1, 2] = Htrans[1, 2] + 139 - 182 * Htrans[1, 0] - 139 * Htrans[1, 1]


    image_warp = inv_warp_image(torch.from_numpy(input_img['img_H']), Htrans)

    image_warp_B_M = inv_warp_image(torch.from_numpy(level_256_B_M), Htrans)

    image_warp_B_M[image_warp_B_M < 255] = 0

    image_warp_B_M[image_warp_B_M >= 255] = 1

    image_warp = image_warp * image_warp_B_M


    center_result = torch.sum(Htrans * torch.tensor([200, 200, 1]), -1)

    # image_warp = np.array(image_warp * 255).astype(np.uint8)

    # imageB = (input_img['img'] * 255).astype(np.uint8)

    image_warp = np.array(image_warp).astype(np.uint8)

    imageB = (input_img['img']).astype(np.uint8)

    b = np.zeros_like(imageB)

    g = image_warp

    r = imageB

    img_match = cv2.merge([r, g, b])

    img_match = crop_img(img_match, center_result)

    score = fm.compare(img_match)  # 58636

    img_match_tmp = img_match.copy()

    img_match[:,:,0] = img_match[:,:,2]

    img_match[:,:,2] = img_match_tmp[:,:,0]

    #img_match = cv2.merge([b, g, r])  # 转换为bgr方便存图

    #img_match = crop_img(img_match, center_result)


    return score, img_match

def cal_overlap_temple(w, h, model, partial_maskB, partial_maskA):
    
    img_coordinate = np.indices((w, h)).reshape(2, w*h).transpose()
    
    img_t = model.inverse(img_coordinate)
    
    overlap_mask = (img_t[:,1] >= 0-0.5) * (img_t[:,1] < h-0.5) * (img_t[:,0] >= 0-0.5) * (img_t[:,0] < w-0.5)
    
    coord_sort = np.floor(img_t[overlap_mask].transpose() + 0.5)
    coord_sort = coord_sort.astype(np.uint)

    partial_mask_H = partial_maskB[coord_sort[1], coord_sort[0]]
    
    mask = np.zeros([h,w],dtype=np.uint8)
    grid = img_coordinate[overlap_mask][partial_mask_H].transpose().reshape(2, -1)
    mask[grid[1], grid[0]] = 1
    mask = mask * partial_maskA

    overlap = np.sum(mask) / (w * h)

    return overlap, mask  #mask为被temple图的mask,此工程中默认为feat1的mask

def main():
    with torch.no_grad():
        
        fm = model_api(0)
        feat_info = feature_info()

        path = r"0000.bmp"
        img = np.array(Img.open(path))
        feat1 = get_feature(img, fm, path)
        
        path = r"0002.bmp"
        img = np.array(Img.open(path))
        feat2 = get_feature(img, fm, path)
        
        result, rec_feat = recognize(feat1, feat2, 0)

        h,w = feat1['h'], feat1['w']
        _, overlap_mask = cal_overlap_temple(w, h, rec_feat['model'], feat2['mask'], feat1['mask'])
        level_16_A = cv2.resize(feat1['level_16_img'], None, fx=2, fy=2)*overlap_mask
        level_16_B = cv2.resize(feat2['level_16_img'], None, fx=2, fy=2)*feat2['mask']

        input_img = {'img': level_16_A, 'img_H': level_16_B}

        Htrans = torch.from_numpy(rec_feat['model'].params)

        image_warp = inv_warp_image(torch.from_numpy(input_img['img_H']), Htrans)
        #image_warp = np.array(image_warp * 255).astype(np.uint8)
        #imageB = (input_img['img'] * 255).astype(np.uint8)
        image_warp = np.array(image_warp ).astype(np.uint8)*feat1['mask']
        imageB = (input_img['img'] ).astype(np.uint8)
        b = np.zeros_like(imageB)
        g = image_warp
        r = imageB
        img_match = cv2.merge([r, g, b])
        score = fm.compare(img_match)
        #print(score)
        cv2.imwrite(path[:-4] + "_match1.bmp", image_warp)
        cv2.imwrite(path[:-4] + "_match2.bmp", imageB)
        img_match = cv2.merge([b, g, r])
        cv2.imwrite(path[:-4] + "_match3_" + str(int(score)) + ".bmp", img_match)

        
if __name__ == "__main__":
    main()
    