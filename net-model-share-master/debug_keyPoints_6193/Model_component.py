# from multiprocessing.connection import deliver_challenge
import os
import cv2
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.linalg import hadamard
from pathlib import Path

class AbstractTransform:

    def transform(self, src):
        pass

    def transform_sequence(self, src_seq):
        return map(lambda x: self.transform(x), src_seq)

    def inverse_transform(self, src):
        pass

    def inverse_transform_sequence(self, src_seq):
        return map(lambda x: self.inverse_transform(x), src_seq)


class WalshHadamardTransform(AbstractTransform):

    def __init__(self, coeff=None):
        self.__coeff = coeff

    # @cached
    def _get_matrix(self, size):
        # Generating Hadamard matrix of size 'size'
        n = int(math.log(size, 2))
        row = [1 / (np.sqrt(2) ** n) for i in range(0, size)]
        matrix = [list(row) for i in range(0, size)]
        for i in range(0, n):
            for j in range(0, size):
                for k in range(0, size):
                    if (j / 2 ** i) % 2 == 1 and (k / 2 ** i) % 2 == 1:
                        matrix[j][k] = -matrix[j][k]
                        if self.__coeff is not None:
                            if matrix[j][k] - self.__coeff < 1e-6:
                                # print('Substitution works! ' + str(matrix[j][k]) + ' ' + str(self.__coeff))
                                matrix[j][k] = 0

        # Producing Walsh-Hadamard matrix by ordering frequencies in ascending order
        matrix.sort(key=lambda x: sum(map(lambda a: a[0] * a[1] < 0, zip(x[1:], x[:-1]))))
        return matrix

    def transform(self, src):
        size = src.shape[0]
        h = np.matrix(self._get_matrix(size))
        return h * src * h

    def inverse_transform(self, src):
        return self.transform(src)

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)    # [64,16,16,64,1]
        spl = t_1.split(self.block_size, 3)     # 在第3维分割成每块包含block_size   len(spl) : 8     spl[0].shape : [64,16,16,8,1]
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]    # stack is a list, stack[0].shape : [64,16,128,1]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)     # [64,128,128,1]
        output = output.permute(0, 3, 1, 2)     # [64,1,128,128]

        return output

# def net_forward(img, model):
#     '''
#     input: 1 1 H W
#     return: 
#     '''
#     H, W = img.shape[2], img.shape[3]
#     with torch.no_grad():
#         model.eval()
#         outs = model.forward(img)
#         semi, coarse_desc = outs['semi'], outs['desc']

#     pass

def L2Norm(x):
    eps = 1e-10
    norm = torch.sqrt(torch.sum(x * x, dim = 1) + eps)
    x= x / norm.unsqueeze(-1).expand_as(x)
    return x

def thresholding_desc(descs):
    norm = torch.sqrt(torch.sum(descs * descs, dim = 1)) * 0.2
    norm = norm.int().unsqueeze(-1).expand_as(descs).float()
    descs = torch.where(descs < norm, torch.sqrt(descs), torch.sqrt(norm)).int()
    return descs

def get_orientation(img, keypoints, patch_size=8):

    esp = 1e-6
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

def get_orientation_batch(img, keypoints, patch_size=16, device='cpu'):
    '''
    img:tensor
    '''
    batch, c, h, w = img.shape
    offset = patch_size // 2
    device = img.device

    Gx=torch.zeros((batch, c, h+patch_size, w+patch_size), dtype=img.dtype, device=img.device)
    Gy=torch.zeros((batch, c, h+patch_size, w+patch_size), dtype=img.dtype, device=img.device)

    Gx0=torch.zeros_like(img)
    Gx2=torch.zeros_like(img)
    Gy0=torch.zeros_like(img)
    Gy2=torch.zeros_like(img)

    Gx0[:,:,:,1:-1] = img[:,:,:,:-2]
    Gx2[:,:,:,1:-1] = img[:,:,:,2:]
    Gx[:,:,offset:-offset,offset:-offset] = (Gx0 - Gx2)

    Gy0[:,:,1:-1,:] = img[:,:,:-2,:]
    Gy2[:,:,1:-1,:] = img[:,:,2:,:]
    Gy[:,:,offset:-offset,offset:-offset] = (Gy2 - Gy0)

    Gxx = Gx*Gx
    Gyy = Gy*Gy
    Gxy = Gx*Gy

    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, patch_size-1), torch.linspace(-1, 1, patch_size-1)), dim=2)  # 产生两个网格
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()
    
    keypoints_num = keypoints.size(1)
    keypoints_correct = torch.round(keypoints.clone())
    keypoints_correct += offset
    
    src_pixel_coords = coor_cells.unsqueeze(0).repeat(batch, keypoints_num,1,1,1)
    src_pixel_coords = src_pixel_coords.float() * (patch_size // 2 - 1) + keypoints_correct.unsqueeze(2).unsqueeze(2).repeat(1,1,patch_size-1,patch_size-1,1)
    
    src_pixel_coords = src_pixel_coords.view([batch, keypoints_num, -1, 2])
    batch_image_coords_correct = torch.linspace(0, (batch-1)*(h+patch_size)*(w+patch_size), batch).long().to(device)
    src_pixel_coords_index = (src_pixel_coords[:,:,:,0] + src_pixel_coords[:,:,:,1]*(w+patch_size)).long()
    src_pixel_coords_index  = src_pixel_coords_index + batch_image_coords_correct[:,None,None]

    a = torch.sum(Gxx.take(src_pixel_coords_index),dim=-1)
    b = torch.sum(Gyy.take(src_pixel_coords_index),dim=-1)
    c = torch.sum(Gxy.take(src_pixel_coords_index),dim=-1)
    
    eps = 1e-12
    degree_value = 2*c / (a - b + eps)
    angle = torch.atan(degree_value)

    angle = angle*57.29578049 #180/(3.1415926)
    a_small_b_mask = (a < b)
    c_mask = (c >= 0)
    angle[~a_small_b_mask] = angle[~a_small_b_mask] / 2
    angle[a_small_b_mask*c_mask] = (angle[a_small_b_mask*c_mask] + 180) /2
    angle[a_small_b_mask*(~c_mask)] = (angle[a_small_b_mask*(~c_mask)] - 180) /2
    angle_mask = (angle > 0)
    angle[angle_mask] = angle[angle_mask] - 90
    angle[~angle_mask] = angle[~angle_mask] + 90

    return angle

def saveImg(img: np.array, filename):
    import cv2
    cv2.imwrite(filename, img)

def is_number(s):
    try:
        int(s)
        return True
    except(TypeError,ValueError):
        pass
    return False

def get_rotation_matrix(theta):
    batchsize = len(theta)
    theta_r = theta*3.14159265/180
    rotate_maxtrix = torch.zeros((batchsize, 3,3)).to(theta.device)
    rotate_maxtrix[:,0,0] = torch.cos(theta_r)
    rotate_maxtrix[:,0,1] = torch.sin(theta_r)
    rotate_maxtrix[:,0,2] = 0
    rotate_maxtrix[:,1,0] = -torch.sin(theta_r)
    rotate_maxtrix[:,1,1] = torch.cos(theta_r)
    rotate_maxtrix[:,1,2] = 0
    rotate_maxtrix[:,2,0] = 0
    rotate_maxtrix[:,2,1] = 0
    rotate_maxtrix[:,2,2] = 1

    return rotate_maxtrix

# from utils.utils import inv_warp_image_batch
def inv_warp_patch_batch(img, points_batch, theta_batch, patch_size=16, sample_size = 16, mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param points:
        batch of points
        tensor [batch_size, N, 2]
    :param theta:
        batch of orientation [-90 +90]
        tensor [batch_size, N]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, patch_size, patch_size]
    '''
    batch_size, points_num = points_batch.size(0),points_batch.size(1)
    points = points_batch.view(-1,2).to(img.device)
    theta = theta_batch.view(-1)
    Batch = len(points)     # (Batch_size * Num) * 2

    mat_homo_inv = get_rotation_matrix(theta)
    # mat_homo_inv = torch.tensor([[1.,0.,0.], [0.,1.,0.], [0., 0., 1.]]).to(img.device)
    # mat_homo_inv = mat_homo_inv.repeat(Batch, 1, 1)

    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)
    device = img.device
    _, channel, H, W = img.shape
    

    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, patch_size), torch.linspace(-1, 1, patch_size)), dim=2)  # 产生两个网格
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv.to(device), device)
    src_pixel_coords = src_pixel_coords.view([Batch, patch_size, patch_size, 2])
    src_pixel_coords = src_pixel_coords.float() * (sample_size // 2) + points.unsqueeze(1).unsqueeze(1).repeat(1, patch_size, patch_size, 1)


    src_pixel_coords_ofs = torch.floor(src_pixel_coords)
    src_pixel_coords_ofs_Q11 = src_pixel_coords_ofs.view([Batch, -1, 2])

    batch_image_coords_correct = torch.linspace(0, (batch_size-1)*H*W, batch_size).long().to(device)

    src_pixel_coords_ofs_Q11 = (src_pixel_coords_ofs_Q11[:,:,0] + src_pixel_coords_ofs_Q11[:,:,1]*W).long()
    src_pixel_coords_ofs_Q21 = src_pixel_coords_ofs_Q11 + 1
    src_pixel_coords_ofs_Q12 = src_pixel_coords_ofs_Q11 + W
    src_pixel_coords_ofs_Q22 = src_pixel_coords_ofs_Q11 + W + 1

    warp_weight = (src_pixel_coords - src_pixel_coords_ofs).view([Batch, -1, 2])

    alpha = warp_weight[:,:,0]
    beta = warp_weight[:,:,1]
    src_Q11 = img.take(src_pixel_coords_ofs_Q11.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q21 = img.take(src_pixel_coords_ofs_Q21.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q12 = img.take(src_pixel_coords_ofs_Q12.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q22 = img.take(src_pixel_coords_ofs_Q22.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)

    warped_img = src_Q11*(1 - alpha)*(1 - beta) + src_Q21*alpha*(1 - beta) + \
        src_Q12*(1 - alpha)*beta + src_Q22*alpha*beta
    warped_img = warped_img.view([Batch, 1, patch_size, patch_size])
    
    return warped_img

def inv_warp_patch_batch_rect(img, points_batch, theta_batch, patch_size=[32, 8], sample_size = [32, 8], mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param points:
        batch of points
        tensor [batch_size, N, 2]
    :param theta:
        batch of orientation [-90 +90]
        tensor [batch_size, N]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, patch_size, patch_size]
    '''
    batch_size, points_num = points_batch.size(0),points_batch.size(1)
    points = points_batch.contiguous().view(-1,2).to(img.device)
    theta = theta_batch.contiguous().view(-1)
    Batch = len(points)     # (Batch_size * Num) * 2

    mat_homo_inv = get_rotation_matrix(theta)
    # mat_homo_inv = torch.tensor([[1.,0.,0.], [0.,1.,0.], [0., 0., 1.]]).to(img.device)
    # mat_homo_inv = mat_homo_inv.repeat(Batch, 1, 1)

    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)
    device = img.device
    _, channel, H, W = img.shape
    

    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, patch_size[1]), torch.linspace(-1, 1, patch_size[0])), dim=2)  # 产生两个网格
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous().float()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv.float(), device)
    src_pixel_coords = src_pixel_coords.view([Batch, patch_size[0], patch_size[1], 2])
    src_pixel_coords = src_pixel_coords.float() * (torch.tensor(sample_size).to(device) / 2) + points.unsqueeze(1).unsqueeze(1).repeat(1, patch_size[0], patch_size[1], 1)

    src_pixel_coords_ofs = torch.floor(src_pixel_coords)
    src_pixel_coords_ofs_Q11 = src_pixel_coords_ofs.view([Batch, -1, 2])

    batch_image_coords_correct = torch.linspace(0, (batch_size-1)*H*W, batch_size).long().to(device)

    src_pixel_coords_ofs_Q11 = (src_pixel_coords_ofs_Q11[:,:,0] + src_pixel_coords_ofs_Q11[:,:,1]*W).long()
    src_pixel_coords_ofs_Q21 = src_pixel_coords_ofs_Q11 + 1
    src_pixel_coords_ofs_Q12 = src_pixel_coords_ofs_Q11 + W
    src_pixel_coords_ofs_Q22 = src_pixel_coords_ofs_Q11 + W + 1

    warp_weight = (src_pixel_coords - src_pixel_coords_ofs).view([Batch, -1, 2])

    alpha = warp_weight[:,:,0]
    beta = warp_weight[:,:,1]
    src_Q11 = img.take(src_pixel_coords_ofs_Q11.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q21 = img.take(src_pixel_coords_ofs_Q21.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q12 = img.take(src_pixel_coords_ofs_Q12.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q22 = img.take(src_pixel_coords_ofs_Q22.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)

    warped_img = src_Q11*(1 - alpha)*(1 - beta) + src_Q21*alpha*(1 - beta) + \
        src_Q12*(1 - alpha)*beta + src_Q22*alpha*beta
    warped_img = warped_img.view([Batch, 1, patch_size[0], patch_size[1]])

    return warped_img

def get_distinction_percent(hist_match: np.array, hist_no_match: np.array, save_csv, scale=1, hit_rate: np.array=None):
    bins = hist_match.shape[0]
    # match = hist
    match_num = 0.
    no_match_num = 0.
    for idx in range(bins):
        match_num += round(hist_match[idx], 4) * scale
        no_match_num += round(hist_no_match[idx], 4) * scale

    gap = 0.05
    percent = 0
    gap_thre_list = [ ]     # 存对应百分比的阈值
    while round(percent, 2) <= 1:
        percent += gap
        num_val = round(no_match_num, 4) * round(percent, 4)

        sum_v = 0
        for v in range(bins):
            sum_v += round(hist_no_match[v], 4)
            if round(sum_v, 4) >= round(num_val, 4):
                gap_thre_list.append(v - 1)     # 直方图，考虑v - 1的阈值，所以减1(不统计当前的阈值)
                break
    assert len(gap_thre_list) == 20, "erro! len(gap_thre_list) != 20"
    print("gap_thre_list: ", gap_thre_list)

    x_axis = [ ]
    total_match_num = [ ]
    for idx in range(len(gap_thre_list)):
        count = 0
        thre_v = gap_thre_list[idx]
        for v in range(thre_v + 1):
            count += hist_match[v]

        total_match_num.append(count / round(match_num, 4))
        x_axis.append(str(round((idx + 1) * 5, 5)) + '%')
        
    match_mean = np.sum(hist_match * np.linspace(0, bins - 1, bins)) / round(np.sum(hist_match), 4)
    no_match_mean = np.sum(hist_no_match * np.linspace(0, bins - 1, bins)) / round(np.sum(hist_no_match), 4)
    for i in range(hist_match.shape[0]):
        if round(hist_match[i], 4) != 0:
            min_match_val = i
            break
    for i in reversed(range(hist_match.shape[0])):
        if round(hist_match[i], 4) != 0:
            max_match_val = i
            break
    for i in range(hist_no_match.shape[0]):
        if round(hist_no_match[i], 4) != 0:
            min_no_match_val = i
            break
    for i in reversed(range(hist_no_match.shape[0])):
        if round(hist_no_match[i], 4) != 0:
            max_no_match_val = i
            break

    content = [[x_axis[i], gap_thre_list[i] + 1, round(total_match_num[i], 6)] for i in range(len(total_match_num))]
    df = pd.DataFrame(content, columns=['x_axis', 'match', 'no_match'])
    df = pd.DataFrame(np.insert(df.values, 0, values=['min', round(min_match_val, 4), round(min_no_match_val, 4)], axis=0))
    df = pd.DataFrame(np.insert(df.values, 1, values=['max', round(max_match_val, 4), round(max_no_match_val, 4)], axis=0))
    df = pd.DataFrame(np.insert(df.values, 2, values=['mean', round(match_mean, 4), round(no_match_mean, 4)], axis=0))
    if hit_rate is not None:
        # df = pd.DataFrame(np.insert(df.values, -1, values=['hit_rate', round(hit_rate, 4), ...], axis=0))
        # df.append(pd.DataFrame(['hit_rate', round(hit_rate, 4)]), ignore_index=True)
        hit_rate_mean = torch.mean(hit_rate).cpu().numpy()
        df.loc[len(df)] = [hit_rate_mean, '/', '/']

    # df.append(['min', round(min_match_val, 4), round(min_no_match_val, 4)], ignore_index=True)
    # df.append(['max', round(max_match_val, 4), round(max_no_match_val, 4)], ignore_index=True)
    # df.append(['mean', round(match_mean, 4), round(no_match_mean, 4)], ignore_index=True)
    df.to_csv(save_csv, index=False, encoding='utf_8_sig')

    pass

def get_distinction_percent_closesampling(hist_match: np.array, hist_no_match: np.array, save_csv, scale=1):
    ''' TODO:输出nomatch每个汉明距下的区分度'''
    bins = hist_match.shape[0]
    # match = hist
    match_num = 0.
    no_match_num = 0.
    for idx in range(bins):
        match_num += round(hist_match[idx], 4) * scale
        no_match_num += round(hist_no_match[idx], 4) * scale

    gap = 0.05
    percent = 0
    gap_thre_list = [ ]     # 存对应百分比的阈值
    while round(percent, 2) <= 1:
        percent += gap
        num_val = round(no_match_num, 4) * round(percent, 4)

        sum_v = 0
        for v in range(bins):
            sum_v += round(hist_no_match[v], 4)
            if round(sum_v, 4) >= round(num_val, 4):
                gap_thre_list.append(v - 1)     # 直方图，考虑v - 1的阈值，所以减1(不统计当前的阈值)
                break
    assert len(gap_thre_list) == 20, "erro! len(gap_thre_list) != 20"
    print("gap_thre_list: ", gap_thre_list)

    x_axis = [ ]
    total_match_num = [ ]
    for idx in range(len(gap_thre_list)):
        count = 0
        thre_v = gap_thre_list[idx]
        for v in range(thre_v + 1):
            count += hist_match[v]

        total_match_num.append(count / round(match_num, 4))
        x_axis.append(str(round((idx + 1) * 5, 5)) + '%')
        

    content = [[x_axis[i], gap_thre_list[i] + 1, round(total_match_num[i], 6)] for i in range(len(total_match_num))]
    df = pd.DataFrame(content, columns=['x_axis', 'thre_val', 'percent'])
    df.to_csv(save_csv, index=False, encoding='utf_8_sig')

    pass

def get_nomatch_percent(no_match_results, match_results, save_csv, dua=0.05):
    # 根据nomatch结果找出占比以0.05为间隔时，汉明距阈值，然后根据这个阈值计算匹配点在该阈值时的总占比
    v_nomatch_list = no_match_results.tolist()
    v_nomatch_list.sort()
    percent = 0
    len_v = len(v_nomatch_list)
    thre_list = [0]
    contents = []
    while percent<0.5:
        percent += dua
        thre_v = len_v * percent
        print(int(thre_v), len_v)
        print(v_nomatch_list[int(thre_v) - 1])
        index = int(thre_v - 1)
        va1 = v_nomatch_list[index]
        va2 = v_nomatch_list[index + 1]
        if va1 != va2:
            thre_list.append(v_nomatch_list[index + 1])
        else:
            thre_list.append(v_nomatch_list[index])

    
    print(thre_list)
    v_match_list = match_results.tolist()
    total_match_num = len(v_match_list)
    total_match_percent = 0
    for i in range(len(thre_list) - 1):
        count = 0
        min_v = thre_list[i]
        max_v = thre_list[i + 1]
        for v_match in v_match_list:
            if v_match >= min_v and v_match < max_v:
                count += 1
    
        # k_name = str(min_v)+'~'+str(max_v)
        k_name = str(max_v)
        x_axis = str(round((i + 1) * 5, 5)) + '%'
        total_match_percent += count
        contents.append(['match', k_name, str(round(count / total_match_num * 100, 5)) + '%', x_axis, str(round(total_match_percent / total_match_num * 100, 5)) + '%'])
    df_s = pd.DataFrame(contents, columns=['type', 'distance', 'percent', 'x_axis', 'total_percent'])

    ''' 分别统计匹配对/非匹配对的mean/max/min '''
    match_mean = match_results.mean()
    match_min = match_results.min()
    match_max = match_results.max()
    no_match_mean = no_match_results.mean()
    no_match_min = no_match_results.min()
    no_match_max = no_match_results.max()
    df_s.loc[df_s.shape[0], :3] = ['/', 'matched', 'no_matched']
    df_s.loc[df_s.shape[0], :3] = ['min', match_min, no_match_min]
    df_s.loc[df_s.shape[0], :3] = ['max', match_max, no_match_max]
    df_s.loc[df_s.shape[0], :3] = ['mean', match_mean, no_match_mean]
    df_s.to_csv(save_csv, index=False, encoding='utf_8_sig')


'''pnts: from func"getPtsFromHeatmap", pnts --> netout_as_label'''
def points_to_2D(pnts, H, W):
    labels = np.zeros((H, W))
    pnts = pnts.astype(int)
    flag = (pnts[:, 1]<136) & (pnts[:, 0]<136) & (pnts[:, 0]>0) & (pnts[:, 1]>0)
    pnts = pnts[flag, :]
    labels[pnts[:, 1], pnts[:, 0]] = 1
    return labels

def homograghy_centorcrop(homograghy, Hdev_top, Wdev_left):
    homograghy_dev = torch.tensor([[0, 0, Wdev_left], [0, 0, Hdev_top], [0, 0, 0]], dtype=torch.float32)
    scale = torch.tensor([[1, 0, Wdev_left], [0, 1, Hdev_top], [0, 0, 1]], dtype=torch.float32)
    homograghy = (homograghy - homograghy_dev) @ scale
    return homograghy

def homograghy_transform(homograghy, Hdev_top, Wdev_left):
    '''
    裁剪: -
    扩边: +
    '''
    scale = torch.tensor([[1, 0, Wdev_left], [0, 1, Hdev_top], [0, 0, 1]], dtype=torch.float32).to(homograghy.device)
    homograghy = scale @ homograghy @ scale.inverse()
    return homograghy

def sample_homography_cv(H, W, max_angle=30, n_angles=25, norm=False):
    scale = 1
    angles = np.linspace(-max_angle, max_angle, num=n_angles)
    angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
    M = [cv2.getRotationMatrix2D((W / 2, H / 2), i, scale) for i in angles]
    # center = np.mean(pts2, axis=0, keepdims=True)
    M = [np.concatenate((m, [[0, 0, 1.]]), axis=0) for m in M]

    valid = np.arange(n_angles)
    idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
    homo = M[idx]

    # if norm:
    #     homo = 

    return homo
    
def flattenDetection(semi, tensor=False):
    '''
    Flatten detection output

    :param semi:
        output from detector head
        tensor [65, Hc, Wc]
        :or
        tensor (batch_size, 65, Hc, Wc)

    :return:
        3D heatmap
        np (1, H, C)
        :or
        tensor (batch_size, 65, Hc, Wc)

    '''
    batch = False
    if len(semi.shape) == 4:
        batch = True
        batch_size = semi.shape[0]      # [batch_size,65,16,16]
    # if tensor:
    #     semi.exp_()
    #     d = semi.sum(dim=1) + 0.00001
    #     d = d.view(d.shape[0], 1, d.shape[1], d.shape[2])
    #     semi = semi / d  # how to /(64,15,20)

    #     nodust = semi[:, :-1, :, :]
    #     heatmap = flatten64to1(nodust, tensor=tensor)
    # else:
    # Convert pytorch -> numpy.
    # --- Process points.
    # dense = nn.functional.softmax(semi, dim=0) # [65, Hc, Wc]
    if batch:
        dense = nn.functional.softmax(semi, dim=1)  # [batch_size, 65, Hc, Wc] 在dim=1维度进行softmax，相当于65个分类
        # Remove dustbin.
        nodust = dense[:, :-1, :, :]                # [batch_size, 64, Hc, Wc]

        # for i in range(nodust.size(2)):
        #     for j in range(nodust.size(3)):
        #         if dense[:, -1, i, j] > (1 / 65):
        #             nodust[:, :, i, j] = 0

    else:
        dense = nn.functional.softmax(semi, dim=0) # [65, Hc, Wc]
        nodust = dense[:-1, :, :].unsqueeze(0)

        # for i in range(nodust.size(2)):
        #     for j in range(nodust.size(3)):
        #         if dense[-1, i, j] > (1 / 65):
        #             nodust[:, :, i, j] = 0
    # Reshape to get full resolution heatmap.
    # heatmap = flatten64to1(nodust, tensor=True) # [1, H, W]
    depth2space = DepthToSpace(8)   # 实例化一个DepthToSpace类
    heatmap = depth2space(nodust)                   # [batch, 1, 128, 128]
    heatmap = heatmap.squeeze(0) if not batch else heatmap      # squeeze 只会对维度为1的维度进行压缩

    return heatmap

def nms_fast(in_corners, H, W, dist_thresh):
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
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.因为grid已经pading，此时pt的坐标就是对应特征点坐标（rc的坐标）
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

def nms_fast_torch(in_corners, H, W, dist_thresh, device='cpu'):
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
    in_corners = in_corners.to(device)
    grid = torch.zeros((H, W), dtype=torch.int).to(device)  # Track NMS data.
    inds = torch.zeros((H, W), dtype=torch.int).to(device)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = torch.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().int()  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return torch.zeros((3, 0)).int().to(device), torch.zeros(0).int().to(device)
    if rcorners.shape[1] == 1:
        out = torch.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, torch.zeros((1)).int().to(device)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = F.pad(grid, (pad, pad, pad, pad), mode='constant').to(device)
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.因为grid已经pading，此时pt的坐标就是对应特征点坐标（rc的坐标）
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = torch.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx].long()
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = torch.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

def soft_nms_fast(in_corners, H, W, conf_thresh, dist_thresh):
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
    thr = np.sqrt(2 * pad**2)
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            x1, y1 = pt[0] - pad, pt[1] - pad
            x2, y2 = pt[0] + pad + 1 , pt[1] + pad + 1
            grid[pt[1], pt[0]] = -1
            for h in range(y1 - pad, y2 - pad):
                for w in range(x1 - pad, x2 - pad):
                    if h == pt[1] - pad and w == pt[0] - pad:
                        continue
                    idx0 = rcorners[0,:] == w
                    idx1 = rcorners[1,:] == h
                    idx = idx0 & idx1   # patch内是否包含特征点

                    grid[h + pad, w + pad] = 0  #该点不是特征点就置0
                    if True in idx:
                        dis = np.linalg.norm(rc - np.stack((w, h)))
                        scor_ori = corners[2, idx]
                        scor_new = scor_ori * np.exp(-(thr - dis) / thr)      #np.exp(-(thr-1.5)/(thr)) 约等于0.7788
                        if scor_new > scor_ori * np.exp(-(thr - 2.5) / thr) and scor_new > conf_thresh:
                            grid[h + pad, w + pad] = -1     # save
                    
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
        
def getPtsFromHeatmap(heatmap, conf_thresh, nms_dist, bord=0, soft_nms=False):
    '''
    :param self:
    :param heatmap:
        np (H, W)
    :return:
    '''
    heatmap = heatmap.squeeze()
    # print("heatmap sq:", heatmap.shape)
    H, W = heatmap.shape[0], heatmap.shape[1]
    ys, xs = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    sparsemap = (heatmap >= conf_thresh)
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = xs # abuse of ys, xs
    pts[1, :] = ys
    pts[2, :] = heatmap[ys, xs]  # check the (x, y) here
    if soft_nms:
        pts, _ = soft_nms_fast(pts, H, W, conf_thresh=conf_thresh, dist_thresh=nms_dist) 
    else:
        pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    # bord = self.border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts

def getPtsFromHeatmap_torch(heatmap, conf_thresh, nms_dist, bord=0, soft_nms=False, device='cpu'):
    '''
    :param self:
    :param heatmap:
        np (H, W)
    :return:
    '''
    heatmap = heatmap.squeeze().to(device)
    # print("heatmap sq:", heatmap.shape)
    H, W = heatmap.shape[0], heatmap.shape[1]
    ys, xs = torch.where(heatmap >= conf_thresh)  # Confidence threshold.
    sparsemap = (heatmap >= conf_thresh)
    if len(xs) == 0:
        return torch.zeros((3, 0)).to(device)
    pts = torch.zeros((3, len(xs))).to(device)  # Populate point data sized 3xN.
    pts[0, :] = xs # abuse of ys, xs
    pts[1, :] = ys
    pts[2, :] = heatmap[ys, xs]  # check the (x, y) here
    if soft_nms:
        pts, _ = soft_nms_fast(pts, H, W, conf_thresh=conf_thresh, dist_thresh=nms_dist) 
    else:
        pts, _ = nms_fast_torch(pts, H, W, dist_thresh=nms_dist, device=device)  # Apply NMS.
    inds = torch.argsort(pts[2, :], descending=True)
    pts = pts[:, inds]  # Sort by confidence.
    # Remove points along border.
    # bord = self.border_remove
    toremoveW = torch.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = torch.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = torch.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts

def inv_warp_image_batch_cv2(img, mat_homo, device='cpu', mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    '''
    # compute inverse warped points
    if len(img.shape) == 2:
        img = img.view(1, 1, img.shape[0], img.shape[1])
    if len(img.shape) == 3:
        img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
    if len(mat_homo.shape) == 2:
        mat_homo = mat_homo.view(1,3,3)

    Batch, channel, H, W = img.shape

    warped_img = cv2.warpPerspective(img.squeeze().numpy(), mat_homo.squeeze().numpy(), (W, H))
    warped_img = torch.tensor(warped_img, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

    # warped_img = cv2.warpAffine(img.squeeze().numpy(), mat_homo_inv[0, :2, :].squeeze().numpy(), (W, H))
    # warped_img = torch.tensor(warped_img, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    return warped_img

def inv_warp_image(img, mat_homo, device='cpu', mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [H, W]
    '''
    warped_img = inv_warp_image_batch_cv2(img, mat_homo, device, mode)
    return warped_img.squeeze()

def compute_valid_mask(image_shape, homography, device='cpu', erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        input_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        `erosion_radius: radius of the margin to be discarded.

    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """
    # mask = H_transform(tf.ones(image_shape), homography, interpolation='NEAREST')
    # mask = H_transform(tf.ones(image_shape), homography, interpolation='NEAREST')
    if homography.dim() == 2:
        homography = homography.view(-1, 3, 3)
    batch_size = homography.shape[0]
    mask = torch.ones(batch_size, 1, image_shape[0], image_shape[1]).to(device)
    mask = inv_warp_image_batch_cv2(mask, homography, device=device, mode='nearest')
    mask = mask.view(batch_size, image_shape[0], image_shape[1])
    mask = mask.cpu().numpy()
    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
        for i in range(batch_size):
            mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)

    return torch.tensor(mask).to(device)

def warp_points(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    # homographies = homographies.unsqueeze(0) if len(homographies.shape) == 2 else homographies
    batch_size = homographies.shape[0]
    points = torch.cat((points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1)    # expand points to (x, y, 1)
    points = points.to(device)
    homographies = homographies.view(batch_size*3,3)
    # warped_points = homographies*points
    # points = points.double()
    warped_points = homographies@points.transpose(0,1)
    # warped_points = np.tensordot(homographies, points.transpose(), axes=([2], [0]))
    '''归一化'''
    # normalize the points
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]   # gui yi hua 

    # toremoveX = torch.logical_or(warped_points[0, :] < -1, warped_points[0, :] > 1)
    # toremoveY = torch.logical_or(warped_points[1, :] < -1, warped_points[1, :] > 1)
    # toremove = torch.logical_or(toremoveX, toremoveY)
    # warped_points = warped_points[:, ~toremove]
    # warped_points = warped_points.view([batch_size, 3, -1])
    # warped_points = warped_points.transpose(2, 1)
    # warped_points = warped_points[:, :, :2]

    return warped_points[0,:,:] if no_batches else warped_points

def filter_points(points, shape, return_mask=False, device='cpu'):
    ### check!
    points = points.float()
    shape = shape.float().to(device)
    mask = (points >= 0) * (points <= shape-1)
    mask = (torch.prod(mask, dim=-1) == 1)
    if return_mask:
        return points[mask], mask
    return points[mask]

def draw_merge_image(input_imgA, input_imgB, save_path):
    # img_2D_warped = sample['imgA_warped'].numpy().squeeze()
    b = np.zeros_like(input_imgA)
    g = input_imgA * 255  #  旋转后的模板
    r = input_imgB * 255    
    image_merge = cv2.merge([b, g, r])
    image_merge = cv2.resize(image_merge, None, fx=3, fy=3)
    cv2.imwrite(save_path, image_merge)

def draw_keypoints_compareNMS(input_img, input_pts, color=(0, 255, 0), radius=3, s=3):

    anchor = int(input_img.shape[1])
    img = np.hstack((input_img, input_img)) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(input_pts['pts']):
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)    # 图像坐标系，先列后行
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    for c in np.stack(input_pts['pts_soft']):
        c[0] += anchor
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=-1)
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    # flag = (label_pts[:, 1]<anchor) & (label_pts[:, 0]<anchor) & (label_pts[:, 0]>0) & (label_pts[:, 1]>0)
    # label_pts = label_pts[flag, :]

    return img

def draw_keypoints_pair(input_img, input_pts, label_pts, color=(0, 255, 0), radius=3, s=3):

    anchor = int(input_img['img_1'].shape[1])
    img = np.hstack((input_img['img_1'], input_img['img_2'])) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(input_pts['pts']):
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[1:]).astype(int)), radius, color, thickness=-1)    # 图像坐标系，先列后行
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    if not (input_pts['pts_B'].size == 0):
        for c in np.stack(input_pts['pts_B']):
            c[1] += anchor
            # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
            cv2.circle(img, tuple((s * c[1:]).astype(int)), radius, (255, 0, 0), thickness=-1)
            # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
        
    # flag = (label_pts[:, 1]<anchor) & (label_pts[:, 0]<anchor) & (label_pts[:, 0]>0) & (label_pts[:, 1]>0)
    # label_pts = label_pts[flag, :]
    if label_pts.size == 0:
        return img

    for c in np.stack(label_pts):
        if c.size == 1:
            break
        c[0] += anchor
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 255, 0), thickness=1)
    
    '''绘制网格线'''
    # for i in range(0, 384, 24):
    #     cv2.line(img, (0, i), (383, i), (192, 192, 192), 1, 1)
    #     cv2.line(img, (i, 0), (i, 383), (192, 192, 192), 1, 1)
    
    return img

def draw_keypoints_pair_tradition(input_img, input_pts, label_pts, color=(0, 255, 0), radius=3, s=3):

    anchor = int(input_img['img_1'].shape[1])
    img = np.hstack((input_img['img_1'], input_img['img_2'])) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(input_pts['tra_pts']):
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)    # 图像坐标系，先列后行
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    for c in np.stack(input_pts['tra_pts_B']):
        c[0] += anchor
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=-1)
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    # flag = (label_pts[:, 1]<anchor) & (label_pts[:, 0]<anchor) & (label_pts[:, 0]>0) & (label_pts[:, 1]>0)
    # label_pts = label_pts[flag, :]
    if label_pts.size == 0:
        return img

    for c in np.stack(label_pts):
        if c.size == 1:
            break
        c[0] += anchor
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 255, 0), thickness=1)
    
    return img

def draw_keypoints(img, corners, label_corners=None, color=(0, 255, 0), radius=3, s=3):

    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners):
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[1:]).astype(int)), radius, color, thickness=-1)
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)

    if label_corners is not None:
        for c in np.stack(label_corners):
            # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
            cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)
            # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    return img

def draw_keypoints_pair_train(input_img, pred, label_pts, color=(0, 255, 0), radius=3, s=3):
    '''

    :param img:
        image:
        numpy [H, W]
    :param corners:
        Points
        numpy [N, 2]
    :param color:
    :param radius:
    :param s:
    :return:
        overlaying image
        numpy [H, W]
    '''
    b = np.zeros_like(input_img['img'])
    g = input_img['img_B'] * 255  #  旋转后的模板
    r = input_img['img_AH'] * 255    
    image_merge = cv2.merge([b, g, r])
    # image_merge = cv2.resize(image_merge, None, fx=3, fy=3)
    # cv2.imwrite(os.path.join(str(self.webdir), str(n_iter) + '_' + str(name[0]) + '_match.bmp'), image_merge)

    anchor = int(input_img['img'].shape[1])
    # img = input_img['img'] * 255
    img = np.hstack((input_img['img'], input_img['img_B'])) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    img = np.hstack((img, image_merge))
    if 'pts' in pred:
        for c in np.stack(pred['pts']):
            if c.size == 1:
                break
            cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)    # 图像坐标系，先列后行
            # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    if pred['lab'].size == 0:
        return img
        
    for c in np.stack(pred['lab']):
        if c.size == 1:
            break
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)

    if 'pts_B' in pred:
        for c in np.stack(pred['pts_B']):
            if c.size == 1:
                break
            c[0] += anchor
            cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)
    for c in np.stack(pred['lab_B']):
        if c.size == 1:
            break
        c[0] += anchor
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)

    if 'L2_list' in pred:
        max_val = np.max(pred["L2_list"])

        L2_norm = pred["L2_list"] / max_val
        inds = L2_norm.argsort()
        thr = L2_norm[inds][int(0.1 * len(inds))]   # 描述子排序 取前10%
        
        mask = L2_norm < thr
        level = (1 - L2_norm) * 255
        level = level.round()

        ''' 画出描述子L2距离小于固定值 0.7 '''
        mask_abs = pred["L2_list"] < 0.2
        for idx, (pt1, pt2) in enumerate(zip(pred['lab'], pred['lab_B'])):
            pt2[0] += anchor
            if mask[idx] == True:
                cv2.line(img, tuple(pt1[:2].astype(int)), tuple(pt2[:2].astype(int)), (0, 255, 0), thickness=1)
            if mask_abs[idx] == True:
                cv2.line(img, tuple(pt1[:2].astype(int)), tuple(pt2[:2].astype(int)), (0, 0, 255), thickness=1)

    return img

def get_dis(p_a, p_b):
    if p_a.shape == torch.Size([]) or p_b.shape[0] == torch.Size([]):
        return torch.tensor([])
    if p_a.shape == torch.Size([2]):
        p_a = p_a.unsqueeze(0)
    if p_b.shape == torch.Size([2]):
        p_b = p_b.unsqueeze(0)
    eps = 1e-12
    x = torch.unsqueeze(p_a[:, 0], 1) - torch.unsqueeze(p_b[:, 0], 0)  # N 2 -> NA 1 - 1 NB -> NA NB
    y = torch.unsqueeze(p_a[:, 1], 1) - torch.unsqueeze(p_b[:, 1], 0)
    dis = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + eps)
    return dis

def get_point_pair_repeat(kptA_tensor, kptB_tensor, correspond=2):
    dis = get_dis(kptA_tensor, kptB_tensor)
    if dis.shape[0] == 0 or dis.shape[1] == 0 or dis.shape == torch.Size([]):
        return torch.tensor([]), torch.tensor([])
    a2b_min_id = torch.argmin(dis, dim=1)
    len_p = len(a2b_min_id)
    ch = dis[list(range(len_p)), a2b_min_id] < correspond
    a2b_min_id = a2b_min_id[ch]

    return a2b_min_id, ch

def get_point_pair_inverse(kptA_tensor, kptB_tensor, correspond=2):
    '''找出次近邻（非匹配点）'''
    dis = get_dis(kptA_tensor, kptB_tensor)
    if dis.shape[0] == 0 or dis.shape[1] == 0 or dis.shape == torch.Size([]):
        return torch.tensor([]), torch.tensor([])
    a2b_min_id = torch.argmin(dis, dim=1)
    len_p = len(a2b_min_id)
    ch = dis[list(range(len_p)), a2b_min_id] > correspond
    a2b_min_id = a2b_min_id[ch]

    return a2b_min_id, ch

def get_point_pair_inverse_all(kptA_tensor, kptB_tensor, correspond=2):
    '''desc is binary'''
    dis = get_dis(kptA_tensor, kptB_tensor)
    if dis.shape[0] == 0 or dis.shape[1] == 0 or dis.shape == torch.Size([]):
        return torch.tensor([]), torch.tensor([])

    a2b_min_id = torch.argmin(dis, dim=1)
    len_p = len(a2b_min_id)
    ch = dis[list(range(len_p)), a2b_min_id] < correspond
    dis_mask = torch.ones_like(dis)

    a2b_min_id_ch = a2b_min_id[ch]
    dis_mask[ch, a2b_min_id_ch] = 0
    # for n in range(len_p):
    #     if ch[n] == True:
    #         dis_mask[n, a2b_min_id[n]] = 0
    
    coor = torch.where(dis_mask == 1)
    
    return coor

def get_hamming_dismat(descA, descB, thr1=None, thr2=None, mask_sign='sift'):
    '''
    input: 256dim desc
    '''
    assert descA.shape[1] == 256 and descB.shape[1] == 256, "the desc's dim is not 256!"

    if mask_sign == 'net':
        # same_mask = (((torch.linspace(16, 1, 16)**4).unsqueeze(0) @ hadamard(16)) == \
        #     ((torch.linspace(1, 16, 16)**4).unsqueeze(0) @ hadamard(16))).long().squeeze(0).unsqueeze(1).repeat(1, 8).view(-1).bool()
        same_mask = torch.tensor([1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1]).unsqueeze(1).repeat(1,8).view(-1).bool()
    else:
        same_mask = (((torch.linspace(32, 1, 32)**8).unsqueeze(0) @ hadamard(32)) == \
            ((torch.linspace(1, 32, 32)**8).unsqueeze(0) @ hadamard(32))).long().squeeze(0).unsqueeze(1).repeat(1, 4).view(-1).bool()   # 同号位

    descA_front_same = descA[:, :128][:, same_mask]     # front_same: 前128维中 64位同号
    descB_front_same = descB[:, :128][:, same_mask]
    descA_front_diff = descA[:, :128][:, ~same_mask]
    descB_front_diff = descB[:, :128][:, ~same_mask]

    descA_back_same = descA[:, 128:][:, same_mask]
    descB_back_same = descB[:, 128:][:, same_mask]
    descA_bask_diff = descA[:, 128:][:, ~same_mask]
    descB_bask_diff = descB[:, 128:][:, ~same_mask]

    descA_same = torch.cat((descA_front_same, descA_back_same), dim=-1) # 128
    descB_same = torch.cat((descB_front_same, descB_back_same), dim=-1)
    descA_diff = torch.cat((descA_front_diff, descA_bask_diff), dim=-1) # 128
    descB_diff = torch.cat((descB_front_diff, descB_bask_diff), dim=-1)

    dis_mat_same = descA_same.shape[1] - descA_same @ descB_same.t() - (1 - descA_same) @ (1 - descB_same.t())
    dis_mat_diff = descA_diff.shape[1] - descA_diff @ descB_diff.t() - (1 - descA_diff) @ (1 - descB_diff.t())

    dis_mat_first = dis_mat_same
    dis_mat_second = torch.where(dis_mat_diff < (128 - dis_mat_diff), dis_mat_diff, 128 - dis_mat_diff)
    dis_mat = dis_mat_first + dis_mat_second

    if thr1 != None and thr2 != None:
        mask_same = dis_mat_same < thr1
        mask_diff = dis_mat < thr2
        dis_mat[~mask_same] = 257
        dis_mat[~mask_diff] = 257

    return dis_mat

def cal_hit_rate(ptsA_warp_masked, pts_B, descA_warp_masked, descB, corr=2):
    '''
    单向匹配A->B
    均满足欧氏距离&汉明距离最小的点数 / 满足欧氏距离最小的点数
    '''
    eps = 1e-6
    device = pts_B.device
    d = descA_warp_masked.shape[1]
    idx_b, mask_a = get_point_pair_repeat(ptsA_warp_masked, pts_B, correspond=corr)

    pts_match_mask = torch.zeros((ptsA_warp_masked.shape[0], pts_B.shape[0])).to(device)
    pts_match_mask[mask_a, idx_b] = 1

    hamming_dis = d - descA_warp_masked @ descB.t() - (1 - descA_warp_masked) @ (1 - descB).t()
    hm_a2b_idx_b = torch.argmin(hamming_dis, dim=1)
    # len_p = len(hm_a2b_idx_b)
    # hm_a2b_mask_a = hamming_dis[list(range(len_p)), hm_a2b_idx_b] < 257
    desc_match_mask = torch.zeros_like(pts_match_mask).to(device)
    desc_match_mask[:, hm_a2b_idx_b] = 1

    inner_mask = pts_match_mask * desc_match_mask
    hit_rate = torch.sum(inner_mask) / (torch.sum(pts_match_mask) + eps)

    dic_info = {}
    dic_info.update({"hit_rate": hit_rate})
    return dic_info

def cal_hit_rate_restrict(ptsA_warp, pts_B, descA, descB, corr=2, thr1=None, thr2=None, mask_sign='sift'):
    '''
    单向匹配A->B
    均满足欧氏距离&汉明距离最小的点数 / 满足欧氏距离最小的点数
    1.阈值卡控30,55  2.最近邻次近邻比值<0.99 3.去重,存在多对一时只保留汉明距离最小的匹配对 4.取前30对
    '''

    eps = 1e-6
    device = pts_B.device

    '''1. 阈值卡控'''
    disA2B_mat = get_hamming_dismat(descA, descB, thr1=thr1, thr2=thr2, mask_sign=mask_sign)

    a2b_min_id = torch.argmin(disA2B_mat, dim=1)
    len_p = len(a2b_min_id)
    reshape_as = torch.tensor(list(range(len_p)), device=device)
    ch = disA2B_mat[list(range(len_p)), a2b_min_id] < 257
    all_mask = torch.zeros_like(disA2B_mat, device=device).bool()
    all_mask[ch, a2b_min_id[ch]] = True

    '''2. 最近邻/次近邻 dis_sim<0.99'''
    disA2B_value, _ = torch.topk(disA2B_mat, k=2, dim=-1, largest=False)
    ch_knn = (disA2B_value[:, 0] / disA2B_value[:, 1]) < 0.99
    knn_mask = torch.zeros_like(disA2B_mat, device=device).bool()
    ch = ch * ch_knn
    knn_mask[ch, :] = True
    all_mask = all_mask * knn_mask

    '''3. 去重'''
    disA2B_rep = torch.ones_like(disA2B_mat, device=device) * 257
    disA2B_rep[all_mask] = disA2B_mat[all_mask]
    a2b_row_min_id = torch.argmin(disA2B_rep, dim=0)    # 其中的0可能是指最小值或257
    ch_rep_b_flag = disA2B_mat[a2b_row_min_id, list(range(len(a2b_row_min_id)))] < 257.
    ch_rep_a_flag = torch.zeros_like(ch, device=device).bool()
    ch_rep_a_flag[a2b_row_min_id[ch_rep_b_flag]] = True
    ch = ch * ch_rep_a_flag

    disA2B_a_idx = reshape_as[ch]
    disA2B_b_idx = a2b_min_id[ch]
    disA2B_line = disA2B_mat[ch, a2b_min_id[ch]]

    '''4. 前30对'''
    if disA2B_line.shape[0] >= 30:
        _, dis_sort_id = torch.topk(disA2B_line, k=30, dim=0, largest=False)
        pts_match_num = 30
    else:
        dis_sort_id = torch.tensor(list(range(disA2B_line.shape[0])), device=device)
        pts_match_num = disA2B_line.shape[0]
    
    disA2B_a_idx = disA2B_a_idx[dis_sort_id]
    disA2B_b_idx = disA2B_b_idx[dis_sort_id]

    idx_b, mask_a = get_point_pair_repeat(ptsA_warp, pts_B, correspond=corr)
    # valid_a = torch.tensor(list(range(len_p)), device=device)
    # valid_a = valid_a[warp_mask][mask_a]
    pts_match_mask = torch.zeros((descA.shape[0], descB.shape[0])).to(device)

    # pts_match_mask[valid_a, idx_b] = 1
    pts_match_mask[mask_a, idx_b] = 1
    desc_match_mask = torch.zeros_like(pts_match_mask).to(device)
    desc_match_mask[disA2B_a_idx, disA2B_b_idx] = 1

    inner_mask = pts_match_mask * desc_match_mask
    # hit_rate = torch.sum(inner_mask) / (torch.sum(pts_match_mask) + eps)
    hit_rate = torch.sum(inner_mask) / (pts_match_num + eps)

    dic_info = {}
    dic_info.update({"hit_rate": torch.tensor([hit_rate]).to(device)})
    dic_info.update({"pt_match_num": torch.tensor([torch.sum(mask_a)]).to(device)})
    dic_info.update({"inner_num": torch.tensor([torch.sum(inner_mask)]).to(device)})

    dic_info.update({"inner_mask": inner_mask})
    dic_info.update({"desc_dis_mat": disA2B_mat})
    
    return dic_info

def cal_hit_rate_with_siftinliers(inliers_mask, ptsA_warp, pts_B, descA, descB, corr=2, thr1=None, thr2=None, mask_sign='sift'):
    '''
    单向匹配A->B
    均满足欧氏距离&汉明距离最小的点数 / 满足欧氏距离最小的点数
    1.阈值卡控30,55  2.最近邻次近邻比值<0.99 3.去重,存在多对一时只保留汉明距离最小的匹配对 4.取前30对
    '''

    eps = 1e-6
    device = pts_B.device

    '''1. 阈值卡控'''
    disA2B_mat = get_hamming_dismat(descA, descB, thr1=thr1, thr2=thr2, mask_sign=mask_sign)

    a2b_min_id = torch.argmin(disA2B_mat, dim=1)
    len_p = len(a2b_min_id)
    reshape_as = torch.tensor(list(range(len_p)), device=device)
    ch = disA2B_mat[list(range(len_p)), a2b_min_id] < 257
    all_mask = torch.zeros_like(disA2B_mat, device=device).bool()
    all_mask[ch, a2b_min_id[ch]] = True

    '''2. 最近邻/次近邻 dis_sim<0.99'''
    disA2B_value, _ = torch.topk(disA2B_mat, k=2, dim=-1, largest=False)
    ch_knn = (disA2B_value[:, 0] / disA2B_value[:, 1]) < 0.99
    knn_mask = torch.zeros_like(disA2B_mat, device=device).bool()
    ch = ch * ch_knn
    knn_mask[ch, :] = True
    all_mask = all_mask * knn_mask

    '''3. 去重'''
    disA2B_rep = torch.ones_like(disA2B_mat, device=device) * 257
    disA2B_rep[all_mask] = disA2B_mat[all_mask]
    a2b_row_min_id = torch.argmin(disA2B_rep, dim=0)    # 其中的0可能是指最小值或257
    ch_rep_b_flag = disA2B_mat[a2b_row_min_id, list(range(len(a2b_row_min_id)))] < 257.
    ch_rep_a_flag = torch.zeros_like(ch, device=device).bool()
    ch_rep_a_flag[a2b_row_min_id[ch_rep_b_flag]] = True
    ch = ch * ch_rep_a_flag

    disA2B_a_idx = reshape_as[ch]
    disA2B_b_idx = a2b_min_id[ch]
    disA2B_line = disA2B_mat[ch, a2b_min_id[ch]]

    '''4. 前30对'''
    if disA2B_line.shape[0] >= 30:
        _, dis_sort_id = torch.topk(disA2B_line, k=30, dim=0, largest=False)
        pts_match_num = 30
    else:
        dis_sort_id = torch.tensor(list(range(disA2B_line.shape[0])), device=device)
        pts_match_num = disA2B_line.shape[0]
    
    disA2B_a_idx = disA2B_a_idx[dis_sort_id]
    disA2B_b_idx = disA2B_b_idx[dis_sort_id]

    idx_b, mask_a = get_point_pair_repeat(ptsA_warp, pts_B, correspond=corr)
    # valid_a = torch.tensor(list(range(len_p)), device=device)
    # valid_a = valid_a[warp_mask][mask_a]
    pts_match_mask = torch.zeros((descA.shape[0], descB.shape[0])).to(device)

    # pts_match_mask[valid_a, idx_b] = 1
    pts_match_mask[mask_a, idx_b] = 1
    desc_match_mask = torch.zeros_like(pts_match_mask).to(device)
    desc_match_mask[disA2B_a_idx, disA2B_b_idx] = 1

    inner_mask = pts_match_mask * desc_match_mask
    # hit_rate = torch.sum(inner_mask) / (torch.sum(pts_match_mask) + eps)

    hit_rate = torch.sum(inliers_mask) / (pts_match_num + eps)

    dic_info = {}
    dic_info.update({"hit_rate": torch.tensor([hit_rate]).to(device)})
    dic_info.update({"pt_match_num": torch.tensor([torch.sum(mask_a)]).to(device)})
    dic_info.update({"inner_num": torch.tensor([torch.sum(inliers_mask)]).to(device)})

    dic_info.update({"inner_mask": inliers_mask})
    dic_info.update({"desc_dis_mat": disA2B_mat})
    
    return dic_info

def Hamming_matched(ptsA_warped, ptsB, descA_masked, descB, correspond=2):
    match_idx_rept, mask_idx = get_point_pair_repeat(ptsA_warped[:, :2], ptsB[:, :2], correspond=correspond)
    desc_matched = descA_masked[mask_idx]
    desc_B_matched = descB[match_idx_rept, :]
    match_dist_list = torch.sum(desc_matched.int() != desc_B_matched.int(), dim=-1)
    match_dist_mean = torch.sum(match_dist_list) / (match_dist_list.shape[0] + 1e-6)

    return match_dist_mean, match_dist_list

def Hamming_no_matched(ptsA_warped, ptsB, descA, descB, correspond=2):
    # from Model_component import get_point_pair_inverse_all
    coor = get_point_pair_inverse_all(ptsA_warped[:, :2], ptsB[:, :2], correspond=correspond)
    desc_no_matched = descA[coor[0]]
    descB_no_matched = descB[coor[1]]
    nomatch_dist_list = torch.sum(desc_no_matched != descB_no_matched, dim=-1)
    no_match_dist_mean = torch.sum(nomatch_dist_list) / (nomatch_dist_list.shape[0] + 1e-6)

    return no_match_dist_mean, nomatch_dist_list

def Hamming_matched_flip_character(ptsA_warped, ptsB, descA, descB, mask_sign='net', correspond=2):
    ''' 256dim '''
    if mask_sign == 'net':
        # same_mask = (((torch.linspace(16, 1, 16)**4).unsqueeze(0) @ hadamard(16)) == \
        #     ((torch.linspace(1, 16, 16)**4).unsqueeze(0) @ hadamard(16))).long().squeeze(0).unsqueeze(1).repeat(1, 8).view(-1).bool()
        same_mask = torch.tensor([1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1]).unsqueeze(1).repeat(1,8).view(-1).bool()
    else:
        same_mask = (((torch.linspace(32, 1, 32)**8).unsqueeze(0) @ hadamard(32)) == \
            ((torch.linspace(1, 32, 32)**8).unsqueeze(0) @ hadamard(32))).long().squeeze(0).unsqueeze(1).repeat(1, 4).view(-1).bool()   # 同号位

    '''### 前128 ###'''
    descA_front = descA[:, :128]
    descB_front = descB[:, :128]
    descA_front_same = descA_front[:, same_mask]    # front_same: 前128维中 64位同号
    descB_front_same = descB_front[:, same_mask]
    descA_front_diff = descA_front[:, ~same_mask]   # front_diff: 前128维中 64位反号
    descB_front_diff = descB_front[:, ~same_mask]

    front_same_dist_mean, front_same_dist = Hamming_matched(ptsA_warped, ptsB, descA_front_same, descB_front_same, correspond=correspond)
    front_diff_dist_mean, front_diff_dist = Hamming_matched(ptsA_warped, ptsB, descA_front_diff, descB_front_diff, correspond=correspond)

    '''### 后128 ###'''
    descA_back = descA[:, 128:]
    descB_back = descB[:, 128:]
    descA_back_same = descA_back[:, same_mask]    # front_same: 前128维中 64位同号
    descB_back_same = descB_back[:, same_mask]    
    descA_back_diff = descA_back[:, ~same_mask]   # front_diff: 前128维中 64位反号
    descB_back_diff = descB_back[:, ~same_mask]

    back_same_dist_mean, back_same_dist = Hamming_matched(ptsA_warped, ptsB, descA_back_same, descB_back_same, correspond=correspond)
    back_diff_dist_mean, back_diff_dist = Hamming_matched(ptsA_warped, ptsB, descA_back_diff, descB_back_diff, correspond=correspond)

    dis1 = front_same_dist + back_same_dist
    dis2 = front_diff_dist + back_diff_dist
    dis2_min = torch.where(dis2 < (128 - dis2), dis2, 128 - dis2)
    dis_all = dis1 + dis2_min

    dis_mean = torch.sum(dis_all) / (dis_all.shape[0] + 1e-12)

    return dis_mean, dis_all

def Hamming_no_matched_flip_character(ptsA_warped, ptsB, descA, descB, mask_sign='net', correspond=2):
    ''' 256dim '''
    if mask_sign == 'net':
        same_mask = torch.tensor([1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1]).unsqueeze(1).repeat(1,8).view(-1).bool()
    else:
        same_mask = (((torch.linspace(32, 1, 32)**8).unsqueeze(0) @ hadamard(32)) == \
            ((torch.linspace(1, 32, 32)**8).unsqueeze(0) @ hadamard(32))).long().squeeze(0).unsqueeze(1).repeat(1, 4).view(-1).bool()   # 同号位

    '''### 前128 ###'''
    descA_front = descA[:, :128]
    descB_front = descB[:, :128]
    descA_front_same = descA_front[:, same_mask]    # front_same: 前128维中 64位同号
    descB_front_same = descB_front[:, same_mask]    
    descA_front_diff = descA_front[:, ~same_mask]   # front_diff: 前128维中 64位反号
    descB_front_diff = descB_front[:, ~same_mask]

    front_same_dist_mean, front_same_dist = Hamming_no_matched(ptsA_warped, ptsB, descA_front_same, descB_front_same, correspond=correspond)
    front_diff_dist_mean, front_diff_dist = Hamming_no_matched(ptsA_warped, ptsB, descA_front_diff, descB_front_diff, correspond=correspond)

    '''### 后128 ###'''
    descA_back = descA[:, 128:]
    descB_back = descB[:, 128:]
    descA_back_same = descA_back[:, same_mask]    # front_same: 前128维中 64位同号
    descB_back_same = descB_back[:, same_mask]    
    descA_back_diff = descA_back[:, ~same_mask]   # front_diff: 前128维中 64位反号
    descB_back_diff = descB_back[:, ~same_mask]

    back_same_dist_mean, back_same_dist = Hamming_no_matched(ptsA_warped, ptsB, descA_back_same, descB_back_same, correspond=correspond)
    back_diff_dist_mean, back_diff_dist = Hamming_no_matched(ptsA_warped, ptsB, descA_back_diff, descB_back_diff, correspond=correspond)

    dis1 = front_same_dist + back_same_dist
    dis2 = front_diff_dist + back_diff_dist
    dis2_min = torch.where(dis2 < (128 - dis2), dis2, 128 - dis2)
    dis_all = dis1 + dis2_min
    # dis_all = front_dis1 + front_dis2 + back_dis1 + back_dis2
    dis_mean = torch.sum(dis_all) / (dis_all.shape[0] + 1e-12)

    return dis_mean, dis_all

def Hamming_no_matched_flip_character_pairwise(desc_t_batch, desc_v_batch, mask_t_batch, mask_v_batch, mask_sign='net'):
    ''' 256dim '''
    if mask_sign == 'net':
        same_mask = torch.tensor([1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1]).unsqueeze(1).repeat(1,8).view(-1).bool()
    else:
        same_mask = (((torch.linspace(32, 1, 32)**8).unsqueeze(0) @ hadamard(32)) == \
            ((torch.linspace(1, 32, 32)**8).unsqueeze(0) @ hadamard(32))).long().squeeze(0).unsqueeze(1).repeat(1, 4).view(-1).bool()   # 同号位

    same_index = torch.cat((same_mask, same_mask), dim=-1)

    descA_same = desc_t_batch[:, :, same_index]    # same: 前256维中 128位同号
    descB_same = desc_v_batch[:, :, same_index]
    descA_diff = desc_t_batch[:, :, ~same_index]   # diff: 前256维中 128位反号
    descB_diff = desc_v_batch[:, :, ~same_index]

    same_dist = descA_same.shape[-1] - torch.einsum('nid,mjd->nmij', descA_same, descB_same) - torch.einsum('nid,mjd->nmij', 1-descA_same, 1-descB_same)    # [14,580,150,150]
    same_disk_mask = torch.einsum('ni,mj->nmij', mask_t_batch, mask_v_batch).bool()
    same_dist = same_dist[same_disk_mask]

    diff_dist = descA_same.shape[-1] - torch.einsum('nid,mjd->nmij', descA_diff, descB_diff) - torch.einsum('nid,mjd->nmij', 1-descA_diff, 1-descB_diff)    # [14,580,150,150]
    diff_disk_mask = torch.einsum('ni,mj->nmij', mask_t_batch, mask_v_batch).bool()
    diff_dist = diff_dist[diff_disk_mask]

    dis1 = same_dist
    dis2 = torch.where(diff_dist < (descA_same.shape[-1] - diff_dist), diff_dist, descA_same.shape[-1] - diff_dist)

    dis_all = dis1 + dis2
    dis_mean = torch.sum(dis_all) / (dis_all.shape[0] + 1e-12)

    return dis_mean, dis_all

def Hamming_calculation_pairwise(desc_t_batch, desc_v_batch, mask_sign='net'):
    ''' 256dim 画图用 '''
    if mask_sign == 'net':
        same_mask = torch.tensor([1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1]).unsqueeze(1).repeat(1,8).view(-1).bool()
    else:
        same_mask = (((torch.linspace(32, 1, 32)**8).unsqueeze(0) @ hadamard(32)) == \
            ((torch.linspace(1, 32, 32)**8).unsqueeze(0) @ hadamard(32))).long().squeeze(0).unsqueeze(1).repeat(1, 4).view(-1).bool()   # 同号位

    same_index = torch.cat((same_mask, same_mask), dim=-1)

    descA_same = desc_t_batch[:, :, same_index]    # same: 前256维中 128位同号
    descB_same = desc_v_batch[:, :, same_index]
    descA_diff = desc_t_batch[:, :, ~same_index]   # diff: 前256维中 128位反号
    descB_diff = desc_v_batch[:, :, ~same_index]

    same_dist = descA_same.shape[-1] - torch.einsum('nid,mjd->nmij', descA_same, descB_same) - torch.einsum('nid,mjd->nmij', 1-descA_same, 1-descB_same)    # [14,580,150,150]
    # same_disk_mask = torch.einsum('ni,mj->nmij', mask_t_batch, mask_v_batch).bool()
    # same_dist = same_dist[same_disk_mask]

    diff_dist = descA_same.shape[-1] - torch.einsum('nid,mjd->nmij', descA_diff, descB_diff) - torch.einsum('nid,mjd->nmij', 1-descA_diff, 1-descB_diff)    # [14,580,150,150]
    # diff_disk_mask = torch.einsum('ni,mj->nmij', mask_t_batch, mask_v_batch).bool()
    # diff_dist = diff_dist[diff_disk_mask]

    dis1 = same_dist
    dis2 = torch.where(diff_dist < (descA_same.shape[-1] - diff_dist), diff_dist, descA_same.shape[-1] - diff_dist)

    dis_all = dis1 + dis2
    dis_mean = torch.sum(dis_all) / (dis_all.shape[0] + 1e-12)

    return dis_mean, dis_all.squeeze()

def European_distance_matched(ptsA_warped, ptsB, descA_masked, descB, correspond=2):
    match_idx_rept, mask_idx = get_point_pair_repeat(ptsA_warped[:, :2], ptsB[:, :2], correspond=correspond)
    desc_matched = descA_masked[mask_idx]
    desc_B_matched = descB[match_idx_rept, :]
    L2_list = torch.sqrt(torch.sum((desc_matched - desc_B_matched)**2, dim=-1))
    L2_mean = torch.sum(L2_list) / (L2_list.shape[0] + 1e-6)

    return L2_mean.numpy(), L2_list
    
def European_distance_no_matched(ptsA_warped, ptsB, descA, descB, correspond=2):
    coor = get_point_pair_inverse_all(ptsA_warped[:, :2], ptsB[:, :2], correspond=correspond)
    desc_no_matched = descA[coor[0]]
    descB_no_matched = descB[coor[1]]
    L2_list = torch.sqrt(torch.sum((desc_no_matched - descB_no_matched)**2, dim=-1))
    L2_mean = torch.sum(L2_list) / (L2_list.shape[0] + 1e-6)
    
    return L2_mean.numpy(), L2_list

'''光学变换'''
from imgaug import augmenters as iaa
class ImgAugTransform:
    def __init__(self, **config):
        from numpy.random import uniform
        from numpy.random import randint

        ## old photometric
        self.aug = iaa.Sequential([     # 数据增强
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),                  # 对batch中的一部分图片应用一部分Augmenters,剩下的图片应用另外的Augmenters
            iaa.Sometimes(0.25,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),   # 每次从一系列Augmenters中选择一个来变换
            iaa.Sometimes(0.25,
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5),
                          )
        ])

        if config['photometric']['enable']:
            params = config['photometric']['params']
            aug_all = []
            if params.get('random_brightness', False):
                change = params['random_brightness']['max_abs_change']
                aug = iaa.Add((-change, change))
                #                 aug_all.append(aug)
                aug_all.append(aug)
            # if params['random_contrast']:
            if params.get('random_contrast', False):
                change = params['random_contrast']['strength_range']
                aug = iaa.LinearContrast((change[0], change[1]))
                aug_all.append(aug)
            # if params['additive_gaussian_noise']:
            if params.get('additive_gaussian_noise', False):
                change = params['additive_gaussian_noise']['stddev_range']
                aug = iaa.AdditiveGaussianNoise(scale=(change[0], change[1]))
                aug_all.append(aug)
            # if params['additive_speckle_noise']:
            if params.get('additive_speckle_noise', False):
                change = params['additive_speckle_noise']['prob_range']
                # aug = iaa.Dropout(p=(change[0], change[1]))
                aug = iaa.ImpulseNoise(p=(change[0], change[1]))
                aug_all.append(aug)
            # if params['motion_blur']:
            if params.get('motion_blur', False):
                change = params['motion_blur']['max_kernel_size']
                if change > 3:
                    change = randint(3, change)
                elif change == 3:
                    aug = iaa.Sometimes(0.5, iaa.MotionBlur(change))
                aug_all.append(aug)

            if params.get('GaussianBlur', False):
                change = params['GaussianBlur']['sigma']
                aug = iaa.GaussianBlur(sigma=(change))
                aug_all.append(aug)

            self.aug = iaa.Sequential(aug_all)

        else:
            self.aug = iaa.Sequential([
                iaa.Noop(),
            ])

    def __call__(self, img):
        img = np.array(img)
        img = (img * 255).astype(np.uint8)
        img = self.aug.augment_image(img)
        img = img.astype(np.float32) / 255
        return img

class customizedTransform:
    def __init__(self):
        pass

    def additive_shade(self, image, nb_ellipses=20, transparency_range=[-0.5, 0.8],
                       kernel_size_range=[250, 350]):
        def _py_additive_shade(img):
            min_dim = min(img.shape[:2]) / 4
            mask = np.zeros(img.shape[:2], np.uint8)
            for i in range(nb_ellipses):
                ax = int(max(np.random.rand() * min_dim, min_dim / 5))
                ay = int(max(np.random.rand() * min_dim, min_dim / 5))
                max_rad = max(ax, ay)
                x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
                y = np.random.randint(max_rad, img.shape[0] - max_rad)
                angle = np.random.rand() * 90
                cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

            transparency = np.random.uniform(*transparency_range)
            kernel_size = np.random.randint(*kernel_size_range)
            if (kernel_size % 2) == 0:  # kernel_size has to be odd
                kernel_size += 1
            mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
#             shaded = img * (1 - transparency * mask[..., np.newaxis] / 255.)
            shaded = img * (1 - transparency * mask[..., np.newaxis] / 255.)
            return np.clip(shaded, 0, 255)

        shaded = _py_additive_shade(image)
        return shaded

    def __call__(self, img, **config):
        if config['photometric']['params']['additive_shade']:
            params = config['photometric']['params']
            img = self.additive_shade(img * 255, **params['additive_shade'])
        return img / 255


def imgPhotometric(img, **config):
    """
    :param img:
        numpy (H, W)
    :return:
    """
    augmentation = ImgAugTransform(**config["augmentation"])
    img = img[:, :, np.newaxis]
    img = augmentation(img)
    cusAug = customizedTransform()
    img = cusAug(img, **config["augmentation"])
    return img

def fix_orient_by_trans_flag(orientation_batch, trans_angle):
    '''
    input:
    orientation_batch: [bs, n]
    trans_angle: [bs]
    return:
    correct_angle: [bs, n]
    trans_angle: φ = real_θ_A - real_θ_B      real_θ_x:真实角
    orien(θ_A) - φ = θ_A - (real_θ_A - real_θ_B)
    '''
    orient_sub_trans = orientation_batch - trans_angle.unsqueeze(1)
    correct_flag = ((orient_sub_trans > 90) + (orient_sub_trans < -90)).float()
    # correct_angles = correct_flag * 180

    return correct_flag

def get_inner_points_info(path, name, sf_ang_a, sf_ang_b, net_ang_a, net_ang_b, trans_ang, sf_inner_mask, net_inner_mask, sf_desc_dismat, net_desc_dismat, output=False):
    '''
    path: save path
    sf_ang_a/sf_ang_b: principal angles of points(sift)[-90, 90]
    net_ang_a/net_ang_b: principal angles of points(net)[-90, 90]
    trans_ang: [-180, 180]
    inner_maks: inner pts matrix(N x M, N: all temple pts num,M: all sample pts num)
    sf_desc_dismat: all sift desc distance matrix
    '''
    # save_info = Path(path, "inner_info")
    os.makedirs(path, exist_ok=True)

    # sf_ang_flag = fix_orient_by_trans_flag(sf_ang_a, trans_ang).squeeze()
    # net_ang_flag = fix_orient_by_trans_flag(net_ang_a, trans_ang).squeeze()

    sf_ang_a_set = sf_ang_a - trans_ang     # [-270, 270]
    net_ang_a_set = net_ang_a - trans_ang
    sf_ang_a_set[sf_ang_a_set < -90] += 180 # [-90, 90]
    sf_ang_a_set[sf_ang_a_set > 90] -= 180
    net_ang_a_set[net_ang_a_set < -90] += 180 # [-90, 90]
    net_ang_a_set[net_ang_a_set > 90] -= 180

    # sf_ang_a_set[sf_ang_a_set < 0] += 360.
    # sf_ang_a_set = sf_ang_flag * 180. + sf_ang_a    # [-90, 90]/[90, 270]
    # net_ang_a_set = net_ang_flag * 180. + net_ang_a

    sf_ang_diff = sf_ang_a_set.unsqueeze(1) - sf_ang_b.unsqueeze(0)     # N x M [-360, 360]
    net_ang_diff = net_ang_a_set.unsqueeze(1) - net_ang_b.unsqueeze(0)

    sf_ang_diff = torch.where(sf_ang_diff.abs() < 180 - sf_ang_diff.abs(), sf_ang_diff.abs(), 180 - sf_ang_diff.abs())
    net_ang_diff = torch.where(net_ang_diff.abs() < 180 - net_ang_diff.abs(), net_ang_diff.abs(), 180 - net_ang_diff.abs())

    # sf_ang_diff[sf_ang_diff < 0] += 360
    # net_ang_diff[net_ang_diff < 0] += 360
    # sf_ang_diff -= trans_ang
    # sf_ang_diff[sf_ang_diff>360] -= 360
    # net_ang_diff -= trans_ang

    # sf_ang_diff[sf_ang_diff > 90] -= 180
    # sf_ang_diff[sf_ang_diff < -90] += 180
    # net_ang_diff[net_ang_diff > 90] -= 180
    # net_ang_diff[net_ang_diff < -90] += 180

    sf_ang_diff = torch.abs(sf_ang_diff)
    net_ang_diff = torch.abs(net_ang_diff)

    sf_inner_ang_ab_diff = sf_ang_diff[sf_inner_mask.bool()]
    net_inner_ang_ab_diff = net_ang_diff[net_inner_mask.bool()]

    y_s, x_s = torch.where(sf_inner_mask == 1.)
    y, x = torch.where(net_inner_mask == 1.)

    sf_inner_ang_a = sf_ang_a[y_s]
    sf_inner_ang_b = sf_ang_b[x_s]
    net_inner_ang_a = net_ang_a[y]
    net_inner_ang_b = net_ang_b[x]

    sf_desc_dis_inner = sf_desc_dismat[sf_inner_mask.bool()]
    net_desc_dis_inner = net_desc_dismat[net_inner_mask.bool()]
    
    trans_list = copy.deepcopy(trans_ang).expand_as(x_s)
    trans_list_net = copy.deepcopy(trans_ang).expand_as(x)

    if output == True:
        content = [
            trans_list.cpu().tolist(),
            sf_inner_ang_a.cpu().tolist(),
            sf_inner_ang_b.cpu().tolist(),
            sf_inner_ang_ab_diff.cpu().tolist(),
            sf_desc_dis_inner.cpu().tolist(),
        ]
        df = pd.DataFrame(
            content,
            index=[
                'trans',
                'sf_inner_ang_a',
                'sf_inner_ang_b',
                'sf_ang_diff',
                'sf_desc_dis_inner',
            ])
        df_T = pd.DataFrame(df.values.T, columns=df.index, index=df.columns)
        df_T.to_csv(os.path.join(path, str(name) + '.csv'))

        content_net = [
            trans_list_net.cpu().tolist(),
            net_inner_ang_a.cpu().tolist(),
            net_inner_ang_b.cpu().tolist(),
            net_inner_ang_ab_diff.cpu().tolist(),
            net_desc_dis_inner.cpu().tolist(),
        ]
        df_net = pd.DataFrame(
            content_net,
            index=[
                'trans',
                'net_inner_ang_a',
                'net_inner_ang_b',
                'net_ang_diff',
                'net_desc_dis_inner',
            ])
        df_net_T = pd.DataFrame(df_net.values.T, columns=df_net.index, index=df_net.columns)
        df_net_T.to_csv(os.path.join(path, str(name) + '_net.csv'))

    output = {}
    output.update({
        'sf_ang_diff': sf_inner_ang_ab_diff.abs(),
        'sf_desc_dis': sf_desc_dis_inner,
        'net_ang_diff': net_inner_ang_ab_diff.abs(),
        'net_desc_dis': net_desc_dis_inner,
        })
    return output