# -*- coding: utf-8 -*-

from desc_patch.utils.desc import Hamming_distance_Hadamard,get_candmask_wb
from desc_patch.utils.desc import Hamming_distance_Hadamard_batch,get_nearest_batch
from desc_patch.utils.draw import draw_keypoints_match
from point_alike.utils.draw import draw_pair_trans, inv_warp_image_bilinear
from SLransac import slransac
from SLransac_batch import slransac_batch, slmodel_batch
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

def get_nearest_matches(feat1, feat2, top_k):

    Hamming_distance_matrix_half, Hamming_distance_matrix = Hamming_distance_Hadamard(feat1['desc'], feat2['desc'],dis12=True)

    nearest_Hamming_mask, nearest_topk = get_candmask_wb(Hamming_distance_matrix_half, Hamming_distance_matrix, 60 , 110, top_k, [feat1['black_num'], feat2['black_num']])
    #print(nearest_topk)
    
    #nearest_match_index = nearest_Hamming_mask.nonzero().cpu()
    mask_256 = (nearest_topk[:,2] == 256)
    nearest_topk = nearest_topk[~mask_256]
    nearest_topk_cpu = nearest_topk.cpu()

    nearest_matches = np.zeros([nearest_topk_cpu.size(0), 5])  # 30x4  最近邻匹配点对
    nearest_matches_angle = np.zeros([nearest_topk_cpu.size(0), 2])  # 30x4  最近邻匹配点对
    
    # nearest_matches[:,0:2] = feat1['pts'][nearest_match_index[:,0]]
    # nearest_matches[:,2:4] = feat2['pts'][nearest_match_index[:,1]]
    nearest_matches[:,0:2] = feat1['pts'][nearest_topk_cpu[:,0]]
    nearest_matches[:,2:4] = feat2['pts'][nearest_topk_cpu[:,1]]
    nearest_matches[:,4] = nearest_topk_cpu[:,2]
    
    nearest_matches_angle[:,0] = feat1['angles'][nearest_topk_cpu[:,0]]
    nearest_matches_angle[:,1] = feat2['angles'][nearest_topk_cpu[:,1]]
    
    return nearest_matches, nearest_matches_angle, Hamming_distance_matrix, nearest_topk_cpu.numpy()

def get_nearest_matches_batch(feat1, feat2, samples, top_k):
    #sample batch化，samples记录对应topk的索引
    Hamming_distance_matrix_half_b, Hamming_distance_matrix_b = Hamming_distance_Hadamard_batch(feat1['desc_b'], feat2['desc_b'][samples],dis12=True)
    Hamming_distance_matrix_half_w, Hamming_distance_matrix_w = Hamming_distance_Hadamard_batch(feat1['desc_w'], feat2['desc_w'][samples],dis12=True)

    b1,b2,Mb,Nb = Hamming_distance_matrix_half_b.size()
    b1,b2,Mw,Nw = Hamming_distance_matrix_half_w.size()
    #无效点对应置为256
    bvalid_mask = torch.einsum('bnd,hmd->bhnm',feat1['bvalid_mask'][:,:,None],feat2['bvalid_mask'][samples][:,:,None]).bool()
    Hamming_distance_matrix_half_b[~bvalid_mask] = 256
    Hamming_distance_matrix_b[~bvalid_mask] = 256
    wvalid_mask = torch.einsum('bnd,hmd->bhnm',feat1['wvalid_mask'][:,:,None],feat2['wvalid_mask'][samples][:,:,None]).bool()
    Hamming_distance_matrix_half_w[~wvalid_mask] = 256
    Hamming_distance_matrix_w[~wvalid_mask] = 256

    nearest_Hamming_value_b, nearest_Hamming_index_b, nearest_Hamming_mask_column_b = get_nearest_batch(Hamming_distance_matrix_half_b, Hamming_distance_matrix_b, 60 , 110)
    nearest_Hamming_value_w, nearest_Hamming_index_w, nearest_Hamming_mask_column_w = get_nearest_batch(Hamming_distance_matrix_half_w, Hamming_distance_matrix_w, 60 , 110)
    
    nearest_Hamming_index = torch.cat([nearest_Hamming_index_b,nearest_Hamming_index_w+Nb],dim=2)
    nearest_Hamming_value = torch.cat([nearest_Hamming_value_b,nearest_Hamming_value_w],dim=2)
    index_w = torch.linspace(0,0.8,Mb+Mw)[None,None,:].repeat(b1,b2,1).to(nearest_Hamming_value.device) #实现稳定排序
    nearest_30_index = torch.argsort((nearest_Hamming_value+index_w), dim=2)[:,:,:top_k] 
    # nearest_30_index = np.argsort(nearest_Hamming_value.cpu().numpy(), kind='mergesort')[:cand_num]
    # nearest_30_index = torch.from_numpy(nearest_30_index).to(nearest_Hamming_value.device)
    # nearest_Hamming_mask_b = torch.zeros((b1,b2,Mb,Nb),device=nearest_30_index.device).bool()
    # nearest_Hamming_mask_w = torch.zeros((b1,b2,Mw,Nw),device=nearest_30_index.device).bool() 
    top_k = nearest_30_index.size(2) #有些图可能点数很少
    nearest_topk = torch.zeros((b1,b2,top_k,3),device=nearest_30_index.device, dtype = torch.int64)
    nearest_topk[:,:,:,0] = nearest_30_index
    nearest_topk[:,:,:,1] = nearest_Hamming_index.gather(2,nearest_30_index)
    nearest_topk[:,:,:,2] = nearest_Hamming_value.gather(2,nearest_30_index)
    
    #nearest_match_index = nearest_Hamming_mask.nonzero().cpu()
    mask_256 = ~(nearest_topk[:,:,:,2] == 256)
    # nearest_topk = nearest_topk[~mask_256]
    # nearest_topk_cpu = nearest_topk.cpu()

    nearest_matches = torch.zeros((b1,b2,top_k,5),device=nearest_30_index.device)  # 30x5  最近邻匹配点对
    nearest_matches_angle = torch.zeros((b1,b2,top_k,2),device=nearest_30_index.device)  # 30x2  最近邻匹配点对
    
    # nearest_matches[:,0:2] = feat1['pts'][nearest_match_index[:,0]]
    # nearest_matches[:,2:4] = feat2['pts'][nearest_match_index[:,1]]
    feat1_pts = torch.cat([feat1['pts_b'],feat1['pts_w']],dim=1)
    feat2_pts = torch.cat([feat2['pts_b'][samples],feat2['pts_w'][samples]],dim=1)

    pts_c = feat1_pts.size(2)
    feat1_pts_matches = feat1_pts[:,None,:,:].repeat(1,b2,1,1).gather(2,nearest_topk[:,:,:,0][:,:,:,None].repeat(1,1,1,pts_c))
    feat2_pts_matches = feat2_pts[None,:,:,:].repeat(b1,1,1,1).gather(2,nearest_topk[:,:,:,1][:,:,:,None].repeat(1,1,1,pts_c))

    nearest_matches[:,:,:,0:2] = feat1_pts_matches[:,:,:,0:2]
    nearest_matches[:,:,:,2:4] = feat2_pts_matches[:,:,:,0:2]
    nearest_matches[:,:,:,4] = nearest_topk[:,:,:,2]
    
    angles_idx = feat1['angles_idx']
    nearest_matches_angle[:,:,:,0] = feat1_pts_matches[:,:,:,angles_idx]
    nearest_matches_angle[:,:,:,1] = feat2_pts_matches[:,:,:,angles_idx]
    
    Hamming_distance_matrix = (Hamming_distance_matrix_b,Hamming_distance_matrix_w)
    return nearest_matches, nearest_matches_angle, Hamming_distance_matrix, nearest_topk, mask_256

def check_inlier_angle(nearest_matches, nearest_matches_angle, model, inliers):
    
    if inliers is None:
        return np.array([]), np.array([])
    
    inliers_matches = nearest_matches[inliers.nonzero()[0]]   #内点对
    inliers_matches_angle = nearest_matches_angle[inliers.nonzero()[0]]  #内点对的角度
    
    pi_coef = 3.1415926
    diff = np.zeros([inliers_matches_angle.shape[0], 4])
    
    diff[:,0] = -model.rotation - inliers_matches_angle[:,0] + inliers_matches_angle[:,1] #[-2pi,2pi]
    #print(model.rotation, inliers_matches_angle[:,0] - inliers_matches_angle[:,1])
    mask = diff[:,0] < 0
    diff[mask,0] = diff[mask,0] + 2*pi_coef  #[0,2pi]
    diff[:,1] = 2*pi_coef - diff[:,0]  #[0,2pi]
    
    diff[:,2] = -model.rotation - inliers_matches_angle[:,0] + inliers_matches_angle[:,1] + pi_coef #避免0,180干扰 [-pi,3pi]
    mask = diff[:,2] < 0
    diff[mask,2] = diff[mask,2] + 2*pi_coef  #[0,3pi]
    mask = diff[:,2] > 2*pi_coef
    diff[mask,2] = diff[mask,2] - 2*pi_coef  #[0,2pi]
    diff[:,3] = 2*pi_coef - diff[:,2]  #[0,2pi]
    
    diff_c = np.min(diff,axis = 1)  #  [0, pi]
    #print(diff_c)
    inliers_matches_mask = diff_c < pi_coef/12  #误差不超过+-15度(model.rotation [-pi,pi])
    #print(np.sum(inliers),np.sum(inliers_matches_mask),  model.rotation)
    #print(np.int32(inliers_matches_mask))
    return inliers_matches[inliers_matches_mask], inliers_matches_mask

def check_inlier_angle_batch(nearest_matches, nearest_matches_angle, model, inliers):
    batchsize, topk_num, c1 = nearest_matches.size()
    _, _, c2 = nearest_matches_angle.size()
    batch_idx = inliers.nonzero()[:,0]
    matches_idx = inliers.nonzero()[:,1]

    inliers_matches = nearest_matches.view(-1,c1)[inliers.view(-1).bool()] #内点对
    inliers_matches_angle = nearest_matches_angle.view(-1,c2)[inliers.view(-1).bool()] #内点对的角度

    pi_coef = 3.1415926
    diff = torch.zeros([inliers_matches_angle.size(0), 4],device=model.device)
    
    H_rotation = torch.atan2(model[:,1,0],model[:,0,0])

    diff[:,0] = -H_rotation[batch_idx] - inliers_matches_angle[:,0] + inliers_matches_angle[:,1] #[-2pi,2pi]
    #print(model.rotation, inliers_matches_angle[:,0] - inliers_matches_angle[:,1])
    mask = diff[:,0] < 0
    diff[mask,0] = diff[mask,0] + 2*pi_coef  #[0,2pi]
    diff[:,1] = 2*pi_coef - diff[:,0]  #[0,2pi]
    
    diff[:,2] = -H_rotation[batch_idx] - inliers_matches_angle[:,0] + inliers_matches_angle[:,1] + pi_coef #避免0,180干扰 [-pi,3pi]
    mask = diff[:,2] < 0
    diff[mask,2] = diff[mask,2] + 2*pi_coef  #[0,3pi]
    mask = diff[:,2] > 2*pi_coef
    diff[mask,2] = diff[mask,2] - 2*pi_coef  #[0,2pi]
    diff[:,3] = 2*pi_coef - diff[:,2]  #[0,2pi]
    
    diff_c = torch.min(diff,axis = 1).values  #  [0, pi]
    #print(diff_c)
    inliers_matches_mask = diff_c < pi_coef/12  #误差不超过+-15度(model.rotation [-pi,pi])

    inliers[batch_idx,matches_idx] = inliers_matches_mask.float()
    inliers_matches = torch.cat([batch_idx[:,None],inliers_matches],dim=1)
    return inliers_matches[inliers_matches_mask], inliers

def cal_overlap(w, h, model):
    
    img_coordinate = np.indices((w, h)).reshape(2, w*h).transpose()
    
    img_t = model(img_coordinate)
    
    overlap_mask = (img_t[:,1] >= 0) * (img_t[:,1] <= h - 1) * (img_t[:,0] >= 0) * (img_t[:,0] <= w - 1)
    
    overlap = np.sum(overlap_mask) / (w * h)
    
    mask = np.zeros([h,w],dtype=np.uint8)
    grid = img_coordinate[overlap_mask].transpose().reshape(2, -1)
    mask[grid[1], grid[0]] = 255
    #print(img_t[0:36], overlap_mask[0:36], img_coordinate[overlap_mask][:20], grid[:,:2], mask[0,:])
    return overlap, mask  #mask为被trans图的mask,此工程中默认为feat2的mask

def DEG2RAD(ori):
    pi_coef = 3.1415926
    return ori * pi_coef / 180

#用的sample贴到temple，跟粗匹配最近邻相同次序，如果颠倒是否更互补
def get_local_nearest_matches(feat1, feat2, model, Hamming_distance_matrix, cand_num = 30):
    
    w,h = feat1['w'], feat1['h']
    Hamming_distance_matrix_nearest = Hamming_distance_matrix.clone()
    
    point2to1 = model(feat2['pts'])  #2to1用正变换
    point1to2 = model.inverse(feat1['pts'])  #1to2用逆变换
    overlap_mask2to1 = (point2to1[:,1] >= 0) * (point2to1[:,1] <= h - 1) * (point2to1[:,0] >= 0) * (point2to1[:,0] <= w - 1)
    overlap_mask1to2 = (point1to2[:,1] >= 0) * (point1to2[:,1] <= h - 1) * (point1to2[:,0] >= 0) * (point1to2[:,0] <= w - 1)

    #cor_distance_matrix = np.square(feat1['pts'][:,None] - point2to1).sum(axis = 2)  #一行为point1[0]到point2to1所有点的距离
    #distance_mask = cor_distance_matrix < 49  #距离阈值为7
    cor_distance_matrix_x = np.abs(feat1['pts'][:,0][:,None] - point2to1[:,0])
    cor_distance_matrix_y = np.abs(feat1['pts'][:,1][:,None] - point2to1[:,1])
    distance_mask = (cor_distance_matrix_x <= 7) * (cor_distance_matrix_y <= 7)   #工程中用的dx,dy

    # Hamming_distance_matrix_nearest[~overlap_mask1to2, :] = 256
    # Hamming_distance_matrix_nearest[:, ~overlap_mask2to1] = 256
    distance_mask = torch.from_numpy(distance_mask)   #避免点数较少时报too many indices错误
    Hamming_distance_matrix_nearest[~distance_mask] = 256

    #黑白点对应
    b1,b2 = feat1['black_num'], feat2['black_num']
    Hamming_distance_matrix_nearest[b1:,0:b2] = 256
    Hamming_distance_matrix_nearest[0:b1,b2:] = 256
    
    M_row, N_col = Hamming_distance_matrix.size()
    #最近邻次近邻比值进行卡控
    asend_Hamming_value, asend_Hamming_index = torch.sort(Hamming_distance_matrix_nearest)
    Nearest_nextNearest_mask = (asend_Hamming_value[:,0] / asend_Hamming_value[:,1]) < 0.99
    Nearest_nextNearest_mask = Nearest_nextNearest_mask.unsqueeze(1).repeat(1,asend_Hamming_value.size(1))
    # Hamming_distance_matrix_nearest[~Nearest_nextNearest_mask] = 256

    #获得行方向最近邻
    nearest_Hamming_index_row = asend_Hamming_index[:,0]
    nearest_Hamming_mask_row = torch.zeros((M_row,N_col),device=Hamming_distance_matrix.device).bool() #行方向
    nearest_Hamming_mask_row[list(range(M_row)),nearest_Hamming_index_row] = 1
    nearest_Hamming_mask_row = Nearest_nextNearest_mask*nearest_Hamming_mask_row  #
    Hamming_distance_matrix_nearest[~nearest_Hamming_mask_row] = 256

    #列方向去重
    nearest_Hamming_index_column = torch.min(Hamming_distance_matrix_nearest,dim=0).indices
    nearest_Hamming_mask_column = torch.zeros((M_row,N_col),device=Hamming_distance_matrix.device).bool()  #行方向
    nearest_Hamming_mask_column[nearest_Hamming_index_column, list(range(N_col))] = 1
    nearest_Hamming_mask_column = nearest_Hamming_mask_row*nearest_Hamming_mask_column
    Hamming_distance_matrix_nearest[~nearest_Hamming_mask_column] = 256

    #获取最近邻匹配matrix
    nearest_Hamming_value = torch.min(Hamming_distance_matrix_nearest,dim=1).values
    nearest_Hamming_index = torch.min(Hamming_distance_matrix_nearest,dim=1).indices  #列号

    # nearest_30_index = torch.topk(nearest_Hamming_value, cand_num, largest=False).indices  #行号
    nearest_30_index = np.argsort(nearest_Hamming_value.cpu().numpy(), kind='mergesort')[:cand_num]
    nearest_30_index = torch.from_numpy(nearest_30_index).to(nearest_Hamming_value.device)
    nearest_Hamming_mask = torch.zeros((M_row,N_col),device=Hamming_distance_matrix.device).bool() 

    nearest_Hamming_mask[nearest_30_index,nearest_Hamming_index[nearest_30_index]] = 1
    nearest_Hamming_mask = nearest_Hamming_mask*nearest_Hamming_mask_column

    nearest_Hamming_mask = nearest_Hamming_mask.bool()

    max_cand_num = min(cand_num, nearest_30_index.shape[0])  #shape[0]小于cand_num时，直接赋值会报错
    nearest_30 = torch.zeros((max_cand_num,3),device=Hamming_distance_matrix.device, dtype = torch.int32)
    nearest_30[:,0] = nearest_30_index
    nearest_30[:,1] = nearest_Hamming_index[nearest_30_index]
    nearest_30[:,2] = nearest_Hamming_value[nearest_30_index]
    
    mask_256 = (nearest_30[:,2] == 256)
    nearest_topk = nearest_30[~mask_256]

    nearest_topk_cpu = nearest_topk.cpu()
    nearest_matches = np.zeros([nearest_topk_cpu.size(0), 5])  # 30x4  最近邻匹配点对
    nearest_matches_angle = np.zeros([nearest_topk_cpu.size(0), 2])  # 30x4  最近邻匹配点对
    
    nearest_matches[:,0:2] = feat1['pts'][nearest_topk_cpu[:,0]]
    nearest_matches[:,2:4] = feat2['pts'][nearest_topk_cpu[:,1]]
    nearest_matches[:,4] = nearest_topk_cpu[:,2]
    
    nearest_matches_angle[:,0] = feat1['angles'][nearest_topk_cpu[:,0]]
    nearest_matches_angle[:,1] = feat2['angles'][nearest_topk_cpu[:,1]]
    
    return nearest_matches, nearest_matches_angle, nearest_topk_cpu.numpy()

def get_local_nearest_matches_batch(feat1, feat2, model, Hamming_distance_matrix, samples, top_k = 30):
    
    def _apply_mat(coords, matrix):
        #coords: [b n ?]
        #matrix: [b 3 3]
        x, y = coords[:,:,0][:,:,None], coords[:,:,1][:,:,None]
        src = torch.cat([x, y, torch.ones_like(x)],dim=2)
        dst = torch.einsum('bnc,bdc->bnd',src.double(), matrix.double())

        zero_add = (dst[:,:,2] == 0)*torch.finfo(float).eps
        dst[:,:,2] += zero_add
        # rescale to homogeneous coordinates
        dst[:,:,:2] /= dst[:,:,2:3]

        return dst[:,:,:2]

    def get_local_nearest_batch(Hamming_distance_matrix_nearest):
        b1, b2, M_row, N_col = Hamming_distance_matrix_nearest.size()

        #最近邻次近邻比值进行卡控
        asend_Hamming_value, asend_Hamming_index = torch.sort(Hamming_distance_matrix_nearest)
        Nearest_nextNearest_mask = (asend_Hamming_value[:,:,:,0] / asend_Hamming_value[:,:,:,1]) < 0.99
        Nearest_nextNearest_mask = Nearest_nextNearest_mask.unsqueeze(3).repeat(1,1,1,N_col)
        # Hamming_distance_matrix_nearest[~Nearest_nextNearest_mask] = 256

        #获得行方向最近邻
        nearest_Hamming_index_row = asend_Hamming_index[:,:,:,0]
        nearest_Hamming_mask_row = torch.linspace(0,N_col-1,N_col)[None,None,None].repeat(b1,b2,M_row,1).to(Hamming_distance_matrix_nearest.device)
        nearest_Hamming_mask_row = (nearest_Hamming_mask_row == nearest_Hamming_index_row[:,:,:,None].repeat(1,1,1,N_col))
        nearest_Hamming_mask_row = Nearest_nextNearest_mask*nearest_Hamming_mask_row  #
        Hamming_distance_matrix_nearest[~nearest_Hamming_mask_row] = 256

        # #获得行方向最近邻
        # nearest_Hamming_index_row = torch.min(Hamming_distance_matrix_nearest,-1).indices
        # nearest_Hamming_mask_row = torch.linspace(0,N_col-1,N_col)[None,None,None].repeat(b1,b2,M_row,1).to(Hamming_distance_matrix_nearest.device)
        # nearest_Hamming_mask_row = (nearest_Hamming_mask_row == nearest_Hamming_index_row[:,:,:,None].repeat(1,1,1,N_col))
        # # nearest_Hamming_mask_row = dis12_mask*Nearest_nextNearest_mask*nearest_Hamming_mask_row  #
        # Hamming_distance_matrix_nearest[~nearest_Hamming_mask_row] = 256

        #列方向去重
        nearest_Hamming_index_column = torch.min(Hamming_distance_matrix_nearest,dim=2).indices
        nearest_Hamming_mask_column = torch.linspace(0,M_row-1,M_row)[None,None,:,None].repeat(b1,b2,1,N_col).to(Hamming_distance_matrix_nearest.device)
        nearest_Hamming_mask_column = (nearest_Hamming_mask_column == nearest_Hamming_index_column[:,:,None,:].repeat(1,1,M_row,1))
        nearest_Hamming_mask_column = nearest_Hamming_mask_row*nearest_Hamming_mask_column
        Hamming_distance_matrix_nearest[~nearest_Hamming_mask_column] = 256

        #获取最近邻匹配matrix
        nearest_Hamming_value = torch.min(Hamming_distance_matrix_nearest,dim=3).values
        nearest_Hamming_index = torch.min(Hamming_distance_matrix_nearest,dim=3).indices  #列号

        return nearest_Hamming_value, nearest_Hamming_index, nearest_Hamming_mask_column

    w,h = feat1['w'], feat1['h']
    Hamming_distance_matrix_nearest_b = Hamming_distance_matrix[0].clone()
    Hamming_distance_matrix_nearest_w = Hamming_distance_matrix[1].clone()
    
    b1,b2,Mb,Nb = Hamming_distance_matrix_nearest_b.size()
    b1,b2,Mw,Nw = Hamming_distance_matrix_nearest_w.size()

    assert len(feat2['pts_b'][samples]) == len(model)
    point2to1_b = _apply_mat(feat2['pts_b'][samples],model)  #2to1用正变换
    # point1to2_b = _apply_mat(feat1['pts_b'].repeat(b2,1,1),model.inverse())  #1to2用逆变换
    # overlap_mask2to1_b = (point2to1_b[:,:,1] >= 0) * (point2to1_b[:,:,1] <= h - 1) * (point2to1_b[:,:,0] >= 0) * (point2to1_b[:,:,0] <= w - 1)
    # overlap_mask1to2_b = (point1to2_b[:,:,1] >= 0) * (point1to2_b[:,:,1] <= h - 1) * (point1to2_b[:,:,0] >= 0) * (point1to2_b[:,:,0] <= w - 1)
    # overlap_mask_b = torch.einsum('bn,bm->bnm',overlap_mask1to2_b,overlap_mask2to1_b)[None,:,:,:]

    cor_distance_matrix_x_b = torch.abs(feat1['pts_b'][:,:,0][:,None,:,None].repeat(1,b2,1,1) - point2to1_b[:,:,0][None,:,None,:].repeat(b1,1,1,1))
    cor_distance_matrix_y_b = torch.abs(feat1['pts_b'][:,:,1][:,None,:,None].repeat(1,b2,1,1) - point2to1_b[:,:,1][None,:,None,:].repeat(b1,1,1,1))
    distance_mask_b = (cor_distance_matrix_x_b <= 7) * (cor_distance_matrix_y_b <= 7)   #工程中用的dx,dy
    
    # Hamming_distance_matrix_nearest_b[~overlap_mask_b] = 256
    Hamming_distance_matrix_nearest_b[~distance_mask_b] = 256

    point2to1_w = _apply_mat(feat2['pts_w'][samples],model)  #2to1用正变换
    # point1to2_w = _apply_mat(feat1['pts_w'].repeat(b2,1,1),model.inverse())  #1to2用逆变换
    # overlap_mask2to1_w = (point2to1_w[:,:,1] >= 0) * (point2to1_w[:,:,1] <= h - 1) * (point2to1_w[:,:,0] >= 0) * (point2to1_w[:,:,0] <= w - 1)
    # overlap_mask1to2_w = (point1to2_w[:,:,1] >= 0) * (point1to2_w[:,:,1] <= h - 1) * (point1to2_w[:,:,0] >= 0) * (point1to2_w[:,:,0] <= w - 1)
    # overlap_mask_w = torch.einsum('bn,bm->bnm',overlap_mask1to2_w,overlap_mask2to1_w)[None,:,:,:]

    cor_distance_matrix_x_w = torch.abs(feat1['pts_w'][:,:,0][:,None,:,None].repeat(1,b2,1,1) - point2to1_w[:,:,0][None,:,None,:].repeat(b1,1,1,1))
    cor_distance_matrix_y_w = torch.abs(feat1['pts_w'][:,:,1][:,None,:,None].repeat(1,b2,1,1) - point2to1_w[:,:,1][None,:,None,:].repeat(b1,1,1,1))
    distance_mask_w = (cor_distance_matrix_x_w <= 7) * (cor_distance_matrix_y_w <= 7)   #工程中用的dx,dy
    
    # Hamming_distance_matrix_nearest_w[~overlap_mask_w] = 256
    Hamming_distance_matrix_nearest_w[~distance_mask_w] = 256


    nearest_Hamming_value_b, nearest_Hamming_index_b, nearest_Hamming_mask_column_b = get_local_nearest_batch(Hamming_distance_matrix_nearest_b)
    nearest_Hamming_value_w, nearest_Hamming_index_w, nearest_Hamming_mask_column_w = get_local_nearest_batch(Hamming_distance_matrix_nearest_w)
    
    nearest_Hamming_index = torch.cat([nearest_Hamming_index_b,nearest_Hamming_index_w+Nb],dim=2)
    nearest_Hamming_value = torch.cat([nearest_Hamming_value_b,nearest_Hamming_value_w],dim=2)
    index_w = torch.linspace(0,0.8,Mb+Mw)[None,None,:].repeat(b1,b2,1).to(nearest_Hamming_value.device) #实现稳定排序
    nearest_30_index = torch.argsort((nearest_Hamming_value+index_w), dim=2)[:,:,:top_k] 
    # nearest_30_index = np.argsort(nearest_Hamming_value.cpu().numpy(), kind='mergesort')[:cand_num]
    # nearest_30_index = torch.from_numpy(nearest_30_index).to(nearest_Hamming_value.device)
    # nearest_Hamming_mask_b = torch.zeros((b1,b2,Mb,Nb),device=nearest_30_index.device).bool()
    # nearest_Hamming_mask_w = torch.zeros((b1,b2,Mw,Nw),device=nearest_30_index.device).bool() 

    top_k = nearest_30_index.size(2) #有些图可能点数很少
    nearest_topk = torch.zeros((b1,b2,top_k,3),device=nearest_30_index.device, dtype = torch.int64)
    nearest_topk[:,:,:,0] = nearest_30_index
    nearest_topk[:,:,:,1] = nearest_Hamming_index.gather(2,nearest_30_index)
    nearest_topk[:,:,:,2] = nearest_Hamming_value.gather(2,nearest_30_index)
    
    #nearest_match_index = nearest_Hamming_mask.nonzero().cpu()
    mask_256 = ~(nearest_topk[:,:,:,2] == 256)
    # nearest_topk = nearest_topk[~mask_256]
    # nearest_topk_cpu = nearest_topk.cpu()

    nearest_matches = torch.zeros((b1,b2,top_k,5),device=nearest_30_index.device)  # 30x5  最近邻匹配点对
    nearest_matches_angle = torch.zeros((b1,b2,top_k,2),device=nearest_30_index.device)  # 30x2  最近邻匹配点对
    
    # nearest_matches[:,0:2] = feat1['pts'][nearest_match_index[:,0]]
    # nearest_matches[:,2:4] = feat2['pts'][nearest_match_index[:,1]]
    feat1_pts = torch.cat([feat1['pts_b'],feat1['pts_w']],dim=1)
    feat2_pts = torch.cat([feat2['pts_b'][samples],feat2['pts_w'][samples]],dim=1)

    pts_c = feat1_pts.size(2)
    feat1_pts_matches = feat1_pts[:,None,:,:].repeat(1,b2,1,1).gather(2,nearest_topk[:,:,:,0][:,:,:,None].repeat(1,1,1,pts_c))
    feat2_pts_matches = feat2_pts[None,:,:,:].repeat(b1,1,1,1).gather(2,nearest_topk[:,:,:,1][:,:,:,None].repeat(1,1,1,pts_c))

    nearest_matches[:,:,:,0:2] = feat1_pts_matches[:,:,:,0:2]
    nearest_matches[:,:,:,2:4] = feat2_pts_matches[:,:,:,0:2]
    nearest_matches[:,:,:,4] = nearest_topk[:,:,:,2]
    
    angles_idx = feat1['angles_idx']
    nearest_matches_angle[:,:,:,0] = feat1_pts_matches[:,:,:,angles_idx]
    nearest_matches_angle[:,:,:,1] = feat2_pts_matches[:,:,:,angles_idx]
    
    Hamming_distance_matrix = (Hamming_distance_matrix_nearest_b,Hamming_distance_matrix_nearest_w)
    
    return nearest_matches, nearest_matches_angle, Hamming_distance_matrix, nearest_topk, mask_256

def estimate_ls(data):
    src, dst = data
    N = len(src)
    A = np.zeros((2*N,6))
    eta = np.eye(6)*1e-6
    b = np.zeros((2*N,1))
    A[:N,:2] = src
    A[:N,2] = 1
    A[N:,3:5] = src
    A[N:,5] = 1
    b[:N,0] = dst[:,0]
    b[N:,0] = dst[:,1]
    h = np.linalg.inv((A.transpose(1,0)@A) + eta)@(A.transpose(1,0))@b
    return h

def estimate_ls_batch(samples, data):
    data = data.double()
    batchsize,_,s_n,_ =  data.size()

    samples_index = samples[:,1:] + (samples[:,0]*s_n)[:,None]
    src = data[:,0].reshape(-1,2)[samples_index]
    dst = data[:,1].reshape(-1,2)[samples_index]
    batchsize, N, d = src.shape
    A = torch.zeros((batchsize,2*N,6)).to(data.device)
    eta = (torch.eye(6)*1e-6).to(data.device)
    b = torch.zeros((batchsize,2*N,1)).to(data.device)
    A[:,:N,:2] = src
    A[:,:N,2] = 1
    A[:,N:,3:5] = src
    A[:,N:,5] = 1
    b[:,:N,0] = dst[:,:,0]
    b[:,N:,0] = dst[:,:,1]
    ATA_inv = (torch.linalg.inv(torch.einsum('bnm,bnd->bmd',A,A) + eta[None,:,:])).double()
    Ab = torch.einsum('bnm,bnd->bmd',A,b).double()
    h = torch.einsum('bnd,bdm->bnm',ATA_inv,Ab)

    H_all = torch.zeros((batchsize,3,3)).double().to(data.device)
    success_all = torch.ones((batchsize)).bool().to(data.device)

    H_all[:,:2] = h.view(batchsize,2,3)
    H_all[:,2,2] = 1
    return H_all, success_all

def rematch(feat1, feat2, model, Hamming_distance_matrix, top_k):  #?????是否也用30 
    
    nearest_matches, nearest_matches_angle, nearest_topk_idx = get_local_nearest_matches(feat1, feat2, model, Hamming_distance_matrix, top_k)
    
    if nearest_matches is None or nearest_matches.shape[0] <= 3:
        return np.array([]), np.array([]), None

    model, inliers, best_inlier_residuals_sum = slransac((nearest_matches[:,2:4], nearest_matches[:,0:2]), AffineTransform, min_samples=3, residual_threshold=2.5,
                        max_trials=500, stop_sample_num= 22, random_state=2000,weight_pairs=None, angles = (nearest_matches_angle[:,1], nearest_matches_angle[:,0]))
    #print(model, nearest_matches[inliers.nonzero()[0]].shape)
    inliers_matches, inliers_matches_mask = check_inlier_angle(nearest_matches, nearest_matches_angle, model, inliers)
    matches_idx = np.array([])
    if inliers is not None and inliers_matches_mask.shape[0] > 0:
        matches_idx = nearest_topk_idx[inliers][inliers_matches_mask]

    # estimate final model using all inliers
    if inliers_matches.shape[0] >= 4 and best_inlier_residuals_sum > 16384./256/256 * np.sum(inliers):    #用所有点重新拟合
        # select inliers for each data array
        data_inliers = [inliers_matches[:,2:4], inliers_matches[:,0:2]]
        model.estimate(*data_inliers)

    return inliers_matches, matches_idx, model

def rematch_batch(feat1, feat2, model, Hamming_distance_matrix, samples, top_k):  #?????是否也用30 
    nearest_matches, nearest_matches_angle, Hamming_distance_matrix, nearest_topk_idx, nearest_topk_mask = get_local_nearest_matches_batch(feat1, feat2, model, Hamming_distance_matrix, samples, top_k)
    
    nearest_matches = nearest_matches[0]
    nearest_matches_angle = nearest_matches_angle[0]
    nearest_topk_mask = nearest_topk_mask[0].float()

    bdata = torch.cat([nearest_matches[:,:,2:4][:,None,:,:],nearest_matches[:,:,0:2][:,None,:,:]],dim=1)
    bangles = torch.cat([nearest_matches_angle[:,:,1][:,None,:],nearest_matches_angle[:,:,0][:,None,:]],dim=1)

    model, inliers, best_inlier_residuals_sum, model_valid = slransac_batch(bdata, nearest_topk_mask, AffineTransform, min_samples=3, residual_threshold=2.5,
                        max_trials=500, stop_sample_num= 22, random_state=2000,weight_pairs=None, bangles = bangles)
    
    if model_valid.sum() == 0:
        assert inliers.sum() == 0
        inliers_matches = torch.zeros((0,6),device=model.device)
        matches_idx = torch.zeros((0,4),device=model.device)
        inliers_matches_mask = inliers
        return inliers_matches, matches_idx, model, model_valid, inliers_matches_mask

    inliers_matches, inliers_matches_mask = check_inlier_angle_batch(nearest_matches, nearest_matches_angle, model, inliers)

    matches_idx = torch.zeros((inliers_matches.size(0),4),device=inliers_matches.device) #N 4(batch_idx,A_idx,B_idx,hamming)
    inliers_matches_mask_idx = inliers_matches_mask.nonzero()
    matches_idx[:,0] = inliers_matches_mask_idx[:,0]
    matches_idx[:,1:] = nearest_topk_idx[0,inliers_matches_mask_idx[:,0],inliers_matches_mask_idx[:,1]]
    
    inliers_matches_num_mask = inliers_matches_mask.sum(1) > 4 #4
    best_inlier_residuals_sum_mask = best_inlier_residuals_sum > 16384./256/256 * inliers.sum(1)

    model_re_estimate = model.clone()
    # sample_model = slmodel_batch()
    #每对匹配对点数不一致，无法并行
    for idx in range(len(inliers_matches_num_mask)):
        if inliers_matches_num_mask[idx] and best_inlier_residuals_sum_mask[idx] and model_valid[idx]:
            samples_batch = torch.cat([torch.tensor([idx],device=model.device),inliers_matches_mask[idx].nonzero().squeeze()])
            model_re_estimate_one, model_re_estimate_success = estimate_ls_batch(samples_batch[None,:],bdata)
            # model_re_estimate_one, model_re_estimate_success = sample_model.estimate_batch(samples_batch[None,:],bdata)
            model_re_estimate[idx] = model_re_estimate_one[0]
            if model_re_estimate_success[0] and torch.linalg.matrix_rank(model_re_estimate[idx]) == 3:   #避免重匹配求逆时出错
                model[idx] = model_re_estimate[idx]

    return inliers_matches, matches_idx, model, model_valid, inliers_matches_mask

def recognize(feat1, feat2, device):
    ''' feat1 temple  feat2 sample
    {'pts':points['pts'],       #numpy Nx2 (uchar)   range : [w,h]
    'prob':points['prob'],      #numpy Nx1  (float)  range : 0-1
    'angles':points['angles'],  #numpy Nx1 (float)   range : -pi/2-pi/2
    'desc':desc,                #numpy Nx(256bit)    range : 0,1
    'enhance_img':enhance_img,  #numpy h*w
    }
    '''
    auth_result = 0
    inliers_matches = np.array([])
    w,h = feat1['w'], feat1['h']  #默认feat1,feat2同宽高
    
    if type(feat1['desc']) == np.ndarray:
        feat1['desc'] = torch.from_numpy(feat1['desc'])
    if type(feat2['desc']) == np.ndarray:
        feat2['desc'] = torch.from_numpy(feat2['desc'])

    feat1['desc'] = feat1['desc'].cuda(device)
    feat2['desc'] = feat2['desc'].cuda(device)

    top_k = 60
    if feat1['quality'] > 50 and feat2['quality'] > 50:
        top_k = 30  #质量低时用60，不过对于质量好的图库用60log量比30反而少，所以增加候选点对不一定是正收益

    nearest_matches, nearest_matches_angle, Hamming_distance_matrix, nearest_topk_idx = get_nearest_matches(feat1, feat2, top_k)
    
    #model, inliers = slransac_random((nearest_matches[:,0:2],nearest_matches[:,2:4]), AffineTransform, min_samples=3, residual_threshold=2.5,
    #                      max_trials=500, stop_sample_num= 22, random_state=2000,weight_pairs=None,)
    model, inliers, best_inlier_residuals_sum = slransac((nearest_matches[:,2:4], nearest_matches[:,0:2]), AffineTransform, min_samples=3, residual_threshold=2.5,
                          max_trials=500, stop_sample_num= 22, random_state=2000,weight_pairs=None, angles = (nearest_matches_angle[:,1], nearest_matches_angle[:,0]))
    
    if model is None:   #or np.linalg.matrix_rank(model.params) < 3:  排除奇异矩阵  是否可以用scale和shear的限制   #放入ransac中判断 
        return auth_result, {'inliers':inliers_matches,'inliers_num':inliers_matches.shape[0],'model':model}

    inliers_matches, inliers_matches_mask = check_inlier_angle(nearest_matches, nearest_matches_angle, model, inliers)

    matches_idx = np.array([])
    if inliers is not None and inliers_matches_mask.shape[0] > 0:
        matches_idx = nearest_topk_idx[inliers][inliers_matches_mask]    # 通过角度卡控后的内点索引 top_k
    
    # estimate final model using all inliers
    if inliers_matches.shape[0] >= 4 and best_inlier_residuals_sum > 16384./256/256 * np.sum(inliers):    #用所有点重新拟合
        # select inliers for each data array
        data_inliers = [inliers_matches[:,2:4], inliers_matches[:,0:2]]
        model_re_estimate = AffineTransform()
        model_re_estimate.estimate(*data_inliers)
        if np.linalg.matrix_rank(model_re_estimate.params) == 3:   #避免重匹配求逆时出错
            model = model_re_estimate
	
    inliers_matches_re = np.array([]) #np.copy(inliers_matches)
    model_re = None   #copy.deepcopy(model)
    matches_idx_re = None   #np.copy(matches_idx)
    model_corse = copy.deepcopy(model)  #备份粗匹配model用于计算特征
    rematchtag = 0

    if inliers_matches.shape[0] > 0 and inliers_matches.shape[0] < 22 :  #大于0小于22个内点走重匹配
        ori = model.rotation
        overlapthr = 40 if abs(ori) <= DEG2RAD(50) else 15
        overlap, mask = cal_overlap(w, h, model)  #还没有考虑mask

        if overlap * 100 > overlapthr:
            inliers_matches_re, matches_idx_re, model_re = rematch(feat1, feat2, model, Hamming_distance_matrix, top_k)
            rematchtag = 1

            if model_re is not None:
                if(inliers_matches.shape[0] < inliers_matches_re.shape[0]):  #重匹配内点更多则采用重匹配结果
                    #inliers_matches = inliers_matches_re
                    model = model_re

    if inliers_matches.shape[0] > 4 or inliers_matches_re.shape[0] > 4:
        auth_result = 1
    
    #   #打印配位图用于debug
    # if inliers_matches.shape[0] > 4:
    #     input_img = {'img':feat1['enhance_img']/255., 'img_H':feat2['enhance_img']/255.}
    #     input_pts = {'matches':inliers_matches } # 'check_angle_mask': inliers_matches_mask}
    #     #model.params = np.array([[np.cos(0.66), -np.sin(0.66),22],[np.sin(0.66), np.cos(0.66), -19],[0,0,1]])
    #     Htrans = torch.from_numpy(model.params)
    #     draw_pair_train_img = draw_pair_trans(input_img, input_pts, None, H = Htrans)  #画点图 + trans图
    #     name = feat1['name'][:-4] + 'and'+ feat2['name'][-8:-4] + '_match2.bmp'
    #     cv2.imwrite(name, draw_pair_train_img)

    return auth_result, {'inliers': inliers_matches,
                         'inliers_num': inliers_matches.shape[0],
                         'inliers_idx': matches_idx,    # 索引基于feat1/feat2
                         'model': model,       #最终trans
                         'model_corse':model_corse,  #粗匹配trans
                         'model_re':model_re,  #重匹配trans
                         'inliers_re': inliers_matches_re,
                         'inliers_num_re': inliers_matches_re.shape[0],
                         'inliers_idx_re': matches_idx_re,
                         'hamming_dis_matrix':Hamming_distance_matrix,    #汉明距矩阵
                         'rematchtag': rematchtag,     #标记是否用重匹配结果
                        }

def recognize_batch_topk(feat1, feat2, samples, top_k=60):
    w,h = feat1['w'], feat1['h']  #默认feat1,feat2同宽高
    batchsize = len(samples)

    device = feat1['desc_b'].device
    nearest_matches, nearest_matches_angle, Hamming_distance_matrix, nearest_topk_idx, nearest_topk_mask = get_nearest_matches_batch(feat1, feat2, samples, top_k)
    nearest_matches = nearest_matches[0]
    nearest_matches_angle = nearest_matches_angle[0]
    nearest_topk_mask = nearest_topk_mask[0].float()

    bdata = torch.cat([nearest_matches[:,:,2:4][:,None,:,:],nearest_matches[:,:,0:2][:,None,:,:]],dim=1)
    bangles = torch.cat([nearest_matches_angle[:,:,1][:,None,:],nearest_matches_angle[:,:,0][:,None,:]],dim=1)

    model, inliers, best_inlier_residuals_sum, model_valid = slransac_batch(bdata, nearest_topk_mask, AffineTransform, min_samples=3, residual_threshold=2.5,
                        max_trials=500, stop_sample_num= 22, random_state=2000,weight_pairs=None, bangles = bangles)
    
    if model_valid.sum() == 0:
        assert inliers.sum() == 0
        inliers_matches = torch.zeros((0,6),device=device)
        inliers_matches_re = inliers_matches.clone()
        matches_idx = torch.zeros((0,4),device=device)
        matches_idx_re = matches_idx.clone()
        inliers_matches_mask = inliers
        model_re = model.clone()
        model_corse = model.clone()  #备份粗匹配model用于计算特征
        model_re_valid = torch.zeros_like(model_valid)
        auth_result = inliers_matches_mask.sum(1)
        inliers_num_re = torch.zeros_like(inliers_matches_mask.sum(1))
        rematchtag = (inliers_matches_mask.sum(1) > 0)*(inliers_matches_mask.sum(1) < 22)
        rematchtag *= model_valid.bool()

        auth_result = auth_result > 4

        return auth_result, {'inliers': inliers_matches,
                    'inliers_num': inliers_matches_mask.sum(1),
                    'inliers_idx': matches_idx,    # 索引基于feat1/feat2
                    'model': model,       #最终trans
                    'model_valid': model_valid,  #trans是否有效
                    'model_coarse':model_corse,  #粗匹配trans
                    'model_re':model_re,  #重匹配trans
                    'model_re_valid':model_re_valid, #重匹配trans是否有效
                    'inliers_re': inliers_matches_re,
                    'inliers_num_re': inliers_num_re,
                    'inliers_idx_re': matches_idx_re,
                    'hamming_dis_matrix':Hamming_distance_matrix,    #汉明距矩阵
                    'rematchtag': rematchtag,     #标记是否用重匹配结果
                    }
    
    
    inliers_matches, inliers_matches_mask = check_inlier_angle_batch(nearest_matches, nearest_matches_angle, model, inliers)

    matches_idx = torch.zeros((inliers_matches.size(0),4),device=device) #N 4(batch_idx,A_idx,B_idx,hamming)
    inliers_matches_mask_idx = inliers_matches_mask.nonzero()
    matches_idx[:,0] = inliers_matches_mask_idx[:,0]
    matches_idx[:,1:] = nearest_topk_idx[0,inliers_matches_mask_idx[:,0],inliers_matches_mask_idx[:,1]]
    
    inliers_matches_num_mask = inliers_matches_mask.sum(1) > 4 #4
    best_inlier_residuals_sum_mask = best_inlier_residuals_sum > 16384./256/256 * inliers.sum(1)

    # sample_model = slmodel_batch()
    #每对匹配对点数不一致，无法并行
    for idx in range(len(inliers_matches_num_mask)):
        if inliers_matches_num_mask[idx] and best_inlier_residuals_sum_mask[idx] and model_valid[idx]:
            samples_batch = torch.cat([torch.tensor([idx],device=device),inliers_matches_mask[idx].nonzero().squeeze()])
            model_re_estimate_one, model_re_estimate_success = estimate_ls_batch(samples_batch[None,:],bdata)
            # model_re_estimate_one, model_re_estimate_success = sample_model.estimate_batch(samples_batch[None,:],bdata)
            if model_re_estimate_success[0] and torch.linalg.matrix_rank(model_re_estimate_one[0]) == 3:   #避免重匹配求逆时出错
                model[idx] = model_re_estimate_one[0]

    model_re = model.clone()
    model_corse = model.clone()  #备份粗匹配model用于计算特征
    model_re_valid = torch.zeros_like(model_valid)
    auth_result = inliers_matches_mask.sum(1)
    inliers_num_re = torch.zeros_like(inliers_matches_mask.sum(1))
    rematchtag = (inliers_matches_mask.sum(1) > 0)*(inliers_matches_mask.sum(1) < 22)
    rematchtag *= model_valid.bool()

    #用原来的代替，只起到占位作用
    inliers_matches_re = inliers_matches.clone()
    matches_idx_re = matches_idx.clone()

    #重匹配
    if rematchtag.sum() > 0:
        H_rotation = torch.atan2(model[:,1,0],model[:,1,0])
        overlap_mask = inv_warp_image_bilinear(torch.ones((rematchtag.sum() ,1,h,w),device=device),model[rematchtag])
        overlap = overlap_mask.mean([1,2,3])

        '''计算是否需要进行重匹配'''
        count = 0
        for idx, rematchtag_one in enumerate(rematchtag):
            if rematchtag_one:  #大于0小于22个内点走重匹配
                ori = H_rotation[idx]
                overlapthr = 40 if abs(ori) <= DEG2RAD(50) else 15

                #进一步计算是否需要重匹配
                if overlap[count] * 100 > overlapthr:
                    rematchtag[idx] = 1
                else:
                    rematchtag[idx] = 0
                count += 1

        if rematchtag.sum() > 0:
            rematch_choose = (rematchtag).nonzero()[:,0]
            samples2re = samples[rematchtag]   
            Hamming_distance_matrix2re = (Hamming_distance_matrix[0][:,rematch_choose],Hamming_distance_matrix[1][:,rematch_choose])
            model2re = model[rematchtag]
            inliers_matches_re, matches_idx_re, model_re_temp, model_re_valid_temp, inliers_matches_mask_re = rematch_batch(feat1, feat2, model2re, Hamming_distance_matrix2re, samples2re, top_k)

            #还原重匹配矩阵在原索引中的位置
            model_re[rematch_choose] = model_re_temp
            model_re_valid[rematch_choose] = model_re_valid_temp
            inliers_matches_re[:,0] = rematch_choose[inliers_matches_re[:,0].long()].float()
            matches_idx_re[:,0] = rematch_choose[matches_idx_re[:,0].long()].float()
            inliers_num_re[rematch_choose] = inliers_matches_mask_re.sum(1)

            #重匹配内点更多则采用重匹配结果
            match_re_updata_mask = inliers_matches_mask[rematch_choose].sum(1) < inliers_matches_mask_re.sum(1)
            match_re_updata_mask *= model_re_valid_temp.bool()

            rematchtag = torch.zeros_like(rematchtag) #清空之前的数据
            model[rematch_choose[match_re_updata_mask]] = model_re_temp[match_re_updata_mask]
            model_valid[rematch_choose[match_re_updata_mask]] = model_re_valid[rematch_choose[match_re_updata_mask]] #存在重匹配矩阵
            rematchtag[rematch_choose[match_re_updata_mask]] = 1 #记录真正使用重匹配结果的mask
            # inliers_matches_mask[rematch_choose[match_re_updata_mask]] = inliers_matches_mask_re[match_re_updata_mask]
            
            auth_result[rematch_choose[match_re_updata_mask]] = inliers_matches_mask_re.sum(1)[match_re_updata_mask]
    
    auth_result = auth_result > 4

    return auth_result, {'inliers': inliers_matches,
                        'inliers_num': inliers_matches_mask.sum(1),
                        'inliers_idx': matches_idx,    # 索引基于feat1/feat2
                        'model': model,       #最终trans
                        'model_valid': model_valid,  #trans是否有效
                        'model_coarse':model_corse,  #粗匹配trans
                        'model_re':model_re,  #重匹配trans
                        'model_re_valid':model_re_valid, #重匹配trans是否有效
                        'inliers_re': inliers_matches_re,
                        'inliers_num_re': inliers_num_re,
                        'inliers_idx_re': matches_idx_re,
                        'hamming_dis_matrix':Hamming_distance_matrix,    #汉明距矩阵
                        'rematchtag': rematchtag,     #标记是否用重匹配结果
                        }

def recognize_batch(feat1_batch, feat2_batch, device):
    ''' feat1 temple  feat2 sample
    {'pts':points['pts'],       #tensor BxNx5 (float)   range : [w,h]
    'prob':points['prob'],      #tensor BxNx1  (float)  range : 0-1
    'angles':points['angles'],  #tensor BxNx1 (float)   range : -pi/2-pi/2
    'desc':desc,                #tensor BxNx(float)    range : 0,1
    'enhance_img':enhance_img,  #tensor Bxh*w
    }
    '''
    feat1 = copy.deepcopy(feat1_batch)
    feat2 = copy.deepcopy(feat2_batch)

    batchsize1 = len(feat1['pts_b'])
    batchsize2 = len(feat2['pts_b'])
    assert batchsize1 == 1

    quality_mask = (feat1['quality'] > 50)*(feat2['quality'] > 50)

    top_k_30 = quality_mask.nonzero()[:,0]
    top_k_60 = (~quality_mask).nonzero()[:,0]
    auth_result = torch.zeros_like(quality_mask).to(device)

    batch = batchsize1*batchsize2
    device = feat1['pts_b'].device
    model_batch = torch.eye(3,device=device)[None,:].repeat(batch,1,1).double()
    model_valid_batch = torch.zeros(batch,device=device)
    model_coarse_batch = model_batch.clone()
    model_re_batch = model_batch.clone()
    model_re_valid_batch = model_valid_batch.clone()
    rematchtag_batch = model_valid_batch.clone().bool()
    inliers_num_batch = torch.zeros(batch,device=device)
    inliers_num_re_batch = torch.zeros(batch,device=device)
    inliers_matches = None
    matches_idx = None
    inliers_matches_re = None
    matches_idx_re = None
    Hamming_distance_matrix = (None,None)

    if len(top_k_30) > 0:
        auth_result_30, match_info_30 = recognize_batch_topk(feat1, feat2, top_k_30, top_k=30)

        auth_result[top_k_30] = auth_result_30
        model_batch[top_k_30] = match_info_30['model']
        model_valid_batch[top_k_30] = match_info_30['model_valid']
        model_coarse_batch[top_k_30] = match_info_30['model_coarse']
        model_re_batch[top_k_30] = match_info_30['model_re']
        model_re_valid_batch[top_k_30] = match_info_30['model_re_valid']
        rematchtag_batch[top_k_30] = match_info_30['rematchtag']
        inliers_num_batch[top_k_30] = match_info_30['inliers_num']
        inliers_num_re_batch[top_k_30] = match_info_30['inliers_num_re']
        Hamming_distance_matrix_30 = match_info_30['hamming_dis_matrix']

        #索引校正：存的索引为选择后，数据的索引，故需要还原为选择前batch的索引，即索引的索引
        inliers_matches_30 = match_info_30['inliers']
        inliers_matches_30[:,0] = top_k_30[inliers_matches_30[:,0].long()].float()
        inliers_idx_30 = match_info_30['inliers_idx']
        inliers_idx_30[:,0] = top_k_30[inliers_idx_30[:,0].long()].float()
        inliers_matches_re_30 = match_info_30['inliers_re']
        inliers_matches_re_30[:,0] = top_k_30[inliers_matches_re_30[:,0].long()].float()
        inliers_idx_re_30 = match_info_30['inliers_idx_re']
        inliers_idx_re_30[:,0] = top_k_30[inliers_idx_re_30[:,0].long()].float()
        
        inliers_matches = inliers_matches_30
        matches_idx = inliers_idx_30
        inliers_matches_re = inliers_matches_re_30
        matches_idx_re = inliers_idx_re_30
        Hamming_distance_matrix = Hamming_distance_matrix_30

    if len(top_k_60) > 0:
        auth_result_60, match_info_60 = recognize_batch_topk(feat1, feat2, top_k_60, top_k=60)

        auth_result[top_k_60] = auth_result_60
        model_batch[top_k_60] = match_info_60['model']
        model_valid_batch[top_k_60] = match_info_60['model_valid']
        model_coarse_batch[top_k_60] = match_info_60['model_coarse']
        model_re_batch[top_k_60] = match_info_60['model_re']
        model_re_valid_batch[top_k_60] = match_info_60['model_re_valid']
        rematchtag_batch[top_k_60] = match_info_60['rematchtag']
        inliers_num_batch[top_k_60] = match_info_60['inliers_num']
        inliers_num_re_batch[top_k_60] = match_info_60['inliers_num_re']
        Hamming_distance_matrix_60 = match_info_60['hamming_dis_matrix']

        #索引校正：存的索引为选择后，数据的索引，故需要还原为选择前batch的索引，即索引的索引
        inliers_matches_60 = match_info_60['inliers']
        inliers_matches_60[:,0] = top_k_60[inliers_matches_60[:,0].long()].float()
        inliers_idx_60 = match_info_60['inliers_idx']
        inliers_idx_60[:,0] = top_k_60[inliers_idx_60[:,0].long()].float()
        inliers_matches_re_60 = match_info_60['inliers_re']
        inliers_matches_re_60[:,0] = top_k_60[inliers_matches_re_60[:,0].long()].float()
        inliers_idx_re_60 = match_info_60['inliers_idx_re']
        inliers_idx_re_60[:,0] = top_k_60[inliers_idx_re_60[:,0].long()].float()
        
        if inliers_matches is not None:
            inliers_matches = torch.cat([inliers_matches,inliers_matches_60],dim=0)
            matches_idx = torch.cat([matches_idx,inliers_idx_60],dim=0)
            inliers_matches_re = torch.cat([inliers_matches_re,inliers_matches_re_60],dim=0)
            matches_idx_re = torch.cat([matches_idx_re,inliers_idx_re_60],dim=0)
            Hamming_distance_matrix_b = torch.cat([Hamming_distance_matrix[0],Hamming_distance_matrix_60[0]],dim=1)
            Hamming_distance_matrix_w = torch.cat([Hamming_distance_matrix[1],Hamming_distance_matrix_60[1]],dim=1)
            Hamming_distance_matrix_b = Hamming_distance_matrix_b[:,torch.cat([top_k_30,top_k_60])]
            Hamming_distance_matrix_w = Hamming_distance_matrix_w[:,torch.cat([top_k_30,top_k_60])]
            Hamming_distance_matrix = (Hamming_distance_matrix_b,Hamming_distance_matrix_w)
        else:
            inliers_matches = inliers_matches_60
            matches_idx = inliers_idx_60
            inliers_matches_re = inliers_matches_re_60
            matches_idx_re = inliers_idx_re_60
            Hamming_distance_matrix = Hamming_distance_matrix_60

    return auth_result, {'inliers': inliers_matches,
                        'inliers_num': inliers_num_batch,
                        'inliers_idx': matches_idx,    # 索引基于feat1/feat2
                        'model': model_batch,       #最终trans
                        'model_valid': model_valid_batch,  #trans是否有效
                        'model_coarse':model_coarse_batch,  #粗匹配trans
                        'model_re':model_re_batch,  #重匹配trans
                        'model_re_valid':model_re_valid_batch, #重匹配trans是否有效
                        'inliers_re': inliers_matches_re,
                        'inliers_num_re': inliers_num_re_batch,
                        'inliers_idx_re': matches_idx_re,
                        'hamming_dis_matrix':Hamming_distance_matrix,    #汉明距矩阵
                        'rematchtag': rematchtag_batch,     #标记是否用重匹配结果
                        }

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
        
        if feat1 is None or feat2 is None:
            return 
        result, rec_feat = recognize(feat1, feat2, 0)
        
        get_match_feat(feat1, feat2, rec_feat, feat_info)

        # dada = {'image1':feat1['enhance_img'],
        #         'image2':feat2['enhance_img'],
        #         'keypoints1':feat1['pts'],
        #         'keypoints2':feat2['pts'],
        #         'matches':rec_feat['inliers'],
        #         'unmatches':np.array([]),
        #        }
        # draw_match_img = draw_keypoints_match(dada)    #画点图
        # cv2.imwrite(path[:-4] + "_match.bmp", draw_match_img)
        if rec_feat['model'] is not None:
            input_img = {'img':feat1['enhance_img']/255., 'img_H':feat2['enhance_img']/255.}
            input_pts = {'matches':rec_feat['inliers']}
            Htrans = torch.from_numpy(rec_feat['model'].params)
            draw_pair_train_img = draw_pair_trans(input_img, input_pts, None, H = Htrans)  #画点图 + trans图
            cv2.imwrite(path[:-4] + "_match.bmp", draw_pair_train_img)
            #cv2.imwrite(path[:-4] + "_overlap.bmp", rec_feat['overlap_mask'])
        
if __name__ == "__main__":
    main()
    