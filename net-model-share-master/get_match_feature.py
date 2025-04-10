# -*- coding: utf-8 -*-

import os
import cv2
import copy
import numpy as np
from PIL import Image as Img
from desc_patch.desc.RotConv import warp_points
import torch
import torchvision.transforms as transforms

def get_dis(p_a, p_b):
    c = 2
    eps = 1e-12
    x = torch.unsqueeze(p_a[:, 0], 1) - torch.unsqueeze(p_b[:, 0], 0)  # N 2 -> NA 1 - 1 NB -> NA NB
    y = torch.unsqueeze(p_a[:, 1], 1) - torch.unsqueeze(p_b[:, 1], 0)
    dis = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + eps)
    return dis

def filter_points(points, shape, return_mask=False):
    ### check!
    points = points.float()
    shape = shape.float()
    mask = (points >= 0) * (points <= shape-1)
    mask = (torch.prod(mask, dim=-1) == 1)
    if return_mask:
        return points[mask], mask
    return points [mask]

def get_linears_density(h, w, model, nInlinersNum):
    # 计算内点密度
    np_flag = np.ones((h, w))
    np_flag_a = cv2.warpAffine(np_flag, model.params[:2], (w,h), flags=cv2.INTER_CUBIC)
    np_flag_a[np_flag_a[:,:]>0.5]=1
    np_flag_a[np_flag_a[:,:]<1]=0
    area = np.sum(np_flag_a)
    nInDenseOrg = (nInlinersNum*256+area/2)/(area+1)
    # 为了后续计算把area返回
    return nInDenseOrg,area

def get_nValidNum(pts_nms_A, h, w, model):
    # 关键点总数大于110,根据dbvalue筛掉20点 
    # 重叠区域点数是在重叠区域的所有关键点
    np_flag = np.ones((h, w))
    np_flag_a = cv2.warpAffine(np_flag, model.params[:2], (w,h), flags=cv2.INTER_CUBIC)
    np_flag_a[np_flag_a[:,:]>0.5]=1
    np_flag_a[np_flag_a[:,:]<1]=0
    
    warped_pnts = warp_points(torch.tensor(pts_nms_A),torch.tensor(model.params)) # 利用变换矩阵变换坐标点
    nValidNum_sample=0
    for point_s in warped_pnts:
        if int(point_s[1]) < 6 or int(point_s[1]) >= h-6 or int(point_s[0]) < 6 or int(point_s[0]) >= w-6:
            continue
        if np_flag_a[int(point_s[1]),int(point_s[0])]==1:
            nValidNum_sample+=1   
    return nValidNum_sample

def get_nInDenseQyh(inlinear_A,inlinear_B,model):
    # 获得第三个内点个数
    warped_pnts = warp_points(torch.tensor(inlinear_A),torch.tensor(model.params))
    nInlinersNumQ=0
    # 点对求距离卡控 
    for idx,pointA in enumerate(warped_pnts):
        pointB = inlinear_B[idx]
        pointA=pointA.numpy()
        # 比较x,y坐标距离
        if abs(pointB[0]-pointA[0])>3 or abs(pointB[1]-pointA[1])>3:
            continue
        dist = np.sum((pointB - pointA) ** 2)
        if dist<=2.5:
            nInlinersNumQ+=1
    return nInlinersNumQ
	
def getIdentityInlinersNum(trans: np.array, inliers_matches: np.array):
    '''
    不动点: 内点数-单位阵内点数
    点坐标放大256倍后满足坐标差在阈值(256*5>>1)范围
    trans@              [3x3]
    inliers_matches@    [Nx5] N is inliers number
    '''
    SLZSIZE = 256
    SLL_ERROR_LIMIT = (SLZSIZE * 5) >> 1

    inlier_num = inliers_matches.shape[0]
    if inlier_num == 0:   #inliers_matches为空
        return 0

    inlier_temp = inliers_matches[:, :2]
    inlier_samp = inliers_matches[:, 2:4]

    diffXY = np.abs(inlier_samp - inlier_temp) * SLZSIZE   # 点坐标放大256
    mask_L1 = np.sum((diffXY > SLL_ERROR_LIMIT), axis=1) == 0
    nErr = np.sum(np.power(diffXY, 2), axis=1)
    mask_L2 = nErr < np.power(SLL_ERROR_LIMIT, 2)
    identity_inlier_num = np.sum(mask_L1 * mask_L2)     # 单位阵内点数

    return inlier_num - identity_inlier_num

def inv_warp_image_batch_cv2(img, mat_homo_inv, mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [3, 3]
    :return:
        batch of warped images
        tensor [H, W]
    '''
    # compute inverse warped points
    assert len(img.shape) == 2
    assert len(mat_homo_inv.shape) == 2

    H, W = img.shape
    warped_img = cv2.warpPerspective(img, mat_homo_inv, (W, H))
    return warped_img

def inv_warp_image(img, mat_homo_inv, mode='bilinear'):
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
    warped_img = inv_warp_image_batch_cv2(img, mat_homo_inv, mode)
    return warped_img
    
def get_binary_img(img, W_ratio, B_ratio, M_ratio):
    h, w = img.shape
    img_hist = np.histogram(img,bins=16, range=(0,256), density=False)[0]
    img_hist_cum = np.cumsum(img_hist/(h*w))
    W_TH = np.argmin(np.abs(img_hist_cum - W_ratio))*16
    B_TH = np.argmin(np.abs(img_hist_cum - B_ratio))*16
    M_TH = np.argmin(np.abs(img_hist_cum - M_ratio))*16
    img_binary_w = (img > W_TH).astype(np.uint8)
    img_binary_b = (img > B_TH).astype(np.uint8)
    img_binary_m = (img > M_TH).astype(np.uint8)
    return img_binary_w, img_binary_b, img_binary_m, img_hist

def getRecuPairs(ptsA:np.array, ptsB:np.array, trans:np.array, H, W, correspond=2.5):
    '''
    复现点数
    '''
    ptsA_warped = warp_points(torch.tensor(ptsA), torch.tensor(trans))
    ptsA_warped = filter_points(ptsA_warped, torch.tensor([W, H]), return_mask=False)

    dis = get_dis(ptsA_warped, torch.tensor(ptsB))
    if dis.shape[0] == 0:
        return 0
    
    a2b_min_id = torch.argmin(dis, dim=1)
    len_p = len(a2b_min_id)
    ch = dis[list(range(len_p)), a2b_min_id] < correspond
    a2b_min_id = a2b_min_id[ch]

    # mask_both = torch.zeros_like(dis)
    # mask_both[ch, a2b_min_id] += 1

    # b2a_min_id = torch.argmin(dis, dim=0)
    # len_p = len(b2a_min_id)
    # ch = dis[b2a_min_id, list(range(len_p))] < correspond
    # b2a_min_id = b2a_min_id[ch]
    # mask_both[b2a_min_id, ch] += 1

    # nRecurNum = torch.sum(mask_both == 2)
    if a2b_min_id.shape[0] == 0:
        return 0
    return a2b_min_id.shape[0]


def get_nn_mask(key_dist):
    #单边最近邻
    n_amin = torch.argmin(key_dist, dim=1)
    mask = torch.ones_like(n_amin).bool()

    # Now `mask` is a binary mask and n_amin[mask] is an index array.
    # We use nonzero to turn `n_amin[mask]` into an index array and return
    nn_idx = torch.stack([torch.nonzero(mask, as_tuple=False)[:, 0],n_amin[mask]], dim=0).cpu()
    nn_mask = torch.zeros_like(key_dist, device=key_dist.device)
    nn_mask[nn_idx[0],nn_idx[1]] = 1
    
    nn_mask = nn_mask.bool()
    return nn_mask

def warp_points_batch(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    # expand points len to (x, y, 1)
    homographies = homographies.double()
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    points = points.unsqueeze(0) if no_batches else homographies
    # homographies = homographies.unsqueeze(0) if len(homographies.shape) == 2 else homographies
    batch_size = homographies.shape[0]
    points = torch.cat((points.double(), torch.ones((points.shape[0], points.shape[1], 1)).to(device)), dim=2)
    points = points.to(device)
    # homographies = homographies.view(batch_size*3,3)
    # warped_points = homographies*points
    # points = points.double()
    homographies = homographies.to(points.device)
    homographies = homographies.unsqueeze(1).repeat(1,points.size(1),1,1)
    points = points.unsqueeze(2).repeat(1,1,3,1)
    warped_points = torch.sum(homographies*points,dim=3)

    # warped_points = np.tensordot(homographies, points.transpose(), axes=([2], [0]))
    # normalize the points
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    warped_points = warped_points.squeeze()
    return warped_points

def get_matches_nMainDirDiff(ptsA,ptsB,angleA,angleB,model,th=1.5):
    
    #根据阈值和最近邻得到匹配对，然后计算主方向差
    #sample=》temple，pts1往pts2上变换
    ptsAT = warp_points_batch(torch.from_numpy(ptsA),torch.from_numpy(model.params))
    key_dist = get_dis(ptsAT,torch.from_numpy(ptsB))
    nn_mask = get_nn_mask(key_dist)
    matches_mask = (key_dist < th) & nn_mask

    assert matches_mask.sum() > 0, '若为0则trans有问题,无后续计算'

    matches_index = matches_mask.nonzero()
    matches_angle = np.zeros([matches_index.shape[0], 2])
    matches_angle[:,0] = angleA[matches_index[:,0]]
    matches_angle[:,1] = angleB[matches_index[:,1]]

    #根据匹配对计算主方向差
    pi_coef = 3.1415926
    diff = np.zeros([matches_index.shape[0], 4])
    
    diff[:,0] = -model.rotation - matches_angle[:,0] + matches_angle[:,1] #[-2pi,2pi]
    #print(model.rotation, inliers_matches_angle[:,0] - inliers_matches_angle[:,1])
    mask = diff[:,0] < 0
    diff[mask,0] = diff[mask,0] + 2*pi_coef  #[0,2pi]
    diff[:,1] = 2*pi_coef - diff[:,0]  #[0,2pi]
    
    diff[:,2] = -model.rotation - matches_angle[:,0] + matches_angle[:,1] + pi_coef #避免0,180干扰 [-pi,3pi]
    mask = diff[:,2] < 0
    diff[mask,2] = diff[mask,2] + 2*pi_coef  #[0,3pi]
    mask = diff[:,2] > 2*pi_coef
    diff[mask,2] = diff[mask,2] - 2*pi_coef  #[0,2pi]
    diff[:,3] = 2*pi_coef - diff[:,2]  #[0,2pi]
    
    diff_c = np.min(diff,axis = 1)  #  [0, pi]

    #转角度制
    diff_c = diff_c*57.29577951
    return diff_c

def getImgSimScoreOPT(ImgB1,ImgB2,mask):
    #输入均为整型
    nSimScoreArr = torch.zeros(4)
    nSimScoreArr[0] = ((ImgB1 == 0)*(ImgB2 == 0)*mask).sum()
    nSimScoreArr[1] = ((ImgB1 == 0)*(ImgB2 == 1)*mask).sum()
    nSimScoreArr[2] = ((ImgB1 == 1)*(ImgB2 == 0)*mask).sum()
    nSimScoreArr[3] = ((ImgB1 == 1)*(ImgB2 == 1)*mask).sum()
    return nSimScoreArr

def get_similarity_net(similarity_model, temple, sample):
    # img_a = Image.open("/home/jianght/000_codes/009_sim/quality_6193/test_img/62_sample.bmp")
    # img_p = Image.open("/home/jianght/000_codes/009_sim/quality_6193/test_img/62_temple.bmp")
    # img_a=np.array(img_a)
    # img_p=np.array(img_p)
    img_a = transforms.ToTensor()(sample)
    img_p = transforms.ToTensor()(temple)
    AB=torch.cat((img_a,img_p),dim=0).cuda()
    with torch.no_grad():
        out=similarity_model(AB.unsqueeze(dim=0))
    return int(out*265+55.5)


def get_match_feat(feat1, feat2, rec_feat, feat_info):
    '''
    利用trans特征提取
    '''
    match_fect = np.zeros((540, 2))       # 第一维为index, 第二维为对应特征值

    model = rec_feat['model']
    # nPhi 斜切角 19761？
    nPhi = int((model.shear / np.pi  + 1) * 20000)  # 65336 范围待定
    match_fect[17, 0] = 17
    match_fect[17, 1] = nPhi
    
    # nRot 旋转角 log里是[0, 90]
    nRot = int((model.rotation / np.pi + 1) * 180) 
    if nRot > 180:
        nRot -= 180
    nRot = np.min([nRot, 180 - nRot])
    match_fect[18, 0] = 18
    match_fect[18, 1] = nRot

    # nImQualitySum 图像质量和 [0, 100] + [0, 100]
    nImQualitySum = int(feat1['quality']) + feat2['quality']
    match_fect[20, 0] = 20
    match_fect[20, 1] = nImQualitySum

    # nImgQuaDif 图像质量差绝对值 abs([0, 100] - [0, 100]) = [0, 100]
    #print(feat1['quality'] - feat2['quality'])
    nImgQuaDif = np.abs(int(feat1['quality']) - feat2['quality'])
    match_fect[27, 0] = 27
    match_fect[27, 1] = nImgQuaDif

    # nMagDet 缩放幅度 log里[0, 30]?
    model_upper_left2x2 = (model.params[:2, :2].T * 256).astype(np.int32).astype(np.float32)
    nMag = np.sum(np.sqrt(np.diag(model_upper_left2x2.T * model_upper_left2x2)).astype(np.int32)) / 2 
    nMagDet = np.abs(nMag - 256)
    match_fect[28, 0] = 28
    match_fect[28, 1] = nMagDet


    model_corse = rec_feat['model_corse']  # 初匹配
    nInliOrig = rec_feat['inliers_num']
    inliers_matches = rec_feat['inliers']

    model = rec_feat['model']    # 最终匹配:可能为初匹配
    nInlinersNumR = rec_feat['inliers_num_re'] if rec_feat['rematchtag'] == 1 else nInliOrig
    inliers_matches_re = rec_feat['inliers_re'] if rec_feat['rematchtag'] == 1 else inliers_matches

    feat_info.set_feature_info_dic({'nInNumO': nInliOrig, 'nInNumR': nInlinersNumR})

    nNoInNumO = getIdentityInlinersNum(model_corse.params, inliers_matches)   # 初匹配不动点 nNoOrg
    nNoInNumR = getIdentityInlinersNum(model.params, inliers_matches_re)   # 重匹配不动点 nNoRem    #不一定走了重匹配 ？？？
    feat_info.set_feature_info_dic({'nNoOrg': nNoInNumO, 'nNoRem': nNoInNumR})
    # feat_info.set_feature_info('nNoRem', nNoInNumR)

    # 初匹配内点数目内点密度
    nInDenseOrg,area_org = get_linears_density(feat1['h'], feat1['w'], model_corse, nInliOrig)    
    feat_info.set_feature_info('nInDenO', nInDenseOrg)

    # 如果重匹配根据两个trans的内点对变换后，计算点对距离，如果单独的x,y大于3pass，距离差小于2.5则是新内点。然后取两个矩阵计算的最大点数作为第三个内点数
    # 如果初匹配点数大于22则赋值0,如果小于22,但是重叠面积不满足阈值没有走重匹配,则粗匹配的内点数目就是q内点数
    # 密度使用的面积有重匹配则是重匹配,没有就用初匹配的
    nInlinersNumQ=0
    nInDenseQyh=0
    if nInliOrig > 0 and nInliOrig < 22:
        nInlinersNumQ=nInliOrig
        nInDenseQyh = nInDenseOrg
        if rec_feat['model_re'] is not None:
            # 为了第三个点计算粗匹配trans
            nInlinersNumQ1 = get_nInDenseQyh(inliers_matches[:,2:4],inliers_matches[:,0:2],model_corse)

            # 重匹配内点及密度
            nInDenseRem, area_rem = get_linears_density(feat1['h'], feat1['w'], rec_feat['model_re'], nInlinersNumR)
            feat_info.set_feature_info('nInDenR', nInDenseRem)

            # 第三个内点数目计算重匹配trans
            nInlinersNumQ2 = get_nInDenseQyh(inliers_matches_re[:,2:4],inliers_matches_re[:,0:2], rec_feat['model_re'])
            nInlinersNumQ = max(nInlinersNumQ1,nInlinersNumQ2)
            nInDenseQyh = (nInlinersNumQ*256+area_rem/2)/(area_rem+1)

    feat_info.set_feature_info('nInDenQ', nInDenseQyh)
    feat_info.set_feature_info('nInNumQ', nInlinersNumQ)
    # 重叠区域点数,应该是sample
    nValidNum = get_nValidNum(feat2['pts'], feat1['h'], feat1['w'], model)
    feat_info.set_feature_info('nValidNum', nValidNum)

    '''变换图像,temple=>sample'''
    Hs, Ws = feat1['level_16_img'].shape
    Homography_resize = np.zeros((3,3))
    Homography_resize[0,0] = Ws / feat1['w']
    Homography_resize[1,1] = Hs / feat1['h']
    Homography_resize[2,2] = 1
    Homography_level_16 = Homography_resize @ model.params @ np.linalg.inv(Homography_resize)
    if np.linalg.matrix_rank(Homography_level_16) < 3:
        return 
    #有效重叠区域 sample转到temple
    OverLap_mask = inv_warp_image(np.ones_like(feat2['level_16_img']),Homography_level_16, mode="bilinear")
    
	#temple 白二值图,黑二值图,中二值图,直方图
    t_binary_img_w, t_binary_img_b, t_binary_img_m, t_level_16_img_hist = get_binary_img(feat1['level_16_img'], 50/255, 205/255, 130/255)
	
	#sample 白二值图,黑二值图,中二值图,直方图
    s_binary_img_w, s_binary_img_b, s_binary_img_m, s_level_16_img_hist = get_binary_img(feat2['level_16_img'], 50/255, 205/255, 130/255)
    s_binary_img_warp_b = inv_warp_image(s_binary_img_b,Homography_level_16, mode="bilinear")
    
    #分别统计00,01,10,11的数量
    nSimScoreArr = getImgSimScoreOPT(t_binary_img_b,s_binary_img_warp_b,OverLap_mask)
    #计算主方向差相关特征：L5,L10,L15,H20,H30
    nMainDirDiff = get_matches_nMainDirDiff(feat1['pts'],feat2['pts'],feat1['angles'],feat2['angles'],model,th=1.5)
    
    nMainDirCalAll = len(nMainDirDiff)
    nMainDirL5Num  = np.sum(nMainDirDiff < 5)
    nMainDirL10Num = np.sum(nMainDirDiff < 10)
    nMainDirL15Num = np.sum(nMainDirDiff < 15)
    nMainDirH20Num = np.sum(nMainDirDiff > 20)
    nMainDirH30Num = np.sum(nMainDirDiff > 30)
    feat_info.set_feature_info('nMainDirL5Ratio', nMainDirL5Num / nMainDirCalAll)
    feat_info.set_feature_info('nMainDirL10Ratio', nMainDirL10Num / nMainDirCalAll)
    feat_info.set_feature_info('nMainDirL15Ratio', nMainDirL15Num / nMainDirCalAll)
    feat_info.set_feature_info('nMainDirH20Ratio', nMainDirH20Num / nMainDirCalAll)
    feat_info.set_feature_info('nMainDirH30Ratio', nMainDirH30Num / nMainDirCalAll)
    feat_info.set_feature_info('nMainDirL5Num', nMainDirL5Num)
    feat_info.set_feature_info('nMainDirL10Num', nMainDirL10Num)
    feat_info.set_feature_info('nMainDirL15Num', nMainDirL15Num)
    feat_info.set_feature_info('nMainDirH20Num', nMainDirH20Num)
    feat_info.set_feature_info('nMainDirH30Num', nMainDirH30Num)
    feat_info.set_feature_info('nMainDirCalAll', nMainDirCalAll)
   
   

    nOverlap = np.mean(OverLap_mask)
    nOverlapAbs = np.sum(OverLap_mask)
    feat_info.set_feature_info('nOverlap', nOverlap)
    feat_info.set_feature_info('nOverlapAbs', nOverlapAbs)

    # 复现点数
    nRecurNum = getRecuPairs(feat2['pts'], feat1['pts'], model.params, feat1['h'], feat1['w'], correspond=2.5)
    feat_info.set_feature_info('nRecurNum', nRecurNum)

    nInlineSum = nInliOrig + nInlinersNumR + nInlinersNumQ
    feat_info.set_feature_info('nInlineSum', nInlineSum)

    # nInliRatio/nInliRatioRem/nInliRatioRecur/nRecuRatio
    nInlinRaito = (nInliOrig << 8) / (nValidNum + 1)
    nInlinRatioRe = (nInlinersNumR << 8) / (nValidNum + 1)
    nInliRatioRecur = (nInlinersNumR << 8) / (nRecurNum + 1)
    nRecuRatio = (nRecurNum * 100) / (nValidNum + 1)
    feat_info.set_feature_info('nInlinRaito', np.int(nInlinRaito + 0.5))
    feat_info.set_feature_info('nInlinRatioRe', np.int(nInlinRatioRe + 0.5))
    feat_info.set_feature_info('nInliRatioRecur', np.int(nInliRatioRecur + 0.5))
    feat_info.set_feature_info('nRecuRatio', np.int(nRecuRatio + 0.5))

    # OrientDiffV:应该是初匹配，后面还有OrientDiffV_ReMatch
    temp_inlier_org_idx = rec_feat['inliers_idx'][:, 0]
    samp_inlier_org_idx = rec_feat['inliers_idx'][:, 1]

    orien_temp = feat1['angles'][temp_inlier_org_idx] * 180. / np.pi    # deg
    orien_samp = feat2['angles'][samp_inlier_org_idx] * 180. / np.pi
    OrientDiffV = np.sqrt(np.mean(np.abs(orien_temp - orien_samp)))
    feat_info.set_feature_info('OrientDiffV', OrientDiffV)

    # devalue相关: DogValueDiffM/DogValueDiffV
    DogValue_temp = feat1['dbvalue'][temp_inlier_org_idx]
    DogValue_samp = feat2['dbvalue'][samp_inlier_org_idx]
    DogValueDiffM = np.mean(np.abs(DogValue_temp - DogValue_samp))
    DogValueDiffV = np.sqrt(DogValueDiffM)
    feat_info.set_feature_info_dic({'DogValueDiffM': DogValueDiffM, 'DogValueDiffV': DogValueDiffV})

    # 计算网络相似度
    mask_temple = cv2.resize(np.array(feat1['mask']*255,dtype=float), (Ws,Hs))
    mask_sample = cv2.resize(np.array(feat2['mask']*255,dtype=float), (Ws,Hs))
    mask_sample = inv_warp_image(mask_sample,Homography_level_16, mode="bilinear")
    mask_sample[mask_temple[:,:]<200]=0
    
    img_temple  = copy.deepcopy(feat1['level_16_img'])
    img_temple[mask_sample[:,:]<200]=0
    img_sample  = copy.deepcopy(feat2['level_16_img'])
    img_sample = inv_warp_image(img_sample,Homography_level_16, mode="bilinear")
    img_sample[mask_sample[:,:]<200]=0
    net_simi= get_similarity_net(feat_info.similarity_model, img_temple, img_sample)
    feat_info.set_feature_info('NetSimi', net_simi)



    pass