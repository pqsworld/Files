from email.mime import base
from sklearn.datasets import load_boston
import utils.correspondence_tools.correspondence_finder as correspondence_finder
import numpy as np
import torch
import torch.nn.functional as F

from utils.homographies import scale_homography_torch
from utils.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss
from utils.utils import getPtsFromLabels2D_torch

def toNumpy(tensor):
    return tensor.detach().cpu().numpy()

'''此方法占用空间严重，速度慢'''
def descriptor_loss_dense_selfsupervised(descriptors, descriptors_warped, homographies, mask_valid=None, 
                    cell_size=1, lamda_d=100, device='cpu', descriptor_dist=0.1, **config):
    homographies = homographies.to(device)
    # config
    from utils.utils import warp_points
    lamda_d = lamda_d # 250
    margin_pos = 1
    margin_neg = 0.2
    batch_size, Hc, Wc = descriptors.shape[0], descriptors.shape[2], descriptors.shape[3]
    #####
    # H, W = Hc.numpy().astype(int) * cell_size, Wc.numpy().astype(int) * cell_size
    H, W = Hc * cell_size, Wc * cell_size
    #####
    with torch.no_grad():
        # shape = torch.tensor(list(descriptors.shape[2:]))*torch.tensor([cell_size, cell_size]).type(torch.FloatTensor).to(device)
        shape = torch.tensor([H, W]).type(torch.FloatTensor).to(device)
        # compute the center pixel of every cell in the image

        coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2)
        coor_cells = coor_cells.type(torch.FloatTensor).to(device)
        coor_cells = coor_cells * cell_size + cell_size // 2
        ## coord_cells is now a grid containing the coordinates of the Hc x Wc
        ## center pixels of the 8x8 cells of the image

        coor_cells = coor_cells.view([1, 1, Hc, Wc, 2])  # be careful of the order  (y, x)
        coor_cells = torch.stack((coor_cells[:, :, :, :, 1], coor_cells[:, :, :, :, 0]), dim=-1)    # be careful of the order(x, y)
        warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)
        warped_coor_cells = warped_coor_cells.view([-1, Hc, Wc, 1, 1, 2])

        weight_field = 3
        pos_offset = torch.stack(torch.meshgrid(torch.arange(weight_field), torch.arange(weight_field)), dim=2).to(device)
        pos_offset -= weight_field // 2     # 邻域内距离中心的offset
        mask = torch.zeros(batch_size, H, W, H, W).to(device)
        for idx in range(warped_coor_cells.shape[0]):   # batch size
            warped_coor_cell = warped_coor_cells[idx, ...]      # (x, y)
            # compute the pairwise distance
            cell_distances = coor_cells - warped_coor_cell      # 广播机制
            cell_distances = torch.norm(cell_distances, dim=-1) # 计算两张图两两点对的距离
            mask_tmp = cell_distances < descriptor_dist            # 0.5
            mask_tmp = mask_tmp.type(torch.FloatTensor).to(device)
            mask[idx, ...] = mask_tmp
            (y_a, x_a, y_b, x_b) = torch.where(mask_tmp == 1)       # return tuple(0,1,2,3)
 
            # desc_warped = descriptors_warped[idx, :, :, :].clone()  # 暂存原数据，后续descriptors_warped会变动
            desc_coefficient = torch.ones_like(descriptors_warped[:, :, :, :])
            for (y, x) in zip(y_b, x_b):    # inner points
                pos_centor = torch.stack((y, x), dim=0).view([1, 1, -1])
                pos = pos_centor + pos_offset   # 邻域坐标矩阵

                count = 0
                # desc_warped_fuse = torch.zeros_like(desc_warped[:, 0, 0])
                desc_coefficient[idx, :, y, x] = 0.
                for p in pos.view([-1, 2]):
                    if (p[0] >= 0 and p[0] <= H - 1) and (p[1] >= 0 and p[1] <= W - 1):
                        count += 1
                        # desc_warped_fuse += 1. * desc_warped[:, p[0], p[1]]     # 邻域融合
                        desc_coefficient[idx, :, y, x] += 1. * descriptors_warped[idx, :, p[0], p[1]]     # 邻域融合
                desc_coefficient[idx, :, y, x] /= (count * (descriptors_warped[idx, :, y, x] + 1e-6))       # 得到的是系数



    descriptors_warped *= desc_coefficient
        ##### check
    #     print("descriptor_dist: ", descriptor_dist)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    # compute the pairwise dot product between descriptors: d^t * d
    descriptors = descriptors.transpose(1, 2).transpose(2, 3)
    descriptors = descriptors.view((batch_size, Hc, Wc, 1, 1, -1))
    descriptors_warped = descriptors_warped.transpose(1, 2).transpose(2, 3)
    descriptors_warped = descriptors_warped.view((batch_size, 1, 1, Hc, Wc, -1))
    # dot_product_desc = torch.zeros(batch_size, H, W, H, W, descriptors.shape[-1]).to(device)
    if mask_valid is None:
        # mask_valid = torch.ones_like(mask)
        mask_valid = torch.ones(batch_size, 1, Hc * cell_size, Wc * cell_size)
    mask_valid = mask_valid.view(batch_size, 1, 1, mask_valid.shape[1], mask_valid.shape[2])

    loss_desc_total = 0.
    pos_sum_total = 0.
    neg_sum_total = 0.
    for idx in range(warped_coor_cells.shape[0]):
        dot_desc = descriptors[idx, ...] * descriptors_warped[idx, ...]
        dot_desc = dot_desc.sum(dim=-1)
        # hinge loss
        positive_dist = torch.max(margin_pos - dot_desc, torch.tensor(0.).to(device))
        negative_dist = torch.max(dot_desc - margin_neg, torch.tensor(0.).to(device))
        loss_desc = lamda_d * mask[idx, ...] * positive_dist + (1 - mask[idx, ...]) * negative_dist
        loss_desc = loss_desc * mask_valid[idx, ...]
        
        normalization = ((mask_valid[idx, ...].sum() + 1) * cell_size * cell_size)
        pos_sum = (lamda_d * mask[idx, ...] * positive_dist / normalization).sum()
        neg_sum = ((1 - mask[idx, ...]) * negative_dist / normalization).sum()
        loss_desc = loss_desc.sum() / normalization

        pos_sum_total += pos_sum
        neg_sum_total += neg_sum
        loss_desc_total += loss_desc

    pos_sum_total /= batch_size
    neg_sum_total /= batch_size
    loss_desc_total /= batch_size
    

    return loss_desc, mask, pos_sum, neg_sum

def descriptor_space_merge(desc, block_size=4):

    block_size_sq = block_size * block_size

    output = desc.permute(0, 2, 3, 1)       # [batch_size, 136, 32, 256]
    (batch_size, s_height, s_width, s_depth) = output.size()

    d_height = int(s_height / block_size)
    t_1 = output.split(block_size, 2)  # 沿第2轴进行拆分,每个划分大小为block_size
    stack = [t_t.reshape(batch_size, d_height, block_size_sq, s_depth) for t_t in t_1]     # 
    # tmp = [m.sum(dim=2) / block_size_sq for m in stack]
    stack_merge = [m.mean(dim=2) for m in stack]
    output = torch.stack(stack_merge, 1)
    output = output.permute(0, 2, 1, 3)
    output = output.permute(0, 3, 1, 2)     # [batch_size, 256, Hc, Wc]
    return output

def descriptor_loss_dense_selfsupervised_new(descriptors, descriptors_warped, homographies, mask_valid=None, 
                    cell_size=4, lamda_d=100, device='cpu', descriptor_dist=2, **config):
    desc = descriptor_space_merge(descriptors, block_size=cell_size)
    desc_warped = descriptor_space_merge(descriptors_warped, block_size=cell_size)

    homographies = homographies.to(device)
    # config
    from utils.utils import warp_points
    lamda_d = lamda_d # 250
    margin_pos = 1
    margin_neg = 0.2
    batch_size, Hc, Wc = desc.shape[0], desc.shape[2], desc.shape[3]
    #####
    # H, W = Hc.numpy().astype(int) * cell_size, Wc.numpy().astype(int) * cell_size
    H, W = Hc * cell_size, Wc * cell_size
    with torch.no_grad():
        shape = torch.tensor([H, W]).type(torch.FloatTensor).to(device)
        # compute the center pixel of every cell in the image

        coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2)
        coor_cells = coor_cells.type(torch.FloatTensor).to(device)
        coor_cells = coor_cells * cell_size + cell_size // 2
        ## coord_cells is now a grid containing the coordinates of the Hc x Wc
        ## center pixels of the 8x8 cells of the image

        coor_cells = coor_cells.view([-1, 1, 1, Hc, Wc, 2])  # be careful of the order  (y, x)
        coor_cells = torch.stack((coor_cells[:,:,:,:,:,1], coor_cells[:,:,:,:,:,0]), dim=-1)    # (x, y)
        warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)

        warped_coor_cells = warped_coor_cells.view([-1, Hc, Wc, 1, 1, 2])
    #     print("warped_coor_cells: ", warped_coor_cells.shape)
        # compute the pairwise distance
        cell_distances = coor_cells - warped_coor_cells     # 用到了广播机制
        cell_distances = torch.norm(cell_distances, dim=-1) # 计算两张图两两点对的距离
        ##### check
    #     print("descriptor_dist: ", descriptor_dist)
        mask = cell_distances <= descriptor_dist # 0.5 # trick

        mask = mask.type(torch.FloatTensor).to(device)

    # compute the pairwise dot product between descriptors: d^t * d
    desc = desc.transpose(1, 2).transpose(2, 3)
    desc = desc.view((batch_size, Hc, Wc, 1, 1, -1))
    desc_warped = desc_warped.transpose(1, 2).transpose(2, 3)
    desc_warped = desc_warped.view((batch_size, 1, 1, Hc, Wc, -1))
    dot_product_desc = desc * desc_warped     # 两两描述子的内积
    dot_product_desc = dot_product_desc.sum(dim=-1)
    ## dot_product_desc.shape = [batch_size, Hc, Wc, Hc, Wc, desc_len]

    # hinge loss 合页损失函数
    positive_dist = torch.max(margin_pos - dot_product_desc, torch.tensor(0.).to(device))
    # positive_dist[positive_dist < 0] = 0
    negative_dist = torch.max(dot_product_desc - margin_neg, torch.tensor(0.).to(device))
    # negative_dist[neative_dist < 0] = 0
    # sum of the dimension

    if mask_valid is None:
        # mask_valid = torch.ones_like(mask)
        mask_valid = torch.ones(batch_size, 1, Hc * cell_size, Wc * cell_size)
    mask_valid = mask_valid.view(batch_size, 1, mask_valid.shape[1], mask_valid.shape[2])
    mask_valid_merge = descriptor_space_merge(mask_valid, block_size=cell_size)     # 把mask压缩到
    mask_valid_merge = mask_valid_merge.view(batch_size, 1, 1, mask_valid_merge.shape[2], mask_valid_merge.shape[3])

    loss_desc = lamda_d * mask * positive_dist + (1 - mask) * negative_dist
    loss_desc = loss_desc * mask_valid_merge
        # mask_validg = torch.ones_like(mask)
    ##### bug in normalization
    # normalization = (batch_size * (mask_valid.sum()+1) * Hc * Wc)
    normalization = ((mask_valid_merge.sum() + 1) * cell_size * cell_size)
    pos_sum = (lamda_d * mask * positive_dist / normalization).sum()
    neg_sum = ((1 - mask) * negative_dist / normalization).sum()
    loss_desc = loss_desc.sum() / normalization
    # loss_desc = loss_desc.sum() / (batch_size * Hc * Wc)
    # return loss_desc, mask, mask_valid, positive_dist, negative_dist
    return loss_desc, mask, pos_sum, neg_sum
    
def descriptor_loss_dense_supervised(desc_labelA, desc_labelB, semi_descA, semi_descB, labelA, labelB, device='cpu'):
    '''
    @desc_labelA:       [batch, num, D]
    @labelA: [batch, 1, H, W] points label 2D
    '''
    from utils.utils import getPtsFromLabels2D_torch, sample_desc_from_points_torch

    loss = 0.
    loss_A = 0.
    loss_B = 0.
    cos_method = False

    batch_size = semi_descA.shape[0]
    for i in range(batch_size):
        
        ptsA = getPtsFromLabels2D_torch(labelA[i, 0, :, :]).transpose(1, 0)
        ptsB = getPtsFromLabels2D_torch(labelB[i, 0, :, :]).transpose(1, 0)

        # select the descriptors from the label position
        desc_A = semi_descA[i, :, ptsA[:, 1].long(), ptsA[:, 0].long()].transpose(1, 0)
        desc_B = semi_descB[i, :, ptsB[:, 1].long(), ptsB[:, 0].long()].transpose(1, 0)

        cut_a, cut_b = ptsA.shape[0], ptsB.shape[0]
        desc_targetA = desc_labelA[i, :cut_a, :] # desc labels
        desc_targetB = desc_labelB[i, :cut_b, :]

        desc_A = torch.sigmoid(desc_A)
        desc_B = torch.sigmoid(desc_B)
        ce_lossA = F.binary_cross_entropy(desc_A, desc_targetA, reduction="none")
        ce_lossB = F.binary_cross_entropy(desc_B, desc_targetB, reduction="none")
        ce_lossA = ce_lossA.sum(dim=1) / ce_lossA.shape[1]
        ce_lossB = ce_lossB.sum(dim=1) / ce_lossB.shape[1]

        loss_A = loss_A + ce_lossA.sum() / (cut_a + 1e-10)
        loss_B = loss_B + ce_lossB.sum() / (cut_b + 1e-10)
        
        if cos_method:
            # norm
            dn_a = torch.norm(desc_A, p=2, dim=1) # Compute the norm of descriptors
            dn_b = torch.norm(desc_B, p=2, dim=1) # Compute the norm of descriptors
            desc_A = desc_A.div(torch.unsqueeze(dn_a, 1))
            desc_B = desc_B.div(torch.unsqueeze(dn_b, 1))

            dn_a = torch.norm(desc_targetA, p=2, dim=1) # Compute the norm of descriptors
            dn_b = torch.norm(desc_targetB, p=2, dim=1) # Compute the norm of descriptors
            desc_targetA = desc_targetA.div(torch.unsqueeze(dn_a, 1))
            desc_targetB = desc_targetB.div(torch.unsqueeze(dn_b, 1))
            

            cos_loss_A = torch.clamp(1. - (desc_A * desc_targetA).sum(dim=-1), min=0)
            cos_loss_B = torch.clamp(1. - (desc_B * desc_targetB).sum(dim=-1), min=0)

            cos_loss_A = 1. / cos_loss_A.shape[0] * cos_loss_A.sum()
            cos_loss_B = 1. / cos_loss_B.shape[0] * cos_loss_B.sum()

            loss_A = loss_A + cos_loss_A
            loss_B = loss_B + cos_loss_B
        
        else:
            innerA_loss = -torch.mul(desc_A,desc_targetA).sum(dim=1)/desc_A.shape[1]
            innerB_loss = -torch.mul(desc_B,desc_targetB).sum(dim=1)/desc_B.shape[1]
            loss_A = loss_A + innerA_loss.sum() / (cut_a + 1e-10)
            loss_B = loss_B + innerB_loss.sum() / (cut_b + 1e-10)

        
    
    loss_A = loss_A / batch_size 
    loss_B = loss_B / batch_size 
    loss =  loss_A + loss_B

    return loss, loss_A, loss_B

def descriptor_loss_sparse_supervised(desc_labelA, desc_labelB, semi_descA, semi_descB, labelA, labelB, device='cpu'):
    '''
    @labelA:            [batch, 1, H, W] points label 2D
    '''
    from utils.utils import getPtsFromLabels2D_torch, sample_desc_from_points_torch

    loss = 0.
    loss_A = 0.
    loss_B = 0.
    batch_size = semi_descA.shape[0]
    for i in range(batch_size):
        
        ptsA = getPtsFromLabels2D_torch(labelA[i, 0, :, :]).transpose(1, 0)
        ptsB = getPtsFromLabels2D_torch(labelB[i, 0, :, :]).transpose(1, 0)
        desc_A = sample_desc_from_points_torch(semi_descA[i, :, :, :].unsqueeze(0), ptsA.transpose(1, 0), device=device).transpose(1, 0)    # net desc
        desc_B = sample_desc_from_points_torch(semi_descB[i, :, :, :].unsqueeze(0), ptsB.transpose(1, 0), device=device).transpose(1, 0)

        cut_a, cut_b = ptsA.shape[0], ptsB.shape[0]
        desc_targetA = desc_labelA[i, :cut_a, :] # desc labels
        desc_targetB = desc_labelB[i, :cut_b, :]

        ce_lossA = F.binary_cross_entropy_with_logits(desc_A, desc_targetA, reduction="none")
        ce_lossB = F.binary_cross_entropy_with_logits(desc_B, desc_targetB, reduction="none")
        ce_lossA = ce_lossA.sum(dim=1) / ce_lossA.shape[1]
        ce_lossB = ce_lossB.sum(dim=1) / ce_lossB.shape[1]

        loss_A = loss_A + ce_lossA.sum() / (cut_a + 1e-10)
        loss_B = loss_B + ce_lossB.sum() / (cut_b + 1e-10)
    
    loss_A = loss_A / batch_size 
    loss_B = loss_B / batch_size 
    loss =  loss_A + loss_B

    return loss, loss_A, loss_B

def get_coor_cells(Hc, Wc, cell_size, device='cpu', uv=False):
    coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2)
    coor_cells = coor_cells.type(torch.FloatTensor).to(device)
    coor_cells = coor_cells.view(-1, 2)
    # change vu to uv
    if uv:
        coor_cells = torch.stack((coor_cells[:,1], coor_cells[:,0]), dim=1) # (y, x) to (x, y)

    return coor_cells.to(device)

def warp_coor_cells_with_homographies(coor_cells, homographies, uv=False, device='cpu'):
    from utils.utils import warp_points
    # warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)
    # warped_coor_cells = normPts(coor_cells.view([-1, 2]), shape)
    warped_coor_cells = coor_cells
    if uv == False:
        warped_coor_cells = torch.stack((warped_coor_cells[:,1], warped_coor_cells[:,0]), dim=1) # (y, x) to (x, y)

    # print("homographies: ", homographies)
    warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

    if uv == False:
        warped_coor_cells = torch.stack((warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]), dim=2)  # (batch, x, y) to (batch, y, x)

    # shape_cell = torch.tensor([H//cell_size, W//cell_size]).type(torch.FloatTensor).to(device)
    # warped_coor_mask = denormPts(warped_coor_cells, shape_cell)

    return warped_coor_cells


def create_non_matches(uv_a, uv_b_non_matches, multiplier):
    """
    Simple wrapper for repeated code
    :param uv_a:
    :type uv_a:
    :param uv_b_non_matches:
    :type uv_b_non_matches:
    :param multiplier:
    :type multiplier:
    :return:
    :rtype:
    """
    uv_a_long = (torch.t(uv_a[0].repeat(multiplier, 1)).contiguous().view(-1, 1),
                 torch.t(uv_a[1].repeat(multiplier, 1)).contiguous().view(-1, 1))

    uv_b_non_matches_long = (uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1))

    return uv_a_long, uv_b_non_matches_long


def descriptor_loss_sparse(descriptors, descriptors_warped, homographies, mask_valid=None,
                           cell_size=8, device='cpu', descriptor_dist=4, lamda_d=250,
                           num_matching_attempts=1000, num_masked_non_matches_per_match=10, 
                           dist='cos', method='1d', **config):
    """
    consider batches of descriptors
    :param descriptors:
        Output from descriptor head
        tensor [descriptors, Hc, Wc]
    :param descriptors_warped:
        Output from descriptor head of warped image
        tensor [descriptors, Hc, Wc]
    """

    def uv_to_tuple(uv):
        return (uv[:, 0], uv[:, 1])

    def tuple_to_uv(uv_tuple):
        return torch.stack([uv_tuple[0], uv_tuple[1]])

    def tuple_to_1d(uv_tuple, W, uv=True):
        if uv:
            return uv_tuple[0] + uv_tuple[1]*W
        else:
            return uv_tuple[0]*W + uv_tuple[1]


    def uv_to_1d(points, W, uv=True):
        # assert points.dim == 2
        #     print("points: ", points[0])
        #     print("H: ", H)
        if uv:
            return points[..., 0] + points[..., 1]*W
        else:
            return points[..., 0]*W + points[..., 1]

    ## calculate matches loss
    def get_match_loss(image_a_pred, image_b_pred, matches_a, matches_b, dist='cos', method='1d'):
        match_loss, matches_a_descriptors, matches_b_descriptors = \
            PixelwiseContrastiveLoss.match_loss(image_a_pred, image_b_pred, 
                matches_a, matches_b, dist=dist, method=method)
        return match_loss

    def get_non_matches_corr(img_b_shape, uv_a, uv_b_matches, num_masked_non_matches_per_match=10, device='cpu'):
        ## sample non matches
        uv_b_matches = uv_b_matches.squeeze()
        uv_b_matches_tuple = uv_to_tuple(uv_b_matches)
        uv_b_non_matches_tuple = correspondence_finder.create_non_correspondences(uv_b_matches_tuple,
                                        img_b_shape, num_non_matches_per_match=num_masked_non_matches_per_match,
                                        img_b_mask=None)

        ## create_non_correspondences
        #     print("img_b_shape ", img_b_shape)
        #     print("uv_b_matches ", uv_b_matches.shape)
        # print("uv_a: ", uv_to_tuple(uv_a))
        # print("uv_b_non_matches: ", uv_b_non_matches)
        #     print("uv_b_non_matches: ", tensorUv2tuple(uv_b_non_matches))
        uv_a_tuple, uv_b_non_matches_tuple = \
            create_non_matches(uv_to_tuple(uv_a), uv_b_non_matches_tuple, num_masked_non_matches_per_match)
        return uv_a_tuple, uv_b_non_matches_tuple

    def get_non_match_loss(image_a_pred, image_b_pred, non_matches_a, non_matches_b, dist='cos'):
        ## non matches loss
        non_match_loss, num_hard_negatives, non_matches_a_descriptors, non_matches_b_descriptors = \
            PixelwiseContrastiveLoss.non_match_descriptor_loss(image_a_pred, image_b_pred,
                                                               non_matches_a.long().squeeze(),
                                                               non_matches_b.long().squeeze(),
                                                               M=0.2, invert=True, dist=dist)
        non_match_loss = non_match_loss.sum()/(num_hard_negatives + 1)
        return non_match_loss

    from utils.utils import filter_points
    from utils.utils import crop_or_pad_choice
    from utils.utils import normPts
    # ##### print configs
    # print("num_masked_non_matches_per_match: ", num_masked_non_matches_per_match)
    # print("num_matching_attempts: ", num_matching_attempts)
    # dist = 'cos'
    # print("method: ", method)

    Hc, Wc = descriptors.shape[1], descriptors.shape[2] #17, 4
    img_shape = (Hc, Wc)
    # print("img_shape: ", img_shape)
    # img_shape_cpu = (Hc.to('cpu'), Wc.to('cpu'))

    # image_a_pred = descriptors.view(1, -1, Hc * Wc).transpose(1, 2)  # torch [batch_size, H*W, D]
    def descriptor_reshape(descriptors):
        descriptors = descriptors.view(-1, Hc * Wc).transpose(0, 1)  # torch [D, H, W] --> [H*W, d]
        descriptors = descriptors.unsqueeze(0)  # torch [1, H*W, D]
        return descriptors

    image_a_pred = descriptor_reshape(descriptors)  # torch [1, H*W, D] ,[1,68,128]
    # print("image_a_pred: ", image_a_pred.shape)
    image_b_pred = descriptor_reshape(descriptors_warped)  # torch [1, H*W, D]

    # matches
    uv_a = get_coor_cells(Hc, Wc, cell_size, uv=True, device='cpu') #[256,2]
    # print("uv_a: ", uv_a[0])

    '''H矩阵是确定的(-1,1)的网格矩阵'''
    # homographies_H = scale_homography_torch(homographies, img_shape, shift=(-1, -1))    #img_shape:16,16

    '''H矩阵是(136,32)的图像位姿'''
    trans = torch.tensor([[8., 0., 0.], [0., 8., 0], [0., 0., 1.]], dtype=torch.float32)
    H_tf = torch.inverse(trans) @ homographies @ trans
    homographies_H = H_tf



    # print("experiment inverse homographies")
    # homographies_H = torch.stack([torch.inverse(H) for H in homographies_H])
    # print("homographies_H: ", homographies_H.shape)
    # homographies_H = torch.inverse(homographies_H)


    uv_b_matches = warp_coor_cells_with_homographies(uv_a, homographies_H.to('cpu'), uv=True, device='cpu')     
    # 此时uv_b_matches是做过单应变换后的uv坐标
    # print("uv_b_matches before round: ", uv_b_matches[0])

    uv_b_matches.round_() # 变成整数
    # print("uv_b_matches after round: ", uv_b_matches[0])
    uv_b_matches = uv_b_matches.squeeze(0)


    # filtering out of range points
    # choice = crop_or_pad_choice(x_all.shape[0], self.sift_num, shuffle=True)
    # uv_b_matches
    uv_b_matches, mask = filter_points(uv_b_matches, torch.tensor([Wc, Hc]).to(device='cpu'), return_mask=True) # mask表示uv_b_matches的有效区域
    # print ("pos mask sum: ", mask.sum())
    uv_a = uv_a[mask]
    
    # crop to the same length
    shuffle = True
    if not shuffle: print("shuffle: ", shuffle)
    choice = crop_or_pad_choice(uv_b_matches.shape[0], num_matching_attempts, shuffle=shuffle)
    choice = torch.tensor(choice)
    uv_a = uv_a[choice]
    uv_b_matches = uv_b_matches[choice]

    if method == '2d':
        matches_a = normPts(uv_a, torch.tensor([Wc, Hc]).float()) # [u, v],归一化到-1~1之间
        matches_b = normPts(uv_b_matches, torch.tensor([Wc, Hc]).float())
    else:
        matches_a = uv_to_1d(uv_a, Wc)
        matches_b = uv_to_1d(uv_b_matches, Wc)

    # print("matches_a: ", matches_a.shape)
    # print("matches_b: ", matches_b.shape)
    # print("matches_b max: ", matches_b.max())

    # 前面计算好1000*2形式的matches a和b(归一化到-1~1之间)，通过这对矩阵计算descriptors和descriptors_warped的匹配loss
    if method == '2d':
        match_loss = get_match_loss(descriptors, descriptors_warped, matches_a.to(device), 
            matches_b.to(device), dist=dist, method='2d')
    else:
        match_loss = get_match_loss(image_a_pred, image_b_pred, 
            matches_a.long().to(device), matches_b.long().to(device), dist=dist)

    # non matches

    # get non matches correspondence
    uv_a_tuple, uv_b_non_matches_tuple = get_non_matches_corr(img_shape,
                                            uv_a, uv_b_matches,
                                            num_masked_non_matches_per_match=num_masked_non_matches_per_match)

    non_matches_a = tuple_to_1d(uv_a_tuple, Wc)
    non_matches_b = tuple_to_1d(uv_b_non_matches_tuple, Wc)

    # print("non_matches_a: ", non_matches_a)
    # print("non_matches_b: ", non_matches_b)

    non_match_loss = get_non_match_loss(image_a_pred, image_b_pred, non_matches_a.to(device),
                                        non_matches_b.to(device), dist=dist)
    # non_match_loss = non_match_loss.mean()

    # loss = lamda_d * match_loss + non_match_loss
    '''修改'''
    loss = match_loss + lamda_d * non_match_loss
    return loss, lamda_d * match_loss, non_match_loss
    pass

"""
img[uv_b_matches.long()[:,1],uv_b_matches.long()[:,0]] = 1
from utils.utils import pltImshow
pltImshow(img.numpy())

"""

def batch_descriptor_loss_sparse(descriptors, descriptors_warped, homographies, **options):
    loss = []
    pos_loss = []
    neg_loss = []
    batch_size = descriptors.shape[0]
    for i in range(batch_size):
        losses = descriptor_loss_sparse(descriptors[i], descriptors_warped[i],
                    # torch.tensor(homographies[i], dtype=torch.float32), **options)
                    homographies[i].type(torch.float32), **options)
        loss.append(losses[0])
        pos_loss.append(losses[1])
        neg_loss.append(losses[2])
    loss, pos_loss, neg_loss = torch.stack(loss), torch.stack(pos_loss), torch.stack(neg_loss)
    return loss.mean(), None, pos_loss.mean(), neg_loss.mean()

if __name__ == '__main__':
    # config
    H, W = 240, 320
    cell_size = 8
    Hc, Wc = H // cell_size, W // cell_size

    D = 3
    torch.manual_seed(0)
    np.random.seed(0)

    batch_size = 2
    device = 'cpu'
    method = '2d'

    num_matching_attempts = 1000
    num_masked_non_matches_per_match = 200
    lamda_d = 1

    homographies = np.identity(3)[np.newaxis, :, :]
    homographies = np.tile(homographies, [batch_size, 1, 1])

    def randomDescriptor():
        descriptors = torch.tensor(np.random.rand(2, D, Hc, Wc)-0.5, dtype=torch.float32)
        dn = torch.norm(descriptors, p=2, dim=1)  # Compute the norm.
        descriptors = descriptors.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return descriptors

    # descriptors = torch.tensor(np.random.rand(2, D, Hc, Wc), dtype=torch.float32)
    # dn = torch.norm(descriptors, p=2, dim=1) # Compute the norm.
    # desc = descriptors.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    descriptors = randomDescriptor()
    print("descriptors: ", descriptors.shape)
    # descriptors_warped = torch.tensor(np.random.rand(2, D, Hc, Wc), dtype=torch.float32)
    descriptors_warped = randomDescriptor()
    descriptor_loss = descriptor_loss_sparse(descriptors[0], descriptors_warped[0],
                                             torch.tensor(homographies[0], dtype=torch.float32),
                                             method=method)
    print("descriptor_loss: ", descriptor_loss)

    # loss = batch_descriptor_loss_sparse(descriptors, descriptors_warped,
    #                                     torch.tensor(homographies, dtype=torch.float32),
    #                                     num_matching_attempts = num_matching_attempts,
    #                                     num_masked_non_matches_per_match = num_masked_non_matches_per_match,
    #                                     device=device,
    #                                     lamda_d = lamda_d, 
    #                                     method=method)
    # print("batch descriptor_loss: ", loss)

    loss = batch_descriptor_loss_sparse(descriptors, descriptors,
                                        torch.tensor(homographies, dtype=torch.float32),
                                        num_matching_attempts = num_matching_attempts,
                                        num_masked_non_matches_per_match = num_masked_non_matches_per_match,
                                        device=device,
                                        lamda_d = lamda_d,
                                        method=method)
    print("same descriptor_loss (pos should be 0): ", loss)

