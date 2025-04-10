from signal import Handlers
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

'''pnts: from func"getPtsFromHeatmap", pnts --> netout_as_label'''
def points_to_2D(pnts, H, W):
    labels = np.zeros((H, W))
    pnts = pnts.astype(int)
    flag = (pnts[:, 1]<136) & (pnts[:, 0]<136) & (pnts[:, 0]>0) & (pnts[:, 1]>0)
    pnts = pnts[flag, :]
    labels[pnts[:, 1], pnts[:, 0]] = 1
    return labels

def sample_homography_cv(H, W, max_angle=30, n_angles=25):
    scale = 1
    angles = np.linspace(-max_angle, max_angle, num=n_angles)
    angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
    M = [cv2.getRotationMatrix2D((W / 2, H / 2), i, scale) for i in angles]
    # center = np.mean(pts2, axis=0, keepdims=True)
    M = [np.concatenate((m, [[0, 0, 1.]]), axis=0) for m in M]

    valid = np.arange(n_angles)
    idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
    homo = M[idx]

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
        # dense = nn.functional.softmax(semi, dim=1)  # [batch_size, 65, Hc, Wc] 在dim=1维度进行softmax，相当于65个分类
        # Remove dustbin.
        nodust = semi[:, :-1, :, :]                # [batch_size, 64, Hc, Wc]
        nodust = nn.functional.softmax(nodust, dim=1)
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
    depth2space = DepthToSpace(4)   # 实例化一个DepthToSpace类
    heatmap = depth2space(nodust)                   # [batch, 1, 128, 128]
    heatmap = heatmap.squeeze(0) if not batch else heatmap      # squeeze 只会对维度为1的维度进行压缩

    return heatmap

def flattenDetection_new(semi, tensor=False):
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
        # dense = nn.functional.softmax(semi, dim=1)  # [batch_size, 65, Hc, Wc] 在dim=1维度进行softmax，相当于65个分类
        # Remove dustbin.
        nodust = semi[:, :-1, :, :]                # [batch_size, 64, Hc, Wc]
        nodust = F.softmax(nodust, dim=1)
        # nodust = F.relu(nodust)

        # for i in range(nodust.size(2)):
        #     for j in range(nodust.size(3)):
        #         if dense[:, -1, i, j] > (1 / 65):
        #             nodust[:, :, i, j] = 0

    else:
        # dense = nn.functional.softmax(semi, dim=0) # [65, Hc, Wc]
        nodust = semi[:-1, :, :].unsqueeze(0)
        nodust = F.softmax(nodust, dim=1)
        # nodust = F.relu(nodust)
        # for i in range(nodust.size(2)):
        #     for j in range(nodust.size(3)):
        #         if dense[-1, i, j] > (1 / 65):
        #             nodust[:, :, i, j] = 0
    # Reshape to get full resolution heatmap.
    # heatmap = flatten64to1(nodust, tensor=True) # [1, H, W]
    depth2space = DepthToSpace(8)   # 实例化一个DepthToSpace类
    heatmap = depth2space(nodust)                   # [batch, 1, 128, 128]
    heatmap = heatmap.squeeze(0) if not batch else heatmap      # squeeze 只会对维度为1的维度进行压缩

    # 显示用
    # temp = heatmap.narrow(0, 0, 1).squeeze(0).squeeze(0)
    # # temp.toNumpy(temp)
    # temp = temp.detach().cpu().numpy()
    # pltImshow(temp)

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

    # 与C代码保持一致
    mask = np.zeros_like(heatmap)
    mask[bord:-bord, bord:-bord] = 1
    heatmap *= mask

    ys, xs = np.where(heatmap >= conf_thresh)  # Confidence threshold.

    # print(torch.sum(torch.tensor(heatmap)))
    sparsemap = (heatmap >= conf_thresh)

    # print(torch.sum(torch.tensor(sparsemap)))
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = xs # abuse of ys, xs
    pts[1, :] = ys
    pts[2, :] = heatmap[ys, xs]  # check the (x, y) here
    # print(torch.sum(torch.tensor(pts[2, :])), xs.shape)
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

def grid_indexes(size):
    weights = np.zeros((size, size, 1, 2), dtype=np.float32)

    columns = []
    for idx in range(1, 1+size):
        columns.append(np.ones((size))*idx)
    columns = np.asarray(columns)

    rows = []
    for idx in range(1, 1+size):
        rows.append(np.asarray(range(1, 1+size)))
    rows = np.asarray(rows)

    weights[:, :, 0, 0] = columns
    weights[:, :, 0, 1] = rows

    return weights.transpose([3, 2, 0, 1])

def ones_multiple_channels(size, num_channels):

    ones = np.ones((size, size))
    weights = np.zeros((size, size, num_channels, num_channels), dtype=np.float32)

    for i in range(num_channels):
        weights[:, :, i, i] = ones
    
    return weights.transpose([3, 2, 0, 1])

def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2

def linear_upsample_weights(half_factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with linear filter
    initialization.
    """

    filter_size = get_kernel_size(half_factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = np.ones((filter_size, filter_size))
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights.transpose([3, 2, 0, 1])

def create_kernels(MSIP_sizes, device):      
    kernels = {}
    for ksize in MSIP_sizes:
        ones_kernel = ones_multiple_channels(ksize, 1)
        indexes_kernel = grid_indexes(ksize)
        upsample_filter_np = linear_upsample_weights(int(ksize / 2), 1)
        
        ones_kernel_t = torch.tensor(ones_kernel, device=device)
        indexes_kernel_t = torch.tensor(indexes_kernel, device=device)
        upsample_filter_t = torch.tensor(upsample_filter_np, device=device)

        kernels['ones_kernel_'+str(ksize)] = ones_kernel_t
        kernels['indexes_kernel_'+str(ksize)] = indexes_kernel_t
        kernels['upsample_filter_'+str(ksize)] = upsample_filter_t
    return kernels

def ip_layer(scores, w_size, kernels):
    eps = 1e-6
    scores_shape = scores.shape     # [b, 1, H, W]
    # maxpool
    scores_pool = F.max_pool2d(scores, kernel_size=w_size, stride=w_size)
    scores_max_unpool = F.conv_transpose2d(scores_pool, kernels['upsample_filter_'+str(w_size)], stride=w_size)
    exp_map = torch.exp(torch.divide(scores, scores_max_unpool + eps)) - 1*(1.-eps)
    sum_exp_map = F.conv2d(exp_map, kernels['ones_kernel_' + str(w_size)], stride=w_size)
    indexes_map = F.conv2d(exp_map, kernels['indexes_kernel_' + str(w_size)], stride=w_size)
    indexes_map = torch.divide(indexes_map, sum_exp_map + eps)

    max_scores_pool = torch.max(torch.max(scores_pool, dim=3, keepdim=True).values, dim=2, keepdim=True).values
    norm_scores_pool= torch.divide(scores_pool, max_scores_pool + eps)
    return indexes_map, [scores_pool, norm_scores_pool]

class NonMaxSuppression(torch.nn.Module):
    '''
        NonMaxSuppression class
    '''
    def __init__(self, thr=0.0, nms_size=5):
        super(NonMaxSuppression, self).__init__()
        padding = nms_size // 2
        self.max_filter = nn.MaxPool2d(kernel_size=nms_size, stride=1, padding=padding)
        self.thr = thr

    def forward(self, scores):

        # local maxima
        maxima = (scores == self.max_filter(scores))

        # remove low peaks
        maxima *= (scores > self.thr)

        return maxima.nonzero().t()[2:4]

def getPtsFromHeatmapByNMS(heatmap, conf_thresh, nms_dist, num_kpts_i=150):
    nms = NonMaxSuppression(thr=conf_thresh, nms_size= 2*nms_dist+1)
    kps = nms(heatmap)
    c = heatmap[0, 0, kps[0], kps[1]]
    sc, indices = torch.sort(c, descending=True)
    indices = indices[torch.where(sc > 0.)]
    kps = kps[:, indices]
    kps_np = torch.cat([kps[1].view(-1, 1).float(), kps[0].view(-1, 1).float()],
        dim=1).detach()
    return kps_np

def getPtsFromHeatmapByCoordinates(heatmap, conf_thresh, w_size, bord=0):
    H, W = heatmap.shape[2], heatmap.shape[3]
    Hc, Wc = heatmap.shape[2] // w_size, heatmap.shape[3] // w_size
    kernels = create_kernels([w_size], heatmap.device)
    indexes_map, _ = ip_layer(heatmap, w_size, kernels)
    coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2) * w_size
    coor_cells = coor_cells.type(torch.FloatTensor).to(heatmap.device).permute(2, 0, 1)
    indexes = indexes_map + coor_cells - 1
    indexes = indexes.view(2, -1)

    # sort 
    labels = torch.zeros(H, W)
    pnts_int = torch.min(
        indexes.transpose(0, 1).round().long(), torch.tensor([[H - 1, W - 1]], device=indexes.device).long()
    )
    # print('--3', pnts_int, pnts_int.size())
    labels[pnts_int[:, 0], pnts_int[:, 1]] = 1
    indexes_score = torch.cat((indexes, heatmap[0, 0, :][labels == 1].view(-1, Wc*Hc)), dim=0)
    
    inds = torch.argsort(indexes_score[2, :], descending=True)
    indexes = indexes_score[:2, inds]
    indexes = indexes[[1, 0], :] # (x, y)
    return indexes    

def inv_warp_image_batch_cv2(img, mat_homo_inv, device='cpu', mode='bilinear'):
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
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)

    Batch, channel, H, W = img.shape

    warped_img = cv2.warpPerspective(img.squeeze().numpy(), mat_homo_inv.squeeze().numpy(), (W, H))
    warped_img = torch.tensor(warped_img, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

    # warped_img = cv2.warpAffine(img.squeeze().numpy(), mat_homo_inv[0, :2, :].squeeze().numpy(), (W, H))
    # warped_img = torch.tensor(warped_img, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    return warped_img

def inv_warp_image(img, mat_homo_inv, device='cpu', mode='bilinear'):
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
    warped_img = inv_warp_image_batch_cv2(img, mat_homo_inv, device, mode)
    return warped_img.squeeze()

def compute_valid_mask(image_shape, inv_homography, device='cpu', erosion_radius=0):
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
    if inv_homography.dim() == 2:
        inv_homography = inv_homography.view(-1, 3, 3)
    batch_size = inv_homography.shape[0]
    mask = torch.ones(batch_size, 1, image_shape[0], image_shape[1]).to(device)
    mask = inv_warp_image_batch_cv2(mask, inv_homography, device=device, mode='nearest')
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
    points = torch.cat((points.float(), torch.ones((points.shape[0], 1), device=points.device)), dim=1)    # expand points to (x, y, 1)
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

def filter_points(points, shape, return_mask=False):
    ### check!
    points = points.float()
    shape = shape.float()
    mask = (points >= 0) * (points <= shape-1)
    mask = (torch.prod(mask, dim=-1) == 1)
    if return_mask:
        return points[mask], mask
    return points[mask]

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

def draw_match_pair_hanming(input_img, input_pts, color=(255, 0, 0), radius=3, s=3):
    anchor = int(input_img['img'].shape[1])
    # img = input_img['img'] * 255
    img = np.hstack((input_img['img'], input_img['img_H'])) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(input_pts['pts']):
        if c.size == 1:
            break
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=-1)    # 图像坐标系，先列后行
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    # if input_pts['lab'].size == 0:
    #     return img
        
    # for c in np.stack(input_pts['lab']):
    #     if c.size == 1:
    #         break
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)

    for c in np.stack(input_pts['pts_H']):
        if c.size == 1:
            break
        c[0] += anchor
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=-1)
    # for c in np.stack(input_pts['lab_H']):
    #     if c.size == 1:
    #         break
    #     c[0] += anchor
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)
    # for c in np.stack(input_pts['pts_TH']):
    #     if c.size == 1:
    #         break
    #     c[0] += anchor
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 0, 255), thickness=-1)

    for cA, cB in zip(np.stack(input_pts['pts_repeatA']), np.stack(input_pts['pts_repeatB'])):
        # c = c[[1,0,3,2]]
        if c.size == 1:
            break
        cB[0] += anchor
        # cb = np.random.randint(0, 256)
        # cg = np.random.randint(0, 256)
        # cr = np.random.randint(0, 256)
        cv2.circle(img, tuple((s * cA[:2]).astype(int)), radius, (0, 0, 255), -1)
        cv2.circle(img, tuple((s * cB[:2]).astype(int)), radius, (0, 0, 255), -1)
        cv2.line(img, tuple((s * cA[:2]).astype(int)),tuple((s * cB[:2]).astype(int)), (0, 0, 255), 1, shift=0)

    # for cA, cB in zip(np.stack(input_pts['pts_nncandA']), np.stack(input_pts['pts_nncandB'])):
    #     # c = c[[1,0,3,2]]
    #     if c.size == 1:
    #         break
    #     cB[0] += anchor
    #     # cb = np.random.randint(0, 256)
    #     # cg = np.random.randint(0, 256)
    #     # cr = np.random.randint(0, 256)
    #     cv2.circle(img, tuple((s * cA[:2]).astype(int)), radius, (0, 255, 255), -1)
    #     cv2.circle(img, tuple((s * cB[:2]).astype(int)), radius, (0, 255, 255), -1)
    #     cv2.line(img, tuple((s * cA[:2]).astype(int)),tuple((s * cB[:2]).astype(int)), (0, 255, 255), 1, shift=0)

    try:
        assert input_pts['pts_nnA'].shape[0] > 0 and input_pts['pts_nnB'].shape[0] > 0
    except:
        print('Hanming distance is greater than threshold all!')
    else:

        for cA, cB in zip(np.stack(input_pts['pts_nnA']), np.stack(input_pts['pts_nnB'])):
            # c = c[[1,0,3,2]]
            if c.size == 1:
                break
            cB[0] += anchor
            # cb = np.random.randint(0, 256)
            # cg = np.random.randint(0, 256)
            # cr = np.random.randint(0, 256)
            cv2.circle(img, tuple((s * cA[:2]).astype(int)), radius, (0, 255, 0), -1)
            cv2.circle(img, tuple((s * cB[:2]).astype(int)), radius, (0, 255, 0), -1)
            cv2.line(img, tuple((s * cA[:2]).astype(int)),tuple((s * cB[:2]).astype(int)), (0, 255, 0), 1, shift=0)

    return img


def draw_keypoints_pair(input_img, input_pts, label_pts, color=(0, 255, 0), radius=3, s=3):

    anchor = int(input_img['img_1'].shape[1])
    img = np.hstack((input_img['img_1'], input_img['img_2'])) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(input_pts['pts']):
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)    # 图像坐标系，先列后行
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    if not (input_pts['pts_B'] == None or input_pts['pts_B'].size() == 0):
        for c in np.stack(input_pts['pts_B']):
            c[0] += anchor
            # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
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
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 0, 255), thickness=1)
    
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

def draw_match_pair_degree_match(input_img, input_pts, color=(255, 0, 0), radius=3, s=3, Htrans=None, is_base=False):
    anchor = int(input_img['img'].shape[1])
    # img = input_img['img'] * 255
    img = np.hstack((input_img['img'], input_img['img_H'])) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    # for c in np.stack(input_pts['pts']):
    #     if c.size == 1:
    #         break
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)    # 图像坐标系，先列后行
    #     cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    if not is_base:
        img = draw_orientation_degree(img, input_pts['pts'], input_pts['pts_degree'], input_pts['pts_degree_label'], color=(0, 255, 0), radius=radius, s=s)
            
        # for c in np.stack(input_pts['lab']):
        #     if c.size == 1:
        #         break
        #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)

        # for c in np.stack(input_pts['pts_H']):
        #     if c.size == 1:
        #         break
        #     c[0] += anchor
        #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=-1)
        img = draw_orientation_degree(img, input_pts['pts_H'], input_pts['pts_H_degree'], input_pts['pts_H_degree_label'], color=(0, 255, 0), radius=radius, s=s, offset=anchor)

    image_warp = inv_warp_image(torch.from_numpy(input_img['img']), Htrans)
    image_warp = (np.repeat(cv2.resize(np.array(image_warp*255), None, fx=s, fy=s)[..., np.newaxis], 1, -1)).astype(np.uint8)
    imageB = (np.repeat(cv2.resize(input_img['img_H']*255, None, fx=s, fy=s)[..., np.newaxis], 1, -1)).astype(np.uint8)
    b = np.zeros_like(imageB)
    g = image_warp
    r = imageB
    img_match = cv2.merge([b, g, r])
    # print(img.shape, img_match.shape)
    img = np.hstack((img, img_match))

    return img


def draw_orientation_degree(img, corners, degrees, degrees_label, color=(0, 255, 0), radius=3, s=3, offset=0):
    # img = image
    # img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    
    deltaX = np.cos(degrees) * 2
    deltaY = np.sin(degrees) * 2

    corners_new = corners.copy()
    corners_new[:, 0] += offset

    start_XY = corners_new.copy()
    end_XY = corners_new.copy()

    start_XY[:,0] -= deltaX
    start_XY[:,1] += deltaY

    end_XY[:,0] += deltaX
    end_XY[:,1] -= deltaY

    deltaX_label = np.cos(degrees_label) * 2
    deltaY_label = np.sin(degrees_label) * 2

    start_XY_label = corners_new.copy()
    end_XY_label = corners_new.copy()

    start_XY_label[:,0] -= deltaX_label
    start_XY_label[:,1] += deltaY_label

    end_XY_label[:,0] += deltaX_label
    end_XY_label[:,1] -= deltaY_label

    
    for c, d, a, e, f in zip(start_XY, end_XY, corners_new, start_XY_label, end_XY_label):
        if a.size == 1:
            break
        cv2.circle(img, tuple((s * a).astype(int)), radius, (255,0,0), thickness=-1)
        # cv2.line(img,tuple((s * c).astype(int)),tuple((s * d).astype(int)), (0,0,255),1,shift=0)
        cv2.arrowedLine(img,tuple((s * c).astype(int)),tuple((s * d).astype(int)), (0,0,255),2,0,0,0.2)
        cv2.arrowedLine(img,tuple((s * e).astype(int)),tuple((s * f).astype(int)), (0,255,0),2,0,0,0.2)
        # cv2.putText(img, str(degree), tuple((s * c).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (255, 255, 255), 1)
    return img


def draw_keypoints(img, corners, label_corners=None, color=(0, 255, 0), radius=3, s=3):

    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners):
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        # print(c.shape)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)

    if label_corners is not None:
        for c in np.stack(label_corners):
            # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
            cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)
            # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
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

def get_point_pair_repeat_double(kptA_tensor, kptB_tensor, correspond=2):
    dis = get_dis(kptA_tensor, kptB_tensor)
    if dis.shape[0] == 0 or dis.shape[1] == 0 or dis.shape == torch.Size([]):
        return torch.tensor([]), torch.tensor([])
    a2b_min_id = torch.argmin(dis, dim=1)
    len_p = len(a2b_min_id)
    ch1 = dis[list(range(len_p)), a2b_min_id] < 1   # correspond
    ch2 = dis[list(range(len_p)), a2b_min_id] < 2
    a2b_min_id1 = a2b_min_id[ch1]
    a2b_min_id2 = a2b_min_id[ch2]

    return a2b_min_id1, ch1, a2b_min_id2, ch2

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
    for n in range(len_p):
        if ch[n] == True:
            dis_mask[n, a2b_min_id[n]] = 0
    
    coor = torch.where(dis_mask == 1)
    
    return coor

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