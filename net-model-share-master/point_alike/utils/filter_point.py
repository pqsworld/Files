import numpy as np
import torch
import torch.nn.functional as F
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

def get_netori(intensity, pnmap, pts_ori):
    H_o, W_o = intensity.shape[0], intensity.shape[1]      # 122x36 
    points_o = pts_ori.transpose(1, 0)[:, [0, 1]].clone().detach().to(device=intensity.device)        # [w, h]
    if intensity is not None and pnmap is not None:
        points_o_normalized = points_o / points_o.new_tensor([W_o - 1, H_o - 1]).to(points_o.device) * 2 - 1  # (w,h) -> (-1~1,-1~1)

        ori_kp_predict_all = F.grid_sample(intensity.view(-1, H_o, W_o).unsqueeze(0),
                                points_o_normalized.float().view(1, 1, -1, 2),
                                mode='bilinear', align_corners=True).squeeze(2).squeeze(0).transpose(0, 1)          # NTA x 2
    
        pn_kp_predict_all = F.grid_sample(pnmap.view(-1, H_o, W_o).unsqueeze(0),
                                points_o_normalized.float().view(1, 1, -1, 2),
                                mode='bilinear', align_corners=True).squeeze(2).squeeze(0).transpose(0, 1)          # NTA x 2
        # 正负号mask
        pn_predict = (pn_kp_predict_all > 0.5).float().squeeze()   

        # print(pn_predict.shape, ori_kp_predict_all[:, 0])
        # 强度loss
        pai_coef = 3.1415926
        net_ori = ori_kp_predict_all[:, 0] * (2 * pn_predict - 1) * pai_coef / 2   # 正负90度

        return net_ori


def filter_net_point(heatmap_in, img_partial_mask, conf_thresh, nms, bord, top_k):
    # 有角度图
    intensity = None
    pnmap = None
    if heatmap_in.shape[0] == 3:
        intensity = heatmap_in[1, 3:-3, 2:-2]
        pnmap = heatmap_in[2, 3:-3, 2:-2]

    heatmap = heatmap_in[0, 3:-3, 2:-2]

    heatmap = heatmap * img_partial_mask.to(heatmap.device)  

    pts = getPtsFromHeatmap(heatmap.squeeze().detach().cpu().numpy(), conf_thresh, nms, soft_nms=False, bord=bord)

    pts = torch.tensor(pts).type(torch.FloatTensor)
    
    if top_k:
        if pts.shape[1] > top_k:
            pts = pts[:, :top_k]

    if intensity is not None:
        net_ori = get_netori(intensity, pnmap, pts)
    
    return pts, net_ori
