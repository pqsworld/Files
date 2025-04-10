import torch
import math
from torch.autograd import Variable
from scipy.linalg import hadamard
import numpy as np 

def get_sift_orientation_batch(img, keypoints, patch_size=19, bin_size=10):
    '''
    img:tensor
    '''
    patch_size=19
    w_gauss = torch.tensor([0.0,0.0,0.0,1.0,1.0,2.0,3.0,5.0,6.0,6.0,6.0,5.0,3.0,2.0,1.0,1.0,0.0,0.0,0.0,
    0.0,0.0,1.0,1.0,3.0,5.0,8.0,10.0,12.0,13.0,12.0,10.0,8.0,5.0,3.0,1.0,1.0,0.0,0.0,
    0.0,1.0,2.0,3.0,6.0,11.0,16.0,22.0,26.0,28.0,26.0,22.0,16.0,11.0,6.0,3.0,2.0,1.0,0.0,
    1.0,1.0,3.0,7.0,13.0,22.0,34.0,45.0,55.0,58.0,55.0,45.0,34.0,22.0,13.0,7.0,3.0,1.0,1.0,
    1.0,3.0,6.0,13.0,24.0,42.0,65.0,88.0,106.0,112.0,106.0,88.0,65.0,42.0,24.0,13.0,6.0,3.0,1.0,
    2.0,5.0,11.0,22.0,42.0,73.0,112.0,151.0,182.0,193.0,182.0,151.0,112.0,73.0,42.0,22.0,11.0,5.0,2.0,
    3.0,8.0,16.0,34.0,65.0,112.0,171.0,232.0,279.0,296.0,279.0,232.0,171.0,112.0,65.0,34.0,16.0,8.0,3.0,
    5.0,10.0,22.0,45.0,88.0,151.0,232.0,314.0,378.0,401.0,378.0,314.0,232.0,151.0,88.0,45.0,22.0,10.0,5.0,
    6.0,12.0,26.0,55.0,106.0,182.0,279.0,378.0,456.0,483.0,456.0,378.0,279.0,182.0,106.0,55.0,26.0,12.0,6.0,
    6.0,13.0,28.0,58.0,112.0,193.0,296.0,401.0,483.0,512.0,483.0,401.0,296.0,193.0,112.0,58.0,28.0,13.0,6.0,
    6.0,12.0,26.0,55.0,106.0,182.0,279.0,378.0,456.0,483.0,456.0,378.0,279.0,182.0,106.0,55.0,26.0,12.0,6.0,
    5.0,10.0,22.0,45.0,88.0,151.0,232.0,314.0,378.0,401.0,378.0,314.0,232.0,151.0,88.0,45.0,22.0,10.0,5.0,
    3.0,8.0,16.0,34.0,65.0,112.0,171.0,232.0,279.0,296.0,279.0,232.0,171.0,112.0,65.0,34.0,16.0,8.0,3.0,
    2.0,5.0,11.0,22.0,42.0,73.0,112.0,151.0,182.0,193.0,182.0,151.0,112.0,73.0,42.0,22.0,11.0,5.0,2.0,
    1.0,3.0,6.0,13.0,24.0,42.0,65.0,88.0,106.0,112.0,106.0,88.0,65.0,42.0,24.0,13.0,6.0,3.0,1.0,
    1.0,1.0,3.0,7.0,13.0,22.0,34.0,45.0,55.0,58.0,55.0,45.0,34.0,22.0,13.0,7.0,3.0,1.0,1.0,
    0.0,1.0,2.0,3.0,6.0,11.0,16.0,22.0,26.0,28.0,26.0,22.0,16.0,11.0,6.0,3.0,2.0,1.0,0.0,
    0.0,0.0,1.0,1.0,3.0,5.0,8.0,10.0,12.0,13.0,12.0,10.0,8.0,5.0,3.0,1.0,1.0,0.0,0.0,
    0.0,0.0,0.0,1.0,1.0,2.0,3.0,5.0,6.0,6.0,6.0,5.0,3.0,2.0,1.0,1.0,0.0,0.0,0.0],device=img.device)

    ori_max = 180
    bins = ori_max // bin_size
    batch, c, h, w = img.shape
    offset = patch_size // 2
    device = img.device

    # img[:,:,:3] = 0
    # img[:,:,:,:8] = 0
    # img[:,:,-3:] = 0
    # img[:,:,:,-8:] = 0

    Gx=torch.zeros((batch, c, h+offset*2, w+offset*2), dtype=img.dtype, device=img.device)
    Gy=torch.zeros((batch, c, h+offset*2, w+offset*2), dtype=img.dtype, device=img.device)
    # Gm=torch.zeros((batch, c, h+patch_size, w+patch_size), dtype=img.dtype, device=img.device)
    # Gm[:,:,patch_size:h+patch_size,patch_size:w+patch_size] = 1

    Gx0=torch.zeros_like(img)
    Gx2=torch.zeros_like(img)
    Gy0=torch.zeros_like(img)
    Gy2=torch.zeros_like(img)

    Gx0[:,:,:,1:-1] = img[:,:,:,:-2]*255
    Gx2[:,:,:,1:-1] = img[:,:,:,2:]*255
    Gx[:,:,offset:-offset,offset:-offset] = (Gx0 - Gx2)

    Gy0[:,:,1:-1,:] = img[:,:,:-2,:]*255
    Gy2[:,:,1:-1,:] = img[:,:,2:,:]*255
    Gy[:,:,offset:-offset,offset:-offset] = (Gy2 - Gy0)

    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, patch_size), torch.linspace(-1, 1, patch_size)), dim=2)  # 产生两个网格
    coor_cells = coor_cells.transpose(0, 1)
    #coor_cells = coor_cells.to(self.device)
    coor_cells = coor_cells.contiguous()
    
    keypoints_num = keypoints.size(1)
    keypoints_correct = torch.round(keypoints.clone())
    keypoints_correct += offset
    
    src_pixel_coords = coor_cells.unsqueeze(0).repeat(batch, keypoints_num,1,1,1)
    src_pixel_coords = src_pixel_coords.float() * (patch_size // 2) + keypoints_correct.unsqueeze(2).unsqueeze(2).repeat(1,1,patch_size,patch_size,1)
    
    src_pixel_coords = src_pixel_coords.view([batch, keypoints_num, -1, 2])
    batch_image_coords_correct = torch.linspace(0, (batch-1)*(h+patch_size-1)*(w+patch_size-1), batch).long().to(device)
    src_pixel_coords_index = (src_pixel_coords[:,:,:,0] + src_pixel_coords[:,:,:,1]*(w+patch_size-1)).long()
    src_pixel_coords_index  = src_pixel_coords_index + batch_image_coords_correct[:,None,None]

    eps = 1e-12
    
    #生成幅值图和角度图
    Grad_Amp = ((torch.sqrt(Gx**2 + Gy**2)) * 256).long()

    #边界反射
    Grad_Amp[:,:,9] = Grad_Amp[:,:,10]
    Grad_Amp[:,:,-10] = Grad_Amp[:,:,-11]
    Grad_Amp[:,:,:,9] = Grad_Amp[:,:,:,10]
    Grad_Amp[:,:,:,-10] = Grad_Amp[:,:,:,-11]

    degree_value = Gy / (Gx + eps)
    Grad_ori = (torch.atan(degree_value)*4096 + 0.5).long() / 4096
    Grad_ori = Grad_ori*180 / math.pi #180/(3.1415926)
    a_mask = (Gx >= 0)
    b_mask = (Gy >= 0)
    apbp_mask = a_mask * b_mask
    apbn_mask = a_mask * (~b_mask)
    anbp_mask = (~a_mask) * b_mask
    anbn_mask = (~a_mask) * (~b_mask)
    Grad_ori[apbp_mask] = Grad_ori[apbp_mask]
    Grad_ori[apbn_mask] = Grad_ori[apbn_mask] + 360
    Grad_ori[anbp_mask] = Grad_ori[anbp_mask] + 180
    Grad_ori[anbn_mask] = Grad_ori[anbn_mask] + 180


    #边界反射
    Grad_ori[:,:,9] = Grad_ori[:,:,10]
    Grad_ori[:,:,-10] = Grad_ori[:,:,-11]
    Grad_ori[:,:,:,9] = Grad_ori[:,:,:,10]
    Grad_ori[:,:,:,-10] = Grad_ori[:,:,:,-11]
    
    Grad_ori = Grad_ori % ori_max

    angle = Grad_ori.take(src_pixel_coords_index)

    #高斯加权
    w_gauss /= 512
    Amp = Grad_Amp.take(src_pixel_coords_index)
    Amp = Amp*w_gauss[None,None,:]
    angle_d = ((angle // bin_size)).long() % bins
    angle_d_onehot = F.one_hot(angle_d,num_classes=bins)
    hist = torch.sum(Amp.unsqueeze(-1)*angle_d_onehot,dim=-2) #[0,pi)

    #平滑
    h_t=torch.zeros((batch, keypoints_num, hist.size(-1)+4), dtype=hist.dtype, device=hist.device)
    h_t[:,:,2:-2] = hist
    h_t[:,:,-2:] = hist[:,:,:2]
    h_t[:,:,:2] = hist[:,:,-2:]

    h_p2=h_t[:,:,4:]
    h_n2=h_t[:,:,:-4]
    h_p1=h_t[:,:,3:-1]
    h_n1=h_t[:,:,1:-3]

    Hist = (h_p2 + h_n2 + 4*(h_p1 + h_n1) + 6*hist) / 16
    Hist = Hist.long()
    
    #获取主方向i
    H_p_i = torch.max(Hist,dim=-1).indices
    H_t=torch.zeros((batch, keypoints_num, Hist.size(-1)+2), dtype=Hist.dtype, device=Hist.device)
    H_t[:,:,1:-1] = Hist
    H_t[:,:,-1:] = Hist[:,:,:1]
    H_t[:,:,:1] = Hist[:,:,-1:]

    H_p1=H_t[:,:,2:]
    H_n1=H_t[:,:,:-2]

    H_i_offset = (H_n1 - H_p1) / (2*(H_n1 + H_p1 - 2*Hist) + eps)
    H_p_i_onehot = F.one_hot(H_p_i,num_classes=bins)
    H_p_offset = torch.sum(H_i_offset*H_p_i_onehot,dim=-1)
    H_p = (H_p_i + H_p_offset + 0.5) * bin_size
    # H_p = (H_p_i + 0.5) * bin_size
    H_p = H_p % 180 - 90


    return H_p


def get_rotation_matrix(theta):
    batchsize = len(theta)
    theta_r = theta*3.14159265/180
    rotate_matrix = torch.zeros((batchsize, 3,3))
    rotate_matrix[:,0,0] = torch.cos(theta_r)
    rotate_matrix[:,0,1] = torch.sin(theta_r)
    rotate_matrix[:,0,2] = 0
    rotate_matrix[:,1,0] = -torch.sin(theta_r)
    rotate_matrix[:,1,1] = torch.cos(theta_r)
    rotate_matrix[:,1,2] = 0
    rotate_matrix[:,2,0] = 0
    rotate_matrix[:,2,1] = 0
    rotate_matrix[:,2,2] = 1

    return rotate_matrix

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
    points = torch.cat((points.double(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)
    homographies = homographies.view(batch_size*3,3)
    # warped_points = homographies*points
    # points = points.double()
    homographies = homographies.to(points.device)
    warped_points = homographies@points.transpose(0,1)
    # warped_points = np.tensordot(homographies, points.transpose(), axes=([2], [0]))
    # normalize the points
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0,:,:] if no_batches else warped_points

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
    points = points_batch.view(-1,2)
    theta = theta_batch.view(-1)

    mat_homo_inv = get_rotation_matrix(theta)
    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)
    device = img.device
    _, channel, H, W = img.shape
    Batch = len(points)
  
    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, patch_size), torch.linspace(-1, 1, patch_size)), dim=2)  # 产生两个网格
    if sample_size == 1:
        coor_cells = torch.zeros_like(coor_cells)
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()
    
    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv.double(), device)
    src_pixel_coords = src_pixel_coords.view([Batch, patch_size, patch_size, 2])
    src_pixel_coords = src_pixel_coords.float() * (sample_size / 2) + points.unsqueeze(1).unsqueeze(1).repeat(1,patch_size,patch_size,1)


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
    warped_img = warped_img.view([Batch, patch_size,patch_size])
    return warped_img

def forward_patches_correct(img, keypoints, descriptor_net, patch_size=16, sample_size=22, correct=False, expand=True, theta=0, _project=93, mode=False):
    # 根据关键点获得patch，并输入网络
    # 返回元组 B kpts_num desc_dim
    # 不满patchsize的patch用0补全
    assert _project in [91,93]
    patch_size = 16

    results = None
    (Batchsize, img_dim, img_H, img_W) = img.size()
    if expand:
        if _project == 91:
            _padH = 4
            _padW = 8
        elif _project == 93:
            _padH = 3
            _padW = 8
    
        img_H = img_H - (_padH * 2)
        img_W = img_W - (_padW * 2)

    assert Batchsize == 1
    assert img_dim == 1

    patch_padding = 60


    img_temp = torch.zeros((1, img_H + patch_padding, img_W + patch_padding),device=img.device)
    add_offset = patch_padding//2
    add_offset_x = patch_padding//2

    if expand:
        img_temp[:,(add_offset - _padH):(img_H+add_offset + _padH),(add_offset_x - _padW):(img_W+add_offset_x + _padW)] = img
    else:
        img_temp[:,add_offset:(img_H+add_offset),add_offset_x:(img_W+add_offset_x)] = img
    
    point_correct = keypoints.clone()
    point_correct[:,0] += add_offset_x
    point_correct[:,1] += add_offset

    if correct:
        if expand:
            point_orientation = keypoints.clone()
            point_orientation[:,0] += _padW
            point_orientation[:,1] += _padH

            orientation_theta = get_sift_orientation_batch(img.clone(), point_orientation.unsqueeze(0),bin_size=10).squeeze(0)
        else:
            orientation_theta = get_orientation_test(img[0,0], keypoints, 16) + theta
    else:
        orientation_theta = torch.zeros(len(point_correct)).to(img.device) + theta

    #print(img_temp.unsqueeze(0), point_correct.unsqueeze(0), (orientation_theta.unsqueeze(0)))
    patch = inv_warp_patch_batch(img_temp.unsqueeze(0), point_correct.unsqueeze(0), (orientation_theta.unsqueeze(0)), patch_size=patch_size, sample_size=sample_size)
    results = patch.unsqueeze(1)

    results_batch = Variable(results)

    # # # #扩边屏蔽
    # mask_extend = img_temp.unsqueeze(0).clone()
    # mask_extend[:] = 0
    # mask_extend[:,:,27:-27,22:-22] = 1

    # patch_mask_0 = inv_warp_patch_batch(mask_extend, point_correct.unsqueeze(0), orientation_theta.unsqueeze(0), patch_size=patch_size, sample_size=sample_size)
    results_batch = results_batch/255.  #totensor
    with torch.no_grad():
        outs,wb_mask = descriptor_net(results_batch)

    return outs, orientation_theta,wb_mask

def get_candmask(Hamming_distance_matrix_half, Hamming_distance_matrix, thre_half=60, thre=110, cand_num=30):
    Hamming_distance_matrix_nearest = Hamming_distance_matrix.clone()

    M_row, N_col = Hamming_distance_matrix.size()
    #进行阈值卡控
    dis1_mask = Hamming_distance_matrix_half < thre_half
    dis2_mask = Hamming_distance_matrix < thre
    dis12_mask = dis1_mask * dis2_mask
    Hamming_distance_matrix_nearest[~dis12_mask] = 256 #大于阈值的匹配对汉明距离置为256

    #最近邻次近邻比值进行卡控
    asend_Hamming_value, asend_Hamming_index = torch.sort(Hamming_distance_matrix_nearest)
    Nearest_nextNearest_mask = (asend_Hamming_value[:,0] / asend_Hamming_value[:,1]) < 0.99
    Nearest_nextNearest_mask = Nearest_nextNearest_mask.unsqueeze(1).repeat(1,dis12_mask.size(1))
    # Hamming_distance_matrix_nearest[~Nearest_nextNearest_mask] = 256

    #获得行方向最近邻
    nearest_Hamming_index_row = asend_Hamming_index[:,0]
    nearest_Hamming_mask_row = torch.zeros((M_row,N_col),device=Hamming_distance_matrix.device).bool() #行方向
    nearest_Hamming_mask_row[list(range(M_row)),nearest_Hamming_index_row] = 1
    nearest_Hamming_mask_row = dis12_mask*Nearest_nextNearest_mask*nearest_Hamming_mask_row
    Hamming_distance_matrix_nearest[~nearest_Hamming_mask_row] = 256

    #列方向去重
    nearest_Hamming_index_column = torch.min(Hamming_distance_matrix_nearest,dim=0).indices
    nearest_Hamming_mask_column = torch.zeros((M_row,N_col),device=Hamming_distance_matrix.device).bool()  #行方向
    nearest_Hamming_mask_column[nearest_Hamming_index_column, list(range(N_col))] = 1
    nearest_Hamming_mask_column = nearest_Hamming_mask_row*nearest_Hamming_mask_column
    Hamming_distance_matrix_nearest[~nearest_Hamming_mask_column] = 256

    #获取最近邻匹配matrix
    nearest_Hamming_value = torch.min(Hamming_distance_matrix_nearest,dim=1).values
    nearest_Hamming_index = torch.min(Hamming_distance_matrix_nearest,dim=1).indices

    nearest_30_index = torch.topk(nearest_Hamming_value, cand_num, largest=False).indices
    nearest_Hamming_mask = torch.zeros((M_row,N_col),device=Hamming_distance_matrix.device).bool() 

    nearest_Hamming_mask[nearest_30_index,nearest_Hamming_index[nearest_30_index]] = 1
    nearest_Hamming_mask = nearest_Hamming_mask*nearest_Hamming_mask_column

    nearest_Hamming_mask = nearest_Hamming_mask.bool()

    return nearest_Hamming_mask
   
def get_candmask_wb(Hamming_distance_matrix_half, Hamming_distance_matrix, thre_half=60, thre=110, cand_num=30, black_num = None):
    Hamming_distance_matrix_nearest = Hamming_distance_matrix.clone()

    M_row, N_col = Hamming_distance_matrix.size()
    
    #黑白点对应
    b1,b2 = black_num
    Hamming_distance_matrix_half[b1:,0:b2] = 256
    Hamming_distance_matrix_half[0:b1,b2:] = 256
    
    #进行阈值卡控
    dis1_mask = Hamming_distance_matrix_half < thre_half
    dis2_mask = Hamming_distance_matrix < thre
    dis12_mask = dis1_mask * dis2_mask
    Hamming_distance_matrix_nearest[~dis12_mask] = 256 #大于阈值的匹配对汉明距离置为256
    Hamming_distance_matrix[~dis12_mask] = 256
    
    #最近邻次近邻比值进行卡控
    asend_Hamming_value, asend_Hamming_index = torch.sort(Hamming_distance_matrix_nearest)
    Nearest_nextNearest_mask = (asend_Hamming_value[:,0] / asend_Hamming_value[:,1]) < 0.99
    Nearest_nextNearest_mask = Nearest_nextNearest_mask.unsqueeze(1).repeat(1,dis12_mask.size(1))
    # Hamming_distance_matrix_nearest[~Nearest_nextNearest_mask] = 256

    #获得行方向最近邻
    nearest_Hamming_index_row = asend_Hamming_index[:,0]
    nearest_Hamming_mask_row = torch.zeros((M_row,N_col),device=Hamming_distance_matrix.device).bool() #行方向
    nearest_Hamming_mask_row[list(range(M_row)),nearest_Hamming_index_row] = 1
    nearest_Hamming_mask_row = dis12_mask*Nearest_nextNearest_mask*nearest_Hamming_mask_row  #
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
    
    return nearest_Hamming_mask, nearest_30

def get_nearest_batch(Hamming_distance_matrix_half, Hamming_distance_matrix, thre_half=60, thre=110):
    Hamming_distance_matrix_nearest = Hamming_distance_matrix.clone()

    b1, b2, M_row, N_col = Hamming_distance_matrix.size()
    
    #进行阈值卡控
    dis1_mask = Hamming_distance_matrix_half < thre_half
    dis2_mask = Hamming_distance_matrix < thre
    dis12_mask = dis1_mask * dis2_mask
    Hamming_distance_matrix_nearest[~dis12_mask] = 256 #大于阈值的匹配对汉明距离置为256
    Hamming_distance_matrix[~dis12_mask] = 256 #对应C,不符合条件没有汉明结果

    #最近邻次近邻比值进行卡控
    asend_Hamming_value, asend_Hamming_index = torch.sort(Hamming_distance_matrix_nearest)
    Nearest_nextNearest_mask = (asend_Hamming_value[:,:,:,0] / asend_Hamming_value[:,:,:,1]) < 0.99
    Nearest_nextNearest_mask = Nearest_nextNearest_mask.unsqueeze(3).repeat(1,1,1,N_col)
    # Hamming_distance_matrix_nearest[~Nearest_nextNearest_mask] = 256

    #获得行方向最近邻
    nearest_Hamming_index_row = asend_Hamming_index[:,:,:,0]
    nearest_Hamming_mask_row = torch.linspace(0,N_col-1,N_col)[None,None,None].repeat(b1,b2,M_row,1).to(Hamming_distance_matrix.device)
    nearest_Hamming_mask_row = (nearest_Hamming_mask_row == nearest_Hamming_index_row[:,:,:,None].repeat(1,1,1,N_col))
    nearest_Hamming_mask_row = dis12_mask*Nearest_nextNearest_mask*nearest_Hamming_mask_row  #
    Hamming_distance_matrix_nearest[~nearest_Hamming_mask_row] = 256

    #列方向去重
    nearest_Hamming_index_column = torch.min(Hamming_distance_matrix_nearest,dim=2).indices
    nearest_Hamming_mask_column = torch.linspace(0,M_row-1,M_row)[None,None,:,None].repeat(b1,b2,1,N_col).to(Hamming_distance_matrix.device)
    nearest_Hamming_mask_column = (nearest_Hamming_mask_column == nearest_Hamming_index_column[:,:,None,:].repeat(1,1,M_row,1))
    nearest_Hamming_mask_column = nearest_Hamming_mask_row*nearest_Hamming_mask_column
    Hamming_distance_matrix_nearest[~nearest_Hamming_mask_column] = 256

    #获取最近邻匹配matrix
    nearest_Hamming_value = torch.min(Hamming_distance_matrix_nearest,dim=3).values
    nearest_Hamming_index = torch.min(Hamming_distance_matrix_nearest,dim=3).indices  #列号

    return nearest_Hamming_value, nearest_Hamming_index, nearest_Hamming_mask_column
   
def _Hamming_Hadamard_one(descs):
    assert descs.size(1) == 128

    Hada = hadamard(128)
    descs = (torch.round(descs*5000).long()+5000).long()
    #门限话
    norm = (torch.sqrt(torch.sum(descs * descs, dim = 1)) * 0.2).long()
    norm = norm.unsqueeze(-1).expand_as(descs)
    descs = torch.where(descs < norm, torch.sqrt(descs).long(), torch.sqrt(norm).long())

    Hada_T = descs.float() @ torch.from_numpy(Hada).float().to(descs.device)
    
    descs_Hamming = (Hada_T.long() > 0).long()

    # descs_Hamming = (descs.long() > 0).long()
    return descs_Hamming

def Hamming_Hadamard(descs):
    (descs_num, descs_dim) = descs.size()

    descA_0, descA_1, descB_0, descB_1 = None, None, None, None
    assert descs_dim in (128,256)
    if descs_dim == 128:
        descs_0, descs_1 = descs, descs
        descs_0_Hamming = _Hamming_Hadamard_one(descs_0)
        descs_1_Hamming = descs_0_Hamming
    elif descs_dim == 256:
        descs = descs.view(-1,16,16)
        descs_0, descs_1 = descs[:,:,:8].reshape(-1,128), descs[:,:,8:].reshape(-1,128)

        descs_0_Hamming = _Hamming_Hadamard_one(descs_0)
        descs_1_Hamming = _Hamming_Hadamard_one(descs_1)
    
    descs_Hamming = torch.cat([descs_0_Hamming, descs_1_Hamming],dim=1)

    return descs_Hamming

def desc_trans(descsA):
    #将0部分和1部分分开的描述子转换为同号位、反号位分开的描述子
    same_mask = torch.tensor([1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1]).unsqueeze(1).repeat(1,8).view(-1).bool()
    # same_mask = (((torch.linspace(32,1,32)**4).unsqueeze(0) @ hadamard(32)) == ((torch.linspace(1,32,32)**4).unsqueeze(0) @ hadamard(32))).long().squeeze(0).unsqueeze(1).repeat(1,4).view(-1).bool()
    descs_dim = descsA.size(1)

    assert descs_dim == 256

    descsA_0, descsA_1 = descsA[:,:128], descsA[:,128:]

    descsA_same    = torch.cat([descsA_0[:,same_mask],descsA_1[:,same_mask]],dim=1)
    descsA_reverse = torch.cat([descsA_0[:,~same_mask],descsA_1[:,~same_mask]],dim=1)
    
    descsA_t = torch.cat([descsA_same, descsA_reverse], dim = 1)
    
    return descsA_t
    
def Hamming_distance_Hadamard(descsA, descsB, dis12=False):
    #计算两组描述子的汉明距离矩阵
    #训练时采用的余弦相似度，这里需要转换
    #Hamming_distance_1计算为1部分的汉明距离
    #Hamming_distance_0计算为0部分的汉明距离
    #同反号分别计算
    factor_dim = 1
    descs_dim = descsA.size(1)

    assert descs_dim == 256
    
    descsA_same, descsA_reverse= descsA[:,:128], descsA[:,128:]
    descsB_same, descsB_reverse= descsB[:,:128], descsB[:,128:]
    
    #计算同号位汉明距离
    Hamming_distance_same_1    = descsA_same.float() @ descsB_same.float().T
    Hamming_distance_same_0    = (descsA_same < 1).float() @ (descsB_same < 1).float().T
    Hamming_distance_same      = 128  - (Hamming_distance_same_1 + Hamming_distance_same_0)

    #计算反号位汉明距离
    Hamming_distance_reverse_1 = descsA_reverse.float() @ descsB_reverse.float().T
    Hamming_distance_reverse_0 = (descsA_reverse < 1).float() @ (descsB_reverse < 1).float().T
    Hamming_distance_reverse   = 128 - (Hamming_distance_reverse_1 + Hamming_distance_reverse_0)
    Hamming_distance_reverse   = torch.where(Hamming_distance_reverse < (128 - Hamming_distance_reverse), Hamming_distance_reverse, (128 - Hamming_distance_reverse))

    if dis12:
        return factor_dim *(Hamming_distance_same), factor_dim *(Hamming_distance_same + Hamming_distance_reverse)

    return factor_dim *(Hamming_distance_same + Hamming_distance_reverse)

def Hamming_distance_Hadamard_batch(descsA, descsB, dis12=False):
    #计算两组描述子的汉明距离矩阵
    #训练时采用的余弦相似度，这里需要转换
    #Hamming_distance_1计算为1部分的汉明距离
    #Hamming_distance_0计算为0部分的汉明距离
    #同反号分别计算
    factor_dim = 1
    descs_dim = descsA.size(2)
    assert len(descsA.size()) == 3
    assert descs_dim == 256
    
    descsA_same, descsA_reverse= descsA[:,:,:128], descsA[:,:,128:]
    descsB_same, descsB_reverse= descsB[:,:,:128], descsB[:,:,128:]
    
    #计算同号位汉明距离
    Hamming_distance_same_1    = torch.einsum('bnd,hmd->bhnm',descsA_same.float(),descsB_same.float())
    Hamming_distance_same_0    = torch.einsum('bnd,hmd->bhnm',(descsA_same < 1).float(), (descsB_same < 1).float())
    Hamming_distance_same      = 128  - (Hamming_distance_same_1 + Hamming_distance_same_0)

    #计算反号位汉明距离
    Hamming_distance_reverse_1 = torch.einsum('bnd,hmd->bhnm',descsA_reverse.float(),descsB_reverse.float())
    Hamming_distance_reverse_0 = torch.einsum('bnd,hmd->bhnm',(descsA_reverse < 1).float(), (descsB_reverse < 1).float())
    Hamming_distance_reverse   = 128 - (Hamming_distance_reverse_1 + Hamming_distance_reverse_0)
    Hamming_distance_reverse   = torch.where(Hamming_distance_reverse < (128 - Hamming_distance_reverse), Hamming_distance_reverse, (128 - Hamming_distance_reverse))

    if dis12:
        return factor_dim *(Hamming_distance_same), factor_dim *(Hamming_distance_same + Hamming_distance_reverse)

    return factor_dim *(Hamming_distance_same + Hamming_distance_reverse)

#计算描述子
#descAs, sift_oriA = FPDST.forward_patches_correct(imgA_expand.to(device), pts_A, patch_size, sample_size, correct=correct_flag,expand=expand_flag, theta=sift_oriA, mode=False)

#哈达玛汉明化
#descAs_Hamming = FPDST.Hamming_Hadamard(descAs)

#两个汉明化的描述子进行汉明距离计算
#Hamming_distance_matrix_half, Hamming_distance_matrix = FPDST.Hamming_distance_Hadamard(descAs_Hamming, descBs_Hamming,dis12=True)

#计算候选匹配对mask
#nearest_Hamming_mask = get_candmask(Hamming_distance_matrix_half, Hamming_distance_matrix, 60 , 110, 30)
