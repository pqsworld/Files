"""util functions
# many old functions, need to clean up
# homography --> homography
# warping
# loss --> delete if useless
"""

#from kornia import warp_frame_depth
import numpy as np
import torch
from pathlib import Path
import datetime
import datetime
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
# from zmq import device
###### check
# from utils.nms_pytorch import box_nms as box_nms_retinaNet
from utils.d2s import DepthToSpace, SpaceToDepth

def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    def to_3d(img):
        if len(img.shape) == 2:
            img = img[np.newaxis, ...]
        return img
    img_r, img_g, img_gray = to_3d(img_r), to_3d(img_g), to_3d(img_gray)
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img

def thd_img(img, thd=0.015):
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()

def save_path_formatter(args, parser):
    print("todo: save path")
    return Path('.')
    pass
'''
def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    # data_folder_name = str(Path(args_dict['data']).normpath().name)
    data_folder_name = str(Path(args_dict['data']))
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['sequence_length'] = 'seq'
    keys_with_prefix['rotation_mode'] = 'rot_'
    keys_with_prefix['padding_mode'] = 'padding_'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'
    keys_with_prefix['photo_loss_weight'] = 'p'
    keys_with_prefix['mask_loss_weight'] = 'm'
    keys_with_prefix['smooth_loss_weight'] = 's'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp

    # return ''
'''

def tensor2array(tensor, max_value=255, colormap='rainbow', channel_first=True):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if int(cv2.__version__[0]) >= 3:
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)
        if channel_first:
            array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
        if not channel_first:
            array = array.transpose(1, 2, 0)
    return array

# from utils.utils import find_files_with_ext
def find_files_with_ext(directory, extension='.npz'):
    # print(os.listdir(directory))
    list_of_files = []
    import os
    if extension == ".npz":
        for l in os.listdir(directory):
            if l.endswith(extension):
                list_of_files.append(l)
                # print(l)
        return list_of_files

def save_checkpoint(save_path, net_state, epoch, filename='checkpoint.pth.tar'):
    file_prefix = ['superPointNet']
    # torch.save(net_state, save_path)
    filename = '{}_{}_{}'.format(file_prefix[0], str(epoch), filename)
    torch.save(net_state, save_path/filename)
    print("save checkpoint to ", filename)
    pass

def load_checkpoint(load_path, filename='checkpoint.pth.tar'):
    file_prefix = ['superPointNet']
    filename = '{}__{}'.format(file_prefix[0], filename)
    # torch.save(net_state, save_path)
    checkpoint = torch.load(load_path/filename)
    print("load checkpoint from ", filename)
    return checkpoint
    pass


def saveLoss(filename, iter, loss, task='train', **options):
    # save_file = save_output / "export.txt"
    with open(filename, "a") as myfile:
        myfile.write(task + " iter: " + str(iter) + ", ")
        myfile.write("loss: " + str(loss) + ", ")
        myfile.write(str(options))
        myfile.write("\n")

        # myfile.write("iter: " + str(iter) + '\n')
        # myfile.write("output pairs: " + str(count) + '\n')

def saveImg(img, filename):
    import cv2
    cv2.imwrite(filename, img)

def pltImshow(img):
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()

def loadConfig(filename):
    import yaml
    with open(filename, 'r') as f:
        config = yaml.load(f)
    return config

def append_csv(file='foo.csv', arr=[]):
    import csv   
    # fields=['first','second','third']
    # pre = lambda i: ['{0:.3f}'.format(x) for x in i]
    with open(file, 'a') as f:
        writer = csv.writer(f)
        if type(arr[0]) is list:
            for a in arr:
                writer.writerow(a)
                # writer.writerow(pre(a))
                # print(pre(a))
        else:
            writer.writerow(arr)


'''
def save_checkpoint(save_path, dispnet_state, exp_pose_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'exp_pose']
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix,filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_model_best.pth.tar'.format(prefix))
'''
import cv2
def sample_homography(inv_scale=3):
  corner_img = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)])
  # offset_r = 1 - 1/inv_scale
  # img_offset = np.array([(-1, -1), (-1, offset_r), (offset_r, -1), (offset_r, offset_r)])
  img_offset = corner_img
  corner_map = (np.random.rand(4,2)-0.5)*2/(inv_scale + 0.01) + img_offset
  matrix = cv2.getPerspectiveTransform(np.float32(corner_img), np.float32(corner_map))
  return matrix


def sample_homographies(batch_size=1, scale=10, device='cpu'):
    ## sample homography matrix
    # mat_H = [sample_homography(inv_scale=scale) for i in range(batch_size)]
    mat_H = [sample_homography(inv_scale=scale) for i in range(batch_size)]
    ##### debug
    # from utils.utils import sample_homo
    # mat_H = [sample_homo(image=np.zeros((1,1))) for i in range(batch_size)]

    # mat_H = [np.identity(3) for i in range(batch_size)]
    mat_H = np.stack(mat_H, axis=0)
    mat_H = torch.tensor(mat_H, dtype=torch.float32)
    mat_H = mat_H.to(device)

    mat_H_inv = torch.stack([torch.inverse(mat_H[i, :, :]) for i in range(batch_size)])
    mat_H_inv = torch.tensor(mat_H_inv, dtype=torch.float32)
    mat_H_inv = mat_H_inv.to(device)
    return mat_H, mat_H_inv

def warpLabels(pnts, homography, H, W):
    import torch
    """
    input:
        pnts: numpy
        homography: numpy
    output:
        warped_pnts: numpy
    """
    from utils.utils import warp_points
    from utils.utils import filter_points
    pnts = torch.tensor(pnts).long()
    homography = torch.tensor(homography, dtype=torch.float32)
    warped_pnts = warp_points(torch.stack((pnts[:, 0], pnts[:, 1]), dim=1),
                              homography)  # check the (x, y)
    warped_pnts = filter_points(warped_pnts, torch.tensor([W, H])).round().long()
    return warped_pnts.numpy()


def warp_points_np(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.
    """
    # expand points len to (x, y, 1)
    batch_size = homographies.shape[0]
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    # points = points.to(device)
    # homographies = homographies.(batch_size*3,3)
    # warped_points = homographies*points
    # warped_points = homographies@points.transpose(0,1)
    warped_points = np.tensordot(homographies, points.transpose(), axes=([2], [0]))
    # normalize the points
    warped_points = warped_points.reshape([batch_size, 3, -1])
    warped_points = warped_points.transpose([0, 2, 1])
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points

def homography_scaling(homography, H, W):
    trans = np.array([[2./W, 0., -1], [0., 2./H, -1], [0., 0., 1.]])
    homography = np.linalg.inv(trans) @ homography @ trans
    return homography

def homography_scaling_torch(homography, H, W):
    trans = torch.tensor([[2./W, 0., -1], [0., 2./H, -1], [0., 0., 1.]])
    homography = (trans.inverse() @ homography @ trans)
    return homography

def homography_centorcrop(homography, Hdev_top, Wdev_left):
    homography_dev = torch.tensor([[0, 0, Wdev_left], [0, 0, Hdev_top], [0, 0, 0]], dtype=torch.float32).to(homography.device)
    scale = torch.tensor([[1, 0, Wdev_left], [0, 1, Hdev_top], [0, 0, 1]], dtype=torch.float32).to(homography.device)
    homography = (homography - homography_dev) @ scale
    return homography

def homography_transform(homography, Hdev_top, Wdev_left):
    scale = torch.tensor([[1, 0, Wdev_left], [0, 1, Hdev_top], [0, 0, 1]], dtype=torch.float32)
    homography = scale @ homography @ scale.inverse()
    return homography


def filter_points(points, shape, return_mask=False):
    ### check!
    points = points.float()
    shape = shape.float()
    mask = (points >= 0) * (points <= shape-1)
    mask = (torch.prod(mask, dim=-1) == 1)
    if return_mask:
        return points[mask], mask
    return points[mask]
    # return points [torch.prod(mask, dim=-1) == 1]

def filter_points_batch(points, shape, return_mask=False, device='cpu'):
    ### check!
    points = points.float()
    shape = shape.float()
    mask = (points >= 0) * (points <= shape-1)
    mask = (torch.prod(mask, dim=-1) == 1)
    flag = torch.nonzero(~mask)
    # flag = ~mask
    # flag = torch.where(flag, )
    pts_coincide = points.clone()
    # pts_coincide.index_fill(2, flag, 0.)
    pts_coincide[flag[:, 0], flag[:, 1], :] = 0.
    return pts_coincide, mask

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


# from utils.utils import inv_warp_image_batch
def inv_warp_image_batch(img, mat_homo_inv, device='cpu', mode='bilinear'):
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
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)

    Batch, channel, H, W = img.shape


    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H)), dim=2)  # 产生两个网格
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv, device)
    src_pixel_coords = src_pixel_coords.view([Batch, H, W, 2])
    src_pixel_coords = src_pixel_coords.float()

    # theta = mat_homo_inv[0,:2,:]
    # grid = F.affine_grid(theta.unsqueeze(0), img.size())
    # grid = F.affine_grid(mat_homo_inv[:, :2, :], [1, 1, H, W], align_corners=True)
    # warped_img = F.grid_sample(img, grid, mode=mode, align_corners=True)    
    
    warped_img = F.grid_sample(img, src_pixel_coords, mode=mode, align_corners=True)    

    # def to3dim(img):
    #     if img.ndim == 2:
    #         img = img[:, :, np.newaxis]
    #     return img
    # img = img.squeeze().numpy()
    # mat_homo_inv = mat_homo_inv.numpy()
    # img = img * 255.

    # warped_img = cv2.warpPerspective(img.squeeze().numpy(), mat_homo_inv.squeeze().numpy(), (W, H))
    # warped_img = torch.tensor(warped_img, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

    # warped_img = cv2.warpAffine(img.squeeze().numpy(), mat_homo_inv[0, :2, :].squeeze().numpy(), (W, H))
    # warped_img = torch.tensor(warped_img, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    return warped_img

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

    _, _, H, W = img.shape

    warped_img = cv2.warpPerspective(img.squeeze().numpy(), mat_homo_inv.squeeze().numpy(), (W, H))
    warped_img = torch.tensor(warped_img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)

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

def get_rotation_matrix(theta):
    batchsize = len(theta)
    theta_r = theta*3.14159265/180
    # print(batchsize)
    rotate_maxtrix = torch.zeros((batchsize,3,3), device=theta_r.device)
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

def get_shear_matrix(shear):
    batchsize = len(shear)
    shear_s = shear*3.14159265/180
    # print(batchsize)
    shear_maxtrix = torch.zeros((batchsize,3,3), device=shear_s.device)
    shear_maxtrix[:,0,0] = 1
    shear_maxtrix[:,0,1] = 0
    shear_maxtrix[:,0,2] = 0
    shear_maxtrix[:,1,0] = torch.tan(shear_s)
    shear_maxtrix[:,1,1] = 1
    shear_maxtrix[:,1,2] = 0
    shear_maxtrix[:,2,0] = 0
    shear_maxtrix[:,2,1] = 0
    shear_maxtrix[:,2,2] = 1

    return shear_maxtrix

# from utils.utils import inv_warp_image_batch
def inv_warp_patch_batch(img, points, theta, patch_size=16, sample_size = 16, shear=None):
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
    mat_homo_inv = get_rotation_matrix(theta)
    if shear is not None:
        mat_shear = get_shear_matrix(shear)
        mat_homo_inv = mat_shear @ mat_homo_inv

    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)
    device = img.device
    B, _, H, W = img.shape
    Batch = len(points)
    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, patch_size), torch.linspace(-1, 1, patch_size)), dim=2)  # 产生两个网格
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous().double()
    src_pixel_coords = warp_points_batch(coor_cells.view([-1, 2]), mat_homo_inv.double(), device) 
    src_pixel_coords = src_pixel_coords.view([Batch, patch_size, patch_size, 2])
    src_pixel_coords = src_pixel_coords.float() * (sample_size / 2) + points.unsqueeze(1).unsqueeze(1).repeat(1,patch_size,patch_size,1)
    src_pixel_coords_ofs = torch.floor(src_pixel_coords)
    src_pixel_coords_ofs_Q11 = src_pixel_coords_ofs.view([Batch, -1, 2])        # (BxM)x(patch_size,patch_size)x2
    batch_offset = H * W * torch.linspace(0, B-1, steps=B, device=device).repeat(Batch//B, 1).transpose(0, 1).unsqueeze(0).repeat(1, 1, patch_size*patch_size).view(Batch, -1).long()
    # print(batch_offset)
    src_pixel_coords_ofs_Q11 = (src_pixel_coords_ofs_Q11[:,:,0] + src_pixel_coords_ofs_Q11[:,:,1]*W).long()
    src_pixel_coords_ofs_Q11 += batch_offset
    src_pixel_coords_ofs_Q21 = src_pixel_coords_ofs_Q11 + 1
    src_pixel_coords_ofs_Q12 = src_pixel_coords_ofs_Q11 + W
    src_pixel_coords_ofs_Q22 = src_pixel_coords_ofs_Q11 + W + 1
    warp_weight = (src_pixel_coords - src_pixel_coords_ofs).view([Batch, -1, 2])

    alpha = warp_weight[:,:,0]
    beta = warp_weight[:,:,1]
    # print(img.shape, src_pixel_coords_ofs_Q11.shape)
    src_Q11 = img.take(src_pixel_coords_ofs_Q11).view(-1, patch_size*patch_size)
    src_Q21 = img.take(src_pixel_coords_ofs_Q21).view(-1, patch_size*patch_size)
    src_Q12 = img.take(src_pixel_coords_ofs_Q12).view(-1, patch_size*patch_size)
    src_Q22 = img.take(src_pixel_coords_ofs_Q22).view(-1, patch_size*patch_size)
    # print(src_Q11.shape)
    warped_img = src_Q11*(1 - alpha)*(1 - beta) + src_Q21*alpha*(1 - beta) + \
        src_Q12*(1 - alpha)*beta + src_Q22*alpha*beta
    warped_img = warped_img.view([Batch, patch_size,patch_size])
    return warped_img

# from utils.utils import inv_warp_image_batch
def inv_warp_patch_batch_rec(img, points, theta, patch_size=(32,8), sample_factor = 1, mode='bilinear'):
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
    batch_size, _, H, W = img.shape
    # points = points_batch.view(-1,2)
    # theta = theta_batch.view(-1)

    mat_homo_inv = get_rotation_matrix(theta)
    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)
    device = img.device
    _, channel, H, W = img.shape
    Batch = len(points)
    points_num = Batch // batch_size

    patch_y = patch_size[0]*sample_factor / 2
    patch_x = patch_size[1]*sample_factor / 2

    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-patch_x, patch_x, patch_size[1]), torch.linspace(-patch_y, patch_y, patch_size[0])), dim=2)  # 产生两个网格
    coor_cells = coor_cells.transpose(1,0)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points_batch(coor_cells.view([-1, 2]).double(), mat_homo_inv.double(), device)
    src_pixel_coords = src_pixel_coords.view([Batch, patch_size[0], patch_size[1], 2])
    src_pixel_coords = src_pixel_coords.float() + points.unsqueeze(1).unsqueeze(1).repeat(1,patch_size[0],patch_size[1],1)


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
    
    # src_Q11 = img.take(src_pixel_coords_ofs_Q11).view(-1, patch_size*patch_size)
    # src_Q21 = img.take(src_pixel_coords_ofs_Q21).view(-1, patch_size*patch_size)
    # src_Q12 = img.take(src_pixel_coords_ofs_Q12).view(-1, patch_size*patch_size)
    # src_Q22 = img.take(src_pixel_coords_ofs_Q22).view(-1, patch_size*patch_size)
    src_Q11 = img.take(src_pixel_coords_ofs_Q11.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q21 = img.take(src_pixel_coords_ofs_Q21.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q12 = img.take(src_pixel_coords_ofs_Q12.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)
    src_Q22 = img.take(src_pixel_coords_ofs_Q22.view(batch_size, points_num, -1) + batch_image_coords_correct[:,None,None]).view(Batch, -1)

    warped_img = src_Q11*(1 - alpha)*(1 - beta) + src_Q21*alpha*(1 - beta) + \
        src_Q12*(1 - alpha)*beta + src_Q22*alpha*beta
    warped_img = warped_img.view([Batch, patch_size[0],patch_size[1]])
    return warped_img


def batch_inv_warp_image(img_batch, mat_homo_inv, device='cpu', mode='billinear'):
    img_batch_shape = img_batch.shape
    for idx in range(img_batch_shape[0]):
        if idx == 0:
            warped_img = inv_warp_image(img_batch[idx,:,:,:].squeeze().squeeze().detach().cpu(), mat_homo_inv[idx].squeeze().cpu(), device, mode).unsqueeze(0).unsqueeze(0)
        else:
            mid_warped_img = inv_warp_image(img_batch[idx,:,:,:].squeeze().squeeze().detach().cpu(), mat_homo_inv[idx].squeeze().cpu(), device, mode).unsqueeze(0).unsqueeze(0)
            warped_img = torch.cat([warped_img, mid_warped_img], dim=0)
    return warped_img

def labels2Dto3D_sort(labels, labels_sort, cell_size, add_dustbin=True):
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    # labels = space2depth(labels).squeeze(0)
    labels = space2depth(labels)        # 执行forward方法     labels.shape:[64,64,16,16]
    labels_sort = space2depth(labels_sort)
    # labels = labels.view(batch_size, H, 1, W, 1)
    # labels = labels.view(batch_size, Hc, cell_size, Wc, cell_size)
    # labels = labels.transpose(1, 2).transpose(3, 4).transpose(2, 3)
    # labels = labels.reshape(batch_size, 1, cell_size ** 2, Hc, Wc)
    # labels = labels.view(batch_size, cell_size ** 2, Hc, Wc)
    if add_dustbin:
        dustbin = labels.sum(dim=1)     # 在第1维上求和：[64,64,16,16] -> [64,16,16] 第一个通道变为1，默认被隐藏
        dustbin = 1 - dustbin           # labels的标签应该是特征点的位置置为1，非特征点为0，所以sum求和后，有特征点的就会大于1（这里不确定，可能是等于1）
        dustbin[dustbin < 1.] = 0       # 有特征点的位置，dustbin是0（也可能是负数，大概率是0）
        # print('dust: ', dustbin.shape)
        # labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
        # labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)    # 增维  labels.shape:[64,65,16,16]
        ## norm
        # dn = labels.sum(dim=1)  # dn.shape:[64,16,16]
        index_conf = cell_size * cell_size - torch.argsort(torch.argsort(labels_sort, dim=1, descending=False), dim=1, descending=False)
        index_conf[labels_sort == 0] = 0
        labels = torch.cat((index_conf, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)    # 增维  labels.shape:[64,65,16,16]
        dn = labels.sum(dim=1)
        labels = labels.div(torch.unsqueeze(dn, 1))
        # labels = labels.div(torch.unsqueeze(dn, 1))         # unsqueeze:扩展维度;  div的原因是因为一个patch可能出现多个标注点，div是用来归一化的
    return labels       # [64,65,16,16]
    pass


def labels2Dto3D(labels, cell_size, add_dustbin=True):
    '''
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    '''
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    # labels = space2depth(labels).squeeze(0)
    labels = space2depth(labels)        # 执行forward方法     labels.shape:[64,64,16,16]
    # labels = labels.view(batch_size, H, 1, W, 1)
    # labels = labels.view(batch_size, Hc, cell_size, Wc, cell_size)
    # labels = labels.transpose(1, 2).transpose(3, 4).transpose(2, 3)
    # labels = labels.reshape(batch_size, 1, cell_size ** 2, Hc, Wc)
    # labels = labels.view(batch_size, cell_size ** 2, Hc, Wc)
    if add_dustbin:
        dustbin = labels.sum(dim=1)     # 在第1维上求和：[64,64,16,16] -> [64,16,16] 第一个通道变为1，默认被隐藏
        dustbin = 1 - dustbin           # labels的标签应该是特征点的位置置为1，非特征点为0，所以sum求和后，有特征点的就会大于1（这里不确定，可能是等于1）
        dustbin[dustbin < 1.] = 0       # 有特征点的位置，dustbin是0（也可能是负数，大概率是0）
        # print('dust: ', dustbin.shape)
        # labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
        labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)    # 增维  labels.shape:[64,65,16,16]
        ## norm
        dn = labels.sum(dim=1)  # dn.shape:[64,16,16]
        labels = labels.div(torch.unsqueeze(dn, 1))         # unsqueeze:扩展维度;  div的原因是因为一个patch可能出现多个标注点，div是用来归一化的
    return labels       # [64,65,16,16]

def labels2Dto3D_flattened(labels, cell_size):
    '''
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    '''
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    # labels = space2depth(labels).squeeze(0)
    labels = space2depth(labels)
    # print("labels in 2Dto3D: ", labels.shape)
    # labels = labels.view(batch_size, H, 1, W, 1)
    # labels = labels.view(batch_size, Hc, cell_size, Wc, cell_size)
    # labels = labels.transpose(1, 2).transpose(3, 4).transpose(2, 3)
    # labels = labels.reshape(batch_size, 1, cell_size ** 2, Hc, Wc)
    # labels = labels.view(batch_size, cell_size ** 2, Hc, Wc)

    dustbin = torch.ones((batch_size, 1, Hc, Wc)).cuda()
    # labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
    labels = torch.cat((labels*2, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
    labels = torch.argmax(labels, dim=1)
    return labels



def old_flatten64to1(semi, tensor=False):
    '''
    Flatten 3D np array to 2D

    :param semi:
        np [64 x Hc x Wc]
        or
        tensor (batch_size, 65, Hc, Wc)
    :return:
        flattened map
        np [1 x Hc*8 x Wc*8]
        or
        tensor (batch_size, 1, Hc*8, Wc*8)
    '''
    if tensor:
        is_batch = len(semi.size()) == 4
        if not is_batch:
            semi = semi.unsqueeze_(0)
        Hc, Wc = semi.size()[2], semi.size()[3]
        cell = 8
        semi.transpose_(1, 2)
        semi.transpose_(2, 3)
        semi = semi.view(-1, Hc, Wc, cell, cell)
        semi.transpose_(2, 3)
        semi = semi.contiguous()
        semi = semi.view(-1, 1, Hc * cell, Wc * cell)
        heatmap = semi
        if not is_batch:
            heatmap = heatmap.squeeze_(0)
    else:
        Hc, Wc = semi.shape[1], semi.shape[2]
        cell = 8
        semi = semi.transpose(1, 2, 0)
        heatmap = np.reshape(semi, [Hc, Wc, cell, cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        # heatmap = np.transpose(heatmap, [2, 0, 3, 1])
        heatmap = np.reshape(heatmap, [Hc * cell, Wc * cell])
        heatmap = heatmap[np.newaxis, :, :]
    return heatmap


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

    # 显示用
    # temp = heatmap.narrow(0, 0, 1).squeeze(0).squeeze(0)
    # # temp.toNumpy(temp)
    # temp = temp.detach().cpu().numpy()
    # pltImshow(temp)

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
        nodust = F.relu(nodust)

        # for i in range(nodust.size(2)):
        #     for j in range(nodust.size(3)):
        #         if dense[:, -1, i, j] > (1 / 65):
        #             nodust[:, :, i, j] = 0

    else:
        # dense = nn.functional.softmax(semi, dim=0) # [65, Hc, Wc]
        nodust = semi[:-1, :, :].unsqueeze(0)
        nodust = F.relu(nodust)
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


def sample_homo(image):
    import tensorflow as tf
    from utils.homographies import sample_homography
    H = sample_homography(tf.shape(image)[:2])
    with tf.Session():
        H_ = H.eval()
    H_ = np.concatenate((H_, np.array([1])[:, np.newaxis]), axis=1)
    # warped_im = tf.contrib.image.transform(image, H, interpolation="BILINEAR")
    mat = np.reshape(H_, (3, 3))
    # for i in range(batch):
    #     np.stack()
    return mat

import cv2

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
    scores_pool = F.max_pool2d(scores.detach(), kernel_size=w_size, stride=w_size)
    scores_max_unpool = F.conv_transpose2d(scores_pool, kernels['upsample_filter_'+str(w_size)], stride=w_size)
    exp_map = torch.exp(torch.divide(scores, scores_max_unpool + eps)) - 1*(1.-eps)
    sum_exp_map = F.conv2d(exp_map, kernels['ones_kernel_' + str(w_size)], stride=w_size)
    indexes_map = F.conv2d(exp_map, kernels['indexes_kernel_' + str(w_size)], stride=w_size)
    indexes_map = torch.divide(indexes_map, sum_exp_map + eps)

    max_scores_pool = torch.max(torch.max(scores_pool, dim=3, keepdim=True).values, dim=2, keepdim=True).values
    norm_scores_pool= torch.divide(scores_pool, max_scores_pool + eps)
    return indexes_map, [scores_pool, norm_scores_pool]

def getPtsFromHeatmapByNMS(heatmap, conf_thresh, nms_dist, num_kpts_i=150):
    nms = NonMaxSuppression(thr=conf_thresh, nms_size= 2*nms_dist+1)
    kps = nms(heatmap)
    c = heatmap[0, 0, kps[0], kps[1]]
    sc, indices = torch.sort(c, descending=True)
    indices = indices[torch.where(sc > 0.)]
    kps = kps[:, indices[:num_kpts_i]]
    kps_np = torch.cat([kps[1].view(-1, 1).float(), kps[0].view(-1, 1).float(), c[indices[:num_kpts_i]].view(-1, 1).float()],
        dim=1).detach()
    return kps_np
    

def getPtsFromHeatmapByCoordinates(heatmap, conf_thresh, w_size, bord=0):
    Hc, Wc = heatmap.shape[2] // w_size, heatmap.shape[3] // w_size
    kernels = create_kernels([w_size], heatmap.device)
    indexes_map, _ = ip_layer(heatmap, w_size, kernels)
    coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2) * w_size
    coor_cells = coor_cells.type(torch.FloatTensor).to(heatmap.device).permute(2, 0, 1)
    indexes = indexes_map + coor_cells - 1
    indexes = indexes[indexes_map >= 1]
    indexes = indexes.view(2, -1)[[1, 0], :]
    return indexes  

def getPtsFromHeatmap(heatmap, conf_thresh, nms_dist):
    '''
    :param self:
    :param heatmap:
        np (H, W)
    :return:
    '''

    border_remove = 0   # 暂时不用

    H, W = heatmap.shape[0], heatmap.shape[1]
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.阈值卡控，并先存到pts中
    sparsemap = (heatmap >= conf_thresh)
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]     # 存在概率很大但很近的点
    '''非极大值抑制'''
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.非极大值抑制
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts

def getPtsFromLabels2D(lables_2D):
        '''
        :param self:
        :param heatmap:
            np (H, W)
        :return:
        '''
        lables_2D = lables_2D.squeeze()
        
        ys, xs = np.where(lables_2D >= 1)  # Confidence threshold.

        if len(ys) == 0:
            return np.zeros((3, 0))
        pts = np.zeros((3, len(ys)))  # Populate point data sized 3xN.
        pts[0, :] = xs # abuse of ys, xs
        pts[1, :] = ys
        pts[2, :] = lables_2D[ys, xs]  # check the (x, y) here
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds]
        
        return pts

def getPtsFromLabels2D_torch(lables_2D):
        '''
        :param self:
        :param heatmap:
            torch (H, W)
        :return:
        '''
        lables_2D = lables_2D.squeeze()
        
        ys, xs = torch.where(lables_2D >= 1)  # Confidence serial number.

        if len(ys) == 0:
            return torch.zeros((3, 0))
        pts = torch.zeros((3, len(ys)))  # Populate point data sized 3xN.
        pts[0, :] = xs # abuse of ys, xs
        pts[1, :] = ys
        pts[2, :] = lables_2D[ys, xs]  # check the (x, y) here
        inds = torch.argsort(pts[2, :])
        pts = pts[:, inds]

        return pts

def getLabel2DFromPts(point, H, W, device='cpu'):
        heatmap = torch.zeros(H, W)
        shape = torch.tensor([W, H]).to(device)

        mask = (point >= 0) * (point <= shape-1)    # 判断边界
        mask = (torch.prod(mask, dim=-1) == 1)
        pts = point[mask]

        x = pts[:, 0].round().long()
        y = pts[:, 1].round().long()
        heatmap[y, x] = 1
        return heatmap

def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    # requires https://github.com/open-mmlab/mmdetection. 
    # Warning : BUILD FROM SOURCE using command MMCV_WITH_OPS=1 pip install -e
    # from mmcv.ops import nms as nms_mmdet 
    from torchvision.ops import nms

    """Performs non maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field). Optionally only keeps the top k detections.
    Arguments:
    prob: the probability heatmap, with shape `[H, W]`.
    size: a scalar, the size of the bouding boxes.
    iou: a scalar, the IoU overlap threshold.
    min_prob: a threshold under which all probabilities are discarded before NMS.
    keep_top_k: an integer, the number of top scores to keep.
    """
    pts = torch.nonzero(prob > min_prob).float() # [N, 2]
    prob_nms = torch.zeros_like(prob)
    if pts.nelement() == 0:
        return prob_nms
    size = torch.tensor(size/2.).cuda()
    boxes = torch.cat([pts-size, pts+size], dim=1) # [N, 4]
    scores = prob[pts[:, 0].long(), pts[:, 1].long()]
    if keep_top_k != 0:
        indices = nms(boxes, scores, iou)
    else:
        raise NotImplementedError
        # indices, _ = nms(boxes, scores, iou, boxes.size()[0])
        # print("boxes: ", boxes.shape)
        # print("scores: ", scores.shape)
        # proposals = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)
        # dets, indices = nms_mmdet(proposals, iou)
        # indices = indices.long()

        # indices = box_nms_retinaNet(boxes, scores, iou)
    pts = torch.index_select(pts, 0, indices)
    scores = torch.index_select(scores, 0, indices)
    prob_nms[pts[:, 0].long(), pts[:, 1].long()] = scores
    return prob_nms

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

    inds1 = np.argsort(-in_corners[2, :])       # in_corners[2]排序
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
        grid[rcorners[1, i], rcorners[0, i]] = 1        #[y, x]
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


def normPts(pts, shape):
    """
    normalize pts to [-1, 1],归一化到-1~1之间
    :param pts:
        tensor (y, x)
    :param shape:
        tensor shape (y, x)
    :return:
    """
    pts = pts/shape*2 - 1
    return pts

def denormPts(pts, shape):
    """
    denormalize pts back to H, W
    :param pts:
        tensor (y, x)
    :param shape:
        numpy (y, x)
    :return:
    """
    pts = (pts+1)*shape/2
    return pts

# def subpixel_loss(image, labels, dense_desc, patch_size=8):
#     # concat image and dense_desc
#     # extract patches

#     # 
#     pass

def sample_desc_from_points(coarse_desc, pts, device='cpu'):
    # --- Process descriptor.
    cell = 8
    H, W = coarse_desc.shape[2]*cell, coarse_desc.shape[3]*cell
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
        desc = np.zeros((D, 0))
    else:
        # Interpolate into descriptor map using 2D point locations.
        samp_pts = torch.from_numpy(pts[:2, :].copy())
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        samp_pts = samp_pts.to(device)
        desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
        desc = desc.data.cpu().numpy().reshape(D, -1)
        desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return desc

def sample_desc_from_points_torch(coarse_desc, pts, device='cpu'):
    # --- Process descriptor.
    cell = 8
    H, W = coarse_desc.shape[2]*cell, coarse_desc.shape[3]*cell
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
        desc = torch.zeros((D, 0))
    else:
        # Interpolate into descriptor map using 2D point locations.
        samp_pts = pts[:2, :].clone()
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        samp_pts = samp_pts.to(device)
        desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
        desc = desc.reshape(D, -1)
        # desc /= torch.norm(desc, p=2, dim=0).unsqueeze(0)
        # desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return desc

def descriptor_loss(descriptors, descriptors_warped, homographies, mask_valid=None, 
                    cell_size=8, lamda_d=100, device='cpu', descriptor_dist=4, **config):

    '''
    Compute descriptor loss from descriptors_warped and given homographies

    :param descriptors:
        Output from descriptor head
        tensor [batch_size, descriptors, Hc, Wc]
    :param descriptors_warped:
        Output from descriptor head of warped image
        tensor [batch_size, descriptors, Hc, Wc]
    :param homographies:
        known homographies
    :param cell_size:
        8
    :param device:
        gpu or cpu
    :param config:
    :return:
        loss, and other tensors for visualization
    '''

    # put to gpu
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

        # coor_cells = coor_cells.view([-1, Hc, Wc, 1, 1, 2])
        coor_cells = coor_cells.view([-1, 1, 1, Hc, Wc, 2])  # be careful of the order  (y, x)
        coor_cells = torch.stack((coor_cells[:,:,:,:,:,1], coor_cells[:,:,:,:,:,0]), dim=-1)
        warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)
        # warped_coor_cells = normPts(coor_cells.view([-1, 2]), shape)
        # warped_coor_cells = torch.stack((warped_coor_cells[:,1], warped_coor_cells[:,0]), dim=1) # (y, x) to (x, y)
        # warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

        # warped_coor_cells = torch.stack((warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]), dim=2)  # (batch, x, y) to (batch, y, x)

        # shape_cell = torch.tensor([H//cell_size, W//cell_size]).type(torch.FloatTensor).to(device)
        # # warped_coor_mask = denormPts(warped_coor_cells, shape_cell)

        # warped_coor_cells = denormPts(warped_coor_cells, shape)
        # warped_coor_cells = warped_coor_cells.view([-1, 1, 1, Hc, Wc, 2])
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
    descriptors = descriptors.transpose(1, 2).transpose(2, 3)
    descriptors = descriptors.view((batch_size, Hc, Wc, 1, 1, -1))
    descriptors_warped = descriptors_warped.transpose(1, 2).transpose(2, 3)
    descriptors_warped = descriptors_warped.view((batch_size, 1, 1, Hc, Wc, -1))
    dot_product_desc = descriptors * descriptors_warped     # 两两描述子的内积
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
    mask_valid = mask_valid.view(batch_size, 1, 1, mask_valid.shape[2], mask_valid.shape[3])

    loss_desc = lamda_d * mask * positive_dist + (1 - mask) * negative_dist
    loss_desc = loss_desc * mask_valid
        # mask_validg = torch.ones_like(mask)
    ##### bug in normalization
    # normalization = (batch_size * (mask_valid.sum()+1) * Hc * Wc)
    normalization = ((mask_valid.sum() + 1) * cell_size * cell_size)
    pos_sum = (lamda_d * mask * positive_dist / normalization).sum()
    neg_sum = ((1 - mask) * negative_dist / normalization).sum()
    loss_desc = loss_desc.sum() / normalization
    # loss_desc = loss_desc.sum() / (batch_size * Hc * Wc)
    # return loss_desc, mask, mask_valid, positive_dist, negative_dist
    return loss_desc, mask, pos_sum, neg_sum

def descriptor_loss_fix(descriptors, descriptors_warped, homographies, mask_valid=None, 
                    cell_size=8, lamda_d=100, device='cpu', descriptor_dist=2, **config):

    '''
    Compute descriptor loss from descriptors_warped and given homographies

    :param descriptors:
        Output from descriptor head
        tensor [batch_size, descriptors, Hc, Wc]
    :param descriptors_warped:
        Output from descriptor head of warped image
        tensor [batch_size, descriptors, Hc, Wc]
    :param homographies:
        known homographies
    :param cell_size:
        8
    :param device:
        gpu or cpu
    :param config:
    :return:
        loss, and other tensors for visualization
    '''
    
    # put to gpu
    homographies = homographies.to(device)
    # config
    from utils.utils import warp_points
    from utils.utils import filter_points

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

        # coor_cells = coor_cells.view([-1, Hc, Wc, 1, 1, 2])
        coor_cells = coor_cells.view([-1, 1, 1, Hc, Wc, 2])  # be careful of the order  (y, x)
        coor_cells = torch.stack((coor_cells[:,:,:,:,:,1], coor_cells[:,:,:,:,:,0]), dim=-1)
        warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)
        # warped_coor_cells, mask_warped = filter_points_batch(warped_coor_cells, torch.tensor([W, H]).to(device), return_mask=True, device=device)
        
        warped_coor_coincide, mask_warped_coincide = filter_points_batch(warped_coor_cells, torch.tensor([W, H]).to(device), return_mask=True, device=device)
        inv_homographies = torch.stack([torch.inverse(homographies[i, :, :]) for i in range(batch_size)])
        # coor_coincide = warp_points(warped_coor_coincide.reshape([-1, 2]), inv_homographies, device)
        coor_coincide = torch.stack([warp_points(warped_coor_coincide[i,:,:].reshape([-1, 2]), inv_homographies[i,:,:], device) for i in range(batch_size)])
        coor_coincide, mask_coincide = filter_points_batch(coor_coincide, torch.tensor([W, H]).to(device), return_mask=True, device=device)
        # sample_desc_from_points()
        # warped_coor_cells = normPts(coor_cells.view([-1, 2]), shape)
        # warped_coor_cells = torch.stack((warped_coor_cells[:,1], warped_coor_cells[:,0]), dim=1) # (y, x) to (x, y)
        # warped_coor_cells = warp_points(warped_coor_cells, homographies, device)
        # warped_coor_cells = torch.stack((warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]), dim=2)  # (batch, x, y) to (batch, y, x)
        # shape_cell = torch.tensor([H//cell_size, W//cell_size]).type(torch.FloatTensor).to(device)
        # # warped_coor_mask = denormPts(warped_coor_cells, shape_cell)
        # warped_coor_cells = denormPts(warped_coor_cells, shape)
        # warped_coor_cells = warped_coor_cells.view([-1, 1, 1, Hc, Wc, 2])

        # warped_coor_cells = warped_coor_cells.view([-1, Hc, Wc, 1, 1, 2])
    #     print("warped_coor_cells: ", warped_coor_cells.shape)

        descriptors = descriptors.transpose(1, 2).transpose(2, 3)
        descriptors = descriptors.view((batch_size, Hc * Wc, -1))
        descriptors_warped = descriptors_warped.transpose(1, 2).transpose(2, 3)
        descriptors_warped = descriptors_warped.view((batch_size, Hc * Wc, -1))
        
    

        warped_coor = warped_coor_cells.reshape([-1, 1, 1, Hc, Wc, 2])
        warped_coor_mirro = warped_coor_cells.reshape([-1, Hc, Wc, 1, 1, 2])
        cell_distances = warped_coor - warped_coor_mirro
        cell_distances = torch.norm(cell_distances, dim=-1) # 计算两张图两两点对的距离
        ##### check
    #     print("descriptor_dist: ", descriptor_dist)
        mask = cell_distances <= descriptor_dist # 0.5 # trick
        mask = mask.type(torch.FloatTensor).to(device)

    # compute the pairwise dot product between descriptors: d^t * d
    descriptors = descriptors.transpose(1, 2).transpose(2, 3)
    descriptors = descriptors.view((batch_size, Hc, Wc, 1, 1, -1))
    descriptors_warped = descriptors_warped.transpose(1, 2).transpose(2, 3)
    descriptors_warped = descriptors_warped.view((batch_size, 1, 1, Hc, Wc, -1))

    dot_product_desc = descriptors * descriptors_warped     # 两两描述子的内积
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
    mask_valid = mask_valid.view(batch_size, 1, 1, mask_valid.shape[2], mask_valid.shape[3])

    loss_desc = lamda_d * mask * positive_dist + (1 - mask) * negative_dist
    loss_desc = loss_desc * mask_valid
        # mask_validg = torch.ones_like(mask)
    ##### bug in normalization
    # normalization = (batch_size * (mask_valid.sum()+1) * Hc * Wc)
    normalization = ((mask_valid.sum() + 1) * cell_size * cell_size)
    pos_sum = (lamda_d * mask * positive_dist / normalization).sum()
    neg_sum = ((1 - mask) * negative_dist / normalization).sum()
    loss_desc = loss_desc.sum() / normalization
    # loss_desc = loss_desc.sum() / (batch_size * Hc * Wc)
    # return loss_desc, mask, mask_valid, positive_dist, negative_dist
    return loss_desc, mask, pos_sum, neg_sum

"""
pos_pairs = mask * positive_dist
pos_pairs = pos_pairs[pos_pairs != 0]
print("pos_pairs mean: ", pos_pairs.mean())
print("pos_pairs max: ", pos_pairs.max())
print("pos_pairs min: ", pos_pairs.min())
===
print("pos_pairs mean: ", pos_pairs.mean())
pos_pairs mean:  tensor(0.6237, device='cuda:0', grad_fn=<MeanBackward1>)
print("pos_pairs max: ", pos_pairs.max())
pos_pairs max:  tensor(1.3984, device='cuda:0', grad_fn=<MaxBackward1>)
print("pos_pairs min: ", pos_pairs.min())
pos_pairs min:  tensor(0.1569, device='cuda:0', grad_fn=<MinBackward1>)
(pos_pairs < 0.3).sum()
Out[9]: tensor(88, device='cuda:0')
(pos_pairs < 0.5).sum()
Out[10]: tensor(393, device='cuda:0')
(pos_pairs < 0.7).sum()
Out[11]: tensor(703, device='cuda:0')

"""


def sumto2D(ndtensor):
    # input tensor: [batch_size, Hc, Wc, Hc, Wc]
    # output tensor: [batch_size, Hc, Wc]
    return ndtensor.sum(dim=1).sum(dim=1)

def mAP(pred_batch, labels_batch):
    pass

def precisionRecall_torch(pred, labels):    
    offset = 10**-6     # 防止分母为0
    assert pred.size() == labels.size(), 'Sizes of pred, labels should match when you get the precision/recall!'
    precision = torch.sum(pred*labels) / (torch.sum(pred)+ offset)      # 精确度计算  sum(pred*labels) 表示预测和标签一致的数量
    recall = torch.sum(pred*labels) / (torch.sum(labels) + offset)      # 召回率计算
    if precision.item() > 1.:
        print(pred)
        print(labels)
        import scipy.io.savemat as savemat
        savemat('pre_recall.mat', {'pred': pred, 'labels': labels})
    assert precision.item() <=1. and precision.item() >= 0.
    return {'precision': precision, 'recall': recall}

def precisionRecall_torch_soft(pred, labels, soft_margin):      # pred预测 labels标签 均为[1,128,128]
    offset = 10**-6
    assert pred.size() == labels.size(), 'Sizes of pred, labels should match when you get the precision/recall!'
    point_count = 0 
    record_cal = -torch.ones_like(pred)  # 记录被重复的次数
    for index_i in range(soft_margin, labels.size(1) - soft_margin):
        for index_j in range(soft_margin, labels.size(2) - soft_margin):
            if labels[:,index_i,index_j] == 1:
                pred_patch = pred[:,index_i - soft_margin:index_i + soft_margin + 1,index_j - soft_margin:index_j + soft_margin + 1]
                if torch.sum(pred_patch) > 0:  # 若标签周围有多个预测点，只计算一次
                    point_count += 1
                    record_cal[:,index_i - soft_margin:index_i + soft_margin + 1,index_j - soft_margin:index_j + soft_margin + 1] += pred_patch
    zero_tensor = torch.zeros_like(record_cal)
    repeat_count = torch.sum(torch.where(record_cal < 1, zero_tensor, record_cal))  # 只保留重复计数的次数
    point_count = point_count - repeat_count

    precision = point_count / (torch.sum(pred)+ offset)
    recall = point_count / (torch.sum(labels) + offset)
   
    assert precision.item() <=1. and precision.item() >= 0.
    assert recall.item() <=1. and recall.item() >= 0.
    return {'precision': precision, 'recall': recall}

def precisionRecall(pred, labels, thd=None):
    offset = 10**-6
    if thd is None:
        precision = np.sum(pred*labels) / (np.sum(pred)+ offset)
        recall = np.sum(pred*labels) / (np.sum(labels) + offset)
    return {'precision': precision, 'recall': recall}

def getWriterPath(task='train', exper_name='', date=True):
    import datetime
    prefix = 'runs/'
    str_date_time = ''
    if exper_name != '':
        exper_name += '_'
    if date:
        str_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return prefix + task + '/' + exper_name + str_date_time

def crop_or_pad_choice(in_num_points, out_num_points, shuffle=False):
    # Adapted from https://github.com/haosulab/frustum_pointnet/blob/635c938f18b9ec1de2de717491fb217df84d2d93/fpointnet/data/datasets/utils.py
    """Crop or pad point cloud to a fixed number; return the indexes
    Args:
        points (np.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order
    Returns:
        np.ndarray: output point cloud
        np.ndarray: index to choose input points
    """
    if shuffle:
        choice = np.random.permutation(in_num_points)
    else:
        choice = np.arange(in_num_points)
    assert out_num_points > 0, 'out_num_points = %d must be positive int!'%out_num_points
    if in_num_points >= out_num_points:
        choice = choice[:out_num_points]
    else:
        num_pad = out_num_points - in_num_points
        pad = np.random.choice(choice, num_pad, replace=True)   #在choice中随机选取（大小为num_pad）数据，replace=True允许重复
        choice = np.concatenate([choice, pad])
    return choice
