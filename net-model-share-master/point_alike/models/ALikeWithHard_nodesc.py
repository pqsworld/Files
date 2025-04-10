import logging
import os
import cv2
import torch
import torch.nn as nn

from copy import deepcopy
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import math
import numpy as np

from models.ALNet import ALNet, ALNet_nodesc
from models.modules import DKD
from models.hardnet_model import HardNet_small, HardNet, HardNet_smaller
# from utils.loss_functions.sparse_loss import D
from utils.utils import inv_warp_image, batch_inv_warp_image, inv_warp_patch_batch
from utils.homographies import sample_homography_cv
import time

configs = {
    'alike-t': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-t.pth')},
    'alike-s': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 96, 'dim': 96, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-s.pth')},
    'alike-n': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dim': 128, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-n.pth')},
    'alike-l': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 128, 'dim': 128, 'single_head': False, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-l.pth')},
}

def sample_descriptor(descriptor_map, kpts, bilinear_interp=False):
    """
    :param descriptor_map: BxCxHxW
    :param kpts: list, len=B, each is Nx2 (keypoints) [h,w]
    :param bilinear_interp: bool, whether to use bilinear interpolation
    :return: descriptors: list, len=B, each is NxD
    """
    batch_size, channel, height, width = descriptor_map.shape

    descriptors = []
    for index in range(batch_size):
        kptsi = kpts[index]  # Nx2,(x,y)

        if bilinear_interp:
            descriptors_ = F.grid_sample(descriptor_map[index].unsqueeze(0), kptsi.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, :, 0, :]  # CxN
        else:
            kptsi = (kptsi + 1) / 2 * kptsi.new_tensor([[width - 1, height - 1]])
            kptsi = kptsi.long()
            descriptors_ = descriptor_map[index, :, kptsi[:, 1], kptsi[:, 0]]  # CxN

        # descriptors_ = F.normalize(descriptors_, p=2, dim=0)
        descriptors.append(descriptors_.t())

    return descriptors

class ALikeWithHard_nodesc(ALNet_nodesc):
    def __init__(self,
                 # ================================== feature encoder
                 input_nc: int = 1, output_nc: int = 64, n_blocks: int = 6, dim: int = 128,
                 single_head: bool = False,
                 # ================================== detect parameterss
                 radius: int = 2,
                 top_k: int = -1, scores_th: float = 0.2,      # default: 1 / ((2*raduis+1) * (2*raduis+1))
                 n_limit: int = 400,
                 device: str = 'cpu',
                 model_path: str = '',
                 phase: str = 'train'
                 ):
        super().__init__(input_nc, output_nc, n_blocks=n_blocks)
        self.radius = radius            # nms radius
        self.top_k = top_k
        self.n_limit = n_limit
        self.scores_th = scores_th
        self.dkd = DKD(radius=self.radius, top_k=self.top_k,
                       scores_th=self.scores_th, n_limit=self.n_limit)
        self.descriptor_net =  nn.DataParallel(HardNet(train_flag=(phase == 'train'), device_ids=[0, 1, 2]))  # HardNet_small(train_flag=(phase == 'train'))  # HardNet(train_flag=(phase == 'train'))   
        self.desc_patch = 16
        self.sample_desc_patch = 16
        self.desc_patch_expand = 28
        self.orient_patch = 17
        self.patch_unfold = nn.Unfold(kernel_size=self.desc_patch, padding=self.desc_patch // 2)
        self.patch_unfold_expand = nn.Unfold(kernel_size=self.desc_patch_expand, padding=self.desc_patch_expand // 2)
        self.patch_unfold_orient = nn.Unfold(kernel_size=self.orient_patch, padding=self.orient_patch // 2)
        self.device = device
        self.phase = phase

        if model_path != '':
            state_dict = torch.load(model_path, self.device)
            self.load_state_dict(state_dict)
            self.to(self.device)
            self.eval()
            logging.info(f'Loaded model parameters from {model_path}')
            logging.info(
                f"Number of model parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e3}KB")

    def extract_dense_map(self, image, ret_dict=False):
        # ====================================================
        # check image size, should be integer multiples of 2^5
        # if it is not a integer multiples of 2^5, padding zeros
        device = image.device
        b, c, h, w = image.shape
        # h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
        # w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w

        # # right bottom padding zero
        # if h_ != h:
        #     h_padding = torch.zeros(b, c, h_ - h, w, device=device)       
        #     image = torch.cat([image, h_padding], dim=2)
        # if w_ != w:
        #     w_padding = torch.zeros(b, c, h_, w_ - w, device=device)
        #     image = torch.cat([image, w_padding], dim=3)
        # ====================================================

        scores_map, descriptor_map = super().forward(image)

        # # ====================================================
        # if h_ != h or w_ != w:
        #     descriptor_map = descriptor_map[:, :, :h, :w]
        #     scores_map = scores_map[:, :, :h, :w]  # Bx1xHxW
        # # ====================================================

        # BxCxHxW
        if descriptor_map is not None:
            descriptor_map = F.normalize(descriptor_map, p=2, dim=1)        # 沿着channel维L2归一化描述子

        if ret_dict:
            return {'descriptor_map': descriptor_map, 'scores_map': scores_map, }
        else:
            return descriptor_map, scores_map

    def generate_homograhy_by_angle(self, H, W, angles):
        scale = 1
        M = [cv2.getRotationMatrix2D((W / 2, H / 2), i, scale) for i in angles]
        # center = np.mean(pts2, axis=0, keepdims=True)
        homo = [np.concatenate((m, [[0, 0, 1.]]), axis=0) for m in M]

        # valid = np.arange(n_angles)
        # idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        # homo = M[idx]

        return homo

    def get_orientation(self, img, keypoints=None, device='cpu'):
        img = np.array(img)
        img = img.astype(np.float)
        # start = timer()
        Gx=np.zeros_like(img)
        Gy=np.zeros_like(img)
        h, w = img.shape
        for i in range(1,h-1):
            Gy[i,:] = img[i-1,:] - img[i+1,:] 
        Gy[0, :] = 2 * (img[0, :] - img[1, :])
        Gy[-1, :] = 2 * (img[-2, :] - img[-1, :])

        for j in range(1,w-1):
            Gx[:,j] = img[:,j+1] - img[:,j-1]
        Gx[:, 0] = 2 * (img[:,1] - img[:,0])
        Gx[:, -1] = 2 * (img[:,-1] - img[:,-2])

        Gxx = Gx*Gx
        Gyy = Gy*Gy
        Gxy = Gx*Gy
        Gxx_unfold = self.patch_unfold_orient(torch.tensor(Gxx).unsqueeze(0).unsqueeze(0).to(device)).view(1, -1, h, w)
        Gyy_unfold = self.patch_unfold_orient(torch.tensor(Gyy).unsqueeze(0).unsqueeze(0).to(device)).view(1, -1, h, w)
        Gxy_unfold = self.patch_unfold_orient(torch.tensor(Gxy).unsqueeze(0).unsqueeze(0).to(device)).view(1, -1, h, w)
        Gxx_unfold_sum = torch.sum(Gxx_unfold, dim=1)
        Gyy_unfold_sum = torch.sum(Gyy_unfold, dim=1)       
        Gxy_unfold_sum = torch.sum(Gxy_unfold, dim=1)

        eps = 1e-12
        degree_value_all = 2 * Gxy_unfold_sum / (Gxx_unfold_sum - Gyy_unfold_sum + eps)
        angle_all = torch.atan(degree_value_all)    
        angle_all = angle_all*57.29578049 #180/(3.1415926)
        cond1 = torch.logical_and(Gxx_unfold_sum - Gyy_unfold_sum < 0, Gxy_unfold_sum >= 0)
        cond2 = torch.logical_and(Gxx_unfold_sum - Gyy_unfold_sum < 0, Gxy_unfold_sum < 0)
        angle_all[cond1] = (angle_all[cond1] + 180) / 2
        angle_all[cond2] = (angle_all[cond2] - 180) / 2
        angle_all[Gxx_unfold_sum - Gyy_unfold_sum >= 0] = angle_all[Gxx_unfold_sum - Gyy_unfold_sum >= 0] / 2
        angle_all += 90
        angle_all[angle_all > 90] = angle_all[angle_all > 90] - 180

        if keypoints is None:
            return angle_all.view(1, 1, -1)
        else:
            angle = F.grid_sample(angle_all.float().unsqueeze(0),
                                keypoints.view(1, 1, -1, 2),
                                mode='bilinear', align_corners=True).squeeze(2).squeeze(0)        # 1 x M

        return angle

    def get_orientation_batch(self, img_batch, keypoints=None, device='cpu'):
        b, _, h, w = img_batch.shape
        img_batch_left = img_batch[:, :, :, :-2]
        img_batch_right = img_batch[:, :, :, 2:]
        img_batch_top = img_batch[:, :, :-2, :]
        img_batch_bottom = img_batch[:, :, 2:, :]
        Gx = torch.zeros_like(img_batch, device=device)
        Gy = torch.zeros_like(img_batch, device=device)
        Gx[:, :, :, 1:-1] = img_batch_right - img_batch_left
        Gy[:, :, 1:-1, :] = img_batch_top - img_batch_bottom       # w-2 
        Gx[:, :, :, 0] = 2 * (img_batch[:, :, :, 1] - img_batch[:, :, :, 0])
        Gx[:, :, :, -1] = 2 * (img_batch[:, :, :,-1] - img_batch[:, :, :,-2])
        Gy[:, :, 0, :] = 2 * (img_batch[:, :, 0, :] - img_batch[:, :, 1, :])
        Gy[:, :, -1, :] = 2 * (img_batch[:, :, -2, :] - img_batch[:, :, -1, :])

        Gxx = Gx*Gx
        Gyy = Gy*Gy
        Gxy = Gx*Gy
        Gxx_unfold = self.patch_unfold_orient(Gxx).view(b, -1, h, w)
        Gyy_unfold = self.patch_unfold_orient(Gyy).view(b, -1, h, w)
        Gxy_unfold = self.patch_unfold_orient(Gxy).view(b, -1, h, w)
        Gxx_unfold_sum = torch.sum(Gxx_unfold, dim=1)
        Gyy_unfold_sum = torch.sum(Gyy_unfold, dim=1)       
        Gxy_unfold_sum = torch.sum(Gxy_unfold, dim=1)
        
        eps = 1e-12
        degree_value_all = 2 * Gxy_unfold_sum / (Gxx_unfold_sum - Gyy_unfold_sum + eps)
        angle_all = torch.atan(degree_value_all)    
        angle_all = angle_all*57.29578049 #180/(3.1415926)
        cond1 = torch.logical_and(Gxx_unfold_sum - Gyy_unfold_sum < 0, Gxy_unfold_sum >= 0)
        cond2 = torch.logical_and(Gxx_unfold_sum - Gyy_unfold_sum < 0, Gxy_unfold_sum < 0)
        angle_all[cond1] = (angle_all[cond1] + 180) / 2
        angle_all[cond2] = (angle_all[cond2] - 180) / 2
        angle_all[Gxx_unfold_sum - Gyy_unfold_sum >= 0] = angle_all[Gxx_unfold_sum - Gyy_unfold_sum >= 0] / 2
        angle_all += 90
        angle_all[angle_all > 90] = angle_all[angle_all > 90] - 180

        if keypoints is None:
            return angle_all.view(b, 1, -1)
        else:
            angle = F.grid_sample(angle_all.float().unsqueeze(0),
                                keypoints.view(1, 1, -1, 2),
                                mode='bilinear', align_corners=True).squeeze(2).squeeze(0)        # 1 x M

        return angle

    def cut_patch(self, img_batch, points, patch_size=16, train_flag=False):
        b, c, h, w = img_batch.shape
        descriptors = []
        results = None 
        # Padding Zero
        pad_size = (patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2)
        img_pad_batch = F.pad(img_batch, pad_size, "constant", 0)
        for batch_idx in range(b):
            keypoints = points[batch_idx]
            keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]]).to(keypoints.device)
            img = img_pad_batch[batch_idx]
            
            for point in keypoints:
                x = int(point[0] + 0.5)
                y = int(point[1] + 0.5)
                # crop_x = 0 if x-patch_size/2<0 else x-patch_size//2
                # crop_y = 0 if y-patch_size/2<0 else y-patch_size//2
                crop_x = x
                crop_y = y
                # print(x, y ,crop_x, crop_y)
                patch = img[:,crop_y:crop_y+patch_size,crop_x:crop_x+patch_size]

                data = patch.unsqueeze(0)
                # print(data.shape)
                if results is None:
                    results = data
                else:
                    results = torch.cat([results, data],dim=0)
        
        # compute output for patch a
        results_batch = Variable(results)   
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
        # outs = torch.chunk(outs, b, dim=0)
    
        return outs    

    def cut_patch_unfold(self, img_batch, points, train_flag=False):
        b, _, h, w = img_batch.shape
        results = None 
        # Padding Zero
        # homography = sample_homography_cv(self.desc_patch, self.desc_patch, max_angle=180, n_angles=360)
        # homography = torch.tensor(homography, dtype=torch.float32)
        img_unfold_batch = self.patch_unfold(img_batch).view(b, -1, h+1, w+1)[:, :, :-1, :-1]     # Bx(patch_size x patch_size) x H x W
        for batch_idx in range(b):
            keypoints = points[batch_idx]
            keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]]).to(keypoints.device)
            img_patches = img_unfold_batch[batch_idx].transpose(0, 2)     # w x h x (patch_size x patch_size)
            # print(img_patches_rotate.shape)
            keypoints = (keypoints + 0.5).long()

            # no rotate
            data = img_patches[keypoints[:, 0], keypoints[:, 1]].view(-1, self.desc_patch, self.desc_patch).unsqueeze(1)    # M x patch_size x patch_size
            if results is None:
                results = data
            else:
                results = torch.cat([results, data],dim=0)

            # data = img_patches[keypoints[:, 0], keypoints[:, 1]].view(-1, self.desc_patch, self.desc_patch).unsqueeze(1)
            # # print(data.shape, homography.shape)
            # data_rotate = batch_inv_warp_image(data.cpu(), homography.repeat(data.shape[0], 1, 1), mode="bilinear").to(data.device)
            # # print(data_rotate.shape)
            # if results is None:
            #     results = data_rotate
            # else:
            #     results = torch.cat([results, data_rotate],dim=0)

        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
        # outs = torch.chunk(outs, b, dim=0)
    
        return outs.squeeze(-1)    

    def cut_patch_unfold_interpolation(self, img_batch, points, train_flag=False):
        b, _, h, w = img_batch.shape
        results = None 
        # Padding Zero
        # homography = sample_homography_cv(self.desc_patch, self.desc_patch, max_angle=180, n_angles=360)
        # homography = torch.tensor(homography, dtype=torch.float32)
        img_unfold_batch = self.patch_unfold(img_batch).view(b, -1, h+1, w+1)     # Bx(patch_size x patch_size) x (H + 1) x (W + 1)
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_offset = keypoints * keypoints.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
            img_patches = F.grid_sample(img_unfold_batch[batch_idx].unsqueeze(0),
                                    keypoints_offset.view(1, 1, -1, 2),
                                    mode='bilinear', align_corners=True).squeeze(2).squeeze(0)
            # no rotate
            data = img_patches.transpose(0, 1).view(-1, self.desc_patch, self.desc_patch).unsqueeze(1)  # M x 1 x patch_size x patch_size
            # data = img_patches[keypoints[:, 0], keypoints[:, 1]].view(-1, self.desc_patch, self.desc_patch).unsqueeze(1)    # M x 1 x patch_size x patch_size
            if results is None:
                results = data
            else:
                results = torch.cat([results, data],dim=0)

            # data = img_patches[keypoints[:, 0], keypoints[:, 1]].view(-1, self.desc_patch, self.desc_patch).unsqueeze(1)
            # # print(data.shape, homography.shape)
            # data_rotate = batch_inv_warp_image(data.cpu(), homography.repeat(data.shape[0], 1, 1), mode="bilinear").to(data.device)
            # # print(data_rotate.shape)
            # if results is None:
            #     results = data_rotate
            # else:
            #     results = torch.cat([results, data_rotate],dim=0)

        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
        # outs = torch.chunk(outs, b, dim=0)
    
        return outs.squeeze(-1)   

    def cut_patch_unfold_interpolation_aligned(self, img_batch, points, train_flag=False):
        b, _, h, w = img_batch.shape
        results = None 
        # Padding Zero
        # homography = sample_homography_cv(self.desc_patch, self.desc_patch, max_angle=180, n_angles=360)
        # homography = torch.tensor(homography, dtype=torch.float32)
        img_unfold_batch_expand = self.patch_unfold_expand(img_batch).view(b, -1, h+1, w+1)     # Bx(expand_patch_size x expand_patch_size) x (H + 1) x (W + 1)
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            keypoints_offset = keypoints * keypoints.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
            img_patches = F.grid_sample(img_unfold_batch_expand[batch_idx].unsqueeze(0),
                                    keypoints_offset.view(1, 1, -1, 2),
                                    mode='bilinear', align_corners=True).squeeze(2).squeeze(0)      # (expand_patch_size x expand_patch_size) x M
            orientation_patch = self.get_orientation(img_batch[batch_idx].squeeze().cpu().numpy(), keypoints, keypoints.device)
            homography_expand_patch = self.generate_homograhy_by_angle(self.desc_patch_expand, self.desc_patch_expand, -orientation_patch.squeeze(0).detach().cpu().numpy())
            homography_expand_patch = torch.tensor(homography_expand_patch, dtype=torch.float32)        # Mx3x3
            data_expand = img_patches.transpose(0, 1).view(-1, self.desc_patch_expand, self.desc_patch_expand).unsqueeze(1) # M x 1 x expand_patch_size x expand_patch_size
            data_expand_rotate = batch_inv_warp_image(data_expand.cpu(), homography_expand_patch, mode="bilinear").to(data_expand.device)
            
            # 24x24 -> 16x16
            data = data_expand_rotate[:, :, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2].contiguous()
  
            if results is None:
                results = data
            else:
                results = torch.cat([results, data],dim=0)

            # data = img_patches[keypoints[:, 0], keypoints[:, 1]].view(-1, self.desc_patch, self.desc_patch).unsqueeze(1)
            # # print(data.shape, homography.shape)
            # data_rotate = batch_inv_warp_image(data.cpu(), homography.repeat(data.shape[0], 1, 1), mode="bilinear").to(data.device)
            # # print(data_rotate.shape)
            # if results is None:
            #     results = data_rotate
            # else:
            #     results = torch.cat([results, data_rotate],dim=0)

        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
        # outs = torch.chunk(outs, b, dim=0)
    
        return outs.squeeze(-1)   

    def cut_patch_unfold_map_interpolation(self, img_batch, train_flag=False):
        b, _, h, w = img_batch.shape
        # Padding Zero
        img_unfold_batch = self.patch_unfold(img_batch).view(b, -1, h+1, w+1)    # Bx(patch_size x patch_size)x H x W 
        x = torch.linspace(0, w-1, w)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        # print(mesh_points.shape, mesh_points)
        mesh_grid = mesh_points / mesh_points.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
        keypoints_offset = mesh_grid * mesh_grid.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
        # print(keypoints_offset.shape)
        data = F.grid_sample(img_unfold_batch,
                        keypoints_offset.unsqueeze(0).repeat(b, 1, 1, 1),
                        mode='bilinear', align_corners=True)            # bx(patch_size x patch_size)x1x(hxw)
        results = data.transpose(1, 3).reshape(-1, self.desc_patch, self.desc_patch).unsqueeze(1).contiguous()
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
                del results_batch
        outs = outs.squeeze(-1).reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w)          # B X dim x H X W
        return outs  

    def cut_patch_unfold_map_interpolation_aligned(self, img_batch, train_flag=False):
        b, _, h, w = img_batch.shape
        # Padding Zero
        img_unfold_batch = self.patch_unfold_expand(img_batch).view(b, -1, h+1, w+1)    # Bx(expand_patch_size x expand_patch_size)x H x W 
        x = torch.linspace(0, w-1, w)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        # print(mesh_points.shape, mesh_points)
        mesh_grid = mesh_points / mesh_points.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
        keypoints_offset = mesh_grid * mesh_grid.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
        # print(keypoints_offset.shape)
        img_patches = F.grid_sample(img_unfold_batch,
                        keypoints_offset.unsqueeze(0).repeat(b, 1, 1, 1),
                        mode='bilinear', align_corners=True)        # B X (expand_patch_size x expand_patch_size) X 1 X (HXW)

        orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        homography_expand_patch = self.generate_homograhy_by_angle(self.desc_patch_expand, self.desc_patch_expand, -orient_batch.view(-1).squeeze().detach().cpu().numpy())
        homography_expand_patch_all = torch.tensor(homography_expand_patch, dtype=torch.float32)        # (BxM)x3x3
        data_expand = img_patches.transpose(1, 3).reshape(-1, self.desc_patch_expand, self.desc_patch_expand).unsqueeze(1) # (BxM) x 1 x expand_patch_size x expand_patch_size
        data_expand_rotate = inv_warp_image(data_expand.cpu(), homography_expand_patch_all)
        # data_expand_rotate = batch_inv_warp_image(data_expand.cpu(), homography_expand_patch_all, mode="bilinear").to(data_expand.device)    
        # 24x24 -> 16x16
        data = data_expand_rotate[:, :, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2].contiguous()
        # print(data.shape)
        # print(img_unfold_batch.shape)
        # results = data.permute(0, 2, 3, 1).reshape(-1, 1, self.desc_patch, self.desc_patch)   
        results = data   
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
                del results_batch
        outs = outs.squeeze(-1).reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w)          # B X dim x H X W
        return outs      

    def cut_patch_unfold_patch_map_interpolation_aligned(self, img_batch,  points, train_flag=False):
        b, _, h, w = img_batch.shape
        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 250
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0)  
    
        orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X (expand_patch_size x expand_patch_size) X 1 X (HXW)
        homography_expand_map = self.generate_homograhy_by_angle(self.desc_patch_expand, self.desc_patch_expand, -orient_batch.view(-1).squeeze().detach().cpu().numpy())
        homography_expand_map_all = torch.tensor(homography_expand_map, dtype=torch.float32)        # (BxHxW)x3x3
        homography_expand_patch = self.generate_homograhy_by_angle(self.desc_patch_expand, self.desc_patch_expand, -orient_batch_kp.view(-1).squeeze().detach().cpu().numpy())
        homography_expand_patch_all = torch.tensor(homography_expand_patch, dtype=torch.float32)        # (BxM)x3x3
        
        # Padding Zero
        img_unfold_batch = self.patch_unfold_expand(img_batch).view(b, -1, h+1, w+1)    # Bx(expand_patch_size x expand_patch_size)x H x W 
        x = torch.linspace(0, w-1, w)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device)
        # print(mesh_points.shape, mesh_points)
        mesh_grid = mesh_points / mesh_points.new_tensor([w - 1, h - 1]).to(img_batch.device) * 2 - 1
        mesh_grid_offset = mesh_grid * mesh_grid.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
        keypoints_expand_offset = keypoints_expand * keypoints_expand.new_tensor([(w - 1) / w, (h - 1) / h]).to(img_batch.device)
        # print(keypoints_offset.shape)
        img_patches = F.grid_sample(img_unfold_batch,
                        mesh_grid_offset.unsqueeze(0).repeat(b, 1, 1, 1),
                        mode='bilinear', align_corners=True)        # B X (expand_patch_size x expand_patch_size) X 1 X (HXW)
        
        img_patches_kp = F.grid_sample(img_unfold_batch,
                        keypoints_expand_offset.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X (expand_patch_size x expand_patch_size) X 1 X M

        data_expand = img_patches.transpose(1, 3).reshape(-1, self.desc_patch_expand, self.desc_patch_expand).unsqueeze(1) # (BxHxW) x 1 x expand_patch_size x expand_patch_size
        data_expand_rotate = batch_inv_warp_image(data_expand.cpu(), homography_expand_map_all, device=data_expand.device)
        data_expand_kp = img_patches_kp.transpose(1, 3).reshape(-1, self.desc_patch_expand, self.desc_patch_expand).unsqueeze(1) # (BxM) x 1 x expand_patch_size x expand_patch_size
        data_expand_rotate_kp = batch_inv_warp_image(data_expand_kp.cpu(), homography_expand_patch_all, device=data_expand_kp.device)
        # data_expand_rotate = batch_inv_warp_image(data_expand.cpu(), homography_expand_patch_all, mode="bilinear").to(data_expand.device)    
        # 24x24 -> 16x16
        data = data_expand_rotate[:, :, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2].contiguous()
        data_kp = data_expand_rotate_kp[expand_mask_all==1,  :, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2, (self.desc_patch_expand-self.desc_patch)//2:-(self.desc_patch_expand-self.desc_patch)//2].contiguous()
        
        # # 画校准后的patch
        # for count in range(data_kp.shape[0]):
        #     if count < 100:
        #         cv2.imwrite('demo/demo_patch_mesh_grid_' + str(count) + '.bmp', 255*data[count, 0, :, :].detach().cpu().numpy())
        #         cv2.imwrite('demo/demo_patch_kp_' + str(count) + '.bmp', 255*data_kp[count, 0, :, :].detach().cpu().numpy())
                
        # print(data.shape)
        # print(img_unfold_batch.shape)
        # results = data.permute(0, 2, 3, 1).reshape(-1, 1, self.desc_patch, self.desc_patch)   
        results = data_kp   
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w)          # B X dim x H X W
        return outs.squeeze(-1), outs_map     

    def cut_patch_unfold_patch_map_interpolation_aligned_batch(self, img_batch, points, train_flag=False, fixed_angle=None):
        b, _, h, w = img_batch.shape
        pad_size = (self.desc_patch_expand // 2, self.desc_patch_expand // 2, self.desc_patch_expand // 2, self.desc_patch_expand // 2)
        img_pad_batch = F.pad(img_batch, pad_size, "constant", 0)

        keypoints_expand = None         # (b x max_num)x2
        expand_mask_all = None          # (b x max_mum)x1
        max_num = 400
        expand = torch.zeros((max_num, 2), device=img_batch.device) 
        for batch_idx in range(b):
            keypoints = points[batch_idx].float()
            expand_mask = torch.zeros(max_num, device=img_batch.device)           # 点坐标补零
            expand_mask[:keypoints.shape[0]] = 1
            if keypoints_expand is None: 
                keypoints_expand = torch.cat((keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = expand_mask
            else:
                keypoints_expand = torch.cat((keypoints_expand, keypoints, expand[expand_mask==0]), dim=0)
                expand_mask_all = torch.cat((expand_mask_all, expand_mask), dim=0) 

        if fixed_angle is not None:
            orient_batch = fixed_angle.repeat(1, 1, h*w)
            # print(orient_batch)
            orient_batch = orient_batch.to(img_batch.device).float()
        else:
            orient_batch = self.get_orientation_batch(img_batch, device=img_batch.device)      # Bx1x(hxw)
        orient_batch_kp = F.grid_sample(orient_batch.view(b, 1, h, w),
                        keypoints_expand.view(b, 1, -1, 2),
                        mode='bilinear', align_corners=True)        # B X 1 X 1 X (HXW)
        # print(orient_batch_old, orient_batch_kp_old)
        # orient_batch = torch.zeros(b, 1, 1, h*w).to(img_batch.device)
        # orient_batch_kp = torch.zeros(b, 1, 1, max_num).to(img_batch.device)
        x = torch.linspace(0, w-1, w)     # ex: [-2, -1, 0, 1, 2]
        y = torch.linspace(0, h-1, h) 
        mesh_points = torch.stack(torch.meshgrid([y, x])).view(2, -1).t()[:, [1, 0]].to(img_batch.device) + self.desc_patch_expand // 2
        keypoints_expand = (keypoints_expand + 1) / 2 * keypoints_expand.new_tensor([w - 1, h - 1]).to(img_batch.device) + self.desc_patch_expand // 2
        data = inv_warp_patch_batch(img_pad_batch, mesh_points.repeat(b, 1, 1).view(-1, 2), orient_batch.view(-1), self.desc_patch, self.sample_desc_patch).unsqueeze(1)   
        data_kp = inv_warp_patch_batch(img_pad_batch, keypoints_expand, orient_batch_kp.view(-1), self.desc_patch, self.sample_desc_patch)[expand_mask_all==1, :, : ].unsqueeze(1)   

        # # 画校准后的patch
        # for count in range(data_kp.shape[0]):            
        #     if count < 100:
        #         cv2.imwrite('demo/demo_patch_mesh_grid_new_' + str(count) + '.bmp', 255*data[count, 0, :, :].detach().cpu().numpy())
        #         cv2.imwrite('demo/demo_patch_kp_new_' + str(count) + '.bmp', 255*data_kp[count, 0, :, :].detach().cpu().numpy())
        # exit()
        # print(data.shape)
        # print(img_unfold_batch.shape)
        # results = data.permute(0, 2, 3, 1).reshape(-1, 1, self.desc_patch, self.desc_patch)   
        results = data_kp
        # compute output for patch a
        results_batch = Variable(results)   
        # print(results_batch.shape)
        if train_flag:
            outs = self.descriptor_net(results_batch)
        else:
            with torch.no_grad():
                outs = self.descriptor_net(results_batch)
                del results_batch
        with torch.no_grad():
            outs_map = self.descriptor_net(data)
            del data
        outs_map = outs_map.squeeze(-1).reshape(b, h*w, -1).transpose(1, 2).view(b, -1, h, w)          # B X dim x H X W

        return outs.squeeze(-1), outs_map     

    def forward(self, img, image_size_max=99999, sort=False, sub_pixel=False):
        """
        :param img: np.array HxWx3, RGB
        :param image_size_max: maximum image size, otherwise, the image will be resized
        :param sort: sort keypoints by scores
        :param sub_pixel: whether to use sub-pixel accuracy
        :return: a dictionary with 'keypoints', 'descriptors', 'scores', and 'time'
        """
        B, C, H, W = img.shape
        # assert three == 3, "input image shape should be [HxWx3]"

        # # ==================== image size constraint
        # image = deepcopy(img)
        # max_hw = max(H, W)
        # if max_hw > image_size_max:
        #     ratio = float(image_size_max / max_hw)
        #     image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)

        # # ==================== convert image to tensor
        # image = torch.from_numpy(image).to(self.device).to(torch.float32).permute(2, 0, 1)[None] / 255.0

        # ==================== extract keypoints
        start = time.time()
        if self.phase == 'train':

            # # Cal orientation
            # keypoints, descriptors, descriptor_map_new, scores, scores_map, scoredispersitys = 0, 0, 0, 0, 0, 0
            # orient_batch = self.get_orientation_batch(img, device=img.device)

            descriptor_map, scores_map = self.extract_dense_map(img)
            keypoints, _, scores, scoredispersitys = self.dkd(scores_map, descriptor_map,
                                                                sub_pixel=sub_pixel)
            # descriptors = self.cut_patch(img, keypoints, self.desc_patch, True)

            # descriptors = self.cut_patch_unfold(img, keypoints, True)
            # descriptors = self.cut_patch_unfold_interpolation(img, keypoints, True)

            del descriptor_map
            # _, _ = self.cut_patch_unfold_patch_map_interpolation_aligned(img, keypoints, True)
            angle_enhance = torch.rand((B, 1, 1), device=img.device) * 360 - 180    # [-180, 180)
            # descriptors, descriptor_map_new = self.cut_patch_unfold_patch_map_interpolation_aligned_batch(img, keypoints, True, angle_enhance)
            descriptors, descriptor_map_new = self.cut_patch_unfold_patch_map_interpolation_aligned_batch(img, keypoints, True)
            # descriptor_map_new = self.cut_patch_unfold_map_interpolation(img, False)

            # Dense
            # descriptors = sample_descriptor(descriptor_map, keypoints, sub_pixel)
            # keypoints, descriptors, scores = keypoints, descriptors, scores
            # print(len(keypoints), keypoints[0].shape, descriptors[0].shape, scores[0].shape)
            # keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W - 1, H - 1]])    # 归一化点坐标
            pass
        else:
            with torch.no_grad():
                descriptor_map, scores_map = self.extract_dense_map(img)
                keypoints, descriptors, scores, scoredispersitys = self.dkd(scores_map, descriptor_map,
                                                                sub_pixel=sub_pixel)
                keypoints, descriptors, scores = keypoints[0], descriptors[0], scores[0]
                keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W - 1, H - 1]])    # 归一化点坐标->尺寸坐标

        # if sort:
        #     indices = torch.argsort(scores, descending=True)        # 置信度降序排序 
        #     keypoints = keypoints[indices]
        #     descriptors = descriptors[indices]
        #     scores = scores[indices]

        end = time.time()

        return {'keypoints': keypoints,
                'descriptors': descriptors,
                'descriptor_map': descriptor_map_new,
                'scores': scores,
                'scores_map': scores_map,
                'scoredispersitys': scoredispersitys,
                # 'orientation': orient_batch,
                # 'time': end - start, 
                }


if __name__ == '__main__':
    import numpy as np
    from thop import profile

    net = ALikeWithHard(c1=32, c2=64, c3=128, c4=128, dim=128, single_head=False)

    image = np.random.random((640, 480, 3)).astype(np.float32)
    flops, params = profile(net, inputs=(image, 9999, False), verbose=False)
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))
