import re
import torch
import copy
from PIL import Image
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import torchvision.transforms as transforms
from scipy.linalg import hadamard

from Model_component import flattenDetection, getPtsFromHeatmap, getPtsFromHeatmap_torch, L2Norm
# from .SuperPointNet_small_128 import SuperPointNet_small_128

class FPDT(object):
    '''Feature Point Detection Test'''
    def __init__(self, name, detector_net_weights=None, device="cpu"):
        print("model: ", name)
        self.device = device
        
        # self.net_index = (((torch.linspace(32,1,32)**8).unsqueeze(0) @ hadamard(32)) == ((torch.linspace(1,32,32)**8).unsqueeze(0) @ hadamard(32))).long().squeeze(0).unsqueeze(1).repeat(1,4).view(-1).bool()
        '''Load Model'''
        if detector_net_weights != None:
            try:
                names = name.split('.')
                if len(names) == 1:
                    net = getattr(__import__('models.{}'.format(name), fromlist=['']), name)
                else:
                    net = getattr(__import__('models.{}'.format(names[0]), fromlist=['']), names[1])

                # self.net = net(train_flag=False).to(self.device)
                self.net = net().to(self.device)
                checkpoint = torch.load(detector_net_weights, map_location=lambda storage, loc: storage)
                self.net.load_state_dict(checkpoint['model_state_dict'], strict = False)
                print("==> Loading {}...".format(detector_net_weights.split('/')[-1]))
                # self.detector_net.load_state_dict(torch.load(detector_net_weights, map_location={'cuda:0':('cuda:' + str(gpu_ids))})['model_state_dict'])
                self.net.eval()
            except Exception:
                print("Load Model fail!")
                raise
            print("==> Successfully loaded pre-trained network.")

            # 如果有reparameterize函数，则会进行重参数化操作
            for module in self.net.modules():
                if hasattr(module, 'reparameterize'):
                    module.reparameterize() # 不需要指定推理开关：inference_mode， 当调用reparameterize方法时就会执行重参数化操作

        '''Init Parameters'''
        self.samples        = None
        self.iterations     = 0

    def test_model(self, x: Image, device='cpu'):
        # evaluate the `model` on 8 rotated versions of the input image `x`     
        # model.eval()
        # wrmup = model(x.to(device))
        # del wrmup
        # x = torch.tensor(np.array(x)).to(device).reshape(1, 1, 29, 29)
        from torchvision.transforms import ToTensor
        totensor = ToTensor()
        # pad = Pad((0, 0, 2, 2), fill=0)
        # x = pad(x)
        print()
        print('##########################################################################################')
        header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(10)])
        print(header)
        with torch.no_grad():
            for r in range(8):
                x_transformed = totensor(x.rotate(r*45., Image.BILINEAR)).reshape(1, 1, 136, 36)
                x_transformed = x_transformed.to(device)
                y = self.net(x_transformed)
                y = y.to('cpu').numpy().squeeze()
                angle = r * 45
                print("{:5d} : {}".format(angle, y))
        print('##########################################################################################')     
        print()

        pass    

    def thresholding_desc(self, descs):
        '''
        @descs:     [N, 128(dim)]
        '''
        norm = (torch.sqrt(torch.sum(descs * descs, dim = 1)) * 0.2).long()
        norm = norm.unsqueeze(-1).expand_as(descs).float()
        descs = torch.where(descs < norm, torch.sqrt(descs).long(), torch.sqrt(norm).long())
        return descs

    def fix_orient_by_trans_flag(self, orientation_batch, trans_angle):
        '''
        input:
        orientation_batch: [bs, n]
        trans_angle: [bs]
        return:
        correct_angle: [bs, n]
        trans_angle: φ = real_θ_A - real_θ_B      real_θ_x:真实角
        orien(θ_A) - φ = θ_A - (real_θ_A - real_θ_B)
        '''
        orient_sub_trans = orientation_batch - trans_angle
        correct_flag = ((orient_sub_trans > 90) + (orient_sub_trans < -90)).float()
        correct_angles = correct_flag * 180

        return correct_angles

    def get_des_hanmingdis_wht_permute(self, des_a, des_b):
        hadama_trans = torch.tensor(hadamard(des_a.shape[1]), device=des_a.device).float()
        wht_desc_a, wht_desc_b = des_a @ hadama_trans, des_b @ hadama_trans
        half_dim =  wht_desc_a.shape[1] // 2 
        desc_binary_a = torch.where(wht_desc_a > 0, torch.ones_like(wht_desc_a), torch.zeros_like(wht_desc_a))  # Nxdim
        desc_binary_b = torch.where(wht_desc_b > 0, torch.ones_like(wht_desc_b), torch.zeros_like(wht_desc_b))  # Mxdim
        # index = (((torch.linspace(16,1,16)**4).unsqueeze(0) @ hadamard(16)) == ((torch.linspace(1,16,16)**4).unsqueeze(0) @ hadamard(16))).long().squeeze(0).unsqueeze(1).repeat(1,8).view(-1).bool()
        hanming_dist_part_begin = half_dim - desc_binary_a[:, self.net_index] @ desc_binary_b[:, self.net_index].t() - (1 - desc_binary_a[:, self.net_index]) @ (1 - desc_binary_b[:, self.net_index].t())       
        hanming_dist_part_end = half_dim - desc_binary_a[:, self.net_index==False] @ desc_binary_b[:, self.net_index==False].t() - (1 - desc_binary_a[:, self.net_index==False]) @ (1 - desc_binary_b[:, self.net_index==False].t())    
        hanming_dist_part_end_min = torch.where(hanming_dist_part_end >= half_dim // 2, half_dim - hanming_dist_part_end, hanming_dist_part_end)        # 0-32
        # hanming_dist = torch.ones((wht_desc_a.shape[0], wht_desc_b.shape[0]), device=self.device) * wht_desc_a.shape[1] - desc_binary_a @ desc_binary_b.t() - (1 - desc_binary_a) @ (1 - desc_binary_b.t())
        return hanming_dist_part_begin, hanming_dist_part_end_min
    
    def run_heatmap(self, img):
        '''
        input: 1 1 H W
        return: heatmap
        '''
        H, W = img.shape[2], img.shape[3]

        with torch.no_grad():
            self.net.eval()
            outs = self.net.forward(img)
            semi, desc = outs['semi'], outs['desc']

        heatmap = flattenDetection(semi)
        # pts = getPtsFromHeatmap(heatmap.squeeze())
        return heatmap, desc

    def run_detec_desc(self, img):
        '''
        input: 1 1 H W
        return: semi, desc
        '''
        H, W = img.shape[2], img.shape[3]

        with torch.no_grad():
            self.net.eval()
            outs = self.net.forward(img)
            semi, desc = outs['semi'], outs['desc']

        return semi, desc

    def run_pts_desc(self, img, conf_thresh, nms):
        '''
        input: 1 1 H W
        return: pts, desc
        '''
        H, W = img.shape[2], img.shape[3]

        with torch.no_grad():
            self.net.eval()
            outs = self.net.forward(img)
            semi, desc = outs['semi'], outs['desc']
        
        heatmap = flattenDetection(semi)
        pts = getPtsFromHeatmap(heatmap.squeeze().detach().cpu().numpy(), conf_thresh, nms)

        return pts, desc

    def run_hardnet_process(self, img_batch, points, patch_size=16, train_flag=False):
        ''' 弃用 '''
        b, _, h, w = img_batch.shape
        results = None
        # Padding Zero
        patch_unfold = nn.Unfold(kernel_size=patch_size, padding=patch_size//2, stride=1)
        img_unfold_batch = patch_unfold(img_batch).view(b, -1, h + 1, w + 1)[:, :, :-1, :-1]     # Bx(patch_size x patch_size)x(HxW) 
        for batch_idx in range(b):
            keypoints = points
            # keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[w - 1, h - 1]]).to(keypoints.device)
            img_patches = img_unfold_batch[batch_idx].transpose(0, 2)     # W x H x (patch_size x patch_size)
            keypoints = (keypoints + 0.5).long()
            data = img_patches[keypoints[:, 0], keypoints[:, 1]].view(-1, patch_size, patch_size).unsqueeze(1)    # M x (patch_size x patch_size)

            '''ttt'''
            # if batch_idx == 1:
            #     img_full = img_batch[batch_idx, 0, :, :].cpu().detach().numpy().squeeze() * 255
            #     img_full = cv2.resize(img_full, None, fx=3, fy=3)
            #     saveImg(img_full, self.webdir + '/images/' + '0_full' + '_' + str(name) + '.bmp')
            #     for num in range(data.shape[0]):
            #         img_patch = data[num]
            #         put_img = img_patch.cpu().detach().numpy().squeeze() * 255
            #         put_img = cv2.resize(put_img, None, fx=3, fy=3)
            #         saveImg(put_img, self.webdir + '/images/' + str(num) + '_' + str(name) + '.bmp')

            if results is None:
                results = data
            else:
                results = torch.cat([results, data],dim=0)
            
            # for point in keypoints:
            #     x = int(point[0] + 0.5)
            #     y = int(point[1] + 0.5)
            #     # crop_x = 0 if x-patch_size/2<0 else x-patch_size//2
            #     # crop_y = 0 if y-patch_size/2<0 else y-patch_size//2
            #     crop_x = x
            #     crop_y = y
            #     # print(x, y ,crop_x, crop_y)
            #     patch = img[:,crop_y:crop_y+patch_size,crop_x:crop_x+patch_size]

            #     data = patch.unsqueeze(0)
            #     # print(data.shape)

        
        # compute output for patch a
        results_batch = Variable(results)   
        
        if train_flag:
            outs = self.net(results_batch)
        else:
            with torch.no_grad():
                outs = self.net(results_batch)
        # outs = torch.chunk(outs, b, dim=0)

        return outs

    def run_hardnet_patch(self, img_batch, points_list, theta_batch, patch_size, sample_size, train_flag=False):
        from torch.nn.utils.rnn import pad_sequence
        from Model_component import inv_warp_patch_batch, get_orientation_batch
        b, _, h, w = img_batch.shape
        # points_batch = points_batch
        results = None

        # 点数补齐，方便进行批量操作
        # num = 150
        # pts_mat = pad_sequence(points_list, batch_first=True, padding_value=-2)
        # mask = (pts_mat[:, :, 0] == -2)
        # pts_mat[mask, :] = torch.tensor([0., 0.], dtype=torch.float32)
        # pts_mask = (~mask).view(-1)    # [batch, N]

        # theta_batch = theta_batch.contiguous()
        # theta_batch = theta_batch.unsqueeze(1).repeat(1, pts_mat.shape[1])

        # padding
        fill_size = 44
        # fill_size = (h_square - w) / 2
        img_batch_pad = F.pad(img_batch, (int(fill_size), int(fill_size), int(fill_size), int(fill_size), 0, 0), 'constant')

        points_list[:, 0] += fill_size
        points_list[:, 1] += fill_size

        theta_batch = get_orientation_batch(img_batch_pad, points_list, patch_size=28)

        img_patches = inv_warp_patch_batch(img_batch_pad, points_list.unsqueeze(0), theta_batch, patch_size=patch_size, sample_size=sample_size)
        # img_patches = img_patches[pts_mask]

        if train_flag:
            outs = self.net(img_patches)
        else:
            with torch.no_grad():
                outs = self.net(img_patches)
        # outs = torch.chunk(outs, b, dim=0)

        '''ttt'''
        # img_full = img_batch[0, 0, :, :].cpu().detach().numpy().squeeze() * 255
        # img_full = np.repeat(cv2.resize(img_full, None, fx=3, fy=3)[..., np.newaxis], 3, -1)
        # for num in range(150):
        #     cv2.circle(img_full, tuple((3 * points_list[0][num].numpy()).astype(int)), 3, (0, 255, 0), -1)
        # saveImg(img_full, self.webdir + '/images/' + '0_full' + '_' + str(name) + '.bmp')
        # for num in range(150):
        #     img_patch = img_patches[num]
        #     put_img = img_patch.cpu().detach().numpy().squeeze() * 255
        #     put_img = np.repeat(cv2.resize(put_img, None, fx=3, fy=3)[..., np.newaxis], 3, -1)
        #     saveImg(put_img, self.webdir + '/images/' + str(points_list[0][num][0].numpy()) + '_' + str(points_list[0][num][1].numpy()) + '_' + str(name) + '.bmp')

        return outs

    def output_pts_desc_batch_net(self, img_batch, conf_thresh, nms, cut=150, quantize=True):
        '''
        input: B 1 H W
        批量操作
        '''
        batch_size = 300
        batch_num = img_batch.shape[0] // batch_size + 1
        img_mini_batch_group = torch.chunk(img_batch, batch_num, dim=0)

        pts_batch = torch.tensor([], dtype=torch.float32).to(self.device)
        desc_batch = torch.tensor([], dtype=torch.float32).to(self.device)
        mask_batch = torch.tensor([], dtype=torch.float32).to(self.device)
        for img_mini_batch in img_mini_batch_group:
            with torch.no_grad():
                self.net.eval()
                outs = self.net.forward(img_mini_batch)
                semi, desc = outs['semi'], outs['desc']
                desc = torch.sigmoid(desc)

            heatmap = flattenDetection(semi)
            for i in range(heatmap.shape[0]):
                pts_single = getPtsFromHeatmap(heatmap[i].cpu().numpy(), conf_thresh, nms).transpose(1, 0)
                pts_single = torch.from_numpy(pts_single[:cut, :]).to(self.device)
                desc_single = desc[i, :, pts_single[:, 1].long(), pts_single[:, 0].long()]
                desc_single = desc_single[:, :cut].to(self.device)

                mask_single = torch.cat((torch.ones(pts_single.shape[0]), torch.zeros(cut - pts_single.shape[0])), dim=0).to(self.device)
                if pts_single.shape[0] < cut:
                    pts_single = torch.cat((pts_single, torch.zeros([cut - pts_single.shape[0], pts_single.shape[1]]).to(self.device)), dim=0)
                    desc_single = torch.cat((desc_single, torch.zeros([desc_single.shape[0], cut - desc_single.shape[1]]).to(self.device)), dim=1)
                
                if quantize == True:
                    desc_single = torch.where(desc_single > 0.5, torch.ones_like(desc_single), torch.zeros_like(desc_single))
                pts_batch = torch.cat((pts_batch, pts_single.unsqueeze(0)), dim=0)
                desc_batch = torch.cat((desc_batch, desc_single.transpose(1, 0).unsqueeze(0)), dim=0)
                mask_batch = torch.cat((mask_batch, mask_single.unsqueeze(0)), dim=0)

            # pts = [getPtsFromHeatmap(copy.deepcopy(heatmap[i].cpu().numpy()), conf_thresh, nms) for i in range(heatmap.shape[0])]
            # pts_torch = [getPtsFromHeatmap_torch(heatmap[i], conf_thresh, nms, device=self.device) for i in range(heatmap.shape[0])]
            # pts_list.extend(pts)
            # netout_heatmap_batch, netout_desc_batch = FPDT.run_heatmap(img_batch)
            # netout_desc_batch = torch.sigmoid(netout_desc_batch)
        
        return pts_batch, desc_batch, mask_batch.bool()

    def output_pts_desc_batch_netV2(self, img_batch, pts_list, theta_batch, patch_size=16, sample_size=22, fill_size=24, trans=None, cut=150, quantize=True, is256=True):
        '''
        切patch方式网络
        input: B 1 H W
        批量操作
        '''
        from Model_component import inv_warp_patch_batch, get_orientation_batch
        ho, wo = 122, 36    # 坐标基准
        b, _, h, w = img_batch.shape    # [b, 1, 128, 52]

        points_list = copy.deepcopy(pts_list)

        fill_size_w = 16  # 122x36 -pad24-> 170x84 - 128x52 -> 42x32 ->need pad 21x16
        fill_size_h = 21
        pad_group = (fill_size_w, fill_size_w, fill_size_h, fill_size_h)
        img_batch_pad = F.pad(img_batch, pad_group, 'constant') # 170x84

        points_list[:, :, 0] += (fill_size_w + 8)       # 122x36->128x52->170x84
        points_list[:, :, 1] += (fill_size_h + 3)
        
        # theta_batch = get_orientation_batch(img_batch_pad, points_list, patch_size=patch_size)
        # angle45 = theta_batch - 45.
        img_patches = inv_warp_patch_batch(img_batch_pad, points_list[:, :, :2], theta_batch, patch_size=patch_size, sample_size=sample_size)
        # img_patches_45 = inv_warp_patch_batch(img_batch_pad, points_list[:, :, :2], angle45, patch_size=patch_size, sample_size=sample_size)

        batch_size = 20000 # 174000
        # if batch_size > img_patches.shape[0]:
        #     batch_size = img_patches.shape[0]
        batch_num = img_patches.shape[0] // batch_size + 1
        img_mini_batch_group = torch.chunk(img_patches, batch_num, dim=0)
        # img_mini_batch_group_45 = torch.chunk(img_patches_45, batch_num, dim=0)

        desc_batch = torch.tensor([], dtype=torch.float32).to(self.device)
        # desc_batch_45 = torch.tensor([], dtype=torch.float32).to(self.device)
        # for (img_mini_batch, img_mini_batch_45) in zip(img_mini_batch_group, img_mini_batch_group_45):
        for img_mini_batch in img_mini_batch_group:
            with torch.no_grad():
                outs = self.net(img_mini_batch) # [N, 128]
                # outs_45 = self.net(img_mini_batch_45)
                # outs_norm = L2Norm(outs)
            desc_batch = torch.cat((desc_batch, outs), dim=0)
            # desc_batch_45 = torch.cat((desc_batch_45, outs_45), dim=0)

        if is256 == True:
            desc_front = desc_batch[:, :128]
            desc_back = desc_batch[:, 128:]
        else:
            desc_front = desc_batch
            desc_back = desc_batch

        if quantize == True:
            # desc_mini = torch.where(outs > 0.5, torch.ones_like(outs), torch.zeros_like(outs))
            assert desc_front.shape[-1] == 128, "desc is not 128!"
            desc_batch_int = torch.round(desc_front * 5000) + 5000
            desc_batch_thre = self.thresholding_desc(desc_batch_int)    # 门限化: base on 128-dim
            # self.get_des_hanmingdis_wht_permute(outs_thre, outs_thre)
            Hada = hadamard(desc_batch_thre.shape[1])
            Hada_T = desc_batch_thre.float() @ torch.from_numpy(Hada).float().to(self.device)
            desc_f = (Hada_T.long() > 0).float().to(self.device)
            if is256 == True:
                desc_back_int_45 = torch.round(desc_back * 5000) + 5000
                desc_back_thre_45 = self.thresholding_desc(desc_back_int_45)    # 门限化: base on 128-dim
                Hada_b_T = desc_back_thre_45.float() @ torch.from_numpy(Hada).float().to(self.device)
                desc_b = (Hada_b_T.long() > 0).float().to(self.device)
                # desc_b = copy.deepcopy(desc_f)
        
        if is256 == True:
            desc_out = torch.cat((desc_front, desc_back), dim=-1).contiguous().view(b, points_list.shape[1], -1)
            desc_hadama = torch.cat((desc_f, desc_b), dim=-1).contiguous().view(b, points_list.shape[1], -1)
        else:
            desc_out = torch.cat((desc_batch, desc_batch), dim=-1).contiguous().view(b, points_list.shape[1], -1)
            desc_hadama = torch.cat((desc_f, desc_f), dim=-1).contiguous().view(b, points_list.shape[1], -1)

        
        return desc_out, _, desc_hadama, theta_batch
    
    def output_pts_desc_batch_netV2_rect(self, img_batch, pts_list, theta_batch, patch_size=[32, 8], sample_size=[32, 8], trans=None, quantize=True, is256=True):
        '''
        切patch方式网络(长条形32x8) 0629
        input: B 1 H W
        批量操作
        '''
        from Model_component import inv_warp_patch_batch_rect, get_orientation_batch
        ho, wo = 122, 36    # 坐标基准
        b, _, h, w = img_batch.shape    # [b, 1, 128, 52]

        points_list = copy.deepcopy(pts_list)

        fill_size_w = 10  # 122x36 -pad18-> 158x72 - 128x52 -> 30x20 ->need pad 15x10
        fill_size_h = 15
        pad_group = (fill_size_w, fill_size_w, fill_size_h, fill_size_h)
        img_batch_pad = F.pad(img_batch, pad_group, 'constant') # 158x72

        points_list[:, :, 0] += (fill_size_w + 8)       # 122x36->128x52->158x72
        points_list[:, :, 1] += (fill_size_h + 3)
        
        # theta_batch = get_orientation_batch(img_batch_pad, points_list, patch_size=patch_size)
        # angle45 = theta_batch - 45.
        img_patches = inv_warp_patch_batch_rect(img_batch_pad, points_list[:, :, :2], theta_batch, patch_size=patch_size, sample_size=sample_size, mode='bilinear')
        # img_patches_45 = inv_warp_patch_batch(img_batch_pad, points_list[:, :, :2], angle45, patch_size=patch_size, sample_size=sample_size)

        batch_size = 20000 # 174000
        # if batch_size > img_patches.shape[0]:
        #     batch_size = img_patches.shape[0]
        batch_num = img_patches.shape[0] // batch_size + 1
        img_mini_batch_group = torch.chunk(img_patches, batch_num, dim=0)
        # img_mini_batch_group_45 = torch.chunk(img_patches_45, batch_num, dim=0)

        desc_batch = torch.tensor([], dtype=torch.float32).to(self.device)
        # desc_batch_45 = torch.tensor([], dtype=torch.float32).to(self.device)
        # for (img_mini_batch, img_mini_batch_45) in zip(img_mini_batch_group, img_mini_batch_group_45):
        for img_mini_batch in img_mini_batch_group:
            with torch.no_grad():
                outs, _ = self.net(img_mini_batch) # [N, 128]
                # outs_45 = self.net(img_mini_batch_45)
                # outs_norm = L2Norm(outs)
            desc_batch = torch.cat((desc_batch, outs), dim=0)
            # desc_batch_45 = torch.cat((desc_batch_45, outs_45), dim=0)

        if is256 == True:
            desc_front = desc_batch[:, :128]
            desc_back = desc_batch[:, 128:]
        else:
            desc_front = desc_batch[:, :128]
            desc_back = desc_batch[:, :128]

        if quantize == True:
            # desc_mini = torch.where(outs > 0.5, torch.ones_like(outs), torch.zeros_like(outs))
            assert desc_front.shape[-1] == 128, "desc is not 128!"
            desc_batch_int = torch.round(desc_front * 5000) + 5000
            desc_batch_thre = self.thresholding_desc(desc_batch_int)    # 门限化: base on 128-dim
            # self.get_des_hanmingdis_wht_permute(outs_thre, outs_thre)
            Hada = hadamard(desc_batch_thre.shape[1])
            Hada_T = desc_batch_thre.float() @ torch.from_numpy(Hada).float().to(self.device)
            desc_f = (Hada_T.long() > 0).float().to(self.device)
            if is256 == True:
                desc_back_int_45 = torch.round(desc_back * 5000) + 5000
                desc_back_thre_45 = self.thresholding_desc(desc_back_int_45)    # 门限化: base on 128-dim
                Hada_b_T = desc_back_thre_45.float() @ torch.from_numpy(Hada).float().to(self.device)
                desc_b = (Hada_b_T.long() > 0).float().to(self.device)
                # desc_b = copy.deepcopy(desc_f)
        
        if is256 == True:
            desc_out = torch.cat((desc_front, desc_back), dim=-1).contiguous().view(b, points_list.shape[1], -1)
            desc_hadama = torch.cat((desc_f, desc_b), dim=-1).contiguous().view(b, points_list.shape[1], -1)
        else:
            desc_out = torch.cat((desc_front, desc_front), dim=-1).contiguous().view(b, points_list.shape[1], -1)
            desc_hadama = torch.cat((desc_f, desc_f), dim=-1).contiguous().view(b, points_list.shape[1], -1)

        
        return desc_out, _, desc_hadama, theta_batch

    def output_e2cnn_forward(self, img_batch, pts_list, theta_batch, patch_size=16, sample_size=32, fill_size=24, trans=None, cut=150, quantize=True):
        '''
        切patch方式网络
        input: B 1 H W
        批量操作
        '''
        from Model_component import inv_warp_patch_batch, get_orientation_batch

        b, _, h, w = img_batch.shape
        points_list = copy.deepcopy(pts_list)
        # fill_size = 44
        # fill_size = (h_square - w) / 2
        ds = 4 # down sample
        fill_size_h = 8
        fill_size_w = fill_size_h + 2
        img_batch_pad = F.pad(img_batch, (fill_size_w, fill_size_w, fill_size_h, fill_size_h, 0, 0), 'constant')

        _, _, uh, uw = img_batch_pad.shape
        mesh_point_big = torch.stack(torch.meshgrid([torch.linspace(0, w - 1, w), torch.linspace(0, h - 1, h)])).to(self.device).view(2, -1).transpose(1, 0).repeat(b, 1, 1)
        mesh_point_big[:, :, 0] += fill_size_w
        mesh_point_big[:, :, 1] += fill_size_h

        points_list[:, :, 0] += fill_size_w    
        points_list[:, :, 1] += fill_size_h
        pts_norm = 2 * points_list / torch.tensor([uw - 1, uh - 1]).to(self.device) - 1
        # theta_batch = get_orientation_batch(img_batch_pad, points_list, patch_size=patch_size)
        theta_batch_mesh = get_orientation_batch(img_batch_pad, mesh_point_big, patch_size=16)
        theta_batch = F.grid_sample(theta_batch_mesh.view(b, 1, h, w), pts_norm.unsqueeze(1), mode='bilinear', align_corners=True).squeeze()

        fix_angle = torch.zeros_like(theta_batch)
        if trans is not None:
            import math
            from skimage.transform import AffineTransform
            mat = AffineTransform(matrix=trans.cpu().numpy())
            angle = mat.rotation * 180 / math.pi
            fix_angle = self.fix_orient_by_trans_flag(theta_batch, angle)
        # angles_90 = theta_batch + torch.tensor([90], dtype=torch.float32).to(self.device)
        # img_patches = inv_warp_patch_batch(img_batch_pad, points_list, theta_batch + fix_angle, patch_size=patch_size, sample_size=sample_size)

        batch_size = 20000 # 174000
        # if batch_size > img_patches.shape[0]:
        #     batch_size = img_patches.shape[0]
        batch_num = img_batch_pad.shape[0] // batch_size + 1
        img_mini_batch_group = torch.chunk(img_batch_pad, batch_num, dim=0)

        desc_batch = torch.tensor([], dtype=torch.float32).to(self.device)
        desc_batch_o = torch.tensor([], dtype=torch.float32).to(self.device)
        for img_mini_batch in img_mini_batch_group:
            with torch.no_grad():
                outs = self.net(img_mini_batch)
                outs_o = F.softmax(outs, dim=1)
            
            desc_batch = torch.cat((desc_batch, outs), dim=0)
            desc_batch_o = torch.cat((desc_batch_o, outs_o), dim=0)

        ds_pad = 2
        outs_pad = F.pad(desc_batch, (ds_pad, ds_pad, ds_pad, ds_pad), 'constant')    # 42x18

        '''mesh_point & pts in feature map 40x16'''
        # mesh_point = torch.stack(torch.meshgrid([torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w)])).view(2, -1).transpose(1, 0)    # [n, [y, x]]
        mesh_point_ds = mesh_point_big / ds + ds_pad  # coor after downsample    [bs, N, [x, y]]
        pts_ds = points_list / ds + ds_pad

        '''grid sample base on [bs, 8, 38, 14]'''
        mesh_point_ds_norm = 2 * (mesh_point_big / ds) / torch.tensor([outs_o.shape[3] - 1, outs_o.shape[2] - 1]).to(self.device) - 1
        featmap_mesh_o = F.grid_sample(outs_o, mesh_point_ds_norm.unsqueeze(1), mode='bilinear', align_corners=True).squeeze()
        featmap_mesh_o = featmap_mesh_o.view(b, outs_o.shape[1], h, w)

        # random_flip = random.choice([True, False])
        angle = theta_batch
        # angle = torch.zeros_like(theta_batch_pts, device=theta_batch_pts.device)
        featmap_patches = inv_warp_patch_batch(outs_pad.contiguous().view(-1, 1, outs_pad.shape[2], outs_pad.shape[3]), pts_ds.unsqueeze(1).repeat(1, 8, 1, 1).view(-1, points_list.shape[1], 2), angle.unsqueeze(1).repeat(1, 8, 1).view(-1, points_list.shape[1]), patch_size=patch_size, sample_size=sample_size)
        featmap_patches = featmap_patches.view(b, 8, points_list.shape[1], 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 16, 8)
        featmap_patches = featmap_patches.view(-1, 128)
        featmap_patches = L2Norm(featmap_patches)
        
        if 0:
            angle_mesh = theta_batch_mesh
            featmap_patches_mesh = inv_warp_patch_batch(outs_pad.contiguous().view(-1, 1, outs_pad.shape[2], outs_pad.shape[3]), mesh_point_ds.unsqueeze(1).repeat(1, 8, 1, 1).view(-1, mesh_point_ds.shape[1], 2), angle_mesh.unsqueeze(1).repeat(1, 8, 1).view(-1, mesh_point_ds.shape[1]), patch_size=patch_size, sample_size=sample_size)
            featmap_patches_mesh = featmap_patches_mesh.view(b, 8, mesh_point_ds.shape[1], 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 16, 8)
            
            featmap_patches_mesh = featmap_patches_mesh.view(-1, 128)
            featmap_patches_mesh = L2Norm(featmap_patches_mesh).contiguous().view(b, h * w, -1).transpose(1, 2).view(b, -1, h, w)

        if quantize == True:
            # desc_mini = torch.where(outs > 0.5, torch.ones_like(outs), torch.zeros_like(outs))
            f_p = torch.round(featmap_patches.cpu() * 10000) + 5000
            f_p_thre = self.thresholding_desc(f_p)    # 门限化: base on 128-dim
            # self.get_des_hanmingdis_wht_permute(outs_thre, outs_thre)
            Hada = hadamard(f_p_thre.shape[1])
            Hada_T = f_p_thre.long() @ torch.from_numpy(Hada)
            f_p_hadama = (Hada_T > 0).long().float().to(self.device)

            if 0:
                f_p_mesh = torch.round(featmap_patches_mesh.cpu() * 10000) + 5000
                f_p_mesh_thre = self.thresholding_desc(f_p_mesh.permute(0, 2, 3, 1).view(-1, 128))    # 门限化: base on 128-dim
                # self.get_des_hanmingdis_wht_permute(outs_thre, outs_thre)
                Hada = hadamard(f_p_mesh_thre.shape[1])
                Hada_T = f_p_mesh_thre.long() @ torch.from_numpy(Hada)
                # f_p_mesh_hadama = (Hada_T > 0).long().float().to(self.device).view(1, h, w, 128).permute(0, 3, 1, 2)
                f_p_mesh_hadama = (Hada_T > 0).long().float().to(self.device)

        featmap_patches = featmap_patches.contiguous().view(points_list.shape[0], points_list.shape[1], -1)
        desc_batch = torch.cat((featmap_patches, featmap_patches), dim=-1).view(points_list.shape[0], points_list.shape[1], -1)
        f_p_hadama = torch.cat((f_p_hadama, f_p_hadama), dim=-1).view(points_list.shape[0], points_list.shape[1], -1)
        if 0:
            f_p_mesh_hadama = torch.cat((f_p_mesh_hadama, f_p_mesh_hadama), dim=-1)
        
        # return desc_batch, featmap_patches_mesh, f_p_hadama, f_p_mesh_hadama
        return desc_batch, _, f_p_hadama, _
    
    def output_e2cnn_forward_V2(self, img_batch, pts_list, theta_batch, patch_size=4, sample_size=4, fill_size=24, trans=None, cut=150, quantize=True):
        '''
        密集描述子网络
        input: B 1 H W
        pts_list: B,150,2  已做过点数补齐 
        批量操作
        '''
        from Model_component import inv_warp_patch_batch, get_orientation_batch

        b, _, h, w = img_batch.shape
        points_list = copy.deepcopy(pts_list)
        # fill_size = 44
        # fill_size = (h_square - w) / 2
        ds = 4 # down sample
        fill_size_h = 8
        fill_size_w = fill_size_h + 2
        img_batch_pad = F.pad(img_batch, (fill_size_w, fill_size_w, fill_size_h, fill_size_h, 0, 0), 'constant')

        '''test'''
        # img_patch_pad_fliped = torch.flip(torch.flip(img_batch_pad, dims=[3]), dims=[2])

        mesh_point_big = torch.stack(torch.meshgrid([torch.linspace(0, w - 1, w), torch.linspace(0, h - 1, h)])).to(self.device).permute(0, 2, 1).contiguous().view(2, -1).transpose(1, 0).repeat(b, 1, 1)

        pts_norm = 2 * points_list / torch.tensor([w - 1, h - 1]).to(self.device) - 1
        theta_batch_mesh = get_orientation_batch(img_batch, mesh_point_big, patch_size=16)
        theta_batch_pts = F.grid_sample(theta_batch_mesh.view(b, 1, h, w), pts_norm.unsqueeze(1), mode='bilinear', align_corners=True).squeeze()

        fix_angle = torch.zeros_like(theta_batch)
        # angles_90 = theta_batch + torch.tensor([90], dtype=torch.float32).to(self.device)
        # img_patches = inv_warp_patch_batch(img_batch_pad, points_list, theta_batch + fix_angle, patch_size=patch_size, sample_size=sample_size)

        batch_size = 2000 # 174000 密集型站空间较大，可用400
        # if batch_size > img_patches.shape[0]:
        #     batch_size = img_patches.shape[0]
        batch_num = img_batch_pad.shape[0] // batch_size + 1
        img_mini_batch_group = torch.chunk(img_batch_pad, batch_num, dim=0)
        # img_mini_batch_group_fliped = torch.chunk(img_patch_pad_fliped, batch_num, dim=0)

        desc_batch = torch.tensor([], dtype=torch.float32).to(self.device)
        # desc_batch_fliped = torch.tensor([], dtype=torch.float32).to(self.device)
        
        for img_mini_batch in img_mini_batch_group:
        # for img_mini_batch, img_mini_batch_fliped in zip(img_mini_batch_group,img_mini_batch_group_fliped):
            with torch.no_grad():
                outs = self.net(img_mini_batch)
                # outs_fliped = self.net(img_mini_batch_fliped)  # [bs, 8, 38, 14]
            desc_batch = torch.cat((desc_batch, outs), dim=0)
            # desc_batch_fliped = torch.cat((desc_batch_fliped, outs_fliped), dim=0)
        

        '''test'''
        # print(outs[0, :, 17, 5], outs_fliped[0, :, 20, 8])
        # print(outs[0, :, 5, 9], outs_fliped[0, :, 32, 4])

        '''mesh_point & pts in feature map 42x18'''
        ds_pad = 2
        outs_pad = F.pad(desc_batch, (ds_pad, ds_pad, ds_pad, ds_pad), 'constant')    # 42x18
        offset = torch.tensor([fill_size_w, fill_size_h], device=self.device).reshape(1, 1, -1)
        pts_ds = (points_list + offset) / ds + ds_pad

        angle = theta_batch_pts
        # angle = torch.zeros_like(theta_batch)
        featmap_patches = inv_warp_patch_batch(outs_pad.contiguous().view(-1, 1, outs_pad.shape[2], outs_pad.shape[3]), pts_ds.unsqueeze(1).repeat(1, 8, 1, 1).view(-1, points_list.shape[1], 2), angle.unsqueeze(1).repeat(1, 8, 1).view(-1, points_list.shape[1]), patch_size=patch_size, sample_size=sample_size)
        featmap_patches = featmap_patches.view(b, 8, points_list.shape[1], 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 16, 8)
        featmap_patches = featmap_patches.view(-1, 128)
        featmap_patches = L2Norm(featmap_patches)

        if quantize == True:
            # desc_mini = torch.where(outs > 0.5, torch.ones_like(outs), torch.zeros_like(outs))
            f_p = torch.round(featmap_patches.cpu() * 10000) + 5000
            f_p_thre = self.thresholding_desc(f_p)    # 门限化: base on 128-dim
            # self.get_des_hanmingdis_wht_permute(outs_thre, outs_thre)
            Hada = hadamard(f_p_thre.shape[1])
            Hada_T = f_p_thre.long() @ torch.from_numpy(Hada)
            f_p_hadama = (Hada_T > 0).long().float().to(self.device)

        # featmap_patches = featmap_patches.contiguous().view(points_list.shape[0], points_list.shape[1], -1)
        desc_batch = torch.cat((featmap_patches, featmap_patches), dim=-1).view(points_list.shape[0], points_list.shape[1], -1)
        f_p_hadama = torch.cat((f_p_hadama, f_p_hadama), dim=-1).view(points_list.shape[0], points_list.shape[1], -1)
        
        '''grid sample base on [bs, 8, 38, 14]'''
        # mesh_point_ds_norm = 2 * (mesh_point_big / ds) / torch.tensor([outs_o.shape[3] - 1, outs_o.shape[2] - 1]).to(self.device) - 1
        # featmap_mesh_o = F.grid_sample(outs_o, mesh_point_ds_norm.unsqueeze(1), mode='bilinear', align_corners=True).squeeze()
        # featmap_mesh_o = featmap_mesh_o.view(b, outs_o.shape[1], h, w)

        # return desc_batch, featmap_patches_mesh, f_p_hadama, f_p_mesh_hadama
        return desc_batch, _, f_p_hadama, _

    def output_e2cnn_forward_V2_resize(self, img_batch, pts_list, theta_batch, patch_size=4, sample_size=4, fill_size=24, trans=None, cut=150, quantize=True):
        '''
        密集描述子网络,输入图像做resize处理  137x37 -> 137x41 -> (137+8x2)x(41+8x2) = 153x57
        input: B 1 H W
        pts_list: B,150,2  已做过点数补齐
        批量操作
        '''
        import torchvision.transforms as transforms
        from Model_component import inv_warp_patch_batch, get_orientation_batch

        b, _, h, w = img_batch.shape
        points_list = copy.deepcopy(pts_list)

        tr_Resize = transforms.Resize((137, 37))
        img_batch_resize = tr_Resize(img_batch)

        ds = 4 # down sample
        fill_size_h = 8
        fill_size_w = fill_size_h + 2
        img_batch_pad = F.pad(img_batch_resize, (fill_size_w, fill_size_w, fill_size_h, fill_size_h, 0, 0), 'constant')

        '''test'''
        # img_patch_pad_fliped = torch.flip(torch.flip(img_batch_pad, dims=[3]), dims=[2])

        mesh_point_big = torch.stack(torch.meshgrid([torch.linspace(0, w - 1, w), torch.linspace(0, h - 1, h)])).to(self.device).permute(0, 2, 1).contiguous().view(2, -1).transpose(1, 0).repeat(b, 1, 1)

        pts_norm = 2 * points_list / torch.tensor([w - 1, h - 1]).to(self.device) - 1   # [136, 36]->[-1, 1]
        theta_batch_mesh = get_orientation_batch(img_batch, mesh_point_big, patch_size=16)
        theta_batch_pts = F.grid_sample(theta_batch_mesh.view(b, 1, h, w), pts_norm.unsqueeze(1), mode='bilinear', align_corners=True).squeeze()

        fix_angle = torch.zeros_like(theta_batch)
        # angles_90 = theta_batch + torch.tensor([90], dtype=torch.float32).to(self.device)
        # img_patches = inv_warp_patch_batch(img_batch_pad, points_list, theta_batch + fix_angle, patch_size=patch_size, sample_size=sample_size)

        batch_size = 2000 # 174000 密集型站空间较大，可用400
        # if batch_size > img_patches.shape[0]:
        #     batch_size = img_patches.shape[0]
        batch_num = img_batch_pad.shape[0] // batch_size + 1
        img_mini_batch_group = torch.chunk(img_batch_pad, batch_num, dim=0)
        # img_mini_batch_group_fliped = torch.chunk(img_patch_pad_fliped, batch_num, dim=0)

        desc_batch = torch.tensor([], dtype=torch.float32).to(self.device)
        # desc_batch_fliped = torch.tensor([], dtype=torch.float32).to(self.device)
        
        for img_mini_batch in img_mini_batch_group:
        # for img_mini_batch, img_mini_batch_fliped in zip(img_mini_batch_group,img_mini_batch_group_fliped):
            with torch.no_grad():
                outs = self.net(img_mini_batch)     # [bs, 8, 39, 15]
                # outs_fliped = self.net(img_mini_batch_fliped)  
            desc_batch = torch.cat((desc_batch, outs), dim=0)
            # desc_batch_fliped = torch.cat((desc_batch_fliped, outs_fliped), dim=0)
        
        '''test'''
        # print(outs[0, :, 17, 5], outs_fliped[0, :, 20, 8])
        # print(outs[0, :, 5, 9], outs_fliped[0, :, 32, 4])

        '''mesh_point & pts in feature map 43x19'''
        ds_pad = 2
        outs_pad = F.pad(desc_batch, (ds_pad, ds_pad, ds_pad, ds_pad), 'constant')    # 43x19
        offset = torch.tensor([fill_size_w, fill_size_h], device=self.device).reshape(1, 1, -1)
        points_list_resize = points_list * points_list.new_tensor([w / (w - 1), h / (h - 1)])
        pts_ds = (points_list_resize + offset) / ds + ds_pad

        angle = theta_batch_pts
        # angle = torch.zeros_like(theta_batch)
        featmap_patches = inv_warp_patch_batch(outs_pad.contiguous().view(-1, 1, outs_pad.shape[2], outs_pad.shape[3]), pts_ds.unsqueeze(1).repeat(1, 8, 1, 1).view(-1, points_list.shape[1], 2), angle.unsqueeze(1).repeat(1, 8, 1).view(-1, points_list.shape[1]), patch_size=patch_size, sample_size=sample_size)
        featmap_patches = featmap_patches.view(b, 8, points_list.shape[1], 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 16, 8)
        featmap_patches = featmap_patches.view(-1, 128)
        featmap_patches = L2Norm(featmap_patches)

        if quantize == True:
            # desc_mini = torch.where(outs > 0.5, torch.ones_like(outs), torch.zeros_like(outs))
            f_p = torch.round(featmap_patches.cpu() * 10000) + 5000
            f_p_thre = self.thresholding_desc(f_p)    # 门限化: base on 128-dim
            # self.get_des_hanmingdis_wht_permute(outs_thre, outs_thre)
            Hada = hadamard(f_p_thre.shape[1])
            Hada_T = f_p_thre.long() @ torch.from_numpy(Hada)
            f_p_hadama = (Hada_T > 0).long().float().to(self.device)

        # featmap_patches = featmap_patches.contiguous().view(points_list.shape[0], points_list.shape[1], -1)
        desc_batch = torch.cat((featmap_patches, featmap_patches), dim=-1).view(points_list.shape[0], points_list.shape[1], -1)
        f_p_hadama = torch.cat((f_p_hadama, f_p_hadama), dim=-1).view(points_list.shape[0], points_list.shape[1], -1)
        
        '''grid sample base on [bs, 8, 38, 14]'''
        # mesh_point_ds_norm = 2 * (mesh_point_big / ds) / torch.tensor([outs_o.shape[3] - 1, outs_o.shape[2] - 1]).to(self.device) - 1
        # featmap_mesh_o = F.grid_sample(outs_o, mesh_point_ds_norm.unsqueeze(1), mode='bilinear', align_corners=True).squeeze()
        # featmap_mesh_o = featmap_mesh_o.view(b, outs_o.shape[1], h, w)

        # return desc_batch, featmap_patches_mesh, f_p_hadama, f_p_mesh_hadama
        return desc_batch, _, f_p_hadama, _

    def output_e2cnn_forward_V2_resize_frandmid_pad(self, img_batch, pts_list, theta_batch, patch_size=4, sample_size=4, fill_size=24, trans=None, cut=150, quantize=True):
        '''
        密集描述子网络,输入图像做resize处理
        front_pad_old: 136x36 =pad=> 144x52 =resize=> 145x53 (odd) =pad12=> 169x77 =ds=> 85x39 =ds=> 43x20
        front_pad_new: 136x36 =pad=> 144x52 =resize=> 145x53 (odd) =pad(8+4)=> 161x61 =ds=> 81x31 =ds=> 41x16
        input: B 1 H W
        pts_list: belong to 144x52  [B,150,2]  已做补齐
        批量操作
        '''
        import torchvision.transforms as transforms
        from Model_component import inv_warp_patch_batch, get_orientation_batch
        ho, wo = 136, 36    # 坐标基准
        b, _, h, w = img_batch.shape    # [b, 1, 144, 52]
        '''还原到136x36'''
        points_list = copy.deepcopy(pts_list)
        points_list -= points_list.new_tensor([(w - wo) // 2, (h - ho) // 2])

        tr_Resize = transforms.Resize((145, 53))
        img_batch_resize = tr_Resize(img_batch)

        mid_pad = True
        ds = 4 # down sample
        if mid_pad:
            '''中置pad'''
            fill_size_h = 0
            fill_size_w = 0
        else:
            fill_size_h = 8     # (144x52)-> 160x60
            fill_size_w = 4
        pad_group = (fill_size_w, fill_size_w, fill_size_h, fill_size_h)    # base:145x53 -> (145+8x2)x(53+4x2) = 161x61
        img_batch_pad = F.pad(img_batch_resize, pad_group, 'constant')

        '''test'''
        # img_patch_pad_fliped = torch.flip(torch.flip(img_batch_pad, dims=[3]), dims=[2])

        pts_norm = 2 * points_list / points_list.new_tensor([wo - 1, ho - 1]) - 1   # [136, 36]->[-1, 1]
        mesh_point_big = torch.stack(torch.meshgrid([torch.linspace(0, wo - 1, wo), torch.linspace(0, ho - 1, ho)])).to(self.device).permute(0, 2, 1).contiguous().view(2, -1).transpose(1, 0).repeat(b, 1, 1)  # 136x36
        mesh_point_big += mesh_point_big.new_tensor([(w - wo) // 2, (h - ho) // 2])     # 136x36 -offset-> 144x52
        # mesh_point_big = mesh_point_big * mesh_point_big.new_tensor([(wo - 1) / (w - 1), (ho - 1) / ( h - 1)]) + mesh_point_big.new_tensor([(w - wo) // 2, (h - ho) // 2])   # 对齐到144x52: 144x52 resize to 136x36,then add to 144x52 : offset[4, 8]
        theta_batch_mesh = get_orientation_batch(img_batch, mesh_point_big, patch_size=16)
        theta_batch_pts = F.grid_sample(theta_batch_mesh.view(b, 1, ho, wo), pts_norm.unsqueeze(1), mode='bilinear', align_corners=True).squeeze()

        batch_size = 2000 # 174000 密集型站空间较大，可用400
        # if batch_size > img_patches.shape[0]:
        #     batch_size = img_patches.shape[0]
        batch_num = img_batch_pad.shape[0] // batch_size + 1
        img_mini_batch_group = torch.chunk(img_batch_pad, batch_num, dim=0)
        # img_mini_batch_group_fliped = torch.chunk(img_patch_pad_fliped, batch_num, dim=0)

        desc_batch = torch.tensor([], dtype=torch.float32).to(self.device)
        # desc_batch_fliped = torch.tensor([], dtype=torch.float32).to(self.device)
        
        for img_mini_batch in img_mini_batch_group:
        # for img_mini_batch, img_mini_batch_fliped in zip(img_mini_batch_group,img_mini_batch_group_fliped):
            with torch.no_grad():
                outs = self.net(img_mini_batch)     # [bs, 1, 161, 61]->[bs, 8, 41, 16]
                # outs_fliped = self.net(img_mini_batch_fliped)  
            desc_batch = torch.cat((desc_batch, outs), dim=0)
            # desc_batch_fliped = torch.cat((desc_batch_fliped, outs_fliped), dim=0)
        
        '''test'''
        # print(outs[0, :, 17, 5], outs_fliped[0, :, 20, 8])
        # print(outs[0, :, 5, 9], outs_fliped[0, :, 32, 4])

        '''pts in feature map 41x16'''
        ds_pad = 0
        outs_pad = F.pad(desc_batch, (ds_pad, ds_pad, ds_pad, ds_pad), 'constant')    # 41x16
        offset = torch.tensor([fill_size_w, fill_size_h], device=self.device).reshape(1, 1, -1)
        points_list_resize = (points_list + points_list.new_tensor([(w - wo) // 2, (h - ho) // 2]).reshape(1, 1, -1)) * points_list.new_tensor([w / (w - 1), h / (h - 1)])
        if mid_pad:
            pts_ds = points_list_resize / ds + torch.tensor([1., 2.]).to(self.device) + ds_pad
        else:
            pts_ds = (points_list_resize + offset) / ds + ds_pad

        angle = theta_batch_pts
        # angle = torch.zeros_like(theta_batch)
        featmap_patches = inv_warp_patch_batch(outs_pad.contiguous().view(-1, 1, outs_pad.shape[2], outs_pad.shape[3]), pts_ds.unsqueeze(1).repeat(1, 8, 1, 1).view(-1, points_list.shape[1], 2), angle.unsqueeze(1).repeat(1, 8, 1).view(-1, points_list.shape[1]), patch_size=patch_size, sample_size=sample_size)
        featmap_patches = featmap_patches.view(b, 8, points_list.shape[1], 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 16, 8)
        featmap_patches = featmap_patches.view(-1, 128)
        featmap_patches = L2Norm(featmap_patches)

        if quantize == True:
            # desc_mini = torch.where(outs > 0.5, torch.ones_like(outs), torch.zeros_like(outs))
            f_p = torch.round(featmap_patches.cpu() * 10000) + 5000
            f_p_thre = self.thresholding_desc(f_p)    # 门限化: base on 128-dim
            # self.get_des_hanmingdis_wht_permute(outs_thre, outs_thre)
            Hada = hadamard(f_p_thre.shape[1])
            Hada_T = f_p_thre.long() @ torch.from_numpy(Hada)
            f_p_hadama = (Hada_T > 0).long().float().to(self.device)

        # featmap_patches = featmap_patches.contiguous().view(points_list.shape[0], points_list.shape[1], -1)
        desc_batch = torch.cat((featmap_patches, featmap_patches), dim=-1).view(points_list.shape[0], points_list.shape[1], -1)
        f_p_hadama = torch.cat((f_p_hadama, f_p_hadama), dim=-1).view(points_list.shape[0], points_list.shape[1], -1)

        # return desc_batch, featmap_patches_mesh, f_p_hadama, f_p_mesh_hadama
        return desc_batch, _, f_p_hadama, _
    
    def output_e2cnn_forward_V2_resize_frandmid_pad_6193(self, img_batch, pts_list, theta_batch, patch_size=4, sample_size=4, fill_size=24, trans=None, cut=150, quantize=True, is256=True):
        '''
        密集描述子网络,输入图像做resize处理
        front_pad_old: 136x36 =pad=> 144x52 =resize=> 145x53 (odd) =pad12=> 169x77 =ds=> 85x39 =ds=> 43x20
        front_pad_new: 136x36 =pad=> 144x52 =resize=> 145x53 (odd) =pad(8+4)=> 161x61 =ds=> 81x31 =ds=> 41x16
        input: B 1 H W
        pts_list: belong to 144x52  [B,150,2]  已做补齐
        批量操作
        '''
        import torchvision.transforms as transforms
        from Model_component import inv_warp_patch_batch, get_orientation_batch
        ho, wo = 122, 36    # 坐标基准
        b, _, h, w = img_batch.shape    # [b, 1, 128, 52]
        points_list = copy.deepcopy(pts_list)
        # '''还原到136x36'''
        # points_list -= points_list.new_tensor([(w - wo) // 2, (h - ho) // 2])

        tr_Resize = transforms.Resize((129, 53))    # 128x52  -> 129x53
        img_batch_resize = tr_Resize(img_batch)

        mid_pad = True
        ds = 4 # down sample
        if mid_pad:
            '''中置pad'''
            fill_size_h = 0
            fill_size_w = 0
        else:
            fill_size_h = 8     # (144x52)-> 160x60
            fill_size_w = 4
        pad_group = (fill_size_w, fill_size_w, fill_size_h, fill_size_h)    # base:128x52
        img_batch_pad = F.pad(img_batch_resize, pad_group, 'constant')

        '''test'''
        # img_patch_pad_fliped = torch.flip(torch.flip(img_batch_pad, dims=[3]), dims=[2])

        pts_norm = 2 * points_list / points_list.new_tensor([wo - 1, ho - 1]) - 1   # [122, 36]->[-1, 1]
        mesh_point_big = torch.stack(torch.meshgrid([torch.linspace(0, wo - 1, wo), torch.linspace(0, ho - 1, ho)])).to(self.device).permute(0, 2, 1).contiguous().view(2, -1).transpose(1, 0).repeat(b, 1, 1)  # 122x36
        mesh_point_big += mesh_point_big.new_tensor([(w - wo) // 2, (h - ho) // 2])     # 122x36 -offset-> 128x52
        # mesh_point_big = mesh_point_big * mesh_point_big.new_tensor([(wo - 1) / (w - 1), (ho - 1) / ( h - 1)]) + mesh_point_big.new_tensor([(w - wo) // 2, (h - ho) // 2])   # 对齐到144x52: 144x52 resize to 136x36,then add to 144x52 : offset[4, 8]
        # theta_batch_mesh = get_orientation_batch(img_batch, mesh_point_big, patch_size=16)
        # theta_batch_pts = F.grid_sample(theta_batch_mesh.view(b, 1, ho, wo), pts_norm.unsqueeze(1), mode='bilinear', align_corners=True).squeeze()

        batch_size = 2000 # 174000 密集型站空间较大，可用400
        # if batch_size > img_patches.shape[0]:
        #     batch_size = img_patches.shape[0]
        batch_num = img_batch_pad.shape[0] // batch_size + 1
        img_mini_batch_group = torch.chunk(img_batch_pad, batch_num, dim=0)
        # img_mini_batch_group_fliped = torch.chunk(img_patch_pad_fliped, batch_num, dim=0)

        desc_batch = torch.tensor([], dtype=torch.float32).to(self.device)
        # desc_batch_fliped = torch.tensor([], dtype=torch.float32).to(self.device)
        
        for img_mini_batch in img_mini_batch_group:
        # for img_mini_batch, img_mini_batch_fliped in zip(img_mini_batch_group,img_mini_batch_group_fliped):
            with torch.no_grad():
                outs = self.net(img_mini_batch)     # [bs, 1, 161, 61]->[bs, 8, 41, 16]
                # outs_fliped = self.net(img_mini_batch_fliped)  
            desc_batch = torch.cat((desc_batch, outs), dim=0)
            # desc_batch_fliped = torch.cat((desc_batch_fliped, outs_fliped), dim=0)
        
        '''test'''
        # print(outs[0, :, 17, 5], outs_fliped[0, :, 20, 8])
        # print(outs[0, :, 5, 9], outs_fliped[0, :, 32, 4])

        '''pts in feature map 41x16'''
        ds_pad = 0
        outs_pad = F.pad(desc_batch, (ds_pad, ds_pad, ds_pad, ds_pad), 'constant')    # 41x16
        offset = torch.tensor([fill_size_w, fill_size_h], device=self.device).reshape(1, 1, -1)
        points_list_resize = (points_list + points_list.new_tensor([(w - wo) // 2, (h - ho) // 2]).reshape(1, 1, -1)) * points_list.new_tensor([w / (w - 1), h / (h - 1)])
        if mid_pad:
            pts_ds = points_list_resize / ds + torch.tensor([1., 2.]).to(self.device) + ds_pad
        else:
            pts_ds = (points_list_resize + offset) / ds + ds_pad

        angle = theta_batch
        # angle = torch.zeros_like(theta_batch)
        featmap_patches = inv_warp_patch_batch(outs_pad.contiguous().view(-1, 1, outs_pad.shape[2], outs_pad.shape[3]), pts_ds.unsqueeze(1).repeat(1, 8, 1, 1).view(-1, points_list.shape[1], 2), angle.unsqueeze(1).repeat(1, 8, 1).view(-1, points_list.shape[1]), patch_size=patch_size, sample_size=sample_size)
        featmap_patches = featmap_patches.view(b, 8, points_list.shape[1], 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 16, 8)
        featmap_patches = featmap_patches.view(-1, 128)
        desc_front = L2Norm(featmap_patches)

        desc_back = None
        if is256:
            angle_s = angle - 45.
            featmap_patches = inv_warp_patch_batch(outs_pad.contiguous().view(-1, 1, outs_pad.shape[2], outs_pad.shape[3]), pts_ds.unsqueeze(1).repeat(1, 8, 1, 1).view(-1, points_list.shape[1], 2), angle_s.unsqueeze(1).repeat(1, 8, 1).view(-1, points_list.shape[1]), patch_size=patch_size, sample_size=sample_size)
            featmap_patches = featmap_patches.view(b, 8, points_list.shape[1], 4, 4).permute(0, 2, 3, 4, 1).contiguous().view(-1, 16, 8)
            featmap_patches = featmap_patches.view(-1, 128)
            desc_back = L2Norm(featmap_patches)

        if quantize == True:
            desc_front_thre = torch.round(desc_front.cpu() * 10000) + 5000
            desc_front_thre = self.thresholding_desc(desc_front_thre)    # 门限化: base on 128-dim
            # self.get_des_hanmingdis_wht_permute(outs_thre, outs_thre)
            Hada = hadamard(desc_front_thre.shape[1])
            Hada_f_T = desc_front_thre.long() @ torch.from_numpy(Hada)
            desc_f = (Hada_f_T > 0).long().float().to(self.device)
            if is256:
                desc_back_thre = torch.round(desc_back.cpu() * 10000) + 5000
                desc_back_thre = self.thresholding_desc(desc_back_thre)    # 门限化: base on 128-dim
                Hada_b_T = desc_back_thre.long() @ torch.from_numpy(Hada)
                desc_b = (Hada_b_T > 0).long().float().to(self.device)

        if is256:
            desc_batch = torch.cat((desc_front, desc_back), dim=-1).view(points_list.shape[0], points_list.shape[1], -1)
            desc_hadama = torch.cat((desc_f, desc_b), dim=-1).view(points_list.shape[0], points_list.shape[1], -1)
        else:
            # featmap_patches = featmap_patches.contiguous().view(points_list.shape[0], points_list.shape[1], -1)
            desc_batch = torch.cat((desc_front, desc_front), dim=-1).view(points_list.shape[0], points_list.shape[1], -1)
            desc_hadama = torch.cat((desc_f, desc_f), dim=-1).view(points_list.shape[0], points_list.shape[1], -1)

        # return desc_batch, featmap_patches_mesh, f_p_hadama, f_p_mesh_hadama
        return desc_batch, _, desc_hadama, angle

    def get_position(self, p_map,  H, W, block_size):
        x = torch.arange(W // block_size, requires_grad=False, device=p_map.device)
        y = torch.arange(H // block_size, requires_grad=False, device=p_map.device)
        y, x = torch.meshgrid([y, x])
        cell = torch.stack([y, x], dim=0)
        res = (cell + p_map) * block_size
       
        return res    

    def to_pnts(self, semi):
        
        if 0:
            import cv2
            import numpy as np
            _score = semi[:,0][0].cpu().numpy()
            
            _score = abs((_score - _score.min()) / (_score.max() - _score.min()) - 0.35)
        
            _score = torch.tensor(_score).unsqueeze(0).to(self.device)
            score = _score.view(semi.size(0),-1)
            
        else:
            score = semi[:,0].view(semi.size(0),-1)    
        
        y_coordinates = semi[:,1].view(semi.size(0),-1)
        x_coordinates = semi[:,2].view(semi.size(0),-1)

        pnts = torch.stack([score, x_coordinates,y_coordinates],dim=2)
        return pnts

    def forward_alike(self, img):
        H, W = img.shape[2], img.shape[3]
        WH = img.new_tensor([[W - 1, H - 1]])

        with torch.no_grad():
            out = self.net(img, sub_pixel=True)

        pts_sc = out['scores']
        score, score_idx = pts_sc[0].sort(descending=True)
        pts = out['keypoints'][0][score_idx]
        score_dispersity = out['score_dispersity'][0][score_idx]

        out['keypoints'] = (pts / 2 + 0.5) * WH      # [-1,1] -> [W, H]
        out['scores'] = score
        out['score_dispersity'] = score_dispersity

        return out

    def forward_self_supervised_160_48_2head(self, img):
        H, W = img.shape[2], img.shape[3]
        WH = img.new_tensor([[W - 1, H - 1]])

        with torch.no_grad():
            outs = self.net.forward(img)
            semi_p, semi_n, coarse_desc = outs['semi_p'], outs['semi_n'], outs['desc']

        cell_size = 8
        correct_position = self.get_position(semi_p[:,1:3,:,:], H, W, cell_size)  # 校准坐标值
        semi_correct = torch.cat(([semi_p[:,0].unsqueeze(dim=1), correct_position]), dim=1)
        pnts = self.to_pnts(semi_correct)  # [B,N,(s,x,y)]
        
        mask_pts = (pnts[:,:,1] >=0) * (pnts[:,:,1] < 48) * (pnts[:,:,2] >=0) * (pnts[:,:,2] < 160)
        pnts_p = pnts[mask_pts]
            

        correct_position = self.get_position(semi_n[:,1:3,:,:], H, W, cell_size)  # 校准坐标值
        semi_correct = torch.cat(([semi_n[:,0].unsqueeze(dim=1), correct_position]), dim=1)
        pnts = self.to_pnts(semi_correct)  # [B,N,(s,x,y)]
        
        mask_pts = (pnts[:,:,1] >=0) * (pnts[:,:,1] < 48) * (pnts[:,:,2] >=0) * (pnts[:,:,2] < 160)
        pnts_n = pnts[mask_pts]

        pnts = torch.cat((pnts_p, pnts_n),0)      # size: 160 * 48

        return pnts_p, pnts_n


    def forward_self_supervised(self, img):
        H, W = img.shape[2], img.shape[3]
        WH = img.new_tensor([[W - 1, H - 1]])

        with torch.no_grad():
            outs = self.net.forward(img)
            semi, coarse_desc = outs['semi'], outs['desc']


        cell_size = 8
        
        correct_position = self.get_position(semi[:,1:3,:,:], H, W, cell_size)  # 校准坐标值
        semi_correct = torch.cat(([semi[:,0].unsqueeze(dim=1), correct_position]), dim=1)
        pnts = self.to_pnts(semi_correct)  # [B,N,(s,x,y)]
        
        mask_pts = (pnts[:,:,1] >=0) * (pnts[:,:,1] < 48) * (pnts[:,:,2] >=0) * (pnts[:,:,2] < 160)
        pnts = pnts[mask_pts]  # 160 * 48

        return pnts

    def forward_self_supervised_96_28(self, img):
        H, W = img.shape[2], img.shape[3]
        WH = img.new_tensor([[W - 1, H - 1]])

        with torch.no_grad():
            outs = self.net.forward(img)
            semi, coarse_desc = outs['semi'], outs['desc']

        cell_size = 4
        
        correct_position = self.get_position(semi[:,1:3,:,:], H, W, cell_size)  # 校准坐标值
        semi_correct = torch.cat(([semi[:,0].unsqueeze(dim=1), correct_position]), dim=1)
        pnts = self.to_pnts(semi_correct)  # [B,N,(s,x,y)]
        
        mask_pts = (pnts[:,:,1] >=0) * (pnts[:,:,1] < 28) * (pnts[:,:,2] >=0) * (pnts[:,:,2] < 96)
        pnts = pnts[mask_pts]  # 160 * 48

        return pnts

    def forward_self_supervised_96_28_2head(self, img):
        H, W = img.shape[2], img.shape[3]
        WH = img.new_tensor([[W - 1, H - 1]])

        with torch.no_grad():
            outs = self.net.forward(img)
            semi_p, semi_n, coarse_desc = outs['semi_p'], outs['semi_n'], outs['desc']

        cell_size = 4
        correct_position = self.get_position(semi_p[:,1:3,:,:], H, W, cell_size)  # 校准坐标值
        semi_correct = torch.cat(([semi_p[:,0].unsqueeze(dim=1), correct_position]), dim=1)
        pnts = self.to_pnts(semi_correct)  # [B,N,(s,x,y)]
        
        mask_pts = (pnts[:,:,1] >=0) * (pnts[:,:,1] < 28) * (pnts[:,:,2] >=0) * (pnts[:,:,2] < 96)
        pnts_p = pnts[mask_pts]
            

        correct_position = self.get_position(semi_n[:,1:3,:,:], H, W, cell_size)  # 校准坐标值
        semi_correct = torch.cat(([semi_n[:,0].unsqueeze(dim=1), correct_position]), dim=1)
        pnts = self.to_pnts(semi_correct)  # [B,N,(s,x,y)]
        
        mask_pts = (pnts[:,:,1] >=0) * (pnts[:,:,1] < 28) * (pnts[:,:,2] >=0) * (pnts[:,:,2] < 96)
        pnts_n = pnts[mask_pts]

        pnts = torch.cat((pnts_p, pnts_n),0)      # size: 160 * 48

        return pnts_p, pnts_n



    def forward_self_Alike(self, img):
        H, W = img.shape[2], img.shape[3]
        WH = img.new_tensor([[W - 1, H - 1]])

        with torch.no_grad():
            outs = self.net.forward(img)
            pnts, scores, scores_map = outs["keypoints"][0], outs["scores"][0], outs["scores_map"]


        scores = scores.view(-1, 1)
        
        pnts = (pnts + 1) / 2 * pnts.new_tensor([[W - 1, H - 1]]).to(pnts.device)

        pnts = torch.cat(([scores, pnts]), dim=1)

        return pnts


    def forward_self_supervised_6193(self, img):
        H, W = img.shape[2], img.shape[3]
        WH = img.new_tensor([[W - 1, H - 1]])

        with torch.no_grad():
            outs = self.net.forward(img)
            semi, coarse_desc = outs['semi'], outs['desc']

        cell_size = 4
        
        correct_position = self.get_position(semi[:,1:3,:,:], H, W, cell_size)  # 校准坐标值
        semi_correct = torch.cat(([semi[:,0].unsqueeze(dim=1), correct_position]), dim=1)
        pnts = self.to_pnts(semi_correct)  # [B,N,(s,x,y)]
        
        mask_pts = (pnts[:,:,1] >=0) * (pnts[:,:,1] < W) * (pnts[:,:,2] >=0) * (pnts[:,:,2] < H)
        pnts = pnts[mask_pts]  # 160 * 48

        return pnts


