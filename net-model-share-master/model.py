# -*- coding: utf-8 -*-

from enhance_cap6193.models.networks import ResnetGenerator323_7_RSG_small2_212
from enhance_cap6193.models.networks import ResnetGenerator323_7_RSG_small2_212_expand
from point_alike.models.ALNet import ALNet_Angle_Short
from point_alike.utils.filter_point import filter_net_point
from point_alike.utils.draw import draw_orientation_net_wb, draw_orientation_net
from desc_patch.desc import HardNet_fast_short
from desc_patch.utils.desc import forward_patches_correct,Hamming_Hadamard, desc_trans
from quality_6193.models.MobileNet import MNV3_bufen_new5
from quality_6193.data.base_dataset import *
from mask_6193.models.MobileNet import MNV3_bufen_new5 as MNV3_bufen_new5_mask
from compare_6193.mobilenetv3 import MNV30811_SMALL_C3 as MNV30811_SMALL_compare
#from pathlib import Path
from PIL import Image as Img
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from get_img_feature import ori_filter

class model_api(object):
    def __init__(self,device):
        self.device = device
        model = ResnetGenerator323_7_RSG_small2_212(1,2)
        # checkpoint = torch.load(r'./enhance_cap6193/checkpoints/neten_m54_0/268_net_G.pth', 'cpu')  #'cpu'map_location=torch.device('cuda:0')
        checkpoint = torch.load(r'./enhance_cap6193/checkpoints/neten_m53_0/258_net_G.pth', 'cpu')  #'cpu'map_location=torch.device('cuda:0')
        model.load_state_dict(checkpoint, strict=True)  #
        self.enhance_model = model.cuda(device)    #经测试，在cpu上跑model会引起进程阻塞，导致多进程变慢很多
        self.enhance_model.eval()
        #print(self.enhance_model.device())
        model = ResnetGenerator323_7_RSG_small2_212_expand(1,2)
        checkpoint = torch.load(r'./enhance_cap6193/checkpoints/desc_exp_use/229_net_G.pth', 'cpu')
        model.load_state_dict(checkpoint, strict=True)  #
        self.expand_model = model.cuda(device).eval()
        model = ALNet_Angle_Short()
        checkpoint = torch.load(r'./point_alike/checkpoints/0811m_angle_184600_point_short.pth.tar', 'cpu')
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)  #
        self.point_model = model.cuda(device).eval()
        model = HardNet_fast_short()
        checkpoint = torch.load(r'./desc_patch/logs/project/93061_short.pth.tar', 'cpu')
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)  #
        self.desc_model = model.cuda(device).eval()
        model = MNV3_bufen_new5(input_nc=1, output_nc=1)
        checkpoint = torch.load(r'./quality_6193/checkpoints/6193quality_10/300_net_G.pth', 'cpu')
        model.load_state_dict(checkpoint, strict=True)  #
        self.quality_model = model.cuda(device).eval()
        model = MNV3_bufen_new5_mask(input_nc=1, output_nc=1)
        checkpoint = torch.load(r'./mask_6193/checkpoints/6193mask_10/256_net_G.pth', 'cpu')
        model.load_state_dict(checkpoint, strict=True)  #
        self.mask_model = model.cuda(device).eval()
        model = MNV30811_SMALL_compare()
        checkpoint = torch.load(r'./compare_6193/ckpt_153.pth', 'cpu')
        model.load_state_dict(checkpoint['net'], strict=True)  #
        self.compare_model = model.cuda(device).eval()
        gabor_delta_xy = self.gabor_kernel()
        gabor_delta_xy[:,:,1] = -gabor_delta_xy[:,:,1]   #dy需要反向
        self.gabor_delta_xy = gabor_delta_xy

    def enhance(self, img):  #img:debase img
        img = (img - 127.5) / 127.5
        img = torch.from_numpy(img)
        img_t = torch.zeros(1,1,122,36)
        img_t[0,0,2:120,2:34] = img
        img_t_c = img_t.cuda(self.device)
        #print(img_t.shape, img_t[0,0,0,0], img_t[0,0,2,2],img[0,0])
        out = self.enhance_model(img_t_c)
        out = out.cpu().detach().numpy()
        out = out.reshape(122,36)
        out = np.uint8((out + 1) * 127.5 + 0.5)  #????
        return out
    
    def expand(self, img):  #img:enhance img
        img_n = (img - 127.5) / 127.5
        img_n = torch.from_numpy(img_n)
        img_t = torch.zeros(1,1,122,36)
        img_t[0,0,:,:] = img_n
        img_t_c = img_t.cuda(self.device)
        out = self.expand_model(img_t_c)
        #img_pad = F.pad(img_t, pad=(8,8,3,3), mode='constant', value=1) 
        out = out.cpu().detach().numpy().reshape(128,52)
        out = np.uint8((out + 1) * 127.5 + 0.5)
        #融合
        ew,eh = 8,3
        w,h = 36,122
        out[eh:eh+h, ew:ew + 4] = out[eh:eh+h, ew:ew + 4] * 0.6 + img[:, 0:4] * 0.4
        out[eh:eh+h, ew+w-4:ew+w] = out[eh:eh+h, ew+w-4:ew+w] * 0.6 + img[:, w-4:w] * 0.4
        out[eh:eh+h, ew+4:ew+w-4] = img[:, 4:w - 4]
        return out
    
    def point(self, img, partial_mask):  #img:enhance img  partial_mask: 1 data 0 null
        img = img / 255.
        img = torch.from_numpy(img)
        img_t = torch.zeros(1,1,128,40)
        img_t[0,0,3:125,2:38] = img
        img_t_c = img_t.cuda(self.device)
        scores_map, points_map = self.point_model(img_t_c)
        partial_mask = torch.from_numpy(partial_mask)
        heatmap = scores_map.squeeze(0)
        pts, net_ori= filter_net_point(heatmap, partial_mask, 0.01, 2, 2, 130)
        pts = pts.cpu().detach().numpy().T
        net_ori = net_ori.cpu().detach().numpy()
        #print(pts, net_ori)
        intensity = heatmap[1, 3:-3, 2:-2]
        pnmap = heatmap[2, 3:-3, 2:-2]
        pn = pnmap > 0.5
        ori_map = (intensity * (2 * pn - 1)) * np.pi / 2
        ori_map = ori_map.cpu().detach().numpy()
        return {'pts':pts[:,0:2], 'prob': pts[:,2],'angles':net_ori, 'ori_map':ori_map}
    
    def decriptor(self, img, points):  #img:expand img
        img = torch.from_numpy(img) 
        img_t = torch.zeros(1,1,128,52)
        img_t[0,0,:,:] = img
        img_t_c = img_t.cuda(self.device)
        keypoints = torch.from_numpy(points['pts']).cuda(self.device)
        angles = torch.from_numpy(points['angles']).cuda(self.device)
        pai_coef = 3.1415926
        #print(keypoints, (angles + pai_coef/2)*4096)
        angles = angles / pai_coef * 180   #转化为角度
        outs, _, wb_mask = forward_patches_correct(img_t_c, keypoints, self.desc_model, theta=angles)
        #print(outs)
        descs_Hamming = Hamming_Hadamard(outs)
        #print(torch.sum(outs))
        #print(outs, descs_Hamming)
        desc = desc_trans(descs_Hamming) #同反号描述分开
        desc = desc.cpu().detach().numpy()
        wb_mask = wb_mask.cpu().detach().numpy()
        return desc, wb_mask

    def quality(self, img):  #img:debase img  #score：0-100
        #print(img.shape)
        img = img / 255.
        img = torch.from_numpy(img).reshape(1,1,118,32)
        img = F.interpolate(img.float(), size=[76, 20], mode="bilinear", align_corners=False)
        img_t_c = img.cuda(self.device)
        score = self.quality_model(img_t_c)
        score = score.cpu().detach().numpy()
        score = np.uint8(score[0][0]*100+0.5)
        return score
    
    def mask(self, img):  #img:debase img  #mask：True为指纹区域 False为无纹路区域   score:0-100 0：正常手指 100：湿手指
        img = img / 255.
        img = torch.from_numpy(img).reshape(1,1,118,32)
        img = F.interpolate(img.float(), size=[80, 24], mode="bilinear", align_corners=False)
        img_t_c = img.cuda(self.device)
        mask,score = self.mask_model(img_t_c)

        mask = F.interpolate((mask), size=[118, 32], mode="bilinear", align_corners=False)
        mask = F.pad(mask, pad=(2,2,2,2), mode='replicate') 
        mask = mask.cpu().detach().numpy().reshape(122,36)
        mask = mask > 0.5

        score = score.cpu().detach().numpy()
        score = np.uint8(score[0][0]*100+0.5)
        return mask, score
    
    def compare(self, img):  #img:对位图  #score:

        img = img / 255.

        img = torch.from_numpy(img)#.reshape(1,3,122,36)

        img = img.transpose(1,2).transpose(0,1).reshape(1,3,122,122)

        img = img[:,:2,:,:].float()

        img_t_c = img.cuda(self.device)

        out = self.compare_model(img_t_c)

        out = F.softmax(out,dim = -1)

        score = out[0][1]*65536

        score = score.cpu().detach().numpy()

        
        return score
    
    def compare_batch(self, img):  #img:对位图  #score:
        img = img / 255.
        # img = torch.from_numpy(img)#.reshape(1,3,122,36)
        # img = img.transpose(1,2).transpose(0,1).reshape(1,3,122,36)
        img = img[:,:2,:,:].float()
        # img_t_c = img.cuda(self.device)
        with torch.no_grad():
            out = self.compare_model(img)
        out = F.softmax(out,dim = -1)
        score = out[:,1]*65536
        # score = score.cpu().detach().numpy()
        
        return score
    
    '''圆形坐标
    def gabor_kernel(self):
        kernel = np.array([1,2,4,8,4,2,1])
        gabor_delta = np.zeros([7,2], dtype=np.float32)
        gabor_delta[:, 0] = np.arange(-3,4)
        theta = np.arange(0, np.pi, np.pi/12)
        cos_sin = np.array([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)], dtype = np.float32)
        trans_all_angle = cos_sin.T.reshape(24,2)
        gabor_delta_xy = trans_all_angle @ gabor_delta.T
        return gb
    '''




    def gabor_kernel(self):   #目前采用方形坐标
        theta = np.arange(0, np.pi, np.pi/12)
        tan = np.tan(theta)
        tan[4:10] = 1/tan[4:10]

        gabor_delta = np.zeros([12,7,2], dtype=np.float32)

        gabor_delta[0:4, :, 0] = np.arange(-3,4)
        gabor_delta[0:4, :, 1] = tan[0:4].reshape(4,1) * gabor_delta[0:4, :, 0]
        gabor_delta[4:7, :, 1] = np.arange(-3,4)
        gabor_delta[4:7, :, 0] = tan[4:7].reshape(3,1) * gabor_delta[4:7, :, 1]
        gabor_delta[7:10, :, 1] = np.arange(3,-4, -1)
        gabor_delta[7:10, :, 0] = tan[7:10].reshape(3,1) * gabor_delta[7:10, :, 1]
        gabor_delta[10:12, :, 0] = np.arange(-3,4)
        gabor_delta[10:12, :, 1] = tan[10:12].reshape(2,1) * gabor_delta[10:12, :, 0]

        gabor_delta_xy = np.int32(np.round(gabor_delta))

        return gabor_delta_xy

def downsample2x2(img):

    h,w = img.shape
    x = range(0,w,2)
    y = range(0,h,2)
    img_downsample = img[y,:][:,x]
    return img_downsample

def main():
    with torch.no_grad():

        path = r"0000.bmp"
        img = np.array(Img.open(path))
        fm = model_api(0)
        
        quality = fm.quality(img)
        partial_mask,wet_score = fm.mask(img)
        cv2.imwrite(path[:-4] + '_mask_' + str(quality) + '_' + str(wet_score) +'.bmp', partial_mask * 255)

        enhance_img = fm.enhance(img)
        cv2.imwrite(path[:-4] + "_enhance.bmp", enhance_img)
        
        point_img = cv2.GaussianBlur(enhance_img, (5,5), 0.5, borderType = cv2.BORDER_REFLECT101)   #(7,7), 0.5
        cv2.imwrite(path[:-4] + "_point.bmp", point_img)

        points = fm.point(point_img, partial_mask)
        #print(points["pts"],points["prob"],points["angles"])
        # #''' todo gabor
        ori_filter_img = ori_filter(enhance_img, points['ori_map'], fm.gabor_delta_xy)
        #获取16级灰度图
        level_16_img = ori_filter(ori_filter_img, points['ori_map'], fm.gabor_delta_xy)
        level_16_img = cv2.GaussianBlur(level_16_img, (5,5), 0.8)  #bordertype 默认reflect
        level_16_img = downsample2x2(level_16_img)   #0,1
        level_16_img = (level_16_img // 16)*16 + 8
        cv2.imwrite(path[:-4] + "_level_16.bmp", level_16_img)
        # ksize,sigma,theta,lambd,gamma,psi = 5, 0.8, 0.7854, 6, 0.5, 0
        # gabor_kernel()
        # kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lambd, gamma, psi = 0)
        # kernel = kernel/kernel.sum()
        # desc_img = cv2.filter2D(enhance_img, cv2.CV_8UC1, kernel,  borderType = cv2.BORDER_REFLECT101)
        # #'''
        expand_img = fm.expand(ori_filter_img)
        cv2.imwrite(path[:-4] + "_expand.bmp", expand_img)

        desc, wb_mask = fm.decriptor(expand_img, points)
        print(desc.dtype)
        img_d = {"img":enhance_img /255.}
        draw_point_img = draw_orientation_net_wb(img_d, points, wb_mask)
        cv2.imwrite(path[:-4] + "_draw_point.bmp", draw_point_img)
        
		
        path = r"./compare_6193/decision_1_temp_overlap_ndx_000040_area_50_simi_222.bmp_merge.bmp"
        img = np.array(Img.open(path))
        s = fm.compare(img)
		
		
if __name__ == "__main__":
    main()
    