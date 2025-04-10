import re
import torch
import pandas as pd
# import torchvision.transforms as transforms

from Model_component import flattenDetection, getPtsFromHeatmap, flattenDetection_new
# from .SuperPointNet_small_128 import SuperPointNet_small_128
from collections import OrderedDict

class FPDT(object):
    '''Feature Point Detection Test'''
    def __init__(self, name, detector_net_weights, device="cpu"):
        print("model: ", name)
        self.device = device
        
        '''Load Model'''
        try:
            net = getattr(__import__('models.{}'.format(name), fromlist=['']), name)
            # print(self.device)
            self.detector_net = net().to(self.device)
            checkpoint = torch.load(detector_net_weights, map_location=lambda storage, loc: storage)

            # print(net)
            DP = True

            is_model_short = True      # short: is_model_short = True is_rec = False
            is_rec = False # True
            with_pnt = True # False
            
            is_pntmodel_short = False # False pntshort: is_pntmodel_short = True, has_teacher = False
            has_teacher = True # True
            use_net_match = False

            # only_pd = True
            

            if DP:
                # print(type(checkpoint['model_state_dict']))
                NoneDP_param = OrderedDict()
                for k in checkpoint['model_state_dict'].keys():
                    if with_pnt or is_pntmodel_short:
                        if 'descriptor_net' not in k and 'point_tea_net' not in k and 'point_fgd_loss' not in k:
                            NoneDP_param['descriptor_net.' + k.replace('.module', '')] = checkpoint['model_state_dict'][k]
                    else:
                        if 'point_tea_net' not in k and 'point_fgd_loss' not in k:
                            NoneDP_param[k.replace('.module', '')] = checkpoint['model_state_dict'][k]
                # print(NoneDP_param.keys())
                # exit()
            else:
                NoneDP_param = checkpoint['model_state_dict']
            
            if use_net_match:
                match_net_weights = "/hdd/file-input/linwc/Descriptor/code/Test_sys/checkpoints/Match/1204m_superGlue_89800_checkpoint.pth.tar"
                match_net_checkpoint = torch.load(match_net_weights, map_location=lambda storage, loc: storage)
                for k in match_net_checkpoint['model_state_dict'].keys():    
                    NoneDP_param['match_net.' + k.replace('.module', '')] = match_net_checkpoint['model_state_dict'][k]

            if is_rec:
                if DP:
                    # print(type(checkpoint['model_state_dict']))
                    checkpoint_assist = torch.load('/hdd/file-input/linwc/Descriptor/code/Test_sys/checkpoints/Des/0505m_ALikeWithHard_patch_group_inner_re_smallc4_ext_rt_nocut_nort_ne_33200_checkpoint.pth.tar', map_location=lambda storage, loc: storage)
                    for k in checkpoint_assist['model_state_dict'].keys():
                        if 'descriptor_net' in k:
                            NoneDP_param[k.replace('.module', '').replace('descriptor_net', 'descriptor_assist_net')] = checkpoint_assist['model_state_dict'][k]

                    # print(NoneDP_param.keys())
                    # exit()
                else:
                    NoneDP_param = checkpoint_assist['model_state_dict']

            if not has_teacher:
                checkpoint_teacher = torch.load('/hdd/file-input/linwc/Descriptor/code/Test_sys/checkpoints/Des/0505m_ALikeWithHard_patch_group_inner_re_smallc4_ext_rt_nocut_nort_ne_33200_checkpoint.pth.tar', map_location=lambda storage, loc: storage)
                for k in checkpoint_teacher['model_state_dict'].keys():
                    if 'descriptor_net' in k:
                        NoneDP_param[k.replace('.module', '').replace('descriptor_net', 'descriptor_tea_net')] = checkpoint_teacher['model_state_dict'][k]        

            if with_pnt:
                pnt_net_weights = '/home/lif/point_train/logs/oppo/1115/checkpoints/superPointNet_148000_checkpoint.pth.tar'# '0811m_angle_184600_point_short.pth.tar' '0821m_63200_checkpoint.pth.tar' '0821m_270600_angle_point_short.pth.tar' '0830n_89400_angle_point_short.pth.tar'
                pnt_net_checkpoint = torch.load(pnt_net_weights, map_location=lambda storage, loc: storage)  
                if DP:
                    for k in pnt_net_checkpoint['model_state_dict'].keys():
                        if 'descriptor' not in k and 'point_tea_net' not in k and 'point_fgd_loss' not in k: 
                            # print(k.replace('.module', ''))
                            NoneDP_param[k.replace('.module', '')] = pnt_net_checkpoint['model_state_dict'][k]
                    # print(NoneDP_param.keys())
                    # exit()
                else:
                    NoneDP_param = checkpoint['model_state_dict']     

            if is_pntmodel_short:
                pnt_short_net_weights = '/home/lif/point_train/logs/oppo/1115/checkpoints/superPointNet_148000_checkpoint.pth.tar'# '0908m_232400_checkpoint.pth.tar' '0821m_164600_angle_point_short.pth.tar' '0906a_115600_angle_point_short.pth.tar'
                pnt_short_net_checkpoint = torch.load(pnt_short_net_weights, map_location=lambda storage, loc: storage)  
                if DP:
                    for k in pnt_short_net_checkpoint['model_state_dict'].keys():
                        if 'descriptor' not in k and 'point_tea_net' not in k and 'point_fgd_loss' not in k: 
                            # print(k.replace('.module', ''))
                            NoneDP_param[k.replace('.module', '')] = pnt_short_net_checkpoint['model_state_dict'][k]
                    # print(NoneDP_param.keys())
                    # exit()
                else:
                    NoneDP_param = checkpoint['model_state_dict']     
   

            if is_model_short and with_pnt is False:
                self.detector_net.descriptor_net.load_state_dict(NoneDP_param)
            else:
                self.detector_net.load_state_dict(NoneDP_param)
                        

            print("==> Loading {}...".format(detector_net_weights.split('/')[-1]))
            # self.detector_net.load_state_dict(torch.load(detector_net_weights, map_location={'cuda:0':('cuda:' + str(gpu_ids))})['model_state_dict'])
            self.detector_net.eval()

        except Exception:
            print("Load Model fail!")
            raise
        print("==> Successfully loaded pre-trained network.")

        '''Init Parameters'''
        self.samples        = None
        self.iterations     = 0

    def run_heatmap(self, img):
        '''
        input: 1 1 H W
        return: heatmap
        '''
        H, W = img.shape[2], img.shape[3]

        with torch.no_grad():
            self.detector_net.eval()
            outs = self.detector_net.forward(img, sub_pixel=False)
            semi, coarse_desc, desc_map, scores, scores_map, scoredispersitys = outs["keypoints"], outs["descriptors"], outs["descriptor_map"], outs["scores"], outs["scores_map"], outs["scoredispersitys"]
            # semi, desc = outs['semi'], outs['desc']
        heatmap = scores_map[0, :, :, :]
        desc = desc_map[0, :, :, :]
        # heatmap = flattenDetection_new(semi)
        # pts = getPtsFromHeatmap(heatmap.squeeze())
        return heatmap, desc
    
    # 返回heatmap的形式
    def run_heatmap_alike(self, img):
        '''
        input: 1 1 H W
        return: heatmap
        '''
        H, W = img.shape[2], img.shape[3]

        with torch.no_grad():
            self.detector_net.eval()
            scores_map = self.detector_net.cut_patch_unfold_patch_map_interpolation_aligned_batch_ext93_onlyP_scoresmap(img, sub_pixel=False)
            # semi, coarse_desc, desc_map, scores, scores_map, scoredispersitys = outs["keypoints"], outs["descriptors"], outs["descriptor_map"], outs["scores"], outs["scores_map"], outs["scoredispersitys"]
            # semi, desc = outs['semi'], outs['desc']
        heatmap = scores_map[0, :, :, :]
        desc = None
        # heatmap = flattenDetection_new(semi)
        # pts = getPtsFromHeatmap(heatmap.squeeze())
        return heatmap, desc 
    
    # 返回heatmap的形式 9800
    def run_heatmap_alike_9800(self, img):
        '''
        input: 1 1 H W
        return: heatmap
        '''
        H, W = img.shape[2], img.shape[3]

        with torch.no_grad():
            self.detector_net.eval()
            scores_map = self.detector_net.cut_patch_unfold_patch_map_interpolation_aligned_batch_ext93_onlyP_scoresmap_9800(img, sub_pixel=False)
            # semi, coarse_desc, desc_map, scores, scores_map, scoredispersitys = outs["keypoints"], outs["descriptors"], outs["descriptor_map"], outs["scores"], outs["scores_map"], outs["scoredispersitys"]
            # semi, desc = outs['semi'], outs['desc']
        heatmap = scores_map[0, :, :, :]
        desc = None
        # heatmap = flattenDetection_new(semi)
        # pts = getPtsFromHeatmap(heatmap.squeeze())
        return heatmap, desc 
    
    # 返回DKD的点
    def run_pts_alike(self, img):
        '''
        input: 1 1 H W
        return: heatmap
        '''
        H, W = img.shape[2], img.shape[3]

        with torch.no_grad():
            self.detector_net.eval()
            keypoints = self.detector_net.cut_patch_unfold_patch_map_interpolation_aligned_batch_ext93_onlyP(img, sub_pixel=False)
            # semi, coarse_desc, desc_map, scores, scores_map, scoredispersitys = outs["keypoints"], outs["descriptors"], outs["descriptor_map"], outs["scores"], outs["scores_map"], outs["scoredispersitys"]
            # semi, desc = outs['semi'], outs['desc']
        # heatmap = scores_map[0, :, :, :]
        # desc = None
        # heatmap = flattenDetection_new(semi)
        # pts = getPtsFromHeatmap(heatmap.squeeze())
        return keypoints
    

    def run_detec_desc(self, img):
        '''
        input: 1 1 H W
        return: semi, desc
        '''
        H, W = img.shape[2], img.shape[3]

        with torch.no_grad():
            self.detector_net.eval()
            outs = self.detector_net.forward(img)
            semi, desc = outs['semi'], outs['desc']

        return semi, desc

    def run_pts_desc(self, img, conf_thresh, nms):
        '''
        input: 1 1 H W
        return: pts, desc
        '''
        H, W = img.shape[2], img.shape[3]

        with torch.no_grad():
            self.detector_net.eval()
            outs = self.detector_net.forward(img, sub_pixel=False)
            semi, coarse_desc, desc_map, scores, scores_map, scoredispersitys = outs["keypoints"], outs["descriptors"], outs["descriptor_map"], outs["scores"], outs["scores_map"], outs["scoredispersitys"]
            # semi, desc = outs['semi'], outs['desc']
        heatmap = scores_map[0, :, :, :]
        desc = desc_map[0, :, :, :]
        # heatmap = flattenDetection(semi)
        pts = getPtsFromHeatmap(heatmap.squeeze().detach().cpu().numpy(), conf_thresh, nms)

        return pts, desc

    def run_pts_desc_dkd(self, img, conf_thresh, nms):
            '''
            input: 1 1 H W
            return: pts, desc
            '''
            H, W = img.shape[2], img.shape[3]

            with torch.no_grad():
                self.detector_net.eval()
                outs = self.detector_net.forward(img, sub_pixel=False)
                semi, coarse_desc, desc_map, scores, scores_map, scoredispersitys = outs["keypoints"], outs["descriptors"], outs["descriptor_map"], outs["scores"], outs["scores_map"], outs["scoredispersitys"]
                # semi, desc = outs['semi'], outs['desc']
            # heatmap = scores_map[0, :, :, :]
            # desc = desc_map[0, :, :, :]
            # heatmap = flattenDetection(semi)
            # pts = getPtsFromHeatmap(heatmap.squeeze().detach().cpu().numpy(), conf_thresh, nms)

            return semi[0], coarse_desc.squeeze(-1)

    def run_pts_desc_dkdwithhard(self, img, conf_thresh, nms):
        '''
        input: 1 1 H W
        return: pts, desc
        '''
        H, W = img.shape[2], img.shape[3]

        with torch.no_grad():
            self.detector_net.eval()
            outs = self.detector_net.forward(img, sub_pixel=False)
            semi, coarse_desc, desc_map, scores, scores_map, scoredispersitys = outs["keypoints"], outs["descriptors"], outs["descriptor_map"], outs["scores"], outs["scores_map"], outs["scoredispersitys"]
            # semi, desc = outs['semi'], outs['desc']
        heatmap = scores_map[0, :, :, :]
        desc = desc_map[0, :, :, :]
        # heatmap = flattenDetection(semi)
        pts = getPtsFromHeatmap(heatmap.squeeze().detach().cpu().numpy(), conf_thresh, nms)

        return pts, desc

    def run_pts_dkd(self, img):
        '''
        input: 1 1 H W
        return: heatmap
        '''
        H, W = img.shape[2], img.shape[3]

        with torch.no_grad():
            self.detector_net.eval()
            outs = self.detector_net.forward(img, sub_pixel=False)
            semi, coarse_desc, desc_map, scores, scores_map, scoredispersitys = outs["keypoints"], outs["descriptors"], outs["descriptor_map"], outs["scores"], outs["scores_map"], outs["scoredispersitys"]
            # semi, desc = outs['semi'], outs['desc']
        # heatmap = flattenDetection_new(semi)
        # pts = getPtsFromHeatmap(heatmap.squeeze())
        return semi[0], coarse_desc[0], scores[0]

    