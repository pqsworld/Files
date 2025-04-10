# -*- coding: utf-8 -*-

import os
import glob
from PIL import Image as Img
import numpy as np
from model import model_api
from get_img_feature import get_feature
from recognize import recognize, recognize_batch
from compare import compare, compare_batch
import time
import datetime
import pandas as pd
from multiprocessing import Pool, Queue, cpu_count
import multiprocessing
import cv2
#from feature_info import feature_info
from name_539 import feature_info_key
from get_match_feature import get_match_feat
import pynvml
import torch
import copy

COLUMNS = ["sample", "temple", "score", "H0", "H1", "H2", "H3", "H4", "H5"]   #, *(feature_info_key.keys())
GPU_ids = [0,1,2,3,4,5,6,7]
Path_out_img = r"../outimg/"
FLAG_SAVE_IMG = 0

class finger(object):
    def __init__(self, template_num_max, model, mode, flag_reserve_fail_num = 0):
        self.enrolled_num = 0
        self.authed_num = 0
        self.auth_succ_num = 0
        self.template = []
        self.template_num_max = template_num_max
        self.model_api = model
        self.mode = mode  #0: first_match  1:all_match
        self.flag_reserve_fail_num = flag_reserve_fail_num
        self.reserve_fail_temple = []

    def enroll(self, img, name):
        feat = get_feature(img, self.model_api, name)
        if feat == None:
            return
        self.template.append([feat, name])
        self.enrolled_num = self.enrolled_num + 1
        #print("enroll", name)
        
    def auth(self, img, name, pdfeat):
        feat = get_feature(img, self.model_api, name)
        if feat == None:
            return 0, None    #跳过异常图，不计数

        flag_auth_succ = 0
        for i in range(len(self.template) - 1, -1, -1):  #reverse
            #feat_info = feature_info()
            auth_result,rec_feat = recognize(copy.deepcopy(self.template[i][0]), copy.deepcopy(feat), self.model_api.device)
            #if auth_result == 0:   #反向
            #    auth_result,rec_feat = recognize(self.template[i][0], feat)
            score, img_match = compare(self.template[i][0], feat, rec_feat, self.model_api)
            auth_result = score > 65535  #58636 为1/1wfa的阈值
            if auth_result > 0 : 
                # get_match_feat(self.template[i][0], feat, rec_feat, feat_info)
                #print("auth", name, self.template[i][1])
                flag_auth_succ = 1
                if FLAG_SAVE_IMG == 1 and img_match is not None:
                    out_img_path = Path_out_img + str(int(score)) + '_' + name[-16:-4].replace('/', '_') + \
                        "_" + self.template[i][1][-16:-4].replace('/', '_') + "_match.bmp"
                    cv2.imwrite(out_img_path, img_match)
                if pdfeat is not None:
                    model_save = rec_feat['model'].params*256
                    trans_save = (model_save[:2].reshape(-1)+0.5).astype(np.int).tolist()
                    log_save = [name, self.template[i][1], score] + trans_save
                    pdfeat.loc[len(pdfeat)] = log_save  #, *(feat_info.feature)

                if self.mode == 0:
                    break

        self.authed_num = self.authed_num + 1
        if flag_auth_succ > 0:
            self.auth_succ_num = self.auth_succ_num + 1
        return flag_auth_succ,feat
    
    def auth_batch(self, fgimage, pdfeat):
        img_h, img_w = self.template[0][0]['h'], self.template[0][0]['w']
        img_L16_h, img_L16_w = self.template[0][0]['level_16_img'].shape
        device = self.model_api.device

        batchsize_s = 200
        bNum_limit = 100
        wNum_limit = 100

        num_s = len(fgimage)
        auth_succ_num = 0
        for batch_samples in [list(range(num_s))[i:i+batchsize_s] for i in range(0,num_s,batchsize_s)]: #sample 数据batch化
            batch_size = len(batch_samples)
            feat_batch_s  =  {
                'h':img_h,
                'w':img_w,
                'prob_idx':2,    #记录存放位置
                'angles_idx':3,
                'dbvalue_idx':4,
                'name':[],
                'quality':torch.zeros(batch_size),
                'mask':torch.zeros((batch_size,1,img_h,img_w)),
                'wet_score':torch.zeros(batch_size),
                'bvalid_mask':torch.zeros(batch_size,bNum_limit),
                'wvalid_mask':torch.zeros(batch_size,wNum_limit),
                'pts_b':torch.zeros((batch_size,bNum_limit,5)),        #torch Nbx5 (float) (x,y,prob,angles,dbvlue)
                'pts_w':torch.zeros((batch_size,wNum_limit,5)),        #torch Nwx5 (float) (x,y,prob,angles,dbvlue)
                'desc_b':torch.zeros((batch_size,bNum_limit,256)),     #torch Nbx(256bit)
                'desc_w':torch.zeros((batch_size,wNum_limit,256)),     #torch Nwx(256bit)
                'enhance_img':torch.zeros((batch_size,1,img_h,img_w)),   #numpy h*w
                'level_16_img':torch.zeros((batch_size,1,img_L16_h,img_L16_w)), #numpy h*w
            }

            bNum_runningMax = 0 #根据实际，动态调整规格化点数
            wNum_runningMax = 0 #根据实际，动态调整规格化点数
            batch_valid = torch.ones(len(batch_samples))
            for idx, idx_sample in enumerate(batch_samples): #sample 数据batch化
                name, img = fgimage[idx_sample]
                feat = get_feature(img, self.model_api, name)
                if feat == None:
                    batch_valid[idx] = 0
                    continue

                black_num = feat['black_num']
                white_num = len(feat['desc']) - black_num
                assert black_num <= bNum_limit
                assert white_num <= wNum_limit

                if black_num > bNum_runningMax:
                    bNum_runningMax = black_num
                if white_num > wNum_runningMax:
                    wNum_runningMax = white_num

                feat_batch_s['bvalid_mask'][idx,:black_num] = 1
                feat_batch_s['wvalid_mask'][idx,:white_num] = 1
                feat_batch_s['pts_b'][idx,:black_num,:2] = torch.from_numpy(feat['pts'][:black_num])
                feat_batch_s['pts_b'][idx,:black_num,2]  = torch.from_numpy(feat['prob'][:black_num])
                feat_batch_s['pts_b'][idx,:black_num,3]  = torch.from_numpy(feat['angles'][:black_num])
                feat_batch_s['pts_b'][idx,:black_num,4]  = torch.from_numpy(feat['dbvalue'][:black_num])
                feat_batch_s['desc_b'][idx,:black_num]   = torch.from_numpy(feat['desc'][:black_num])

                feat_batch_s['pts_w'][idx,:white_num,:2] = torch.from_numpy(feat['pts'][black_num:])
                feat_batch_s['pts_w'][idx,:white_num,2]  = torch.from_numpy(feat['prob'][black_num:])
                feat_batch_s['pts_w'][idx,:white_num,3]  = torch.from_numpy(feat['angles'][black_num:])
                feat_batch_s['pts_w'][idx,:white_num,4]  = torch.from_numpy(feat['dbvalue'][black_num:])
                feat_batch_s['desc_w'][idx,:white_num]   = torch.from_numpy(feat['desc'][black_num:])

                
                feat_batch_s['name'].append(name)
                feat_batch_s['quality'][idx] = feat['quality']
                feat_batch_s['wet_score'][idx] = feat['wet_score']
                feat_batch_s['mask'][idx,:] = torch.from_numpy(feat['mask'])
                feat_batch_s['enhance_img'][idx,:] = torch.from_numpy(feat['enhance_img'])
                feat_batch_s['level_16_img'][idx,:] = torch.from_numpy(feat['level_16_img'])
            
            batch_valid = batch_valid.nonzero()[:,0]
            #最小化数量，有利于减少计算量
            feat_batch_s['bvalid_mask'] = feat_batch_s['bvalid_mask'][:,:bNum_runningMax]
            feat_batch_s['pts_b']       = feat_batch_s['pts_b'][:,:bNum_runningMax]
            feat_batch_s['desc_b']      = feat_batch_s['desc_b'][:,:bNum_runningMax]
            feat_batch_s['wvalid_mask'] = feat_batch_s['wvalid_mask'][:,:wNum_runningMax]
            feat_batch_s['pts_w']       = feat_batch_s['pts_w'][:,:wNum_runningMax]
            feat_batch_s['desc_w']      = feat_batch_s['desc_w'][:,:wNum_runningMax]

            #数据全部加载到GPU
            feat_batch_s['quality'] = feat_batch_s['quality'][batch_valid].cuda(device)
            feat_batch_s['wet_score'] = feat_batch_s['wet_score'][batch_valid].cuda(device)
            feat_batch_s['pts_b'] = feat_batch_s['pts_b'][batch_valid].cuda(device)
            feat_batch_s['pts_w'] = feat_batch_s['pts_w'][batch_valid].cuda(device)
            feat_batch_s['desc_b'] = feat_batch_s['desc_b'][batch_valid].cuda(device)
            feat_batch_s['desc_w'] = feat_batch_s['desc_w'][batch_valid].cuda(device)
            feat_batch_s['bvalid_mask'] = feat_batch_s['bvalid_mask'][batch_valid].cuda(device)
            feat_batch_s['wvalid_mask'] = feat_batch_s['wvalid_mask'][batch_valid].cuda(device)
            feat_batch_s['mask'] = feat_batch_s['mask'][batch_valid].cuda(device)
            feat_batch_s['enhance_img'] = feat_batch_s['enhance_img'][batch_valid].cuda(device)
            feat_batch_s['level_16_img'] = feat_batch_s['level_16_img'][batch_valid].cuda(device)

            for i in range(len(self.template) - 1, -1, -1):  #reverse

                feat_t = self.template[i][0]
                feat_t_pts_b = torch.from_numpy(feat_t['pts'][:feat_t['black_num']]).unsqueeze(0)
                feat_t_pts_w = torch.from_numpy(feat_t['pts'][feat_t['black_num']:]).unsqueeze(0)
                feat_t_pts_b = torch.cat([feat_t_pts_b,torch.from_numpy(feat_t['prob'][:feat_t['black_num']])[None,:,None]],dim=2)
                feat_t_pts_b = torch.cat([feat_t_pts_b,torch.from_numpy(feat_t['angles'][:feat_t['black_num']])[None,:,None]],dim=2)
                feat_t_pts_b = torch.cat([feat_t_pts_b,torch.from_numpy(feat_t['dbvalue'][:feat_t['black_num']])[None,:,None]],dim=2)
                feat_t_pts_w = torch.cat([feat_t_pts_w,torch.from_numpy(feat_t['prob'][feat_t['black_num']:])[None,:,None]],dim=2)
                feat_t_pts_w = torch.cat([feat_t_pts_w,torch.from_numpy(feat_t['angles'][feat_t['black_num']:])[None,:,None]],dim=2)
                feat_t_pts_w = torch.cat([feat_t_pts_w,torch.from_numpy(feat_t['dbvalue'][feat_t['black_num']:])[None,:,None]],dim=2)

                b_num_t = feat_t['black_num']
                w_num_t = len(feat_t['desc']) - feat_t['black_num']
                assert len(feat_t['desc']) > 0
                feat_t_desc_b = torch.from_numpy(feat_t['desc'][:feat_t['black_num']]).unsqueeze(0)
                feat_t_desc_w = torch.from_numpy(feat_t['desc'][feat_t['black_num']:]).unsqueeze(0)
                bvalid_mask_t = torch.ones(1,b_num_t)
                wvalid_mask_t = torch.ones(1,w_num_t)

                if b_num_t == 0:
                    bvalid_mask_t = torch.zeros_like(wvalid_mask_t)
                    feat_t_pts_b  = feat_t_pts_w.clone()
                    feat_t_desc_b = feat_t_desc_w.clone()
                if w_num_t == 0:
                    wvalid_mask_t = torch.zeros_like(bvalid_mask_t)
                    feat_t_pts_w  = feat_t_pts_b.clone()
                    feat_t_desc_w = feat_t_desc_b.clone() 

                feat_batch_t  =  {
                    'h':img_h,
                    'w':img_w,
                    'prob_idx':2,    #记录存放位置
                    'angles_idx':3,
                    'dbvalue_idx':4,
                    'name':feat_t['name'],
                    'quality':torch.tensor(feat_t['quality']),
                    'mask':torch.from_numpy(feat_t['mask'])[None,None,:,:],
                    'wet_score':torch.tensor(feat_t['wet_score']),
                    'bvalid_mask':bvalid_mask_t,
                    'wvalid_mask':wvalid_mask_t,
                    'pts_b':feat_t_pts_b,        #torch Nbx5 (float) (x,y,prob,angles,dbvlue)
                    'pts_w':feat_t_pts_w,        #torch Nwx5 (float) (x,y,prob,angles,dbvlue)
                    'desc_b':feat_t_desc_b,     #torch Nbx(256bit)
                    'desc_w':feat_t_desc_w,     #torch Nwx(256bit)
                    'enhance_img':torch.from_numpy(feat_t['enhance_img'])[None,None,:,:],   #tensor 1*1*h*w
                    'level_16_img':torch.from_numpy(feat_t['level_16_img'])[None,None,:,:], #tensor 1*1*h*w
                }

                #数据全部加载到GPU
                feat_batch_t['pts_b']  = feat_batch_t['pts_b'].cuda(device)
                feat_batch_t['pts_w']  = feat_batch_t['pts_w'].cuda(device)
                feat_batch_t['desc_b'] = feat_batch_t['desc_b'].cuda(device)
                feat_batch_t['desc_w'] = feat_batch_t['desc_w'].cuda(device)
                feat_batch_t['bvalid_mask'] = feat_batch_t['bvalid_mask'].cuda(device)
                feat_batch_t['wvalid_mask'] = feat_batch_t['wvalid_mask'].cuda(device)
                feat_batch_t['mask'] = feat_batch_t['mask'].cuda(device)
                feat_batch_t['enhance_img'] = feat_batch_t['enhance_img'].cuda(device)

                auth_result_batch,rec_feat_batch = recognize_batch(feat_batch_t, feat_batch_s, self.model_api.device)
                
                #trans合理性判定
                H_rotation =  torch.atan2(rec_feat_batch['model'][:,1,0], rec_feat_batch['model'][:,0,0])

                H_shear = torch.atan2(- rec_feat_batch['model'][:,0,1], rec_feat_batch['model'][:,1,1])
                H_shear = H_shear - H_rotation

                ss = torch.sum(rec_feat_batch['model'] ** 2, axis=1)
                ss[:,1] = ss[:,1] / (torch.tan(H_shear)**2 + 1)
                H_scale =  torch.sqrt(ss)[:,:2]

                sx, sy = H_scale[:,0], H_scale[:,1]
                sx_mask = (sx >= 0.4)*(sx <= 2)
                sy_mask = (sy >= 0.6)*(sy <= 1.5)
                shear_mask = (H_shear >= -1)*(H_shear <= 1)

                score_batch = torch.zeros_like(auth_result_batch).float()
                model_valid = rec_feat_batch['model_valid']
                model_valid = model_valid*sx_mask*sy_mask*shear_mask
                cls_samples = model_valid.nonzero()[:,0]
                if len(cls_samples) > 0: #跳过没trans的

                    score_batch_samples, img_merge_batch = compare_batch(feat_batch_t, feat_batch_s, rec_feat_batch, cls_samples, self.model_api)

                    # score_batch_samples = self.model_match.rf_cls_batch(feat_batch_t, feat_batch_s, cls_samples, None).squeeze()
                    score_batch[cls_samples] = score_batch_samples
                auth_result_batch = score_batch > 65535

                remain_samples = (~auth_result_batch).nonzero()[:,0]
                score_batch = score_batch.cpu().numpy()
                auth_result_batch = auth_result_batch.cpu().numpy()
                model_save = rec_feat_batch['model'].cpu().numpy()*256

                auth_succ_num += auth_result_batch.sum()
                for idx, (name, auth_result, score) in enumerate(zip(feat_batch_s['name'],auth_result_batch,score_batch)):
                    if auth_result > 0 and pdfeat is not None: 
                        # get_match_feat(self.template[i][0], feat, rec_feat, feat_info)
                        #print("auth", name, self.template[i][1])
                        trans_save = (model_save[idx][:2].reshape(-1)+0.5).astype(np.int).tolist()
                        log_save = [name, self.template[i][1], score] + trans_save
                        pdfeat.loc[len(pdfeat)] = log_save  #, *(feat_info.feature)
                    if FLAG_SAVE_IMG == 1 and torch.where(cls_samples==idx)[0].shape[0] == 1:
                        img_match = img_merge_batch[torch.where(cls_samples==idx)].squeeze(0).cpu().numpy().astype(np.uint8).transpose(1,2,0)
                        out_img_path = Path_out_img + str(int(score)) + '_' + name[-16:-4].replace('/', '_') + \
                            "_" + self.template[i][1][-16:-4].replace('/', '_') + "_match.bmp"
                        cv2.imwrite(out_img_path, img_match)

                if self.mode == 0: 
                    if len(remain_samples) == 0:
                        break
                    remain_name = []
                    for idx in remain_samples:
                        remain_name.append(feat_batch_s['name'][idx])
                    feat_batch_s['name'] = remain_name
                    for key in list(feat_batch_s.keys()):
                        if torch.is_tensor(feat_batch_s[key]):
                            if feat_batch_s[key].shape[0] > 0:
                                feat_batch_s[key] = feat_batch_s[key][remain_samples]

        self.authed_num = self.authed_num + num_s
        self.auth_succ_num = self.auth_succ_num + auth_succ_num
        if len(auth_result_batch) > 0:
            auth_result = auth_result_batch[0]
        else:
            auth_result = 0

        if auth_result == 0 and self.flag_reserve_fail_num > 0 and feat is not None:  #失败缓存
            self.reserve_fail_temple.append([feat, fgimage[0][0]])
            if len(self.reserve_fail_temple) > self.flag_reserve_fail_num:
                self.reserve_fail_temple.pop(0) 

        return auth_result, feat
    
    def reserve_fail(self):
        img_h, img_w = self.template[0][0]['h'], self.template[0][0]['w']
        img_L16_h, img_L16_w = self.template[0][0]['level_16_img'].shape
        device = self.model_api.device

        batchsize_s = 200
        bNum_limit = 100
        wNum_limit = 100
        batch_size = len(self.reserve_fail_temple)
        feat_batch_s  =  {
            'h':img_h,
            'w':img_w,
            'prob_idx':2,    #记录存放位置
            'angles_idx':3,
            'dbvalue_idx':4,
            'name':[],
            'quality':torch.zeros(batch_size),
            'mask':torch.zeros((batch_size,1,img_h,img_w)),
            'wet_score':torch.zeros(batch_size),
            'bvalid_mask':torch.zeros(batch_size,bNum_limit),
            'wvalid_mask':torch.zeros(batch_size,wNum_limit),
            'pts_b':torch.zeros((batch_size,bNum_limit,5)),        #torch Nbx5 (float) (x,y,prob,angles,dbvlue)
            'pts_w':torch.zeros((batch_size,wNum_limit,5)),        #torch Nwx5 (float) (x,y,prob,angles,dbvlue)
            'desc_b':torch.zeros((batch_size,bNum_limit,256)),     #torch Nbx(256bit)
            'desc_w':torch.zeros((batch_size,wNum_limit,256)),     #torch Nwx(256bit)
            'enhance_img':torch.zeros((batch_size,1,img_h,img_w)),   #numpy h*w
            'level_16_img':torch.zeros((batch_size,1,img_L16_h,img_L16_w)), #numpy h*w
        }

        bNum_runningMax = 0 #根据实际，动态调整规格化点数
        wNum_runningMax = 0 #根据实际，动态调整规格化点数
        batch_valid = torch.ones(len(self.reserve_fail_temple))
        for idx, idx_sample in enumerate(self.reserve_fail_temple): #sample 数据batch化
            feat = self.reserve_fail_temple[idx][0]
            name = self.reserve_fail_temple[idx][1]
            if feat == None:
                batch_valid[idx] = 0
                continue

            black_num = feat['black_num']
            white_num = len(feat['desc']) - black_num
            assert black_num <= bNum_limit
            assert white_num <= wNum_limit

            if black_num > bNum_runningMax:
                bNum_runningMax = black_num
            if white_num > wNum_runningMax:
                wNum_runningMax = white_num

            feat_batch_s['bvalid_mask'][idx,:black_num] = 1
            feat_batch_s['wvalid_mask'][idx,:white_num] = 1
            feat_batch_s['pts_b'][idx,:black_num,:2] = torch.from_numpy(feat['pts'][:black_num])
            feat_batch_s['pts_b'][idx,:black_num,2]  = torch.from_numpy(feat['prob'][:black_num])
            feat_batch_s['pts_b'][idx,:black_num,3]  = torch.from_numpy(feat['angles'][:black_num])
            feat_batch_s['pts_b'][idx,:black_num,4]  = torch.from_numpy(feat['dbvalue'][:black_num])
            feat_batch_s['desc_b'][idx,:black_num]   = torch.from_numpy(feat['desc'][:black_num])

            feat_batch_s['pts_w'][idx,:white_num,:2] = torch.from_numpy(feat['pts'][black_num:])
            feat_batch_s['pts_w'][idx,:white_num,2]  = torch.from_numpy(feat['prob'][black_num:])
            feat_batch_s['pts_w'][idx,:white_num,3]  = torch.from_numpy(feat['angles'][black_num:])
            feat_batch_s['pts_w'][idx,:white_num,4]  = torch.from_numpy(feat['dbvalue'][black_num:])
            feat_batch_s['desc_w'][idx,:white_num]   = torch.from_numpy(feat['desc'][black_num:])

            
            feat_batch_s['name'].append(name)
            feat_batch_s['quality'][idx] = feat['quality']
            feat_batch_s['wet_score'][idx] = feat['wet_score']
            feat_batch_s['mask'][idx,:] = torch.from_numpy(feat['mask'])
            feat_batch_s['enhance_img'][idx,:] = torch.from_numpy(feat['enhance_img'])
            feat_batch_s['level_16_img'][idx,:] = torch.from_numpy(feat['level_16_img'])
        
        batch_valid = batch_valid.nonzero()[:,0]
        #最小化数量，有利于减少计算量
        feat_batch_s['bvalid_mask'] = feat_batch_s['bvalid_mask'][:,:bNum_runningMax]
        feat_batch_s['pts_b']       = feat_batch_s['pts_b'][:,:bNum_runningMax]
        feat_batch_s['desc_b']      = feat_batch_s['desc_b'][:,:bNum_runningMax]
        feat_batch_s['wvalid_mask'] = feat_batch_s['wvalid_mask'][:,:wNum_runningMax]
        feat_batch_s['pts_w']       = feat_batch_s['pts_w'][:,:wNum_runningMax]
        feat_batch_s['desc_w']      = feat_batch_s['desc_w'][:,:wNum_runningMax]

        #数据全部加载到GPU
        feat_batch_s['quality'] = feat_batch_s['quality'][batch_valid].cuda(device)
        feat_batch_s['wet_score'] = feat_batch_s['wet_score'][batch_valid].cuda(device)
        feat_batch_s['pts_b'] = feat_batch_s['pts_b'][batch_valid].cuda(device)
        feat_batch_s['pts_w'] = feat_batch_s['pts_w'][batch_valid].cuda(device)
        feat_batch_s['desc_b'] = feat_batch_s['desc_b'][batch_valid].cuda(device)
        feat_batch_s['desc_w'] = feat_batch_s['desc_w'][batch_valid].cuda(device)
        feat_batch_s['bvalid_mask'] = feat_batch_s['bvalid_mask'][batch_valid].cuda(device)
        feat_batch_s['wvalid_mask'] = feat_batch_s['wvalid_mask'][batch_valid].cuda(device)
        feat_batch_s['mask'] = feat_batch_s['mask'][batch_valid].cuda(device)
        feat_batch_s['enhance_img'] = feat_batch_s['enhance_img'][batch_valid].cuda(device)
        feat_batch_s['level_16_img'] = feat_batch_s['level_16_img'][batch_valid].cuda(device)

        for i in range(len(self.template) - 1, len(self.template) - 2, -1):  #reverse

            feat_t = self.template[i][0]
            feat_t_pts_b = torch.from_numpy(feat_t['pts'][:feat_t['black_num']]).unsqueeze(0)
            feat_t_pts_w = torch.from_numpy(feat_t['pts'][feat_t['black_num']:]).unsqueeze(0)
            feat_t_pts_b = torch.cat([feat_t_pts_b,torch.from_numpy(feat_t['prob'][:feat_t['black_num']])[None,:,None]],dim=2)
            feat_t_pts_b = torch.cat([feat_t_pts_b,torch.from_numpy(feat_t['angles'][:feat_t['black_num']])[None,:,None]],dim=2)
            feat_t_pts_b = torch.cat([feat_t_pts_b,torch.from_numpy(feat_t['dbvalue'][:feat_t['black_num']])[None,:,None]],dim=2)
            feat_t_pts_w = torch.cat([feat_t_pts_w,torch.from_numpy(feat_t['prob'][feat_t['black_num']:])[None,:,None]],dim=2)
            feat_t_pts_w = torch.cat([feat_t_pts_w,torch.from_numpy(feat_t['angles'][feat_t['black_num']:])[None,:,None]],dim=2)
            feat_t_pts_w = torch.cat([feat_t_pts_w,torch.from_numpy(feat_t['dbvalue'][feat_t['black_num']:])[None,:,None]],dim=2)

            b_num_t = feat_t['black_num']
            w_num_t = len(feat_t['desc']) - feat_t['black_num']
            assert len(feat_t['desc']) > 0
            feat_t_desc_b = torch.from_numpy(feat_t['desc'][:feat_t['black_num']]).unsqueeze(0)
            feat_t_desc_w = torch.from_numpy(feat_t['desc'][feat_t['black_num']:]).unsqueeze(0)
            bvalid_mask_t = torch.ones(1,b_num_t)
            wvalid_mask_t = torch.ones(1,w_num_t)

            if b_num_t == 0:
                bvalid_mask_t = torch.zeros_like(wvalid_mask_t)
                feat_t_pts_b  = feat_t_pts_w.clone()
                feat_t_desc_b = feat_t_desc_w.clone()
            if w_num_t == 0:
                wvalid_mask_t = torch.zeros_like(bvalid_mask_t)
                feat_t_pts_w  = feat_t_pts_b.clone()
                feat_t_desc_w = feat_t_desc_b.clone() 

            feat_batch_t  =  {
                'h':img_h,
                'w':img_w,
                'prob_idx':2,    #记录存放位置
                'angles_idx':3,
                'dbvalue_idx':4,
                'name':feat_t['name'],
                'quality':torch.tensor(feat_t['quality']),
                'mask':torch.from_numpy(feat_t['mask'])[None,None,:,:],
                'wet_score':torch.tensor(feat_t['wet_score']),
                'bvalid_mask':bvalid_mask_t,
                'wvalid_mask':wvalid_mask_t,
                'pts_b':feat_t_pts_b,        #torch Nbx5 (float) (x,y,prob,angles,dbvlue)
                'pts_w':feat_t_pts_w,        #torch Nwx5 (float) (x,y,prob,angles,dbvlue)
                'desc_b':feat_t_desc_b,     #torch Nbx(256bit)
                'desc_w':feat_t_desc_w,     #torch Nwx(256bit)
                'enhance_img':torch.from_numpy(feat_t['enhance_img'])[None,None,:,:],   #tensor 1*1*h*w
                'level_16_img':torch.from_numpy(feat_t['level_16_img'])[None,None,:,:], #tensor 1*1*h*w
            }

            #数据全部加载到GPU
            feat_batch_t['pts_b']  = feat_batch_t['pts_b'].cuda(device)
            feat_batch_t['pts_w']  = feat_batch_t['pts_w'].cuda(device)
            feat_batch_t['desc_b'] = feat_batch_t['desc_b'].cuda(device)
            feat_batch_t['desc_w'] = feat_batch_t['desc_w'].cuda(device)
            feat_batch_t['bvalid_mask'] = feat_batch_t['bvalid_mask'].cuda(device)
            feat_batch_t['wvalid_mask'] = feat_batch_t['wvalid_mask'].cuda(device)
            feat_batch_t['mask'] = feat_batch_t['mask'].cuda(device)
            feat_batch_t['enhance_img'] = feat_batch_t['enhance_img'].cuda(device)

            auth_result_batch,rec_feat_batch = recognize_batch(feat_batch_t, feat_batch_s, self.model_api.device)
            
            #trans合理性判定
            H_rotation =  torch.atan2(rec_feat_batch['model'][:,1,0], rec_feat_batch['model'][:,0,0])

            H_shear = torch.atan2(- rec_feat_batch['model'][:,0,1], rec_feat_batch['model'][:,1,1])
            H_shear = H_shear - H_rotation

            ss = torch.sum(rec_feat_batch['model'] ** 2, axis=1)
            ss[:,1] = ss[:,1] / (torch.tan(H_shear)**2 + 1)
            H_scale =  torch.sqrt(ss)[:,:2]

            sx, sy = H_scale[:,0], H_scale[:,1]
            sx_mask = (sx >= 0.4)*(sx <= 2)
            sy_mask = (sy >= 0.6)*(sy <= 1.5)
            shear_mask = (H_shear >= -1)*(H_shear <= 1)

            score_batch = torch.zeros_like(auth_result_batch).float()
            model_valid = rec_feat_batch['model_valid']
            model_valid = model_valid*sx_mask*sy_mask*shear_mask
            cls_samples = model_valid.nonzero()[:,0]
            if len(cls_samples) > 0: #跳过没trans的

                score_batch_samples, img_merge_batch = compare_batch(feat_batch_t, feat_batch_s, rec_feat_batch, cls_samples, self.model_api)

                # score_batch_samples = self.model_match.rf_cls_batch(feat_batch_t, feat_batch_s, cls_samples, None).squeeze()
                score_batch[cls_samples] = score_batch_samples
            auth_result_batch = score_batch > 65535
        # print(len(self.reserve_fail_temple), auth_result_batch.shape[0], auth_result_batch)

        if auth_result_batch.sum() > 0:
            reserve_fail_temple_succ = [self.reserve_fail_temple[i] for i in range(auth_result_batch.shape[0]) if auth_result_batch[i] == True]
            reserve_fail_temple_fail = [self.reserve_fail_temple[i] for i in range(auth_result_batch.shape[0]) if auth_result_batch[i] == False]

            self.reserve_fail_temple = reserve_fail_temple_fail
            for _tpl in reserve_fail_temple_succ:
                self.study(_tpl[0], _tpl[1])

    def study(self, feat, name):
        #print("study", name)
        if len(self.template) < self.template_num_max:
            self.template.append([feat, name])  #todo:study strategy like Htrans guide     
            self.reserve_fail()
        else:
            self.template.append([feat, name])
            self.reserve_fail()
            self.template.pop(0)    #todo: del strategy

def finger_enroll_auth(fgimage, opt):

    device = GPU_ids[os.getpid() % len(GPU_ids)]

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    pdfeat = pd.DataFrame(columns = COLUMNS)

    try:
        model93 = model_api(device)
    except:
        print(fgimage[0][0], "initialize model failed", os.getpid(), device, meminfo.free/1024**2)
        return pdfeat,0,0
    else:
        print(fgimage[0][0], "initialize model success", os.getpid(), device, meminfo.free/1024**2)
    
    fg = finger(opt['template_num_max'], model93, opt['mode'], opt['fail_reserve_num'])

    # for _name, _img in fgimage:
    #     if fg.enrolled_num < opt['enroll_num_max']:
    #         fg.enroll(_img, _name)
    #     else:
    #         auth_result, feat = fg.auth(_img, _name, pdfeat)
    #         if (auth_result == 1 and opt['study_enable'] == 1) or (opt['study_force'] == 1):
    #             fg.study(feat, _name)

    for enroll_img_index, (_name, _img) in enumerate(fgimage):
        fg.enroll(_img, _name)
        if fg.enrolled_num == opt['enroll_num_max']:
            break

    if (opt['study_enable'] == 1) or (opt['study_force'] == 1):
        for _name, _img in fgimage[enroll_img_index+1:]:
            # auth_result, feat = fg.auth(_img, _name, pdfeat)
            auth_result, feat = fg.auth_batch([[_name, _img]], pdfeat)
            if (auth_result == 1 and opt['study_enable'] == 1) or (opt['study_force'] == 1):
                fg.study(feat, _name)
    else:
        auth_result, feat = fg.auth_batch(fgimage[enroll_img_index+1:], pdfeat)

    print('end', fgimage[0][0], fg.authed_num, fg.auth_succ_num, len(fg.template))

    return pdfeat, fg.authed_num, fg.auth_succ_num

def frr_test(path, opt, p_num, device):

    p = Pool(p_num)
    time_start = time.time()
    processes = []

    for pid in sorted(os.listdir(path)):
        if not os.path.isdir(path + '/' + pid):
            continue
        if 'X' in pid:
            continue
        person_result = [pid, []]

        for fid in sorted(os.listdir(path + '/' + pid)):
            if 'X' in fid:
                continue
            fgimage = []

            for name in sorted(glob.glob(path + '/' + pid + '/' + fid + '/*.bmp' )):
                img = np.array(Img.open(name))
                fgimage.append([name,img])

            # ret = finger_enroll_auth(fgimage, opt['template_num_max'], 1, opt, device%8)
            # pdfeat = pd.concat([pdfeat, ret])
            ret = p.apply_async(finger_enroll_auth, args=(fgimage, opt))
            # ret = finger_enroll_auth(fgimage, opt)
            person_result[1].append(ret)
            #print(pid, fid,"start")

        processes.append(person_result)

    p.close()
    p.join()
    print("join ok ")

    authed_num = 0
    auth_succ_num = 0
    for person_result in processes:
        pdfeat = pd.DataFrame(columns = COLUMNS)
        for ret in person_result[1]:
            try:
                fingerpd, _authed_num, _auth_succ_num = ret.get()
            except:
                import traceback
                print("#### error ####: ",person_result[0])
                traceback.print_exc(file=open(os.path.join(opt['pd_path'],'frr_error.txt'),'a'))
            else:
                pdfeat = pd.concat([pdfeat, fingerpd])
                authed_num = authed_num + _authed_num
                auth_succ_num = auth_succ_num + _auth_succ_num
        pdname = path.split('/')[-1] +'_'+person_result[0] +"_frr_"+ str(len(pdfeat)) + ".csv"  #h,w = df.shape
        pdfeat.to_csv(opt['pd_path']+"/"+ pdname)
        
    print("authed_num", authed_num, "auth_fail_num", authed_num - auth_succ_num, "frr", 1 - auth_succ_num / authed_num)   
    print("authed_num", authed_num, "auth_fail_num", authed_num - auth_succ_num, "frr", auth_succ_num / authed_num, file=open(os.path.join(opt['pd_path'],path.split('/')[-1] +"_"+ 'frlog.txt'),'a'))
    time_frr_end = time.time()
    print("frr test time:", (time_frr_end - time_start) / 60.)

    return

def finger_auth(template_fg, fgimage):

    device = GPU_ids[os.getpid() % len(GPU_ids)]   #device与pid绑定，可以避免某pid在不同device重复申请空间，nidia-smi出现冗余进程

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    print('start', fgimage[0][0], os.getpid(), meminfo.free/1024**2)

    pdfeat = pd.DataFrame(columns = COLUMNS)
    try:
        model93 = model_api(device)
    except:
        print("initialize model failed", os.getpid(), device, meminfo.free/1024**2)
        return pdfeat,0,0
    else:
        print("initialize model success", os.getpid(), device, meminfo.free/1024**2)

    template_fg.model_api = model93


    template_fg.auth_batch(fgimage, pdfeat)
    # for image in fgimage:
    #     template_fg.auth(image[1], image[0], pdfeat)

    print('end', fgimage[0][0], os.getpid(), template_fg.authed_num, template_fg.auth_succ_num, len(template_fg.template))

    return pdfeat, template_fg.authed_num, template_fg.auth_succ_num

def far_test(path, opt, p_num, device):
    
    p = Pool(p_num)
    time_start = time.time()
    processes = []
    template_pool = []
    model93 = model_api(device%8)
    
    authed_num = 0
    auth_succ_num = 0
    for pid in sorted(os.listdir(path)):
        if not os.path.isdir(path + '/' + pid):
            continue

        if 'X' in pid:
            continue

        template_pool_person = []

        for fid in sorted(os.listdir(path + '/' + pid)):
            if 'X' in fid:
                continue

            fg = finger(opt['template_num_max'], model93, opt["mode"], opt["fail_reserve_num"])

            for name in sorted(glob.glob(path + '/' + pid + '/' + fid + '/*.bmp' )): #[:opt["enroll_num_max"]]:
                img = np.array(Img.open(name))

                if fg.enrolled_num < opt['enroll_num_max']:
                    fg.enroll(img, name)
                else:
                    auth_result, feat = fg.auth_batch([[name, img]], None)
                    if (auth_result == 1 and opt['study_enable'] == 1) or (opt['study_force'] == 1):
                        fg.study(feat, name)
            authed_num = authed_num + fg.authed_num
            auth_succ_num = auth_succ_num + fg.auth_succ_num
            fg.model_api = None
            fg.auth_succ_num = 0
            fg.authed_num = 0
            template_pool_person.append(fg)
            template_pool.append(fg)
            print("enroll", fg.template[0][1], len(fg.template))

        if opt["far_mode_same_person_cross_finger"]:
            person_result = [pid, []]
            finger_id = 0
            for fid in sorted(os.listdir(path + '/' + pid)):
                if 'X' in fid:
                    continue
                _template_pool_person = template_pool_person.copy()
                _template_pool_person.pop(finger_id)
                fgimage = []

                for name in sorted(glob.glob(path + '/' + pid + '/' + fid + '/*.bmp' ))[opt["far_start_idx"]:]:  #后xxx张进行far比对
                    img = np.array(Img.open(name))
                    fgimage.append([name,img])

                for template_fg in _template_pool:
                    ret = p.apply_async(finger_auth, args=(template_fg, fgimage))
                    person_result[1].append(ret)

                finger_id = finger_id + 1

            processes.append(person_result)

    print("authed_num", authed_num, "auth_fail_num", authed_num - auth_succ_num, "frr", 1 - auth_succ_num / authed_num)   
    print("authed_num", authed_num, "auth_fail_num", authed_num - auth_succ_num, "frr", auth_succ_num / authed_num, file=open(os.path.join(opt['pd_path'],path.split('/')[-1] +"_"+ 'frlog.txt'),'a'))

    if opt["far_mode_cross_person_cross_finger"]:
        finger_id = 0
        for pid in sorted(os.listdir(path)):
            if not os.path.isdir(path + '/' + pid):
                continue
            if 'X' in pid:
                continue
            person_result = [pid, []]

            for fid in sorted(os.listdir(path + '/' + pid)):
                if 'X' in fid:
                    continue

                _template_pool = template_pool.copy()
                _template_pool.pop(finger_id)
                fgimage = []

                for name in sorted(glob.glob(path + '/' + pid + '/' + fid + '/*.bmp' ))[opt["far_start_idx"]:]:  #后xxx张进行far比对
                    img = np.array(Img.open(name))
                    fgimage.append([name,img])

                for template_fg in _template_pool:
                    ret = p.apply_async(finger_auth, args=(template_fg, fgimage))
                    # ret = finger_auth(template_fg, fgimage)
                    person_result[1].append(ret)
                    # time.sleep(5)
                    # print("auth start template_fg 分析open files ")
                    # os.system('ls /proc/'+str(os.getpid())+'/fd | wc -l')
                    # lsof -p 1776338 >openfiles.log  从log里面分析fd的增量来源

                finger_id = finger_id + 1

            processes.append(person_result)

    p.close()
    p.join()
    print("join ok ")

    authed_num = 0
    auth_succ_num = 0
    for person_result in processes:
        pdfeat = pd.DataFrame(columns = COLUMNS)

        for ret in person_result[1]:
            try:
                fingerpd, _authed_num, _auth_succ_num = ret.get()
            except:
                import traceback
                traceback.print_exc(file=open(os.path.join(opt['pd_path'],'far_error.txt'),'a'))
            else:
                pdfeat = pd.concat([pdfeat, fingerpd])
                authed_num = authed_num + _authed_num
                auth_succ_num = auth_succ_num + _auth_succ_num

        pdname = path.split('/')[-1] +'_'+person_result[0] +"_far_"+ str(len(pdfeat)) + ".csv"  #h,w = df.shape
        pdfeat.to_csv(opt['pd_path']+"/"+ pdname)

    print("authed_num", authed_num, "auth_succ_num", auth_succ_num, "far", auth_succ_num / authed_num)   
    print("authed_num", authed_num, "auth_succ_num", auth_succ_num, "far", auth_succ_num / authed_num, file=open(os.path.join(opt['pd_path'],path.split('/')[-1] +"_"+ 'falog.txt'),'a'))
    time_frr_end = time.time()
    print("far test time:", (time_frr_end - time_start) / 60.)

    return

#study_enable = 0 study_force = 0  关学习模式，录入x张，后面150-x张比对
#enroll_num_max = 1 template_num_max = 150 study_force = 1 强制学习，相当于两两匹配
#mode: 0表示sample配上某个子模版就退出，  1表示sample与所有的子模板都比对一遍
#fail_reserve_num  失败缓存数量，设置为0等于关闭，6195设置为20
def main():
    now = datetime.datetime.now()
    Path_out_pd = r'../result'
    if not os.path.isdir(Path_out_pd):
        os.mkdir(Path_out_pd)
    Path_out_pd = Path_out_pd + '/' + now.strftime('%Y-%m-%d-%H-%M-%S')+'-frr-开学习-关失败缓存-v3-65535'
    if not os.path.isdir(Path_out_pd):
        os.mkdir(Path_out_pd)
    if FLAG_SAVE_IMG == 1 and not os.path.isdir(Path_out_img):
        os.mkdir(Path_out_img)
    root = r"/data/yey/work/FPRDL/dataset"#"../../img_base/img_ori_data""../test_img/6193连续采图-debase" 
    datasets = [
        r'6193_DK7_random_merge_test',
        # r'6193_DK7_XA_rot_test',
        # r'6193-DK7-140-8-powder_test',
        # r'6193-DK7-140-8-rotate_test',
        # r'6193_DK7_CDXA_normal_merge_test',
        # r'6193_DK7_merge_test2',
        # r'6193-DK7-140-8-wet2_clear_test',
        # r'6193_DK7_partialpress_test',
        ]
    optfrr = {'enroll_num_max':20,'template_num_max':94, 'study_enable': 1,'study_force' : 0,'mode' : 0,
           "far_mode_same_person_cross_finger":0, "far_mode_cross_person_cross_finger":0, "far_start_idx":0, "fail_reserve_num":0, 'pd_path':Path_out_pd}
    optfar = {'enroll_num_max':20,'template_num_max':94, 'study_enable': 1,'study_force' : 0, 'mode' : 0,
           "far_mode_same_person_cross_finger":0, "far_mode_cross_person_cross_finger":1, "far_start_idx":0, "fail_reserve_num":0, 'pd_path':Path_out_pd}
    flag_frr_test = 1
    flag_far_test = 0
    # p = Pool(1)   #经测试，单进程显存占用1.7G, 单GPU(3090)最多12进程，可能会超出GPU memory,所以多进程需管控每张卡的进程上限  2个cpu，每个有20个物理核
    device = 0

    multiprocessing.set_start_method('spawn')
    if flag_frr_test == 1:
        for dataset_name in datasets:
            path = os.path.join(root,dataset_name)
            frr_test(path, optfrr, 8, device)
    if flag_far_test == 1: #循环跑FA会有显存释放问题，出现OOM，需要修改
        for dataset_name in datasets:
            path = os.path.join(root,dataset_name)
            far_test(path, optfar, 8, device)


if __name__ == "__main__":
    main()
