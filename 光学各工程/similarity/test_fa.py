'''
慧婷测试代码，跑fa
'''
'''
Author: lif fu.li@gigadevice.com
Date: 2023-09-21 19:36:52
LastEditors: lif fu.li@gigadevice.com
LastEditTime: 2023-11-06 12:01:02
FilePath: /Auto_code_trans/get_test.py
Description: 

Copyright (c) 2005-2022 GigaDevice/Silead Inc.
All rights reserved
The present software is the confidential and proprietary information of GigaDevice/Silead Inc.
You shall not disclose the present software and shall use it only in accordance with the terms of the license agreement you entered into with GigaDevice/Silead Inc.
This software may be subject to export or import laws in certain countries.
'''
#from MobileNet import *
from pathlib import Path
from collections import deque
import numpy as np
from pathlib import Path
import sys
sys.path.append('..')
# from simplejson import OrderedDict
import argparse
import torch
import os
from tensorboard.compat.proto.graph_pb2 import *
from torch.utils.tensorboard._pytorch_graph import *

import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from models.MobileNet import *
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import shutil
# os.environ['CUDA_VISIBLE_DEVICES']='5'

def get_warning(img_dir,save_dir):
    all_img =[str(f) for f in Path(img_dir).rglob('*.bmp')]
    abs_list= []
    w_l = 0
    for idx,img in enumerate(all_img):
        img_list = (img.split('/')[-1]).split('_')
        pred = int(img_list[0])
        label = int(img_list[1])
        # label = int(img_list[1][2:])
        print(idx, img)
        cha = abs(pred-label)
        abs_list.append(cha)
        if cha>10:
            w_l+=1
            shutil.copy(img, os.path.join(save_dir,str(cha)+'_'+img.split('/')[-1]))
    print('mean:', np.mean(np.array(abs_list)), 'std:', np.std(np.array(abs_list)))
    print(w_l)

def resize_tensor(img, ow, oh):
    img = img.unsqueeze(dim=0)
    img = F.interpolate(img, size=[ow, oh], mode="bilinear", align_corners=False)
    return img.squeeze(dim=0)
    # return img
import cv2
def main():
    net =   MNV3_bufen_new5(2, 1, 4, n_blocks=1).cuda()
    saveimg = 0
    # pthpath = '/home/jianght/000_codes/009_sim/quality_6193/checkpoints/feaure_20231117/270_net_G.pth'
    # pthpath = '/home/jianght/000_codes/009_sim/quality_6193/checkpoints/feaure_1127_wet/270_net_G.pth'
    
    # pthpath = '/home/jianght/000_codes/009_sim/quality_6193/checkpoints/feaure_20231117_change/270_net_G.pth'
    # pthpath = "/home/jianght/000_codes/009_sim/quality_6193/checkpoints/feaure_1201_changelabel/270_net_G.pth"
    # pthpath = "/home/jianght/000_codes/009_sim/quality_6193/checkpoints/feaure_1206/270_net_G.pth"
    # 增强图模型
    # pthpath = "/home/zhangsn/simi_net/quality_6193/checkpoints/enhance_0440/270_net_G.pth"
    # pthpath = "/home/jianght/000_codes/009_sim/quality_6193/checkpoints/enhance_0411lianxu/270_net_G.pth"
    pthpath = "/home/zhangsn/simi_net/quality_6193/checkpoints/6193simi_en1_417/270_net_G.pth"
    
    
    net.load_state_dict(torch.load(pthpath))
    # path = '/home/jianght/000_codes/009_sim/quality_6193/test_img/'
    # img_dir = "/home/zhangsn/simi_net/000_data/testdata_level16_far/valid/"
    # img_dir = "/home/jianght/001_data/003_featuredata_20231117/train/normal-pair-sele/"
    # img_dir = "/home/zhangsn/simi_net/000_data/000_alldata/03transpair/"
    # img_dir = "/home/zhangsn/simi_net/000_data/000_alldata/03transpair/far/DK4_C3U/"
    # img_dir = "/home/zhangsn/simi_net/000_data/000_alldata/03transpair/far/DK7/"
    # add_data new-lianxu-succes /data/guest/simi_data/test/add_data/
    img_dir = "/home/jianght/004_simi_results/2_q/outimg_q_pred_enhance/"
    img_dir = "/home/jianght/004_simi_results/2_q/outimg_q_pred_enhance_65536/"
    # img_dir = "/data/guest/jht/6195_fa/"
    # img_dir = "/data/guest/simi_data/test/bad_cases/"
    all_list = [str(f) for f in Path(img_dir).rglob('*.bmp')]
    # save_dir = '/home/jianght/001_results_simi/feaure_1201_changelabel/test/'
    save_dir ="/data/guest/simi_data/6195_fa/"
    warn_dir = os.path.join(save_dir,'warn')
    
    # img_csv = "/home/jianght/001_data/003_featuredata_20231117/test_info/all_data_change.csv"
    # df_r = pd.read_csv(img_csv)
    # all_list=df_r['img_path'].to_list()
    # all_label= df_r['raw_label'].to_list()
    # img_dir = '/data/guest/simi_data/test/'
    
    net.eval()
    content =[]
    len_all = len(all_list)
    
    img_name_list=[]
    net_simi_list=[]
    net_class_list = []
    
    for idx, img in enumerate(all_list):
        img_all = Image.open(img).convert('RGB')
        # img_tensor = torch.from_numpy(np.array(img_PIL) / 255).cuda()
        img_tensor = torch.from_numpy(np.array(img_all))
        # all = img_tensor.chunk(4,dim=1)
        # img_tensor = all[3] #第四张为融合图
        # mask = np.array(Image.fromarray(all[2].numpy()).convert('L'))
        
        # img_p = img_tensor[:,:,0].cpu().numpy()
        # img_a = img_tensor[:,:,1].cpu().numpy() 
        # img_p[mask[:,:]==0]=0
        # img_a[mask[:,:]==0]=0
        

        img_p = img_tensor[:,:,0].cpu().numpy()
        img_a = img_tensor[:,:,1].cpu().numpy()
        img_p = cv2.resize(img_p,(18, 61))
        img_a = cv2.resize(img_a,(18, 61))
        
        # save_mg = Image.new("L", (img_a.shape[1], img_a.shape[0]))
        # b = np.array(save_mg)
        # img_merge = cv2.merge([b, img_a,img_p])
        
        # img_p = img_tensor[:,:,0].cpu()
        # img_a = img_tensor[:,:,1].cpu()
        
        img_a = transforms.ToTensor()(img_a)
        img_p = transforms.ToTensor()(img_p)

        
        AB=torch.cat((img_a,img_p),dim=0)
        # AB=torch.cat((img_p,img_a),dim=0)
        with torch.no_grad():
            out1=net(AB.unsqueeze(dim=0).cuda())
      
        pred = int(out1*265+55.5)
        
        #为csv存储做准备
        img_name = os.path.split(img)[1]
        # net_simi = int(img_name.split('_')[0])
        net_class = int(img_name.split('_')[0])
        img_name_list.append(img)
        net_simi_list.append(pred)
        net_class_list.append(net_class)
        
        if idx%100 ==0:
            print(idx,'/',len_all,':',pred)
        # rename = str(pred)+'_'+str(all_label[idx])+'_'+img.split('/')[-1]
        if saveimg == 1:
            rename = str(pred)+'_'+img.split('/')[-1]
            save_path = os.path.join(img[:-len(img.split('/')[-1])],rename)
            save_path=save_path.replace(img_dir, save_dir)
            Path(save_path).parent.mkdir(parents=True,exist_ok=True)
            # print(save_path)
            shutil.copy(img, save_path)
            # shutil.move(img,save_path)
            # cv2.imwrite(save_path,img_merge)
    
    
    data={'name':img_name_list,'net_simi':net_simi_list,'net_class':net_class_list}  #
    df=pd.DataFrame(data)
    df.to_csv('/data/guest/simi_data/out_far_simi.csv',index=True)
    
    net_simi_list= np.sort(net_simi_list)
    unique_elements,counts = np.unique(net_simi_list,return_counts=True)
    cal_counts = np.zeros_like(counts)
    cal_counts[0]=counts[0]
    for i in range(len(counts)-1):
        cal_counts[i+1]=cal_counts[i]+counts[i+1]
    ratio = cal_counts/len_all*100
    data={'net_simi':unique_elements,'counts':counts,'cal_counts':cal_counts,'ratio':ratio}  #
    df=pd.DataFrame(data)
    df.to_csv('/data/guest/simi_data/out_far_simi_counts.csv',index=True)
       
    #     content.append([img, pred])
    # df = pd.DataFrame(content, columns=['img_path', 'pred'])
    # # df.to_csv(os.path.join(save_dir,'yym_trainfr_results_change.csv'))
    # df.to_csv(os.path.join(save_dir,'tmp.csv'))
    # if not os.path.exists(warn_dir):
    #     os.mkdir(warn_dir)
    # print(warn_dir)
    # get_warning(save_dir,warn_dir)

def main_csv():
    net =   MNV3_bufen_new5(2, 1, 4, n_blocks=1).cuda()
    pthpath =  ''  
    net.load_state_dict(torch.load(pthpath))
    csv_path = ''
    # all_list = [str(f) for f in Path(img_dir).rglob('*.bmp')]
    df=pd.read_csv(csv_path)
    all_list=df['img_path'].to_list()
    net.eval()
    content =[]
    len_all = len(all_list)
    for idx, img in enumerate(all_list):
        img_all = Image.open(img).convert('RGB')
        # img_tensor = torch.from_numpy(np.array(img_PIL) / 255).cuda()
        img_tensor = torch.from_numpy(np.array(img_all))
        # all = img_tensor.chunk(4,dim=1)
        # img_tensor = all[3] #第四张为融合图
        # mask = np.array(Image.fromarray(all[2].numpy()).convert('L'))
        
        # img_p = img_tensor[:,:,0].cpu().numpy()
        # img_a = img_tensor[:,:,1].cpu().numpy() 
        # img_p[mask[:,:]==0]=0
        # img_a[mask[:,:]==0]=0
        

        img_p = img_tensor[:,:,0].cpu().numpy()
        img_a = img_tensor[:,:,1].cpu().numpy() 
        img_a = transforms.ToTensor()(img_a)
        img_p = transforms.ToTensor()(img_p)
        AB=torch.cat((img_a,img_p),dim=0)
        with torch.no_grad():
            out1=net(AB.unsqueeze(dim=0).cuda())
      
        pred = int(out1*265+55.5)
        print(idx,'/',len_all,':',pred)
        content.append([img, pred])
    df = pd.DataFrame(content, columns=['img_path', 'pred'])
    # df.to_csv(os.path.join(save_dir,'yym_trainfr_results_change.csv'))
    # df.to_csv(os.path.join(save_dir,'yym_dk7fa_results.csv'))

def static():
    # save_dir = '/home/jianght/000_codes/009_sim/quality_6193/results/fa/'
    # cvs=pd.read_csv("/home/jianght/000_codes/009_sim/quality_6193/results/fa/fa_results.csv")
    # fa_pred = cvs['pred'].to_list()
    # fa_grid_score_list =  cvs['img_path'].to_list()
    # fa_grid_score_list=[int(f.split('/')[-1].split('_')[-3][1:]) for f in fa_grid_score_list]

    # cvs_train=pd.read_csv("/home/jianght/000_codes/009_sim/quality_6193/results/fa/train_results.csv")

    # yym_trainfr_results_change
    # cvs_train=pd.read_csv("/home/jianght/000_codes/009_sim/quality_6193/results/fa/yym_trainfr_results.csv")
    cvs_train=pd.read_csv("/home/jianght/000_codes/009_sim/quality_6193/results/fa/yym_trainfr_results_change.csv")
    train_pred = cvs_train['pred'].to_list()
    img_path_list =  cvs_train['img_path'].to_list()
    grid_score_list =  cvs_train['img_path'].to_list()
    grid_score_list=[int(f.split('/')[-1].split('_')[-3][1:]) for f in grid_score_list]
    # print(len(train_pred),min(train_pred),max(train_pred))

    # 网格相似度
    # train_pred = grid_score_list

    new_pred = []
    new_img_path = []
    for idx_img,img_path in img_path_list:
        if 'DK4' in img_path:
            new_pred.append(train_pred[idx_img])
            new_img_path.append(img_path)
    train_pred = new_pred
    img_path = new_img_path

    
    new_ =[]
    
    for idx_c,pred in enumerate(train_pred):
        if 'wet' in img_path_list[idx_c]:
            new_.append(pred)
    
    fa_pred =[]
    for idx_c,pred in enumerate(train_pred):
        if 'wet' not in img_path_list[idx_c]:
            fa_pred.append(pred)
    fr_pred = new_

    print(len(fr_pred),min(fr_pred),max(fr_pred))
    print(len(fa_pred),min(fa_pred),max(fa_pred))
    keys = range(50,320,10)
    # keys = range(120,260,5)
    dict = {}
    for key_v in keys:
        dict[key_v]=0

    for pred in fr_pred:
        for idx in range(len(keys)-1):
            if pred<keys[idx+1] and pred>=keys[idx]:
                dict[keys[idx]]+=1
    dict_fa = {}
    for key_v in keys:
        dict_fa[key_v]=0
        
    for pred1 in fa_pred:
        for idx in range(len(keys)-1):
            if pred1<keys[idx+1] and pred1>=keys[idx]:
                dict_fa[keys[idx]]+=1

    
    content_fr=[]
    label = []
    for key in dict.keys():
        label.append(key)
        content_fr.append(dict[key])
        # content.append([key, dict[key]])
    print(sum(content_fr))
    total_ = sum(content_fr)
    # content_train=content_train/total_


    content_fa=[]
    # label = []
    for key in dict_fa.keys():
        # label.append(key)
        content_fa.append(dict_fa[key])
        # content.append([key, dict[key]])
    print('fa:', sum(content_fa))
    total1 = sum(content_fa)
    # content_fa=content_fa/total1


    #     print(key, dict[key])
    # df = pd.DataFrame(content, columns=['img_path', 'pred'])
    # df.to_csv(os.path.join(save_dir,'train_static.csv'))

    # a = 0
    # # 分组柱状图绘制
    # plt.rcParams['font.sans-serif']=['KaiTi']
    fig,ax = plt.subplots()
    # plt.rcParams['font.sans-serif']=['SimiHei']
    width=0.35
    x= np.arange(len(label))
    # rect1 = ax.bar(x-width/2,np.array(content_train)/total_,width,label='fr_wet')
    # rect2 = ax.bar(x+width/2,np.array(content_fa)/total1,width,label='fr_other')
    rect1 = ax.bar(x-width/2,np.array(content_fr),width,label='fr_wet')
    rect2 = ax.bar(x+width/2,np.array(content_fa),width,label='fr_other')
    # ax.set_ylabel('数据量')
    ax.set_xlabel('pred')
    ax.set_xticks(x)
    ax.set_xticklabels(np.array(label),rotation=90)
    # ax.set_title('grid_scores')
    ax.set_title('net_scores')
    ax.legend()
    plt.savefig("/home/jianght/tmp/re/net_fr_change.jpg")




    # x= [f for f in range(len(label))]
    # y = np.array(content_train)
    # plt.xticks(x,np.array(label),fontsize=10,rotation=90)
    # plt.bar(x,y,width=0.2,color='r')
    # plt.savefig("/home/jianght/tmp/1_wet.jpg")
        # data=np.random.randn(1000)
    # # plt.hist(np.array(data,dtype=float),hist=30,density=False,alpha=0.5,color='red')
    # plt.plot(data,color='red')
    # x = np.arange(0, 5, 0.1)
    # y = np.sin(x)
    # plt.plot(x, y)
    # plt.xlabel('value')
    # plt.ylabel('Fre')
    # plt.title('Hist')
    # plt.plot()
    # plt.legend()
    
def static_fafr():  
    save_dir= "/home/jianght/000_codes/009_sim/quality_6193/results/static/"
    # cvs_fa=pd.read_csv("/home/jianght/000_codes/009_sim/quality_6193/results/fa/yym_dk4fa_results.csv")
    cvs_fa=pd.read_csv("/home/jianght/000_codes/009_sim/quality_6193/results/fa/yym_dk7fa_results.csv")
    fa_pred = cvs_fa['pred'].to_list()
    fa_img_path_list =  cvs_fa['img_path'].to_list()
    fa_grid_score_list =  cvs_fa['img_path'].to_list()
    fa_grid_score_list=[int(f.split('/')[-1].split('_')[-3][1:]) for f in fa_grid_score_list]

    cvs_fa1=pd.read_csv("/home/jianght/000_codes/009_sim/quality_6193/results/fa/yym_dk4fa_results.csv")
    fa_pred1 = cvs_fa1['pred'].to_list()
    fa_img_path_list1 =  cvs_fa1['img_path'].to_list()
    fa_grid_score_list1 =  cvs_fa1['img_path'].to_list()
    fa_grid_score_list1=[int(f.split('/')[-1].split('_')[-3][1:]) for f in fa_grid_score_list1]
    fa_img_path_list.extend(fa_img_path_list1[:])
    fa_grid_score_list.extend(fa_grid_score_list1[:])
    print(len(fa_pred))
    fa_pred.extend(fa_pred1[:])
    print(len(fa_pred))

    

    # cvs_train=pd.read_csv("/home/jianght/000_codes/009_sim/quality_6193/results/fa/train_results.csv")

    # yym_trainfr_results_change
    cvs_fr=pd.read_csv("/home/jianght/000_codes/009_sim/quality_6193/results/fa/yym_trainfr_results.csv")
    # cvs_fr=pd.read_csv("/home/jianght/000_codes/009_sim/quality_6193/results/fa/yym_trainfr_results_change.csv")
    fr_pred = cvs_fr['pred'].to_list()
    img_fr_path_list =  cvs_fr['img_path'].to_list()
    grid_score_list =  cvs_fr['img_path'].to_list()
    grid_score_list=[int(f.split('/')[-1].split('_')[-3][1:]) for f in grid_score_list]
    # 网格相似度
    # fa_pred=fa_grid_score_list
    # fr_pred = grid_score_list
    print('fa_pred:',len(fa_pred),min(fa_pred),max(fa_pred))
    print('fr_pred:',len(fr_pred),min(fr_pred),max(fr_pred))
    
    # # 要区分湿手指及DK图库
    # # 1区分图库
    # new_pred = []
    # new_img_path = []
    # for idx_img,img_path in enumerate(fa_img_path_list):
    #     if 'DK7' in img_path:
    #         new_pred.append(fa_pred[idx_img])
    #         new_img_path.append(img_path)
    # fa_pred = new_pred
    # fa_img_path_list = new_img_path

    # new_pred = []
    # new_img_path = []
    # for idx_img,img_path in enumerate(img_fr_path_list):
    #     if 'DK7' in img_path:
    #         new_pred.append(fr_pred[idx_img])
    #         new_img_path.append(img_path)
    # fr_pred = new_pred
    # img_fr_path_list = new_img_path

    
    fa_wet_pred =[]
    
    for idx_c,pred in enumerate(fa_pred):
        if 'wet' in fa_img_path_list[idx_c]:
            fa_wet_pred.append(pred)
    
    fa_nowet_pred =[]
    for idx_c,pred in enumerate(fa_pred):
        if 'wet' not in fa_img_path_list[idx_c]:
            fa_nowet_pred.append(pred)


    fr_wet_pred =[]
    
    for idx_c,pred in enumerate(fr_pred):
        if 'wet' in img_fr_path_list[idx_c]:
            fr_wet_pred.append(pred)
    
    fr_nowet_pred =[]
    for idx_c,pred in enumerate(fr_pred):
        if 'wet' not in img_fr_path_list[idx_c]:
            fr_nowet_pred.append(pred)

    # 现在有四种数据，fafr以及是否湿手指
    # print(len(fr_pred),min(fr_pred),max(fr_pred))
    # print(len(fa_pred),min(fa_pred),max(fa_pred))

    # print('fa wet:',len(fa_wet_pred),min(fa_wet_pred),max(fa_wet_pred))
    # print('fa no wet:',len(fa_nowet_pred),min(fa_nowet_pred),max(fa_nowet_pred))
    # print('fr wet:',len(fr_wet_pred),min(fr_wet_pred),max(fr_wet_pred))
    # print('fr no wet:',len(fr_nowet_pred),min(fr_nowet_pred),max(fr_nowet_pred))

    keys = range(50,320,1)
    # keys = range(120,260,1)
    dict_fa_wet = {}
    for key_v in keys:
        dict_fa_wet[key_v]=0
    for pred in fa_wet_pred:
        for idx in range(len(keys)-1):
            if pred<keys[idx+1] and pred>=keys[idx]:
                dict_fa_wet[keys[idx]]+=1
    dict_fa_nowet = {}
    for key_v in keys:
        dict_fa_nowet[key_v]=0 
    for pred1 in fa_nowet_pred:
        for idx in range(len(keys)-1):
            if pred1<keys[idx+1] and pred1>=keys[idx]:
                dict_fa_nowet[keys[idx]]+=1

    dict_fr_wet = {}
    for key_v in keys:
        dict_fr_wet[key_v]=0
    for pred2 in fr_wet_pred:
        for idx in range(len(keys)-1):
            if pred2<keys[idx+1] and pred2>=keys[idx]:
                dict_fr_wet[keys[idx]]+=1
    dict_fr_nowet = {}
    for key_v in keys:
        dict_fr_nowet[key_v]=0 
    for pred1 in fr_nowet_pred:
        for idx in range(len(keys)-1):
            if pred1<keys[idx+1] and pred1>=keys[idx]:
                dict_fr_nowet[keys[idx]]+=1

    # draw_info= []
    content_fa_wet=[]
    label = []
    for key in dict_fa_wet.keys():
        label.append(key)
        content_fa_wet.append(dict_fa_wet[key])
        # content.append([key, dict[key]])
    print('content_fa_wet', sum(content_fa_wet))
    total1 = sum(content_fa_wet)
    content_fa_wet=list(np.array(content_fa_wet)/total1)
    


    content_fa_nowet=[]
    # label = []
    for key in dict_fa_nowet.keys():
        # label.append(key)
        content_fa_nowet.append(dict_fa_nowet[key])
        # content.append([key, dict[key]])
    print('content_fa_nowet:', sum(content_fa_nowet))
    total2 = sum(content_fa_nowet)
    content_fa_nowet=list(np.array(content_fa_nowet)/total2)

    content_fr_wet=[]
    # label = []
    for key in dict_fr_wet.keys():
        # label.append(key)
        content_fr_wet.append(dict_fr_wet[key])
        # content.append([key, dict[key]])
    print('content_fr_wet',sum(content_fr_wet))
    total3 = sum(content_fr_wet)
    content_fr_wet=list(np.array(content_fr_wet)/total3)
    


    content_fr_nowet=[]
    # label = []
    for key in dict_fr_nowet.keys():
        # label.append(key)
        content_fr_nowet.append(dict_fr_nowet[key])
        # content.append([key, dict[key]])
    print('content_fr_nowet:', sum(content_fr_nowet))
    total4 = sum(content_fr_nowet)
    content_fr_nowet=list(np.array(content_fr_nowet)/total4)
    # print(content_fr_nowet)
    
 #     print(key, dict[key])
# #  计算累加
    
#     sum_v = []
#     sum1=0
#     for idx,val in enumerate(content_fr_nowet):
#         sum1+=val
#         sum_v.append(sum1)
#     content_fr_nowet = sum_v
#     sum_v = []
#     sum1=0
#     for idx,val in enumerate(content_fr_wet):
#         sum1+=val
#         sum_v.append(sum1)
#     content_fr_wet = sum_v

#     sum_v = []
#     sum1=0
#     for idx,val in enumerate(content_fa_nowet):
#         sum1+=val
#         sum_v.append(sum1)
#     content_fa_nowet = sum_v

#     sum_v = []
#     sum1=0
#     for idx,val in enumerate(content_fa_wet):
#         sum1+=val
#         sum_v.append(sum1)
#     content_fa_wet = sum_v
#     print(sum_v)

    
    save_info= {'label':label, 'fr_nowet':list(content_fr_nowet),'fr_wet':list(content_fr_wet),'fa_nowet':list(content_fa_nowet),
    'fa_wet':list(content_fa_wet)}
    df = pd.DataFrame(save_info)
    # ,columns=['label', 'fr_nowet','fr_wet','fa_nowet','fa_wet']
    df.to_csv(os.path.join(save_dir,'net_static_all.csv'))
    # save_info= {'content_fr_nowet':list(content_fr_nowet)}
    # df = pd.DataFrame(save_info,columns=['label'])
    # df.to_csv(os.path.join(save_dir,'net_static.csv'))

    # # # 分组柱状图绘制
    # fig,ax = plt.subplots()
    # width=0.35
    # x= np.arange(len(label))
    # # rect1 = ax.bar(x-width/2,np.array(content_train)/total_,width,label='fr_wet')
    # # rect2 = ax.bar(x+width/2,np.array(content_fa)/total1,width,label='fr_other')
    # # rect1 = ax.bar(x-width/2,np.array(content_fr_nowet),width,label='fr_other')
    # # rect2 = ax.bar(x+width/2,np.array(content_fa_nowet),width,label='fa_other')
    # # rect3 = ax.bar(x-width/2,np.array(content_fr_wet),width,label='fr_wet')
    # # rect4 = ax.bar(x+width/2,np.array(content_fa_wet),width,label='fa_wet')

    # rect1 = ax.bar(x-width/2,np.array(content_fr_nowet),width,label='fr_other')
    # rect2 = ax.bar(x+width/2,np.array(content_fa_nowet),width,label='fa_other')
    # rect3 = ax.bar(x-width/2,np.array(content_fr_wet),width,label='fr_wet')
    # rect4 = ax.bar(x+width/2,np.array(content_fa_wet),width,label='fa_wet')

    # # ax.set_ylabel('数据量')
    # ax.set_xlabel('pred')
    # ax.set_xticks(x)
    # ax.set_xticklabels(np.array(label),rotation=90,fontsize=6)
    # # ax.set_title('grid_scores')
    # ax.set_title('net_scores')
    # ax.legend()
    # # plt.savefig(os.path.join(save_dir,"1117_dk4_nowet_fafr_grid.jpg"))
    # plt.savefig(os.path.join(save_dir,"1117_dk4_all_fafr_net.jpg"))
    # # plt.savefig(os.path.join(save_dir,"tmp.jpg"))


    # # # 分组柱状图绘制
    # fig,ax = plt.subplots()
    # width=0.35
    # x= np.arange(len(label))
    # # rect1 = ax.bar(x-width/2,np.array(content_train)/total_,width,label='fr_wet')
    # # rect2 = ax.bar(x+width/2,np.array(content_fa)/total1,width,label='fr_other')
    # rect1 = ax.bar(x-width/2,np.array(content_fr_wet),width,label='fr_wet')
    # rect2 = ax.bar(x+width/2,np.array(content_fa_wet),width,label='fa_wet')
    # # ax.set_ylabel('数据量')
    # ax.set_xlabel('pred')
    # ax.set_xticks(x)
    # ax.set_xticklabels(np.array(label),rotation=90,fontsize=6)
    # ax.set_title('grid_scores')
    # # ax.set_title('net_scores')
    # ax.legend()
    # plt.savefig(os.path.join(save_dir,"1117_dk4_wet_fafr_grid.jpg"))
    # # plt.savefig(os.path.join(save_dir,"tmp1.jpg"))




    


def static_label():
    save_dir = '/home/jianght/001_data/003_feature_duibi/'
    # cvs_fr=pd.read_csv("/home/jianght/001_data/003_feature_duibi/frr/info.csv")
    cvs_fr=pd.read_csv("/home/jianght/001_data/003_featuredata_20231117/add_data/info_frr.csv")
    fr_pred = cvs_fr['label'].to_list()
    fr_img_path = cvs_fr['img_path'].to_list()

    # cvs_fa=pd.read_csv("/home/jianght/001_data/003_feature_duibi/far/info.csv")
    cvs_fa=pd.read_csv("/home/jianght/001_data/003_featuredata_20231117/add_data/info_far.csv")
    fa_pred = cvs_fa['label'].to_list()
    fa_img_path = cvs_fa['img_path'].to_list()

    # 
    new_label = []
    for idx, imgpath in enumerate(fr_img_path):
        if 'wet' in imgpath:
            new_label.append(fr_pred[idx])
    fr_pred=new_label

    new_label = []
    for idx, imgpath in enumerate(fa_img_path):
        if 'wet' in imgpath:
            new_label.append(fa_pred[idx])
    fa_pred=new_label
    
    print(len(fr_pred),min(fr_pred),max(fr_pred))
    print(len(fa_pred),min(fa_pred),max(fa_pred))

    keys = range(50,320,10)
    # keys = range(120,260,5)
    dict_fr = {}
    for key_v in keys:
        dict_fr[key_v]=0
        
    for pred1 in fr_pred:
        for idx in range(len(keys)-1):
            if pred1<keys[idx+1] and pred1>=keys[idx]:
                dict_fr[keys[idx]]+=1

    dict_fa = {}
    for key_v in keys:
        dict_fa[key_v]=0
        
    for pred1 in fa_pred:
        for idx in range(len(keys)-1):
            if pred1<keys[idx+1] and pred1>=keys[idx]:
                dict_fa[keys[idx]]+=1

    
    content_fr=[]
    label = []
    for key in dict_fr.keys():
        label.append(key)
        content_fr.append(dict_fr[key])
        # content.append([key, dict[key]])
    print(sum(content_fr))
    total_ = sum(content_fr)
    content_fr=np.array(content_fr)/total_

    content_fa=[]
    label = []
    for key in dict_fa.keys():
        label.append(key)
        content_fa.append(dict_fa[key])
        # content.append([key, dict[key]])
    print(sum(content_fa))
    total2 = sum(content_fa)
    content_fa=np.array(content_fa)/total2


   
    save_info= {'label':label, 'fr_label':list(content_fr),'fa_label':list(content_fa)}
    df = pd.DataFrame(save_info)
    df.to_csv(os.path.join(save_dir,'cal_label_addnowet.csv'))
    


def static_label_train():
    save_dir = '/home/jianght/001_data/003_feature_duibi/'
    cvs_fr=pd.read_csv("/home/jianght/001_data/003_feature_duibi/frr/info.csv")
    fr_pred = cvs_fr['label'].to_list()

    cvs_fa=pd.read_csv("/home/jianght/001_data/003_feature_duibi/far/info.csv")
    fa_pred = cvs_fa['label'].to_list()
    
    print(len(fr_pred),min(fr_pred),max(fr_pred))
    print(len(fa_pred),min(fa_pred),max(fa_pred))

    keys = range(50,320,10)
    # keys = range(120,260,5)
    dict_fr = {}
    for key_v in keys:
        dict_fr[key_v]=0
        
    for pred1 in fr_pred:
        for idx in range(len(keys)-1):
            if pred1<keys[idx+1] and pred1>=keys[idx]:
                dict_fr[keys[idx]]+=1

    dict_fa = {}
    for key_v in keys:
        dict_fa[key_v]=0
        
    for pred1 in fa_pred:
        for idx in range(len(keys)-1):
            if pred1<keys[idx+1] and pred1>=keys[idx]:
                dict_fa[keys[idx]]+=1

    
    content_fr=[]
    label = []
    for key in dict_fr.keys():
        label.append(key)
        content_fr.append(dict_fr[key])
        # content.append([key, dict[key]])
    print(sum(content_fr))
    total_ = sum(content_fr)
    content_fr=np.array(content_fr)/total_

    content_fa=[]
    label = []
    for key in dict_fa.keys():
        label.append(key)
        content_fa.append(dict_fa[key])
        # content.append([key, dict[key]])
    print(sum(content_fa))
    total2 = sum(content_fa)
    content_fa=np.array(content_fa)/total2


   
    save_info= {'label':label, 'fr_label':list(content_fr),'fa_label':list(content_fa)}
    df = pd.DataFrame(save_info)
    df.to_csv(os.path.join(save_dir,'cal_label.csv'))



    

    
       
if __name__ == '__main__':
    main()
    # static()
    # static_fafr()
    # static_label()
