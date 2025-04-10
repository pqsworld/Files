
#from MobileNet import *
from pathlib import Path
from collections import deque
import numpy as np
from pathlib import Path
import sys
import cv2
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
import random


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

def crop_img(img_p,img_a,in_w,in_h):
    edge_w,edge_h=[],[]
    for i in range(in_w):
        for j in range(in_h):
            if img_p[j][i]>0:
                edge_w.append(i)
                edge_h.append(j)
    left=min(edge_w)
    right = max(edge_w)
    bottom=min(edge_h)
    top = max(edge_h)
    w = right-left+1
    h = top-bottom+1
    w_new =max(w,24)#21)  #多一个maxpool
    h_new =max(h,24)#9)
    #判断以免多加的几像素超出最大范围,判断加在哪边
    if right+w_new-w>in_w-1:
        left = left-(w_new-( right - left + 1))
    if top+h_new-h>in_h-1:
        bottom = bottom-(h_new- (top - bottom + 1))
    img_p_crop = img_p[bottom:bottom+h_new,left:left+w_new]
    img_a_crop = img_a[bottom:bottom+h_new,left:left+w_new]
    return img_p_crop,img_a_crop


def cal_count_curve(net_simi_list_frr0,img_name_list_frr0):
    net_simi_list_frr0= np.sort(net_simi_list_frr0)
    unique_elements,counts = np.unique(net_simi_list_frr0,return_counts=True)
    cal_counts = np.zeros_like(counts)
    cal_counts[0]=counts[0]
    for i in range(len(counts)-1):
        cal_counts[i+1]=cal_counts[i]+counts[i+1]
    ratio = cal_counts/(len(img_name_list_frr0))*100
    data={'net_simi':unique_elements,'counts':counts,'cal_counts':cal_counts,'ratio':ratio}  #
    df=pd.DataFrame(data)
    return df

def main():
    # net = MNV3_bufen_new5_bin_reg_pool2(2, 1, 4, n_blocks=1).cuda()
    net =  MNV3_bufen_new5_bin_reg(2, 1, 4, n_blocks=1).cuda()
    # pthpath="./checkpoints/st_simi_21_1220_out2_sz72_newlabel_crop/284_net_G.pth"
    # pthpath="./checkpoints/st_simi_1210_out2_sz72_fa5_fr5_moredata_crop_teacher/227_net_G.pth"
    pthpath="./checkpoints/st_simi_st32_0110_out2_sz72_crop/296_net_G.pth"
    # net =  MNV3_bufen_new5_bin_reg_small1(2, 1, 4, n_blocks=1).cuda()
    # pthpath="/hdd/file-input/zhangsn/light_simi/checkpoints/st_simi_1211_out2_sz72_small1_fa5_fr5_moredata_crop_teacher/216_net_G.pth"

    net.load_state_dict(torch.load(pthpath))
    # img_dir = "/data/guest/zsn/simi_data_light/000_light_simi_test_class/test_data_sele0.1"#/sm"#/frr"
    img_dir = "/data/guest/zsn/simi_data_light/003_ryttl_test/"#/sm"#/frr"
    all_list = sorted([str(f) for f in Path(img_dir).rglob('*.bmp')])

    saveimg = 0
    savecsv = 1
    Crop_flag = 1


    save_dir_frr0 ="/data/guest/zsn/simi_data_light/003_ryttl_test/res/fr/0"
    save_dir_frr1="/data/guest/zsn/simi_data_light/003_ryttl_test/res/fr/1"
    save_dir_fa ="/data/guest/zsn/simi_data_light/003_ryttl_test/res/fa"
    # warn_dir = os.path.join(save_dir,'warn')
    save_csv_path="/data/guest/zsn/simi_data_light/003_ryttl_test/res"

    
    net.eval()
    content =[]
    len_all = len(all_list)
    
    img_name_list_far=[]
    net_simi_list_far=[]
    net_class_list_far = []
    img_name_list_frr=[]
    net_simi_list_frr=[]
    net_class_list_frr = []
    img_name_list_frr0=[]
    net_simi_list_frr0=[]
    net_class_list_frr0 = []
    
    test_correct_1,test_correct_0,test_correct_fa,all1,all0,allfa=0,0,0,0,0,0
    eps=1e-10
    for idx, img in enumerate(all_list):
        if '/fa/' in img:
            classfr = 0
            # if idx%10!=0: #只测试1/10
            #     continue
        else:
            classfr = 1
            # if idx%10!=0: 
            #     continue
        # img='/hdd/file-input/zhangsn/data/000_light_simi_test_class/test_fail_data_frr/0013_L2_0020_22_sim152_simpro0_score65522.36_fr.bmp'
        img_all = Image.open(img).convert('RGB')
        # print(img)
        # img_tensor = torch.from_numpy(np.array(img_PIL) / 255).cuda()
        img_tensor = torch.from_numpy(np.array(img_all))

        in_w,in_h = 72,72#96,96
        img_p = img_tensor[:,:,0].cpu().numpy()
        img_a = img_tensor[:,:,1].cpu().numpy()
        img_p = cv2.resize(img_p,(in_w,in_h))
        img_a = cv2.resize(img_a,(in_w,in_h))   
        # img_p = cv2.resize(img_p,(64,64))
        # img_a = cv2.resize(img_a,(64,64))  

        #重叠区域外边框
        if Crop_flag==1:
            img_p_crop,img_a_crop=crop_img(img_p,img_a,in_w,in_h)
            img_a = transforms.ToTensor()(img_a_crop)  #size(h,w)
            img_p = transforms.ToTensor()(img_p_crop)
        else:
            img_a = transforms.ToTensor()(img_a)
            img_p = transforms.ToTensor()(img_p)

        
        AB=torch.cat((img_a,img_p),dim=0)
        # AB=torch.cat((img_p,img_a),dim=0)
        with torch.no_grad():
            out1,net_class_t=net(AB.unsqueeze(dim=0).cuda())

        net_class_t0 = F.softmax(net_class_t,dim = -1)
        clc_score = net_class_t0[0][1]*10000
        clc_score=int(clc_score.cpu().detach().numpy())
        pred = int(out1*265+55.5)
        
        prediction = torch.max(net_class_t, 1)
        pre_res=prediction[1].cpu().detach().numpy()  

        #为csv存储做准备
        
        classscore = int(img.split('_')[-4])#float(img.split('_')[-3][5:])
        
        if classfr==0:  #far
            img_name_list_far.append(img)
            net_simi_list_far.append(pred)
            net_class_list_far.append(clc_score)
            #统计分类头准确率
            test_correct_fa += np.sum(pre_res[0] == classfr)
            allfa+=1
        elif classscore>0:#64480:  #frr succes
            img_name_list_frr.append(img)
            net_simi_list_frr.append(pred)
            net_class_list_frr.append(clc_score)  
            #统计分类头准确率
            test_correct_1 += np.sum(pre_res[0] == classfr)
            all1+=1
        else:   #frr fail
            img_name_list_frr0.append(img)
            net_simi_list_frr0.append(pred)
            net_class_list_frr0.append(clc_score)  
            #统计分类头准确率
            test_correct_0 += np.sum(pre_res[0] == classfr)
            all0+=1
        
        if idx%100 ==0:
            print(idx,'/',len_all,':',pred)
        # rename = str(pred)+'_'+str(all_label[idx])+'_'+img.split('/')[-1]
        if saveimg == 1:
            rename = str(pred)+'_'+str(clc_score)+'_'+img.split('/')[-1]
            save_path = os.path.join(img[:-len(img.split('/')[-1])],rename)
            if classfr == 0:
                save_path=save_path.replace(img_dir, save_dir_fa)
            elif classscore>64480:
                save_path=save_path.replace(img_dir, save_dir_frr1)
            else:
                save_path=save_path.replace(img_dir, save_dir_frr0)
            Path(save_path).parent.mkdir(parents=True,exist_ok=True)
            # print(save_path)
            shutil.copy(img, save_path)
            # shutil.move(img,save_path)
            # cv2.imwrite(save_path,img_merge)s
    
    print('!res of acc:',(test_correct_0+test_correct_1+test_correct_fa)/(all0+all1+allfa))
    print('!res of acc_fr_succ:',test_correct_1/(all1+eps),'fr_all_num:',all1)
    print('!res of acc_fr_fail:',test_correct_0/(all0+eps),'fr_all_num:',all0)
    print('!res of acc_fa:',test_correct_fa/(allfa+eps),'fa_all_num:',allfa)
    # far
    if savecsv==1:
        if allfa>0:
            data={'name':img_name_list_far,'net_simi':net_simi_list_far,'net_class':net_class_list_far}  #
            df=pd.DataFrame(data)
            df.to_csv(save_csv_path+'/st32/all_far_simi.csv',index=True)
            df=cal_count_curve(net_simi_list_far,img_name_list_far)
            df.to_csv(save_csv_path+'/st32/all_far_simi_counts.csv',index=True)

            df=cal_count_curve(net_class_list_far,img_name_list_far)
            df.to_csv(save_csv_path+'/st32/all_far_cls_counts.csv',index=True)

        # frr 1
        if all1>0:
            data={'name':img_name_list_frr,'net_simi':net_simi_list_frr,'net_class':net_class_list_frr}  #
            df=pd.DataFrame(data)
            df.to_csv(save_csv_path+'/st32/all_frr_simi_succ.csv',index=True)

            df=cal_count_curve(net_simi_list_frr,img_name_list_frr)
            df.to_csv(save_csv_path+'/st32/all_frr_simi_succ_counts.csv',index=True)

            df=cal_count_curve(net_class_list_frr,img_name_list_frr)
            df.to_csv(save_csv_path+'/st32/all_frr_cls_succ_counts.csv',index=True)

        # frr 0
        if all0>0:
            data={'name':img_name_list_frr0,'net_simi':net_simi_list_frr0,'net_class':net_class_list_frr0}  #
            df=pd.DataFrame(data)
            df.to_csv(save_csv_path+'/st32/all_frr_simi_fail.csv',index=True)

            df=cal_count_curve(net_simi_list_frr0,img_name_list_frr0)
            df.to_csv(save_csv_path+'/st32/all_frr_simi_fail_counts.csv',index=True)  

            df=cal_count_curve(net_class_list_frr0,img_name_list_frr0)
            df.to_csv(save_csv_path+'/st32/all_frr_cls_fail_counts.csv',index=True)    


        
def main_2():
    # net =   MNV3_bufen_new5_out2(2, 1, 4, n_blocks=1).cuda()
    net =  MNV3_bufen_new5_bin_reg(2, 1, 4, n_blocks=1).cuda()
    # net =  MNV3_bufen_new5_bin_reg_small1(2, 1, 4, n_blocks=1).cuda()

    saveimg = 0
    savecsv = 1
    savecurve=1
    Crop_flag = 1
    # pthpath="./checkpoints/st_simi_1210_out2_sz72_fa5_fr5_moredata_crop_teacher/227_net_G.pth"
    # pthpath="./checkpoints/st_simi_21_1220_out2_sz72_newlabel_crop/284_net_G.pth"
    pthpath="./checkpoints/st_simi_st31_0109_out2_sz72_crop/296_net_G.pth"

  
    net.load_state_dict(torch.load(pthpath))
    net.eval()
    
    # img_dir = "/data/guest/zsn/simi_data_light/000_light_simi_test_frr_1217version/simi1210/"#/sm"#/frr"
    # all_list = sorted([str(f) for f in Path(img_dir).rglob('*.bmp')])
    # len_all = len(all_list)

    # save_dir ="/data/guest/zsn/simi_data_light/000_light_simi_test_frr_1217version/simi1220_21/"
    
    img_dir = "/data/guest/zsn/simi_data_light/003_ryttl_test/fr"#/sm"#/frr"
    all_list = sorted([str(f) for f in Path(img_dir).rglob('*.bmp')])
    len_all = len(all_list)

    save_dir ="/data/guest/zsn/simi_data_light/003_ryttl_test/simi1210_1/"

    img_name_list_frr=[]
    net_simi_list_frr=[]
    net_class_list_frr = []
    img_name_list_frr0=[]
    net_simi_list_frr0=[]
    net_class_list_frr0 = []
    img_name_list_far=[]
    net_simi_list_far=[]
    net_class_list_far = []

    net_simi_list_frr_old=[]
    net_class_list_frr_old = []
    net_simi_list_frr0_old=[]
    net_class_list_frr0_old = []

    test_correct_1,test_correct_0,test_correct_fa,all1,all0,allfa=0,0,0,0,0,0
    eps=1e-10
    class_thre = 65450
    in_w,in_h = 72,72#96,96
    for idx, img in enumerate(all_list):
        # img='/hdd/file-input/zhangsn/data/002_test_frr_newmodel/test_suc/dust/164_9585_188_0012_L1_0026_11_sim188_simpro2085_score65530.41_fr.bmp'
        #class inlinearm simiscore quality
        classscore = int(img.split('_')[-4])
        simi =int(img.split('_')[-2])
        # classscore = float(img.split('_')[-2][5:])
        # simi =int(img.split('_')[-3][3:])
        # if '/1/'in img:#
        if 'fa' in img:
            classfr = 0
        else:
            classfr = 1

        # img_new=str(simi)+'_'+img.split('/')[-1]
        # save_path = os.path.join(img[:-len(img.split('/')[-1])],img_new)
        # save_path=img
        # if classscore>0:#class_thre:  #frr succes
        #     save_path = save_path.replace('/fr/','/fr_suc/')#os.path.join(img[:-len(img.split('/')[-1])],img_new)
        #     Path(save_path).parent.mkdir(parents=True,exist_ok=True)
        #     # print(save_path)
        #     shutil.move(img, save_path)
        # else:
        #     save_path = save_path.replace('/fr/','/fr_fail/')#os.path.join(img[:-len(img.split('/')[-1])],img_new)
        #     Path(save_path).parent.mkdir(parents=True,exist_ok=True)
        #     # print(save_path)
        #     shutil.move(img, save_path)
     
        img_all = Image.open(img).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img_all))
        img_p = img_tensor[:,:,0].cpu().numpy()
        img_a = img_tensor[:,:,1].cpu().numpy()
        img_p = cv2.resize(img_p,(in_w,in_h))
        img_a = cv2.resize(img_a,(in_w,in_h))   

        #重叠区域外边框
        if Crop_flag==1:
            img_p_crop,img_a_crop=crop_img(img_p,img_a,in_w,in_h)
            img_a = transforms.ToTensor()(img_a_crop)  #size(h,w)
            img_p = transforms.ToTensor()(img_p_crop)
        else:
            img_a = transforms.ToTensor()(img_a)
            img_p = transforms.ToTensor()(img_p)

        
        AB=torch.cat((img_a,img_p),dim=0)
        # AB=torch.cat((img_p,img_a),dim=0)
        with torch.no_grad():
            out1,net_class_t=net(AB.unsqueeze(dim=0).cuda())

        net_class_t0 = F.softmax(net_class_t,dim = -1)
        clc_score = net_class_t0[0][1]*10000+0.5
        clc_score=int(clc_score.cpu().detach().numpy())
        pred = int(out1*265+55.5)
        
        prediction = torch.max(net_class_t, 1)
        pre_res=prediction[1].cpu().detach().numpy()  

        clc_score_old=int(img.split('/')[-1].split('_')[1])
        pred_old=int(img.split('/')[-1].split('_')[0])

        if saveimg == 1:
            rename = str(pred)+'_'+str(clc_score)+'_'+img.split('/')[-1]
            save_path = os.path.join(img[:-len(img.split('/')[-1])],rename)
            save_path=save_path.replace(img_dir, save_dir)
            Path(save_path).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy(img, save_path)

            
            # rename = str(pred)+'_'+str(clc_score)+'_'+img.split('/')[-1][4:]
            # save_path = os.path.join(img[:-len(img.split('/')[-1])],rename)
            # # save_path=save_path.replace(img_dir, save_dir)
            # os.rename(img,save_path)

        #为csv存储做准备
        if classscore>0:#class_thre:  #frr succes
            img_name_list_frr.append(img)
            net_simi_list_frr.append(pred)
            net_class_list_frr.append(clc_score)  
            net_simi_list_frr_old.append(pred_old)
            net_class_list_frr_old.append(clc_score_old)  
            #统计分类头准确率
            test_correct_1 += np.sum(pre_res[0] == classfr)
            all1+=1

        else:   #frr fail
            img_name_list_frr0.append(img)
            net_simi_list_frr0.append(pred)
            net_class_list_frr0.append(clc_score)  
            net_simi_list_frr0_old.append(pred_old)
            net_class_list_frr0_old.append(clc_score_old)  
            #统计分类头准确率
            test_correct_0 += np.sum(pre_res[0] == classfr)
            all0+=1
        

        if idx%100 ==0:
            print(idx,'/',len_all,':',pred)
        # rename = str(pred)+'_'+str(all_label[idx])+'_'+img.split('/')[-1]

    print('!res of acc:',(test_correct_0+test_correct_1+test_correct_fa)/(all0+all1+allfa))
    print('!res of acc_fr_succ:',test_correct_1/(all1+eps),'fr_all_num:',all1)
    print('!res of acc_fr_fail:',test_correct_0/(all0+eps),'fr_all_num:',all0)
    
    # plot
    if savecurve==1:
        simi_new=np.array(net_simi_list_frr)
        simi_old=np.array(net_simi_list_frr_old)
        # alldata=np.c_[resnumpy,gtnumpy]
        # print(alldata.shape)
        plt.figure()
        plt.title('res')
        plt.hist(simi_new,bins=210, rwidth=0.8, range=(90,300),label='simi21', align='left',alpha=0.5)
        plt.hist(simi_old,bins=210, rwidth=0.8, range=(90,300), label='simi1', align='left',alpha=0.5)
        # plt.hist(alldata,bins=65,color=['c','r'], rwidth=0.5, range=(55,315), label=['net_res','gt_label'], align='left',alpha=0.5,stacked=False)
        plt.legend()
        plt.savefig(save_dir+'/frr_simi_curve.png')
    
    # far
    if savecsv==1:
        # frr 1
        if all1>0:
            data={'name':img_name_list_frr,'net_simi':net_simi_list_frr,'net_class':net_class_list_frr,'net_simi_old':net_simi_list_frr_old,'net_class_old':net_class_list_frr_old}  #
            df=pd.DataFrame(data)
            df.to_csv(save_dir+'/all_frr_simi_succ.csv',index=True)

            df=cal_count_curve(net_simi_list_frr,img_name_list_frr)
            df.to_csv(save_dir+'/all_frr_simi_succ_counts.csv',index=True)

            df=cal_count_curve(net_class_list_frr,img_name_list_frr)
            df.to_csv(save_dir+'/all_frr_cls_succ_counts.csv',index=True)
        # frr 0
        if all0>0:
            data={'name':img_name_list_frr0,'net_simi':net_simi_list_frr0,'net_class':net_class_list_frr0,'net_simi_old':net_simi_list_frr0_old,'net_class_old':net_class_list_frr0_old}  #
            df=pd.DataFrame(data)
            df.to_csv(save_dir+'/all_frr_simi_fail.csv',index=True)

            df=cal_count_curve(net_simi_list_frr0,img_name_list_frr0)
            df.to_csv(save_dir+'/all_frr_simi_fail_counts.csv',index=True)  

            df=cal_count_curve(net_class_list_frr0,img_name_list_frr0)
            df.to_csv(save_dir+'/all_frr_cls_fail_counts.csv',index=True)    

def main_save_curve():
    img_dir = "/data/guest/zsn/simi_data_light/000_light_simi_test_frr_1217version/simi1220_21/"#/sm"#/frr"
    all_list = sorted([str(f) for f in Path(img_dir).rglob('*.bmp')])
    len_all = len(all_list)

    # save_dir ="/data/guest/zsn/simi_data_light/000_light_simi_test_frr_1217version/simi1220_21/"

    
    img_name_list_frr=[]
    net_simi_list_frr=[]
    net_class_list_frr = []
    img_name_list_frr0=[]
    net_simi_list_frr0=[]
    net_class_list_frr0 = []


    net_simi_list_frr_old=[]
    net_class_list_frr_old = []
    net_simi_list_frr0_old=[]
    net_class_list_frr0_old = []

    class_thre = 65450
    for idx, img in enumerate(all_list):
        # img='/hdd/file-input/zhangsn/data/002_test_frr_newmodel/test_suc/dust/164_9585_188_0012_L1_0026_11_sim188_simpro2085_score65530.41_fr.bmp'
        classscore = float(img.split('_')[-2][5:])
        # simi =int(img.split('_')[-3][3:])
 
        clc_score=int(img.split('/')[-1].split('_')[1])
        pred=int(img.split('/')[-1].split('_')[0])
        clc_score_old=int(img.split('/')[-1].split('_')[3])
        pred_old=int(img.split('/')[-1].split('_')[2])

        
        #为csv存储做准备
        if classscore>class_thre:  #frr succes
            img_name_list_frr.append(img)
            net_simi_list_frr.append(pred)
            net_class_list_frr.append(clc_score)  
            net_simi_list_frr_old.append(pred_old)
            net_class_list_frr_old.append(clc_score_old)  
        else:   #frr fail
            img_name_list_frr0.append(img)
            net_simi_list_frr0.append(pred)
            net_class_list_frr0.append(clc_score)  
            net_simi_list_frr0_old.append(pred_old)
            net_class_list_frr0_old.append(clc_score_old)  

        

        if idx%100 ==0:
            print(idx,'/',len_all,':',pred)
        # rename = str(pred)+'_'+str(all_label[idx])+'_'+img.split('/')[-1]
  
    # plot
    simi_new=np.array(net_simi_list_frr)
    simi_old=np.array(net_simi_list_frr_old)
    alldata=np.c_[simi_new,simi_old]
    print(alldata.shape)
    plt.figure()
    plt.title('frr_simi_succ_curve')
    plt.hist(alldata,bins=list(range(100,300,10)),color=['c','r'], rwidth=0.8, range=(100,300), label=['simi21','simi1'], align='left',alpha=0.5,stacked=False)
    
    # plt.hist(simi_new,bins=list(range(100,300,10)), rwidth=0.8, range=(100,300),label='simi21', align='left',alpha=0.5)
    # plt.hist(simi_old,bins=list(range(100,300,10)), rwidth=0.8, range=(100,300), label='simi1', align='left',alpha=0.5)
    my_x_ticks=np.arange(100,300,20)  #坐标轴刻度
    plt.xticks(my_x_ticks)
    plt.legend()
    plt.savefig(img_dir+'/frr_simi_succ_curve.png')
    
    simi_new=np.array(net_simi_list_frr0)
    simi_old=np.array(net_simi_list_frr0_old)
    alldata=np.c_[simi_new,simi_old]
    print(alldata.shape)
    plt.figure()
    plt.title('frr_simi_fail_curve')
    plt.hist(alldata,bins=list(range(100,300,10)),color=['c','r'], rwidth=0.8, range=(100,300), label=['simi21','simi1'], align='left',alpha=0.5,stacked=False)
    my_x_ticks=np.arange(100,300,20)  #坐标轴刻度
    plt.xticks(my_x_ticks)
    plt.legend()
    plt.savefig(img_dir+'/frr_simi_fail_curve.png')

def main_csv():
    net =  MNV3_bufen_new5_bin_reg(2, 1, 4, n_blocks=1).cuda()  
    pthpath="./checkpoints/st_simi_1210_out2_sz72_fa5_fr5_moredata_crop_teacher/227_net_G.pth"
    net.load_state_dict(torch.load(pthpath))
    net.eval()
    save_path='/data/guest/zsn/simi_data_light/train1219_add'
    
    datasets_use = [
        # "1_2",
        # "3_4",
        # "45",
        # "belly",
        # "tip",
        # "dust",
        # "dw",
        # "lgm",
        # "msm",
        # "wet",

        # "nova_dust",
        # "nova_dw",
        # "nova_dw-bad",
        # "nova_lgm",
        # "nova_msm",
        # "nova_sun-part",
        # "nova_wet",
        # "nova_snm",
        # "nova_sun-all",
        # 'nova-cdw',
        
        #small ttl
        # "TM-CWBF",
        "TM-DW",
        "TM-GSZ",
        "TM-QG",
        # "TM-QGBF",
        "TM-SSZ",
        "TM-XS",
    ]
    # ori_path='/data/guest/zsn/simi_data_light/001_light_simi_0108add_sttl'
    for phase in ['0']:#,'1']:
        for dataset_t in datasets_use:
        # for phase in opt.phase:
            # csvpath = os.path.join(ori_path,phase,dataset_t,'train1101.csv')  #trans_choose.txt
            csvpath = os.path.join(save_path,phase,dataset_t+'_train1219.csv')  #trans_choose.txt
            # savecsv = os.path.join(save_path,phase,dataset_t+'_train1219.csv')
            # shutil.move(csvpath,savecsv)

            df = pd.read_csv(
                        csvpath,
                        header=0,
                        encoding = "gb18030", #"gb2312",
                        # names=['img_path', 'ori','ham','ham_thre','ssim','label_new1219','label','grid_score','temp','samp','trans'],
                        names=['img_path', 'ori','ham','ham_thre','ssim','label','grid_score','temp','samp','trans']
                        # index_col=0,
                        )    # 'gb2312'
            
            # del df['level_0']
            # del df['label_new1219']
            
            print(csvpath,len(df),df.shape) 
            
            # print(csvpath,'sele:',len(df)) 
            df['img_path'] = df['img_path'].str.replace('/data/guest/zsn/simi_data_light/001_light_simi_0108add_sttl','/data/guest/zsn/simi_data_light/001_light_simi_0108add_sttl/')#,inplace=True)
            df['img_path']=df['img_path'].str.replace('/hdd/file-input/zhangsn/data/000_light_simi_1127add_nvwa/','/data/guest/zsn/simi_data_light/')#,inplace=True)
            # df['img_path']=(df['img_path'].str).replace('/hdd/file-input/zhangsn/data/000_light_simi_1127add_nvwa/','/data/guest/zsn/simi_data_light/001_traindata/')
            # df['img_path']=df['img_path'].str.replace('/data/guest/zsn/simi_data_light/1/','/data/guest/zsn/simi_data_light/001_traindata/1/')
            AB_paths = df['img_path'].to_list()
            sim_list=df['label'].to_list()
       
            new_label_list =[]
            len_all = len(AB_paths)
            for idx, img in enumerate(AB_paths):  
                sim=sim_list[idx]
                # if sim<150:
                #     save_img=img.replace('simi_data_light','simi_data_light_bad')
                #     Path(save_img).parent.mkdir(parents=True,exist_ok=True)
                #     shutil.copy(img,save_img)
                img_all = Image.open(img).convert('RGB')
                # print(img)
                img_tensor = torch.from_numpy(np.array(img_all))

                in_w,in_h = 72,72#96,96
                img_p = img_tensor[:,:,0].cpu().numpy()
                img_a = img_tensor[:,:,1].cpu().numpy()
                img_p = cv2.resize(img_p,(in_w,in_h))
                img_a = cv2.resize(img_a,(in_w,in_h))  

                img_a = transforms.ToTensor()(img_a)
                img_p = transforms.ToTensor()(img_p)

                
                AB=torch.cat((img_a,img_p),dim=0)
                # AB=torch.cat((img_p,img_a),dim=0)
                with torch.no_grad():
                    out1,net_class_t=net(AB.unsqueeze(dim=0).cuda())
                pred = int(out1*265+55.5)
                new_label_list.append(pred)
                
                
                if idx%5000==0:
                    print(idx,'/',len_all,':',pred)
            
            df.insert(loc=5,column='label_new1219',value=new_label_list)

            df.to_csv(csvpath)#.replace('/train1219/','/train1219_t/'))

'''统计数据最大外界四边形'''
def main_fa_size():  

    img_dir = "/hdd/file-input/zhangsn/data/000_light_simi_test_class/test_data"#/sm"#/frr"
    all_list = [str(f) for f in Path(img_dir).rglob('*.bmp')]
    # warn_dir = os.path.join(save_dir,'warn')
    save_csv_path="/hdd/file-input/zhangsn/data/000_light_simi_test_class/res"
     
    w_list_fr=[]
    h_list_fr=[]
    w_list_fa=[]
    h_list_fa=[]
    len_all=len(all_list)
    for idx, img in enumerate(all_list):
        if idx%100 ==0:
            print(idx,'/',len_all)
        if 'far' in img:
            classfr = 0
        else:
            classfr = 1
        img_all = Image.open(img).convert('RGB')
        # print(img)
        # img_tensor = torch.from_numpy(np.array(img_PIL) / 255).cuda()
        img_tensor = torch.from_numpy(np.array(img_all))

        in_w,in_h = 96,96
        img_p = img_tensor[:,:,0].cpu().numpy()
        img_a = img_tensor[:,:,1].cpu().numpy()
        img_p = cv2.resize(img_p,(in_w,in_h))
        img_a = cv2.resize(img_a,(in_w,in_h))   
        
        #重叠区域外边框
        edge_w,edge_h=[],[]
        for i in range(in_w):
            for j in range(in_h):
                if img_p[i][j]>0:
                    edge_w.append(i)
                    edge_h.append(j)
        left=min(edge_w)
        right = max(edge_w)
        bottom=min(edge_h)
        top = max(edge_h)
        w = right-left
        h = top-bottom
        if classfr==1:
            w_list_fr.append(w)
            h_list_fr.append(h)
        else:
            w_list_fa.append(w)
            h_list_fa.append(h)

    data={'w':w_list_fr,'h':h_list_fr}  #
    df=pd.DataFrame(data)
    df.to_csv(save_csv_path+'/all_frr_size.csv',index=True)
    data={'w':w_list_fa,'h':h_list_fa}  #
    df=pd.DataFrame(data)
    df.to_csv(save_csv_path+'/all_far_size.csv',index=True)

'''统计训练数据的label'''
def main_traindata_label():
        dir_AB = '/data/guest/zsn/simi_data_light/001_light_simi_0108add_sttl/1'  
        all_list = sorted([str(f) for f in Path(dir_AB).rglob('*.bmp')])
        len_all = len(all_list)
        label_list=[]
        i=0
        for img_path in all_list:
            if i%1000==0:
                print(i,len_all,img_path)
            label=int(img_path.split('label')[0].split('_')[-1])
            if label<155:#>180:
                save_path=img_path.replace('000_light_simi_1127add_nvwa','000_light_simi_1127add_nvwa_bad')
                Path(save_path).parent.mkdir(parents=True,exist_ok=True)
                shutil.copy(img_path,save_path)
            i+=1

'''刷图'''
def main_img_pair():
    # net =   MNV3_bufen_new5_out2(2, 1, 4, n_blocks=1).cuda()
    net =  MNV3_bufen_new5_bin_reg(2, 1, 4, n_blocks=1).cuda()
    # net =  MNV3_bufen_new5_bin_reg_small1(2, 1, 4, n_blocks=1).cuda()

    saveimg = 1
    savecsv = 0
    savecurve=0
    Crop_flag = 1
    # pthpath="./checkpoints/st_simi_1210_out2_sz72_fa5_fr5_moredata_crop_teacher/227_net_G.pth"
    # pthpath="./checkpoints/st_simi_21_1220_out2_sz72_newlabel_crop/284_net_G.pth"
    pthpath="./checkpoints/st_simi_st31_0109_out2_sz72_crop/296_net_G.pth"

  
    net.load_state_dict(torch.load(pthpath))
    net.eval()
    
    
    img_dir = "/data/guest/zsn/simi_data_light/003_ryttl_test/simi_wrong_fail/SSZ/"#/sm"#/frr"
    all_list = sorted([str(f) for f in Path(img_dir).rglob('*.bmp')])
    len_all = len(all_list)

    save_dir ="/data/guest/zsn/simi_data_light/003_ryttl_test/simi_wrong_fail/simi0109_31/"

    

    img_name_list_frr=[]
    net_simi_list_frr=[]
    net_class_list_frr = []
    img_name_list_frr0=[]
    net_simi_list_frr0=[]
    net_class_list_frr0 = []


    net_simi_list_frr_old=[]
    net_class_list_frr_old = []
    net_simi_list_frr0_old=[]
    net_class_list_frr0_old = []

    test_correct_1,test_correct_0,test_correct_fa,all1,all0,allfa=0,0,0,0,0,0
    eps=1e-10
    class_thre = 65450
    in_w,in_h = 72,72#96,96
    for idx, img in enumerate(all_list):
        # img='/hdd/file-input/zhangsn/data/002_test_frr_newmodel/test_suc/dust/164_9585_188_0012_L1_0026_11_sim188_simpro2085_score65530.41_fr.bmp'
     
        img_all = Image.open(img).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img_all))
        img_p = img_tensor[:,:,0].cpu().numpy()
        img_a = img_tensor[:,:,1].cpu().numpy()
        img_p = cv2.resize(img_p,(in_w,in_h))
        img_a = cv2.resize(img_a,(in_w,in_h))   

        #重叠区域外边框
        if Crop_flag==1:
            img_p_crop,img_a_crop=crop_img(img_p,img_a,in_w,in_h)
            img_a = transforms.ToTensor()(img_a_crop)  #size(h,w)
            img_p = transforms.ToTensor()(img_p_crop)
        else:
            img_a = transforms.ToTensor()(img_a)
            img_p = transforms.ToTensor()(img_p)

        
        AB=torch.cat((img_a,img_p),dim=0)
        # AB=torch.cat((img_p,img_a),dim=0)
        with torch.no_grad():
            out1,net_class_t=net(AB.unsqueeze(dim=0).cuda())

        net_class_t0 = F.softmax(net_class_t,dim = -1)
        clc_score = net_class_t0[0][1]*10000+0.5
        clc_score=int(clc_score.cpu().detach().numpy())
        pred = int(out1*265+55.5)
        
        prediction = torch.max(net_class_t, 1)
        pre_res=prediction[1].cpu().detach().numpy()  

        # clc_score_old=int(img.split('/')[-1].split('_')[1])
        # pred_old=int(img.split('/')[-1].split('_')[0])

        if saveimg == 1:
            rename = str(pred)+'_'+str(clc_score)+'_'+img.split('/')[-1]
            save_path = os.path.join(img[:-len(img.split('/')[-1])],rename)
            save_path=save_path.replace(img_dir, save_dir)
            Path(save_path).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy(img, save_path) 
            # rename = str(pred)+'_'+str(clc_score)+'_'+img.split('/')[-1][4:]
            # save_path = os.path.join(img[:-len(img.split('/')[-1])],rename)
            # # save_path=save_path.replace(img_dir, save_dir)
            # os.rename(img,save_path)
   

if __name__ == '__main__':
    main()
    # main_img_pair()
    # main_save_curve()
    # main_csv()
    # main_traindata_label()
    # main_fa_size()
    # static()
    # static_fafr()
    # static_label()
