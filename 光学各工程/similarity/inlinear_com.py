
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from models.MobileNet import *
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import shutil
import random
# os.environ['CUDA_VISIBLE_DEVICES']='5'

def ct_inlinear(txt_fn):
    all,fail,suc60=0,0,0
    pairname_n=[]
    InlinersNumR_n=[]
    with open(txt_fn, "r") as file:
        envirname=os.path.basename(txt_fn).split('Frr')[0]
        # path_img, f_n = os.path.split(txt_fn)
        lines = file.readlines()
        # lines = lines[1:]
        for line in lines:
            line=line.replace("\\","_")
            line=line.replace(".bmp","")
            sampname = envirname+'_'+line.split(',')[0].split(' ')[0].split(':')[-1]
            tempname = envirname+'_'+line.split(',')[0].split(' ')[1].split(':')[-1]
            sampname=sampname[:-7]+'_'+sampname[-7:]
            tempname=tempname[:-7]+'_'+tempname[-7:]
            pairname=sampname+'_'+tempname[-4:]
            InlinersNumR = int(line.split(',')[9].split(':')[-1])
            InlinersNumR_n.append(InlinersNumR)
            # sampname_list.append(sampname)
            # tempname_list.append(tempname)  
            pairname_n.append(pairname)  
            all+=1
            if InlinersNumR<=4:
                fail+=1
            elif InlinersNumR==60:
                suc60+=1
    return InlinersNumR_n,pairname_n,all,fail,suc60

def net_inlinear(txt_fn):
    all,fail,suc60=0,0,0
    with open(txt_fn, "r") as file:
        envirname=os.path.basename(os.path.dirname(txt_fn))
        # path_img, f_n = os.path.split(txt_fn)
        lines = file.readlines()
        # lines = lines[1:]
        pairname_n=[]
        InlinersNumR_n=[]
        overlap_n=[]
        for line in lines:
            line = line.replace("\\","_")
            line = line.replace('\n','')
            sampname = line.split(',')[0].split(':')[-1]
            tempid = line.split(',')[1].split(':')[-1]
            tempname = sampname[:-4]+tempid.zfill(4)
            pairname = sampname+'_'+tempid.zfill(4)
            InlinersNumR = int(line.split(',')[2].split(':')[-1])
            overlap = int(line.split(',')[3].split(':')[-1])
            InlinersNumR_n.append(InlinersNumR)
            overlap_n.append(overlap)
            # sampname_list.append(sampname)
            # tempname_list.append(tempname)    
            pairname_n.append(pairname)
            all+=1
            if InlinersNumR<=4:
                fail+=1
            elif InlinersNumR==60:
                suc60+=1
    return InlinersNumR_n,overlap_n,pairname_n,all,fail,suc60


def net_inlinear_suc(txt_fn):
    all,fail,suc60=0,0,0
    with open(txt_fn, "r") as file:
        envirname=os.path.basename(os.path.dirname(txt_fn))
        # path_img, f_n = os.path.split(txt_fn)
        lines = file.readlines()
        # lines = lines[1:]
        pairname_n=[]
        InlinersNumR_n=[]
        for line in lines:
            line=line.replace("\\","_")
            line=line.replace('\n','')
            sampname = line.split(',')[0].split(':')[-1]
            tempid=line.split(',')[1].split(':')[-1]
            tempname = sampname[:-4]+tempid.zfill(4)
            pairname = sampname+'_'+tempid.zfill(4)
            InlinersNumR = int(line.split(',')[2].split(':')[-1])
            rec = int(line.split(',')[3].split(':')[-1])
            if rec<=0:
                continue
            InlinersNumR_n.append(InlinersNumR)
            # sampname_list.append(sampname)
            # tempname_list.append(tempname)    
            pairname_n.append(pairname)
            all+=1
            if InlinersNumR<=4:
                fail+=1
            elif InlinersNumR==60:
                suc60+=1
    return InlinersNumR_n,pairname_n,all,fail,suc60

def main():
    # dataroot='/data/guest/zsn/simi_data_light/002_compare_inlinear_num/ct_nolearn'
    dataroot='/data/guest/zsn/simi_data_light/002_compare_inlinear_num/do_not_update_new/net_nolearn_overlap/dust'
    all_list = sorted([str(f) for f in Path(dataroot).rglob('*.txt')])
    # len_all = len(all_list)
    # all_list=['/data/guest/zsn/simi_data_light/002_compare_inlinear_num/ct_nolearn/dustFrrMatchInfoTrans000_064.txt']
    InlinersNumR_list=[]
    overlap_list=[]
    # sampname_list=[]
    # tempname_list=[]
    pairname_list=[]
    allnum=0
    failnum=0
    suc60num=0
    for idx, txt_fn in enumerate(all_list):
        print(txt_fn)
        # InlinersNumR_list,pairname_n,all_n,fail_n,suc60_n = ct_inlinear(txt_fn)
        InlinersNumR_n,overlap_n,pairname_n,all_n,fail_n,suc60_n = net_inlinear(txt_fn)
        # InlinersNumR_n,pairname_n,all_n,fail_n,suc60_n = net_inlinear_suc(txt_fn)
        allnum = allnum+all_n
        failnum = failnum+fail_n
        suc60num=suc60num+suc60_n
        pairname_list=pairname_list+pairname_n
        InlinersNumR_list=InlinersNumR_list+InlinersNumR_n
        overlap_list=overlap_list+overlap_n
        
    data={'name':pairname_list,'InlinersNumR':InlinersNumR_list,'overlap':overlap_list}  #
    df=pd.DataFrame(data)
    df.to_csv(dataroot+'/InlinersNumR_dust.csv',index=True)
    
    myInlinersNumR=np.array(InlinersNumR_list)
    # alldata=np.c_[resnumpy,gtnumpy]
    # print(alldata.shape)
    plt.figure()
    plt.title('inlinear_num')
    plt.hist(myInlinersNumR,bins=60, rwidth=0.8, range=(0,60),label='inlinear_num', align='left',alpha=0.5)
    # plt.hist(alldata,bins=65,color=['c','r'], rwidth=0.5, range=(55,315), label=['net_res','gt_label'], align='left',alpha=0.5,stacked=False)
    plt.legend()
    plt.savefig(dataroot+'/InlinersNumR_curve_dust.png')
        
    print('!num<=4:',failnum,'/',allnum,'=',failnum/allnum)
    print('!num=60:',suc60num,'/',allnum,'=',suc60num/allnum)

def compare_two_csv():
    path1='/data/guest/zsn/simi_data_light/002_compare_inlinear_num/do_not_update_new/net_nolearn_overlap/InlinersNumR_sun_part_45.csv'
    path2='/data/guest/zsn/simi_data_light/002_compare_inlinear_num/do_not_update_new/net_nolearn_overlap_suc/InlinersNumR_sun_part_45.csv'
    df1=pd.read_csv(path1)
    df2=pd.read_csv(path2)
    index = df1['name'].isin(df2['name'])
    # index = df1['name'].isin(df2['name'])
    outfile_suc=df1[index]
    outfile_fail=df1[~index]
    outfile_suc.to_csv('/data/guest/zsn/simi_data_light/002_compare_inlinear_num/do_not_update_new/net_nolearn_overlap/InlinersNumR_suc_sun_part_45.csv',index=False,encoding='gbk')
    outfile_fail.to_csv('/data/guest/zsn/simi_data_light/002_compare_inlinear_num/do_not_update_new/net_nolearn_overlap/InlinersNumR_fail_sun_part_45.csv',index=False,encoding='gbk')

if __name__ == '__main__':
    # main()
    compare_two_csv()
    