"""
整理fa以及fr图像的工具
"""

from __future__ import print_function

import glob
import os
import re
import shutil
import random
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from pathlib import Path
OLDDATA=0


def csv_rename_transpair(filepath):
    df = pd.read_csv(
            filepath,
            header=0,
            index_col=0,
            # encoding = "gb2312",
            names = ['img_path','him','ori','ssim','raw_label','change_label','flag']
            )
    all_num = len(df)
    for i in range(len(df)-1,-1,-1):
        
        # if len(df['name'][i]) != 33:
        #     name = os.path.join(df['root'][i],df['name'][i])
        #     print(name,'wrong',file)
        img_path = df['img_path'][i]
        
        img = cv2.imread(img_path)
        h,w = img.shape[0],img.shape[1]
        # print('000',img.shape)
        if w!=18:
            w2 = int(w / 4)
            # A = AB.crop((w2*2,0, w2*3, h))
            # B = AB.crop((w2, 0, w2*2, h))
            # ABmerge = AB.crop((w2*3,0,w, h))
            ABmerge = img[0:h,w2*3:w]
            mask = img[0:h,w2*2:w2*3,0]
            A_level_16_img = np.where(mask<10,0,ABmerge[:,:,1])
            B_level_16_img = np.where(mask<10,0,ABmerge[:,:,2])
            zeros = np.zeros_like(A_level_16_img)   
            newimg = cv2.merge([zeros, A_level_16_img, B_level_16_img])
            new_img_path = img_path.replace('/home/jianght/001_data/003_featuredata_20231117/','/data/guest/simi_data/')
            img_name = new_img_path.split('/')[-1]
            new_img_name = img_name[-37:]
            new_img_path = new_img_path.replace(img_name,new_img_name)
            df['img_path'][i]=new_img_path
            
        else:
            newimg=img
            new_img_path = img_path.replace('/home/zhangsn/simi_net/000_data/000_alldata/03transpair/','/data/guest/simi_data/train/add_data/')
            # img_name = new_img_path.split('/')[-1]
            # new_img_name = img_name
            # new_img_path = new_img_path.replace(img_name,new_img_name)
            df['img_path'][i]=new_img_path
        na=new_img_path.split('/')[:]
        save_data_path = na[0]
        for i_na in range(1, len(na) - 1):
            save_data_path = save_data_path + "/" + na[i_na]
            if not os.path.exists(save_data_path):
                os.mkdir(save_data_path) 
        print('%d/%d: %s,%s' % (i, all_num,img_path,new_img_path))       
        cv2.imwrite(new_img_path, newimg)
        shutil.copy(img_path[:-4]+'.npy',new_img_path[:-4]+'.npy')
        
    df.to_csv(filepath.replace('.csv','_change.csv'),index=False)

def loadimg_rename_transpair(Dpath):
    all_img=[str(f) for f in Path(Dpath).rglob('*.bmp')]
    all_num = len(all_img) 
    index=0
    for img_path in all_img:    
        index+=1          
        img = cv2.imread(img_path)
        h,w = img.shape[0],img.shape[1]
        # print('000',img.shape)
        w2 = int(w / 4)
        # A = AB.crop((w2*2,0, w2*3, h))
        # B = AB.crop((w2, 0, w2*2, h))
        # ABmerge = AB.crop((w2*3,0,w, h))
        ABmerge = img[0:h,w2*3:w]
        mask = img[0:h,w2*2:w2*3,0]
        A_level_16_img = np.where(mask<10,0,ABmerge[:,:,1])
        B_level_16_img = np.where(mask<10,0,ABmerge[:,:,2])
        zeros = np.zeros_like(A_level_16_img)   
        newimg = cv2.merge([zeros, A_level_16_img, B_level_16_img])
        new_img_path = img_path.replace('/home/zhangsn/simi_net/000_data/traindata_level16_1116/valid','/data/guest/simi_data/test')
        img_name = new_img_path.split('/')[-1]
        new_img_name = img_name[-37:]
        new_img_path = new_img_path.replace(img_name,new_img_name)

            
        na=new_img_path.split('/')[:]
        save_data_path = na[0]
        for i_na in range(1, len(na) - 1):
            save_data_path = save_data_path + "/" + na[i_na]
            if not os.path.exists(save_data_path):
                os.mkdir(save_data_path) 
        print('%d/%d: %s,%s' % (index, all_num,img_path,new_img_path))       
        cv2.imwrite(new_img_path, newimg)
        shutil.copy(img_path[:-4]+'.npy',new_img_path[:-4]+'.npy')       

                                                                           
if __name__ == '__main__':
    # datapath='/home/zhangsn/simi_net/000_data/traindata_level16_1116/train_1127/train_new.csv'
    datapath='/home/zhangsn/simi_net/000_data/traindata_level16_1116/valid/new-lianxu-succes'
    loadimg_rename_transpair(datapath)
  