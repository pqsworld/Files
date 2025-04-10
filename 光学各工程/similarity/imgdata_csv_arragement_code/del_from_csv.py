"""
整理fa以及fr图像的工具
删除csv中某些行
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

OLDDATA=0
       
def del_csv(file):
    print(file)
    filepath = file
    df = pd.read_csv(
            filepath,
            header=0,
            index_col=0,
            # encoding = "gb2312",
            # names=['img_path','him','ori','ssim','raw_label','change_label','flag'],
            names =['img_path','him','ori','ssim','raw_label','change_label','flag','new_label']
            # names=['root','name','simi_label2','ham','ssim',]
            )
    # df_new = df["normal-pair-sele" in df['img_path']]
    del_index=[]
    for i in range(len(df)):
        name = df['img_path'][i]
        if "new-lianxu-succes" in name or "old-lianxu-identity" in name:
            print(name)
            del_index.append(i)
    df = df.drop(del_index)
    
    df.to_csv("train_move_badceses_newlabel_changewet_nolianxu.csv",index=True)
                                                                           
if __name__ == '__main__':
    datapath='/home/zhangsn/simi_net/000_data/traindata_level16_1116/train_1130/train_move_badceses_newlabel_changewet.csv'
    del_csv(datapath)
  