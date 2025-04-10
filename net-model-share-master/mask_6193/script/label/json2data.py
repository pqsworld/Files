#!/usr/bin/python
# -*- coding: utf-8 -*-
#Filename:cal2!base.py

import numpy as np

import stat
import pdb

# export CUDA_VISIBLE_DEVICES=3

import itertools
import os
import subprocess
from multiprocessing import Pool
'''
将json转化为data
'''


def zhuan(file_path):
    if file_path.find('.json') != -1:
        os.system("labelme_json_to_dataset %s" % file_path)
if __name__=='__main__':
    prename = r""
    with open('./config.txt',"r") as f:    #设置文件对象
        configs = f.readlines()    #可以是随便对文件的操作
    for config in configs:
        config_dict = config.split(":")
        if config_dict[0] == "work_dir":
            work_dir = config_dict[-1].split("\\")
            work_dir = "/".join(work_dir).strip('\n')
            work_dir = prename + work_dir
        elif config_dict[0] == "save_dir":
            save_dir = config_dict[-1].split("\\")
            save_dir = "/".join(save_dir).strip('\n')
            save_dir = prename + save_dir
        elif config_dict[0] == "mask_dir":
            mask_dir = config_dict[-1].split("\\")
            mask_dir = "/".join(mask_dir).strip('\n')
            mask_dir = prename + mask_dir
        else:
            print("Config error!")
            exit()
            
    p= Pool(10)
    for parent, dirnames, filenames in os.walk(work_dir,  followlinks=True):
        for filename in filenames:
            file_path = os.path.join(parent, filename)    
            p.apply_async(zhuan, (file_path,))
    p.close()
    p.join()