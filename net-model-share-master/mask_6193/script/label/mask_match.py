#!/usr/bin/python
# -*- coding: utf-8 -*-
#Filename:floders.py

import numpy as np


import stat
import pdb

# export CUDA_VISIBLE_DEVICES=3

import itertools
import os
from PIL import Image
import util
from multiprocessing import Pool

'''
将当前文件夹内所有文件递归的和上一目录的mask的标签图结合，存在result文件夹中
'''

def fuse_mask(source_path, target_path, mask_path):
   
    # image_ = Image.open(source_path).convert('L')
    # image_mask = Image.open(mask_path).convert('L')
    # ima_np = np.array(image_)
    # ima_mask_np = np.array(image_mask)
    # ima_np = np.hstack((ima_np,ima_mask_np))

    # image_c = Image.fromarray(ima_np)
    # image_c = image_c.convert('L')
    # image_c.save(os.path.join(target_path, source_path.split('/')[-1]))

    image_ = Image.open(source_path).convert('L')
    
    ima_np = np.array(image_)
    ima_mask_np = np.zeros((188,188))
    ima_np = np.hstack((ima_np,ima_mask_np))

    image_c = Image.fromarray(ima_np)
    image_c = image_c.convert('L')
    image_c.save(os.path.join(target_path, source_path.split('/')[-1]))

def main(work_dir, save_dir, mask_dir):
    for parent, dirnames, filenames in os.walk(work_dir,  followlinks=True):
        for filename in filenames:
            file_path = os.path.join(parent, filename)
            if filename.endswith(".bmp"):
                # if re.match("00[0-7][0-9].bmp",filename) or re.match("008[0-4].bmp",filename):   #手指图库
                print('文件完整路径：%s\n' % file_path)
                out_path = util.mkdir_files(save_dir, file_path)
                fuse_mask(file_path, out_path, mask_dir)

if __name__=='__main__':
    #prename = r"/hdd/file-input/guest/ben/mask/pycode/mask/datasets"
    prename = r""
    with open('/hdd/file-input/guest/ben/mask/pycode/mask/datasets/scripts/config.txt',"r") as f:    #设置文件对象
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

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    p= Pool(10)
    for parent, dirnames, filenames in os.walk(work_dir,  followlinks=True):
        for filename in filenames:
            file_path = os.path.join(parent, filename)
            if filename.endswith(".bmp"):
                # if re.match("00[0-7][0-9].bmp",filename) or re.match("008[0-4].bmp",filename):   #手指图库
                print('文件完整路径：%s\n' % file_path)
                out_path = util.mkdir_files(save_dir, file_path, work_dir)
                p.apply_async(fuse_mask, (file_path,out_path,mask_dir,))
    p.close()
    p.join()
              