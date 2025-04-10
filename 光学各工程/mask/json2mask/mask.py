#!/usr/bin/python
# -*- coding: utf-8 -*-
# Filename:floders.py

import numpy as np

import stat
import pdb

# export CUDA_VISIBLE_DEVICES=3

import itertools
import os
from math import exp
import re
from PIL import Image
import time
import sys
sys.path.append("/hdd/file-input/yey/scripts")
import util
from multiprocessing import Pool
import cv2

'''
将图库中的label图像挑出来
'''


def convert_label(source_dir, target_path):
    # source_dir = r'X:\jht\异物\process\shiyan\data\mask'
    # list_dir = os.listdir(source_dir)
    # target_path = r'X:\jht\异物\process\shiyan\data\convert_mask'

    mask_ = Image.open(source_dir).convert('L')
    mask_np = np.array(mask_)
    mask_np[mask_np[:, :] != 0] = 255

    # 核的大小和形状
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 开操作
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel, iterations=3)

    image_path = source_dir.split("/")
    image_path = "/".join(image_path[:-1]) + "/img.png"
    image_ = Image.open(image_path).convert('L')
    ima_np = np.array(image_)

    image_c = np.hstack((ima_np,mask_np))

    image_c = Image.fromarray(image_c)
    image_c = image_c.convert('L')
 

    # save_name = source_dir.split('/')[-2].split('_')[0] + ".bmp"
    save_name = source_dir.split('/')[-2].split('_')
    save_name = save_name[:-1]  # 去除json
    save_name = "_".join(save_name) + ".bmp"
    image_c.save(os.path.join(target_path, save_name))

def convert_(parent, filename, save_dir, work_dir):
    file_path = os.path.join(parent, filename)
    if filename.endswith(".png"):
        if re.match("label.png", filename):  # 标注
            print('文件完整路径：%s\n' % file_path)
            path_name = file_path[len(work_dir):].split("/")
            mkdir_name = save_dir
            if not os.path.isdir(mkdir_name):
                os.mkdir(mkdir_name)
            for path_level in path_name:
                if (path_level == path_name[0]) or (path_level == path_name[-1]) or (path_level == path_name[-2]):
                    continue
                mkdir_name += "/" + path_level
                if not os.path.isdir(mkdir_name):
                    os.mkdir(mkdir_name)
            out_path = mkdir_name
            
            convert_label(file_path, out_path)

def find_label(work_dir, save_dir):
    p = Pool(10)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for parent, dirnames, filenames in os.walk(work_dir, followlinks=True):
        for filename in filenames:
            p.apply_async(convert_, (parent, filename, save_dir, work_dir))
    p.close()
    p.join()




if __name__ == '__main__':
    # work_dir = '.\\'
    # save_dir = '../mask'
    prename = r"/hdd/file-input"
    with open('/hdd/file-input/yey/scripts/config.txt',"r") as f:    #设置文件对象
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

    find_label(work_dir, save_dir)