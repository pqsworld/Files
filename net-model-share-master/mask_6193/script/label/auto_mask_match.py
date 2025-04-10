#!/usr/bin/python
# -*- coding: utf-8 -*-
#Filename:floders.py

import numpy as np


import stat
import pdb

# export CUDA_VISIBLE_DEVICES=3

import itertools
import os
import util
from PIL import Image
import time
from multiprocessing import Pool

'''
将当前文件夹内所有文件的和指定文件夹内名字相同（带mask）的文件的mask结合，存在result文件夹中
'''
def fuse_mask2(source_path, target_path, mask_path):
   
    # image_ = Image.open(source_path).convert('L')
    # image_mask = Image.open(mask_path).convert('L')
    # ima_np = np.array(image_)
    # ima_mask_np = np.array(image_mask)
    # ima_np = np.hstack((ima_np,ima_mask_np))

    # image_c = Image.fromarray(ima_np)
    # image_c = image_c.convert('L')
    # image_c.save(os.path.join(target_path, source_path.split('/')[-1]))

    image_ = Image.open(source_path).convert('L')
    w,h = image_.size
    
    ima_np = np.array(image_)
    ima_mask_np = np.zeros((h,w))
    ima_mask_np[ima_mask_np==0]=255
    ima_np = np.hstack((ima_np,ima_mask_np))

    image_c = Image.fromarray(ima_np)
    image_c = image_c.convert('L')
    image_c.save(os.path.join(target_path, source_path.split('/')[-1]))

def fuse_mask_one(source_path, save_dir, mask_path, work_dir): #单图模式
    
    image_ = Image.open(source_path).convert('L')
    try:
        image_mask_temp = Image.open(mask_path).convert('L')
        ima_np = np.array(image_)
        ima_mask_np = np.array(image_mask_temp)
        ima_mask_np[ima_mask_np > 128] =128
        ima_mask_np[ima_mask_np < 128] =255
        ima_mask_np[ima_mask_np == 128] =0
        ima_np = np.hstack((ima_np,ima_mask_np))
        image_c = Image.fromarray(ima_np)
        image_c = image_c.convert('L')
        target_path = util.mkdir_files(save_dir, source_path, work_dir) #创建文件夹，并返回存储路径
        image_c.save(os.path.join(target_path, source_path.split('/')[-1]))
    except FileNotFoundError: 
        #image_mask_temp = Image.open(mask_path).convert('L')
        print("Not Found!：%s" % source_path)
        target_path = util.mkdir_files(save_dir, source_path, work_dir)
        fuse_mask2(source_path,target_path,mask_path)
        '''
        ima_np = np.array(image_)
        print("qqq")
        ima_np2 = np.copy(ima_np)
        print("sss")
        ima_np2=255
        print("ggg")
        ima_np = np.hstack((ima_np,ima_np2))
        print("lll")
        image_c = Image.fromarray(ima_np)
        print("ooo")
        image_c = image_c.convert('L')
        print("ppp")
        target_path = util.mkdir_files(save_dir, source_path, work_dir) #创建文件夹，并返回存储路径
        print("www")
        image_c.save(os.path.join(target_path, source_path.split('/')[-1]))
        print("vvv")
        print("Not Found!：%s" % source_path)
        '''

def fuse_mask(source_path, save_dir, mask_path, work_dir): #合并图模式
   
    image_ = Image.open(source_path).convert('L')
    try:
        image_mask_temp = Image.open(mask_path).convert('L')
        ima_np = np.array(image_)
        ima_mask_temp_np = np.array(image_mask_temp)
        _, ima_mask_np = np.hsplit(ima_mask_temp_np, 2)
        # ima_mask_np, _ = np.hsplit(ima_mask_temp_np, 2)
        ima_np = np.hstack((ima_np,ima_mask_np))

        image_c = Image.fromarray(ima_np)
        image_c = image_c.convert('L')
        target_path = util.mkdir_files(save_dir, source_path, work_dir) #创建文件夹，并返回存储路径
        image_c.save(os.path.join(target_path, source_path.split('/')[-1]))
    except FileNotFoundError: 
        print("Not Found!：%s" % source_path)
        #print('文件完整路径：%s\n' % file_path)



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

    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for parent, dirnames, filenames in os.walk(work_dir,  followlinks=True):
        for filename in filenames:
            file_path = os.path.join(parent, filename)
            if filename.endswith(".bmp"):
                # if re.match("00[0-7][0-9].bmp",filename) or re.match("008[0-4].bmp",filename):   #手指图库
                #print('文件完整路径：%s\n' % file_path)
                
                # fuse_mask(file_path, save_dir, mask_dir + file_path[len(work_dir):], work_dir)
                p.apply_async(fuse_mask_one,(file_path,save_dir,mask_dir + file_path[len(work_dir):],work_dir,))
                # fuse_mask_one(file_path, save_dir, mask_dir + parent[1:])
    p.close()
    p.join()