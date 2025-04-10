# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:49:21 2023

@author: fengshilin
"""
import PIL
import cv2
import os
import glob
from tqdm import tqdm
from PIL import Image 

img_file_list = glob.glob(r"/home/zhangsn/enhance/datasets/004_light124/test/0/!base/*.bmp")
replace_ori = r'/home/zhangsn/enhance/datasets/004_light124/test'
replace_dst = r'/home/zhangsn/enhance/datasets/004_light124/testcut'


for img_file in tqdm(img_file_list):
#     img = Image.open(img_file)
#     img = cv2.imread(img_file)
#     width, height = img.shape
#     img1 = img.crop((5,15,width-5,height-15))
#     img1 = img1.resize((124,124),resample=PIL.Image.BILINEAR)   #1
#     os.makedirs(os.path.split(img_file.replace(replace_ori,replace_dst))[0],exist_ok=True)
#     img1.save(img_file.replace(replace_ori,replace_dst))
     img = Image.open(img_file)
     width, height = img.size
     img1 = img.crop((10,20,width-10,height-20))
     img1 = img1.resize((124,124),resample=PIL.Image.BILINEAR)   #1
     os.makedirs(os.path.split(img_file.replace(replace_ori,replace_dst))[0],exist_ok=True)
     img1.save(img_file.replace(replace_ori,replace_dst))
     