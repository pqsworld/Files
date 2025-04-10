'''
Author: qi.pan qi.pan@gigadevice.com
Date: 2023-08-25 18:03:46
LastEditTime: 2024-04-07 22:07:42
LastEditors: qi.pan qi.pan@gigadevice.com
Description:
FilePath: /vendor/autils/change_base.py

Copyright (c) 2005-2022 GigaDevice/Silead Inc.
All rights reserved
The present software is the confidential and proprietary information of GigaDevice/Silead Inc.
You shall not disclose the present software and shall use it only in accordance with the terms of the license agreement you entered into with GigaDevice/Silead Inc.
This software may be subject to export or import laws in certain countries.
'''
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:35:50 2021

@author: suanfa
"""
import os
import shutil
import re
from shutil import move
from shutil import copyfile, rmtree,copytree
import argparse
import random
import pathlib
from pathlib import PureWindowsPath


def parse_args():
    parser = argparse.ArgumentParser(description="divide database")
    parser.add_argument(
        '-p',
        "--path",
        # default=r"/home/panq/dataset/spoof/amz/product/0922/silead/silead - 副本",
        default=r"/home/panq/dataset/spoof/s3/product/0415wxn/212raw-base199",
        help="DB path")
    parser.add_argument(
        "--path_base",
        # default=r"/home/panq/dataset/spoof/amz/product/0922/silead/silead - 副本",
        default=r"/home/panq/dataset/spoof/omegac2/product/0325icon-back/20240322-iconbase/20240322-新旧坐标校准图/omgS3/旧坐标",
        help="DB path")
    parser.add_argument('-r',
                        "--r",
                        default=0.1,
                        help="division ratio",
                        type=float)
    args = parser.parse_args()
    return args


def change_base(path,path_base):
    # for root, dirs, files in os.walk(path_base):
    filelist=  os.listdir(path_base)
    list_mtn =[os.path.join(path_base ,file) for file in filelist]
    list_mtn.sort()

    for root, dirs, files in os.walk(path):

        for d in dirs:
            # !!!TODO:此处暴力地认定 base所在的路径中 代表机器的格式为 /2# 格式
            basePath = os.path.join(root, d)
            if basePath.find('!base') != -1:
                num_mtn = basePath.split('#')[0].split('/')[-1]+'#'
                for i in range(len(list_mtn)):
                    if num_mtn in list_mtn[i]:
                        print(num_mtn+' : '+list_mtn[i])
                        rmtree(basePath)
                        copytree(list_mtn[i],basePath)





if __name__ == '__main__':
    args = parse_args()
    path = args.path
    change_base(args.path,args.path_base)
