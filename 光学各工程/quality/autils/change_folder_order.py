'''
Author: qi.pan qi.pan@gigadevice.com
Date: 2023-08-25 18:03:46
LastEditTime: 2024-04-01 21:51:54
LastEditors: qi.pan qi.pan@gigadevice.com
Description:
FilePath: /vendor/autils/change_folder_order.py

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
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="divide database")
    parser.add_argument(
        '-p',
        "--path",
        # default=r"/home/panq/dataset/spoof/amz/product/0922/silead/silead - 副本",
        default=r"/home/panq/dataset/spoof/omegac2/product/0401icon/214-base202-changebase",
        help="DB path")

    parser.add_argument('-r',
                        "--r",
                        default=0.1,
                        help="division ratio",
                        type=float)
    args = parser.parse_args()
    return args


def change_base(path):
    # for root, dirs, files in os.walk(path_base):
    filelist=  os.listdir(path)
    list_mtn =[os.path.join(path ,file) for file in filelist]
    list_mtn.sort()

    for root, dirs, files in os.walk(path):

        for d in dirs:
            # !!!TODO:此处暴力地认定 base所在的路径中 代表机器的格式为 /2# 格式
            basePath = os.path.join(root, d)
            path_change = Path(basePath)
            if basePath.find('0_') != -1 or basePath.find('粉色硅胶') != -1:
                os.rename(basePath,basePath.replace('0_','7_').replace('粉色硅胶','7_粉色硅胶'))
                print(basePath.replace('0_','7_').replace('粉色硅胶','7_粉色硅胶'))
            if basePath.find('1_') != -1 or basePath.find('黑色硅胶') != -1:
                os.rename(basePath,basePath.replace('1_','6_').replace('黑色硅胶','6_黑色硅胶'))
                print(basePath.replace('1_','6_').replace('粉色硅胶','6_黑色硅胶'))
            if basePath.find('2_') != -1 or basePath.find('黑色木胶') != -1:
                os.rename(basePath,basePath.replace('2_','5_').replace('黑色木胶','5_黑色木胶'))
                print(basePath.replace('2_','5_').replace('黑色木胶','5_黑色木胶'))
            if basePath.find('3_') != -1 or basePath.find('红色打印纸') != -1:
                os.rename(basePath,basePath.replace('3_','4_').replace('红色打印纸','4_红色打印纸'))
                print(basePath.replace('3_','4_').replace('红色打印纸','4_红色打印纸'))
                        # rmtree(basePath)
                        # copytree(list_mtn[i],basePath.split('')[0])


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    change_base(args.path)
