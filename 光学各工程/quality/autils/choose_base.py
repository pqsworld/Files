'''
Author: qi.pan qi.pan@gigadevice.com
Date: 2023-08-25 18:03:46
LastEditTime: 2024-04-01 20:39:51
LastEditors: qi.pan qi.pan@gigadevice.com
Description:
FilePath: /vendor/autils/choose_base.py

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
from shutil import copyfile, rmtree
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
        default=r"/home/panq/dataset/spoof/amz/data/data-6-1027/",
        help="DB path")
    parser.add_argument('-r',
                        "--r",
                        default=0.1,
                        help="division ratio",
                        type=float)
    args = parser.parse_args()
    return args


def choose_base(path):
    for root, dirs, files in os.walk(path):
        debasePath = os.path.join(root, "!base")
        for d in dirs:
            if os.path.exists(debasePath) and len(os.listdir(debasePath)) == 3:
                print("break")
                break
            basePath = os.path.join(root, d)
            if basePath.find('!base') != -1:
                continue
            list1 = os.listdir(basePath)
            list1.sort()
            baseCount = 3
            for sub in list1:
                subPath = os.path.join(basePath, sub)
                if os.path.isdir(subPath):  #不是最深层文件夹 跳过
                    continue
                elif subPath.find('.fmi') == -1 and subPath.find(
                        '.bmp') == -1:  #删除非图像文件:
                    os.remove(subPath)
                    continue
                else:
                    print(subPath)
                    debasePath = os.path.join(root, "!base")
                    if not os.path.exists(debasePath):
                        os.makedirs(debasePath)
                    print("{}:\n{}".format(debasePath,
                                           len(os.listdir(debasePath))))

                    if baseCount > 0 and len(os.listdir(debasePath)) < 3:
                        print("choose base/n")
                        baseCount = baseCount - 1
                        copyfile(subPath, os.path.join(debasePath, sub))
                    else:
                        break


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    choose_base(path)
