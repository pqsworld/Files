"""
Author: qi.pan qi.pan@gigadevice.com
Date: 2023-09-18 17:41:49
LastEditors: qi.pan qi.pan@gigadevice.com
LastEditTime: 2024-07-19 17:37:52
FilePath: /vendor/autils/devidePy.py
Description: 

Copyright (c) 2005-2022 GigaDevice/Silead Inc.
All rights reserved
The present software is the confidential and proprietary information of GigaDevice/Silead Inc.
You shall not disclose the present software and shall use it only in accordance with the terms of the license agreement you entered into with GigaDevice/Silead Inc.
This software may be subject to export or import laws in certain countries.
"""
"""
Author: qi.pan qi.pan@gigadevice.com
Date: 2023-09-18 17:41:49
LastEditTime: 2024-05-27 10:34:47
LastEditors: qi.pan qi.pan@gigadevice.com
Description:
FilePath: /vendor/autils/devidePy.py

Copyright (c) 2005-2022 GigaDevice/Silead Inc.
All rights reserved
The present software is the confidential and proprietary information of GigaDevice/Silead Inc.
You shall not disclose the present software and shall use it only in accordance with the terms of the license agreement you entered into with GigaDevice/Silead Inc.
This software may be subject to export or import laws in certain countries.
"""
"""
Author: qi.pan qi.pan@gigadevice.com
Date: 2023-09-18 17:41:49
LastEditors: qi.pan qi.pan@gigadevice.com
LastEditTime: 2024-04-10 23:08:39
FilePath: /vendor/autils/devidePy.py
Description:

Copyright (c) 2005-2022 GigaDevice/Silead Inc.
All rights reserved
The present software is the confidential and proprietary information of GigaDevice/Silead Inc.
You shall not disclose the present software and shall use it only in accordance with the terms of the license agreement you entered into with GigaDevice/Silead Inc.
This software may be subject to export or import laws in certain countries.
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:35:50 2021

@author: suanfa
"""
import os
import shutil
import re
from shutil import move, copy
from shutil import copyfile
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser(description="divide database")
    parser.add_argument(
        "-p",
        "--path",
        default=r"/home/panq/dataset/spoof/omegac2/product/711underwater/20240718-C2-水下-raw-cp/0press",
        help="DB path",
    )
    # parser.add_argument('-p',"--path",default=r"\\?\E:\077\tina\Tina-all\1spoof\Tina-20210802", help = "DB path")
    parser.add_argument("-r", "--r", default=0.2, help="division ratio", type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    path = args.path
    ratio = args.r
    traindir = os.path.join(path, "train-4-0908")
    validdir = os.path.join(path, "valid-4-0908")
    if not os.path.isdir(traindir):
        os.makedirs(traindir)
    if not os.path.isdir(validdir):
        os.makedirs(validdir)
    for root, dirs, files in os.walk(path):
        for d in dirs:
            basePath = os.path.join(root, d)
            if basePath.find("train") != -1 or basePath.find("valid") != -1:
                continue
            list1 = os.listdir(basePath)
            rel = os.path.relpath(root, path)  # 获得对于源路径的相对路径
            if random.random() > ratio:
                newPath = os.path.join(traindir, "train-" + rel)
            else:
                newPath = os.path.join(validdir, "valid-" + rel)
            for sub in list1:
                subPath = os.path.join(basePath, sub)
                if os.path.isdir(subPath):  # 不是最深层文件夹 跳过
                    continue
                # elif subPath.find(".py") == -1:
                #     continue
                else:
                    newFilePath = os.path.join(newPath, d)

                    if not os.path.exists(newFilePath):
                        os.makedirs(newFilePath)

                    try:
                        copy(subPath, os.path.join(newFilePath, sub))
                    except:
                        continue
