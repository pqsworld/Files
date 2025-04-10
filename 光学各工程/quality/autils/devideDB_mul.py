"""
Author: qi.pan qi.pan@gigadevice.com
Date: 2024-03-07 18:46:56
LastEditTime: 2024-03-07 18:48:47
LastEditors: qi.pan qi.pan@gigadevice.com
Description:
FilePath: /vendor/autils/devideDB_mul.py

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
import argparse
import os
import random
import re
import shutil
from pathlib import Path
from shutil import copyfile, move


def parse_args():
    parser = argparse.ArgumentParser(description="divide database")
    # parser.add_argument('-p',"--path",default=r"\\?\E:\077\tina\Tina-all\1spoof\Tina-20210802", help = "DB path")
    parser.add_argument(
        "-p",
        "--path",
        default=r"/ssd/share/spoof/s3/data/0fp/valid-0410",
        help="DB path",
    )
    parser.add_argument(
        "-r", "--r", default=0.001, help="division ratio", type=float
    )  # 1 : 9
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    path = args.path
    ratio = args.r
    traindir = os.path.join(path, "train-04110")
    validdir = os.path.join(path, "valid-04110")
    if not os.path.isdir(traindir):
        os.makedirs(traindir)
    if not os.path.isdir(validdir):
        os.makedirs(validdir)
    path = Path(path)
    for p in path.glob(r"*"):
        val_dir_num = 1
        for root, dirs, files in os.walk(p):
            for f in files:
                file = Path(root) / f
                if (
                    str(file).find(".fmi") == -1 and str(file).find(".bmp") == -1
                ):  # 删除非图像文件:
                    try:
                        if file.exists():
                            file.remove()
                    except:
                        continue
            for d in dirs:
                jump_flag = False
                basePath = os.path.join(root, d)
                source = Path(basePath)
                target = Path(basePath)
                if (
                    basePath.find("train-04110") != -1
                    or basePath.find("valid-04110") != -1
                    or basePath.find("!base") != -1
                ):
                    continue
                list1 = os.listdir(basePath)
                rel = os.path.relpath(basePath, path)  # 获得对于源路径的相对路径
                if random.random() > ratio:  # 随机划分训练测试集
                    target = os.path.join(traindir, rel)
                else:
                    target = os.path.join(validdir, rel)
                for sub in list1:  # 判断是否为最深层文件夹
                    subPath = os.path.join(basePath, sub)
                    if os.path.isdir(subPath):  # 不是最深层文件夹 跳过
                        jump_flag = True
                        break
                    else:
                        source = Path(basePath)
                if jump_flag:
                    continue
                if val_dir_num:  # 各分类至少划分n个文件夹为测试集
                    # print("enter val_dir_num:{}".format(val_dir_num))
                    target = target.replace("train", "valid")
                    val_dir_num = 0
                    # print("out val_dir_num:{}".format(val_dir_num))
                for f in source.rglob("*.bmp"):
                    rel = f.relative_to(source)
                    newFile = Path(target) / rel
                    if not newFile.parent.exists():
                        newFile.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        move(f, newFile)
                    except:
                        print(f)
                        print(newFile)
