"""
Author: qi.pan qi.pan@gigadevice.com
Date: 2023-08-18 11:22:24
LastEditors: qi.pan qi.pan@gigadevice.com
LastEditTime: 2024-04-10 23:04:13
FilePath: /vendor/autils/devideDB.py
Description:

Copyright (c) 2005-2022 GigaDevice/Silead Inc.
All rights reserved
The present software is the confidential and proprietary information of GigaDevice/Silead Inc.
You shall not disclose the present software and shall use it only in accordance with the terms of the license agreement you entered into with GigaDevice/Silead Inc.
This software may be subject to export or import laws in certain countries.
"""
"""
Author: qi.pan qi.pan@gigadevice.com
Date: 2023-08-18 11:22:24
LastEditTime: 2024-04-08 00:16:46
LastEditors: qi.pan qi.pan@gigadevice.com
Description:
FilePath: /vendor/autils/devideDB.py

Copyright (c) 2005-2022 GigaDevice/Silead Inc.
All rights reserved
The present software is the confidential and proprietary information of GigaDevice/Silead Inc.
You shall not disclose the present software and shall use it only in accordance with the terms of the license agreement you entered into with GigaDevice/Silead Inc.
This software may be subject to export or import laws in certain countries.
"""
import os
import shutil
import re
from shutil import move
from shutil import copyfile
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser(description="divide database")
    parser.add_argument(
        "-p",
        "--path",
        default=r"/home/panq/dataset/spoof/omegac2/product/711underwater/20240718-C2-水下-raw/0press",
        help="DB path",
    )
    # parser.add_argument('-p',"--path",default=r"\\?\E:\077\tina\Tina-all\1spoof\Tina-20210802", help = "DB path")
    parser.add_argument("-r", "--r", default=0.1, help="division ratio", type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    path = args.path
    ratio = args.r
    traindir = os.path.join(path, "train")
    validdir = os.path.join(path, "valid")
    if not os.path.isdir(traindir):
        os.makedirs(traindir)
    if not os.path.isdir(validdir):
        os.makedirs(validdir)
    for root, dirs, files in os.walk(path):
        for d in dirs:
            basePath = os.path.join(root, d)
            if basePath.find("train") != -1 or basePath.find("train") != -1:
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
                elif (
                    subPath.find(".fmi") == -1 and subPath.find(".bmp") == -1
                ):  # 删除非图像文件:
                    try:
                        os.remove(subPath)
                    except:
                        continue
                    continue
                else:
                    newFilePath = os.path.join(newPath, d)

                    if not os.path.exists(newFilePath):
                        os.makedirs(newFilePath)

                    try:
                        move(subPath, os.path.join(newFilePath, sub))
                        copyfile(
                            # subPath.replace(
                            #     "train-1-0825-rty", "train-1-0825-bak"
                            # ).replace(".bmp", ".fmi"),
                            os.path.join(
                                newFilePath,
                                sub.replace(".bmp", ".fmi")
                                .replace("-1", "")
                                .replace("-2", ""),
                            ),
                        )
                    except:
                        continue
