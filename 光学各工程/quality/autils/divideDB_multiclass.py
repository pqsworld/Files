# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:35:50 2021

@author: suanfa
"""
import os
import shutil
import re
from shutil import move
from shutil import copyfile
import argparse
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="divide database")
    # parser.add_argument('-p',"--path",default=r"\\?\E:\077\tina\Tina-all\1spoof\Tina-20210802", help = "DB path")
    # parser.add_argument('-p',"--path",default=r"/hdd/file-input/qint/data/parallel_data/huahai_ori_bandpass/bandpass/", help = "DB path")
    # parser.add_argument('-p',"--path",default=r"/hdd/file-input/qint/data/parallel_data/OmegaC2/traC2/", help = "DB path")
    # parser.add_argument('-p',"--path",default=r"/hdd/file-input/qint/data/parallel_data/Fusion_ori_bandpass/huahai_ori_bandpass/HH_2D_25D/bandpass_2d_25d/", help = "DB path")
    # parser.add_argument('-p',"--path",default=r"/hdd/file-input/qint/data/parallel_data/Fusion_ori_bandpass/cmr_ori_bandpass/bandpass/", help = "DB path")
    parser.add_argument(
        "-p",
        "--path",
        default=r"/home/panq/dataset/spoof/omegac2/product/711underwater/20240718-C2-水下-raw-cp/1press",
        help="DB path",
    )
    parser.add_argument(
        "-r", "--r", default=0.1, help="division ratio", type=float
    )  # 1 : 9
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
                    basePath.find("train") != -1
                    or basePath.find("valid") != -1
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
