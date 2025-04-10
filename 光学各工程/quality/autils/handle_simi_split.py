'''
Author: qi.pan qi.pan@gigadevice.com
Date: 2023-10-26 12:17:37
LastEditTime: 2023-10-30 19:40:10
LastEditors: qi.pan qi.pan@gigadevice.com
Description:
FilePath: /vendor/autils/handle_simi_split.py

Copyright (c) 2005-2022 GigaDevice/Silead Inc.
All rights reserved
The present software is the confidential and proprietary information of GigaDevice/Silead Inc.
You shall not disclose the present software and shall use it only in accordance with the terms of the license agreement you entered into with GigaDevice/Silead Inc.
This software may be subject to export or import laws in certain countries.
'''

import os
import shutil
import argparse
from pathlib import Path
import xlsxwriter


def mkdir_before_rename(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="ASP Train Network")
    parser.add_argument(
        '-p',
        '--path',
        default=
        r"/home/panq/dataset/siminet/new/6193-DK4-110-zise-12-ori/0002_new/R")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    path = args.path
    count = 0
    path = Path(path)
    row = 1

    for p in path.rglob("*.bmp"):

        score = int(p.stem.split("_")[-3].split("g")[-1])
        # if score<=31:
        # if score <= 30 or score >= 50:

        score_10 = score // 10 * 10
        parts = list(p.parts)
        parts.append(parts[10])
        parts[10] = parts[9] + '-' + str(score_10)

        path_newfile = Path(*parts)
        try:
            p.rename(mkdir_before_rename(path_newfile))
        except FileExistsError:
            continue

        # p.unlink()
        # print(score)
