'''
Author: qi.pan qi.pan@gigadevice.com
Date: 2023-08-18 11:22:24
LastEditTime: 2023-10-30 19:03:35
LastEditors: qi.pan qi.pan@gigadevice.com
Description:
FilePath: /vendor/autils/handle_folder.py

Copyright (c) 2005-2022 GigaDevice/Silead Inc.
All rights reserved
The present software is the confidential and proprietary information of GigaDevice/Silead Inc.
You shall not disclose the present software and shall use it only in accordance with the terms of the license agreement you entered into with GigaDevice/Silead Inc.
This software may be subject to export or import laws in certain countries.
'''
import os
# import xlwt
import numpy as np
import time
import sys
import re
import xlsxwriter
import os
from os.path import join as Join
import subprocess
from subprocess import call
from pathlib import Path
from glob import glob

path_sub1 = "/home/panq/dataset/spoof/amz/train/valid-4-split/0fp/BOE-1#-20230824/BOE-1#-20230824-副本"
path_sub2 = "/home/panq/dataset/spoof/amz/train/valid-4-split/0fp/BOE-1#-20230824/BOE-1#-20230824-副本/FingerCollect/0003"
# path="/home/panq/dataset/spoof/6250/0705sh/0705sh/1hdg/1hdg.log"


def get_folder_sub2(path):
    listp = [p for p in os.listdir(path)]
    return listp


def get_folder_depth(path, depth=3):
    path = Path(path)
    assert path.exists(), f'Path:{path} does not exist!'
    ds = '*/' * depth
    search_pattern = Join(path, ds)
    return list(glob(f'{search_pattern}'))


if __name__ == '__main__':
    list_sub2 = get_folder_sub2(path_sub2)
    print(list_sub2)

    path_sub2_tidy = Join(path_sub1, '-tidy')
    if not os.path.exists(path_sub2_tidy):
        os.makedirs(path_sub2_tidy)

    # for type in list_sub2:
    #     if not os.path.exists(Join(path_sub2_tidy,'-',type)):
    #         os.makedirs(Join(path_sub2_tidy,'-',type))

    delta = len(Path(path_sub2).parts) - len(Path(path_sub1).parts)
    list_a = get_folder_depth(path_sub1, delta + 1)
    list_depth = call('find ' + path_sub1 + ' -type d -mindepth 3 -maxdepth 3',
                      shell=True)
    a = 1
