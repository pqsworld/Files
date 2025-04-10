'''
Author: qi.pan qi.pan@gigadevice.com
Date: 2023-08-18 11:22:24
LastEditTime: 2023-10-30 20:21:02
LastEditors: qi.pan qi.pan@gigadevice.com
Description:
FilePath: /vendor/autils/handle_folder_rename.py

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

path_sub1 = r"/home/panq/dataset/spoof/amz/valid/1spoof/"
dir = Path(path_sub1)
root_parts = list(dir.parts)
num_sub1 = len(root_parts)

l_key = ['常温', '低温', '洗手', '强光', 'normal', 'wash', 'cold', 'light','硅胶','明胶','树脂','数值','乳胶','吉利丁','石墨','打印','红底','拍照','印尼','印泥','碳粉']


def mkdir_before_rename(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


for d in dir.glob("**/*"):
    if d.is_dir():
        for idx in l_key:
            if idx in d._str:
                print(d)
                parts = list(d.parts)
                list_key_recusive = parts[num_sub1:-1]
                name_key_recusive = parts[-1] + '_' + '_'.join(
                    list_key_recusive)
                name_key = parts[-1]
                new_parts = root_parts[:-1]
                new_parts.append(parts[-1])
                key_parts = root_parts[:-1]
                new_parts.append(name_key_recusive)
                key_parts.append(name_key)
                path_newfile = Path(*new_parts)
                path_newkey = Path(*key_parts)
                # if not path_newkey.exists:
                path_newkey.mkdir(parents=True, exist_ok=True)
                try:
                    d.rename(mkdir_before_rename(path_newfile))
                except (FileExistsError , FileNotFoundError):
                    continue
