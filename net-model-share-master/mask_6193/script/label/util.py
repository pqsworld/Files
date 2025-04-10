#!/usr/bin/python
# -*- coding: utf-8 -*-
#Filename:functions.py


import os


'''
本文件定义了一些通用模块供scripts下的脚本进行调用
'''

def mkdir_files(mkdir_name, file_path, work_dir):  # 根据file_path的目录结构，在mkdir_name位置构建文件夹， 同时返回相应的存储路径

    path_name = file_path[len(work_dir):].split("/")
    if not os.path.isdir(mkdir_name):
        os.mkdir(mkdir_name)
    for path_level in path_name:
        if path_level == path_name[-1]: #跳过初始的 . 和 文件名
            continue
        mkdir_name += "/" + path_level
        if not os.path.isdir(mkdir_name):
            os.mkdir(mkdir_name)
    dst_path = mkdir_name
    return dst_path
