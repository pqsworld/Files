#!/usr/bin/python3 -utt
# -*- coding: utf-8 -*-
import cv2
import os
import yaml
import torch
import random
# import pynvml
import logging
from models import FPDT
import pandas as pd
from get_Net_Points_6193 import get_module

default_config = {

    'testclass': 'get_net_pts_6193',    # 真实复现率: 122x36，测试自监督大模型 81300
    'model': 'UnSuperPointNet_small_8_theta_student_block_v2_0814',
    
    'img_h':122,
    'img_w':36,
    'img_resize_h':96,
    'img_resize_w':28,
    'top_k': 130,
    'border_remove': 2,
    'nms': 2,
}

def processing(FPDT, default_config, device="cpu"):

    Getclass = get_module(default_config['testclass'])
    TestComponent = Getclass(default_config['image_path'], device, **default_config)  # 初始化测试类
    res = TestComponent.test_process(FPDT)
    
    name, pts, img = res['name'], res['pts'], res['img']
    
    # save img
    save_img_pth = default_config['output_dir'] + '/' +  (name + '_pts.bmp')
    save_pts_pth = default_config['output_dir'] + '/' +  (name + '_pts.csv')
    cv2.imwrite(str(save_img_pth), img)
        
    # save pts
    df = pd.DataFrame(pts, columns=['score', 'x', 'y' ])
    df.to_csv(save_pts_pth)    

    pass


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES']='5'

    lucky_gpu = random.choice([5])
    device = torch.device('cuda:' + str(lucky_gpu))
    print("==> Use GPU: {}".format(lucky_gpu))
    
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    image_path = './6193_test_img.bmp'
    
    '''net UnSuperPointNet_small_8_theta_student_block_v2_0814'''
    model_weights_detector = "./net_result/6193/superPointNet_29700_checkpoint.pth.tar" #目前最优版本
    
    ''' 存储路径 '''    
    output_dir = "./res"
    os.makedirs(output_dir, exist_ok=True)
    print('Output-> ', output_dir)

    default_config.update({
        'model_weights': model_weights_detector,
        'image_path': image_path,
        'output_dir': output_dir})
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)

    FPDT = FPDT(default_config['model'], model_weights_detector, device=device)

    processing(FPDT, default_config, device=device)

