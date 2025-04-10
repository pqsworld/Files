import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from Qmodels.PTQ.ptq_config import Config
from Qmodels.Qhardnet_model import QHardNet_small

def main():
    
    cfg = Config()
    device = torch.device('cuda:2')
    net_weight = "/data/yey/work/6191/unsuperpoint/logs/0701_RL_4_Qhardnetsmall/checkpoints/superPointNet_200_desc.pth.tar"
    valdir = "/data/yey/work/6191/datasets/6191_DK4-110_rot_p10s200/images"
    quant_save_path = "/data/yey/quant.pth"
    param_save_path = "/data/yey/quant_param.h"

    model = QHardNet_small(cfg=cfg)
    model.load_state_dict(torch.load(net_weight)['model_state_dict'])
    model = model.to(device)
    # switch to evaluate mode
    model.eval()

    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((16,16)),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.ImageFolder(valdir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


    
    # Get calibration set.
    image_list = []
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        image_list.append(data)

    print('Calibrating...')
    if os.path.exists(quant_save_path):
        model.model_load_quant_param(quant_save_path,device)
    else:
        model.model_open_calibrate()
        with torch.no_grad():
            for i, image in enumerate(image_list):
                if i == len(image_list) - 1:
                    model.model_open_last_calibrate()
                output = model(image)
        model.model_close_calibrate()
        model.model_save_quant_param(quant_save_path)
    model.model_quant()

    model.get_parameters(param_save_path,begin_flag=True, end_flag=True)

    with torch.no_grad():
        for i, image in enumerate(image_list):
            output = model(image)


if __name__ == '__main__':
    main()
