"""
Author: qi.pan qi.pan@gigadevice.com
Date: 2023-09-08 18:00:09
LastEditors: qi.pan qi.pan@gigadevice.com
LastEditTime: 2024-05-15 17:49:01
FilePath: /vendor/autils/check_bmp_features.py
Description:

Copyright (c) 2005-2022 GigaDevice/Silead Inc.
All rights reserved
The present software is the confidential and proprietary information of GigaDevice/Silead Inc.
You shall not disclose the present software and shall use it only in accordance with the terms of the license agreement you entered into with GigaDevice/Silead Inc.
This software may be subject to export or import laws in certain countries.
"""
"""
Author: qi.pan qi.pan@gigadevice.com
Date: 2023-09-08 18:00:09
LastEditTime: 2024-01-04 14:05:31
LastEditors: qi.pan qi.pan@gigadevice.com
Description:
FilePath: /vendor/autils/check_bmp_features.py

Copyright (c) 2005-2022 GigaDevice/Silead Inc.
All rights reserved
The present software is the confidential and proprietary information of GigaDevice/Silead Inc.
You shall not disclose the present software and shall use it only in accordance with the terms of the license agreement you entered into with GigaDevice/Silead Inc.
This software may be subject to export or import laws in certain countries.
"""
import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="ASP Train Network")
    parser.add_argument(
        "-p",
        "--path",
        # default=r"~/dataset/spoof/amz/product/0922/silead/0922/0922fa")
        default="/hdd/share/quality/optic_quality/datasets/cw/valid/0000/L0",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    path = args.path
    print(path)
    count = 0
    path = Path(path)
    for p in path.rglob("*.bmp"):
        bmp = cv2.imread(p.as_posix())
        h, w, c = bmp.shape
        if h != 118:
            print(p)
            cropped_bmp = bmp[20 :180, 10 : 170].astype(np.float32)

            # cv2.imwrite(p.as_posix(), cropped_bmp)

            ori_tensor = torch.from_numpy(cropped_bmp).permute(2,0,1).unsqueeze(0)
            res_tensor = F.interpolate(ori_tensor,size=(96,96),mode='bilinear',align_corners=False)
            res_array = res_tensor.permute(1,2,0).numpy()
        if w != 26:
            print(p)
