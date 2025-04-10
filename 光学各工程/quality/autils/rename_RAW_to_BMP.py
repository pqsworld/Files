"""
Author: qi.pan qi.pan@gigadevice.com
Date: 2024-01-04 13:47:40
LastEditors: qi.pan qi.pan@gigadevice.com
LastEditTime: 2024-09-14 18:54:56
FilePath: /vendor/autils/rename_RAW_to_BMP.py
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
LastEditTime: 2024-03-07 22:18:33
LastEditors: qi.pan qi.pan@gigadevice.com
Description:
FilePath: /vendor/autils/rename_RAW_to_BMP.py

Copyright (c) 2005-2022 GigaDevice/Silead Inc.
All rights reserved
The present software is the confidential and proprietary information of GigaDevice/Silead Inc.
You shall not disclose the present software and shall use it only in accordance with the terms of the license agreement you entered into with GigaDevice/Silead Inc.
This software may be subject to export or import laws in certain countries.
"""
import os
import shutil
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="ASP Train Network")
    parser.add_argument(
        "-p",
        "--path",
        # default=r"~/dataset/spoof/amz/product/0922/silead/0922/0922fa")
        default="/ssd/share/6193-spoof/6193dk7",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    path = args.path
    print(path)
    count = 0
    path = Path(path)
    for p in path.rglob("*.raw"):
        print(p)
        p.rename(p.with_suffix(".bmp"))
