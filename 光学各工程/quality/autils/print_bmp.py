import os
import shutil
import argparse
from pathlib import Path
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="ASP Train Network")
    parser.add_argument(
        "-p",
        "--path",
        # default=r"~/dataset/spoof/amz/product/0922/silead/0922/0922fa")
        default="/home/panq/tool/aosp/vendor/net_forward/test/resource/6195/8317_.bmp",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    path = args.path
    print(path)
    count = 0
    bmp = cv2.imread(path)
    h, w, c = bmp.shape
    for ih in range(h):
        for iw in range(w):
            print(bmp[ih, iw, 0], end=", ")
        print()
