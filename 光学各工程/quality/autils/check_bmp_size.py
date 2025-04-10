import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0,'/hdd/file-input/panq/.local/lib/python3.8/site-packages')

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="ASP Train Network")
    parser.add_argument(
        '-p',
        '--path',
        # default=r"~/dataset/spoof/amz/product/0922/silead/0922/0922fa")
        # default="/ssd/share/6195-spoof/data/1028add")
        default="/hdd/share/quality/optic_quality/RankIQA-master/RankIQA-master/data/train3")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    print(path)
    count = 0
    count_26 = 0
    path = Path(path)
    for p in path.rglob('*.bmp'):
        count+=1
    with tqdm(total=count, position=0, ncols=80) as pbar:
        for p in path.rglob('*.bmp'):

            bmp = cv2.imread(p.as_posix())
            try  :
                h,w,c=bmp.shape
            except:
                print(p._str)

            if w!=180 or h!=200:
            # if  h==119:
                # cropped_bmp=bmp[0:0+118,0:0+32]
                # cv2.imwrite(p.as_posix(),cropped_bmp)
                os.remove(p._str)
                # cou
                # print(p._str)

            pbar.update(1)
