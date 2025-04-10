
import os
import shutil
import argparse
from pathlib import Path
def parse_args():
    parser = argparse.ArgumentParser(description="ASP Train Network")
    parser.add_argument('-p','--path', default=r"/home/panq/dataset/spoof/amz/data-2-0828-rty0/valid-rty0-bin")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    count = 0
    path = Path(path)
    for p in path.rglob("*.fmi"):
        if p.name.find("base")==-1:
            # print(p.parts)
            p.unlink()
            continue
        # str1 = p.stem.split("_quality_")
        # dst = p.parent/(str1[-1]+"_"+str1[0]+".bmp")
        # print(dst)
        # os.rename(p,dst)
