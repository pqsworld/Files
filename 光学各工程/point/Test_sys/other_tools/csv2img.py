# from labelme import utils
import pandas as pd
import numpy as np
from PIL import Image
import os
from pathlib import Path
# import base64
# import util
# import shutil
# import csv

if __name__ == "__main__":

    # save_dir = "D:/data/DeepLearning/处理网络预测结果/net_out_points/11_28/result/train"
    save_dir = "/hdd/file-input/qint/data/toQt_Jy/newLabelLog/6159_cd_p11s400/train/pic"  # 读/存 同地址
    
    e = [str(p) for p in Path(save_dir).iterdir()]
    n = [(p.split('/')[-1])[:-4] for p in e]

    for idx, i_csv in enumerate(e):
        print("progress:{0}%".format(round((idx + 1) * 100 / len(e))), end="\r")

        img = pd.read_csv(i_csv, header=None).to_numpy()
        bmp_PIL = Image.fromarray(np.uint8(img))
        bmp_PIL = bmp_PIL.convert('L')

        out_path = Path(save_dir, n[idx] + '.bmp')
        bmp_PIL.save(out_path)