from labelme import utils
import numpy as np
from PIL import Image
import os
from pathlib import Path
# import base64
# import util
# import shutil
# import json
import pandas as pd


# detection_file = 'C:/Users/54754/Desktop/network_npy/0_0_cnn.npy'
#
# detections = None
# if detection_file is not None:
#     detections = np.load(detection_file)  # .npy文件
#
# name = ['y', 'x']
#
# test = pd.DataFrame(columns=name, data=detections)
# test.to_csv('C:/Users/54754/Desktop/network_npy/0_0_cnn.csv')
#
# # np.savetxt('C:/Users/54754/Desktop/network_npy/0_0_cnn.csv', detections, fmt='%0.18f')
# #print(detections)
#




if __name__ == "__main__":

    # data_path = '/hdd/file-input/qint/data/toQt_Jy/newLabelLog/2022_4_01_NewData/'
    data_path = '/hdd/file-input/qint/data/toQt_Jy/DescribeToJy_Qt/Descriptor_deletefrr/'
    # save_dir = '/hdd/file-input/qint/data/toQt_Jy/newLabelLog/2022_4_01_NewData/'

    path_list=os.listdir(data_path)
    for filename in path_list:
        if filename == "Frr_Log":
            continue

        f_train = [str(p) for p in Path(data_path, filename, "desc", "val").iterdir() if str(p)[-4:] == '.npy']
        fname = [str(n).split('/')[-1][:-4] for n in f_train]

        count = 0
        print(filename)
        for f, n in zip(f_train, fname):
            print("progress:{0}%".format(round((count + 1) * 100 / len(f_train))), end="\r")
            if os.path.splitext(f)[1] =='.npy':
                f_data = np.load(f).astype(np.int)     # (y, x)
                f_data = f_data[:, [1, 0]]
                f_save = Path(os.path.splitext(f)[0] + ".csv")

                output = pd.DataFrame(data=f_data, columns=None)
                output.to_csv(f_save, index=False, header=None)
                
                count += 1
        