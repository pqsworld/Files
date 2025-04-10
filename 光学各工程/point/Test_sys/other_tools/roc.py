import os
import pandas as pd
from PIL import Image
import shutil
import glob
import torch
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
"""
处理文件内所有csv(必须是 get_THtable得到的阈值表 不做异常文件处理)
"""


def get_ROC(work_dir, save_dir, FA_show_num = 400, check_num = 154):
    start_time = timer()

    content = []
    content_show = []
    #循环取csv
    for parent, dirnames, filenames in os.walk(work_dir,  followlinks=True):
        filenames = sorted(filenames)
        for filename in filenames:
            file_path = os.path.join(parent, filename)
            if file_path.endswith(".csv"):
                print('文件完整路径：%s\n' % file_path)
                df = pd.read_csv(file_path, header=None)
                mat_info = df.to_numpy()
                mat_info = mat_info[2:,:].astype(np.float)
                frr = np.array(sorted(mat_info[:,1]))
                far = np.array(sorted(mat_info[:,4]))
                show_maxindex = np.sum(far < FA_show_num)
                frr = frr[:show_maxindex]
                far = far[:show_maxindex]
                content.append([far,frr,filename])
                
                content_far = []
                content_frr = []
                for off in range(-2,3):
                    check_mask = (far == (check_num + off))
                    if check_mask.sum() > 0:
                        far_show = far[check_mask]
                        frr_show = frr[check_mask]
                        content_far += list(far_show)
                        content_frr += list(frr_show)
                far_check_mean = np.mean(np.array(content_far))
                frr_check_mean = np.mean(np.array(content_frr))
                content_show.append([far_check_mean,frr_check_mean])



    colors = ['r','g','b','c','m','y','k','w']
    show_num = len(content)
    if show_num > len(colors):
        print("待比较文件过多!")
        exit()
    plt.subplot(2,1,1)
    plt.grid(linestyle='--')
    for item_idx, item in enumerate(content):
        plt.plot(item[0],item[1],colors[item_idx],alpha=1,linewidth=1,label=item[2])
    plt.legend()
    plt.title("compare")
    plt.xlabel('FA_log')
    plt.ylabel('FR_log')

    show_name = ['1','2','3','4','5','6','7','8']
    col_table = ['FA_log',"FR_log"]
    row_tabel = show_name[:show_num]

    content_table = np.array(content_show)
    plt.subplot(2,1,2)
    tab_show = plt.table(cellText=content_table,
                        colLabels=col_table,
                        rowLabels=row_tabel,
                        loc='center',
                        cellLoc='center',
                        rowLoc='center',
                        rowColours=colors[:show_num])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc.png'))


if __name__ == "__main__":
    work_dir = "/data/yey/temp/roc/compare"
    save_dir = "/data/yey/temp/roc/results"
    get_ROC(work_dir, save_dir, FA_show_num=400, check_num=100)
       