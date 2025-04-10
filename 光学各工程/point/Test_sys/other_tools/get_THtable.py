import os
import pandas as pd
from PIL import Image
import shutil
import glob
import torch
import numpy as np
from timeit import default_timer as timer
"""
获得阈值表
"""


def get_THtable(work_dir, model_csv, save_dir):
    #获得模型
    wf = pd.read_csv(model_csv, header=None)
    w_mat_info = wf.to_numpy()
    weight = torch.tensor(w_mat_info[1:, 0].astype(np.float)).type(torch.FloatTensor)
    
    FA_score = []
    FR_score = []
    start_time = timer()
    #循环取csv
    for parent, dirnames, filenames in os.walk(work_dir,  followlinks=True):
        filenames = sorted(filenames)
        for filename in filenames:
            file_path = os.path.join(parent, filename)
            if file_path.endswith(".csv"):
                print('文件完整路径：%s\n' % file_path)
                df = pd.read_csv(file_path, header=None)
                mat_info = df.to_numpy()
                mat_info = mat_info[1:,2:].astype(np.float)
                data_info0 = mat_info[(mat_info[:, 40] == 0) * (mat_info[:, 43] <= 0), :]

                data_all = np.zeros((len(data_info0),609))
                data_all[:,:391] = data_info0[:,:391]
                data_all[:,391:393] = data_info0[:,74:76]
                data_all[:,393] = data_info0[:,74]+data_info0[:,75]
                data_all[:,394] = data_info0[:,78]
                data_all[:,395:404] = data_info0[:,391:400]
                data_all[:,404:413] = data_info0[:,176:185]
                data_all[:,413:] = data_info0[:,400:]

                data = torch.tensor(data_all).type(torch.FloatTensor)
                data_pick = [0,1,2,3,4,5,6,7,9,10,21,22,23,24,25,26,27,29,30,31,32,33,34,51,52,53,54,55,56,57,58,59,66]
                data_pick +=  list(range(83, 176)) + list(range(186, 195)) + list(range(196, 404)) + list(range(413, 609))
                data_539 = data[:,data_pick]
                data_539 = torch.cat([torch.ones(len(data_539)).unsqueeze(1),data_539],dim=1)
                score_info0 = torch.einsum('nk,k->n',data_539,weight)
                score_info0 = torch.sigmoid(score_info0).tolist()

                if "_Far_" in file_path:
                    FA_score += score_info0
                elif "_Frr_" in file_path:
                    FR_score += score_info0
                else:
                    print("fail")
                    exit()

    FA_score = np.array(FA_score)
    FR_score = np.array(FR_score)

    FA_total_num = len(FA_score)
    FR_total_num = len(FR_score)
    content = []
    content.append(["rate_1","rate_2","rate_3","rate_4","rate_5","rate_6","rate_7"])
    content.append(["概率阈值","frr的检出数量","frr的损失数量","frr的检出率%","far的误检数量","far的检出数量","far的误检率%"])

    for score_th in range(0,1001,1):
        score_th /= 1000
        FA_num = np.sum(FA_score >= score_th)
        FR_num = np.sum(FR_score >= score_th)
        FA_res = FA_total_num-FA_num
        FR_res = FR_total_num-FR_num
        content.append([score_th,FR_num,FR_res,FR_res/FR_total_num,FA_num,FA_res,FA_num/FA_total_num*50000])

    df = pd.DataFrame(content)
    df.to_csv(os.path.join(save_dir, 'th_table.csv'),header=None,index=None,encoding='utf_8_sig')
    
    end_time = timer()
    print("Process Consuming: {:.2f}m".format((end_time - start_time)/60))

if __name__ == "__main__":
    # model_csv = '/data/yey/temp/roc/models/table_v9800fix_t1t2wet_len539_patch-Y93054_info0_t20_TH3-55_exe0328_Coef.csv'
    model_csv = '/data/yey/temp/roc/models/table_v9800fix_t1t2wet_len539_patch-Q0425_info0_t20_TH3-55_exe0328_Coef.csv'
    work_dir = "/data/yey/temp/netL"
    save_dir = "/data/yey/temp/roc/THtable"
    get_THtable(work_dir, model_csv, save_dir)
       