# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:06:29 2023

@author: zhaotiantian
"""

import pandas as pd
import numpy as np
import os

imgall_dir = 'oppo_hepburn_boe_dry'
trans_txt = r"F:\zhangderong\data\smallTTL\datasets_deepDescrp\wxn\wxn_dw\wxn_dw_MatchTrans_Frr000_288.txt"
out_trans_txt = r"F:\zhangderong\data\smallTTL\datasets_deepDescrp\wxn\wxn_dw\trans_choose.txt"
#frr_path = r"F:\ztt1\Log\端叉点\20211201\FirstCh\frr_total.csv"
with open(trans_txt, mode='r') as trans_f:
    with open(out_trans_txt, mode='w', encoding='utf-8') as out_trans_f:
        for line in trans_f:
            if 'Trans:' in line and 'ENROLL' in line and 'verify' in line:
                line = line.replace('\n', '')
                ori_verify_path = line.split('verify:')[1].split(',')[0]
                trans_ori = line.split('Trans:')[-1].split('ENROLL:')[0][:-1]
                enroll_ori = line.split('ENROLL:')[1].split(' verify:')[0]
                enroll = os.path.join(enroll_ori.split("\\")[-3],enroll_ori.split("\\")[-2],enroll_ori.split("\\")[-1])
                verify = int(ori_verify_path.split('\\')[-1].replace('.bmp', ''))
                path = os.path.join(ori_verify_path.split("\\")[-4], ori_verify_path.split("\\")[-3],ori_verify_path.split("\\")[-2],ori_verify_path.split("\\")[-1])
                trans = trans_ori.split(",")[0] + "," + trans_ori.split(",")[1] + "," + trans_ori.split(",")[2] + "," + trans_ori.split(",")[3] + "," + trans_ori.split(",")[4] + "," + trans_ori.split(",")[5]
                message = {
                        'ENROLL':enroll,
                        'verify':verify,
                        'path':path,
                        'Trans':trans,
                        'nMag':trans_ori.split(",")[6],
                        'nPhi':trans_ori.split(",")[7],
                        'nReverse':trans_ori.split(",")[8],
                        'imgQDif':trans_ori.split(",")[9],
                        'nSNRDif':trans_ori.split(",")[10],
                        'nGridSimi':trans_ori.split(",")[11],
                        'nScoreL':trans_ori.split(",")[12],
                        'score=': '0',
                        'up=': '1'
                        }
                message_str = str(message).replace('{','').replace('}', '').replace("'", "")
                message_str = message_str.replace(" Trans: ", "Trans:").replace(" nMag: ", "nMag:").replace(" nPhi: ", "nPhi:").replace(" nReverse: ", "nReverse:")
                message_str = message_str.replace(" imgQDif: ", "nImgQDif:").replace(" nSNRDif: ", "nSNRDif:").replace(" nGridSimi: ", "nGridSimi:").replace(" nScoreL: ", "nScoreL:")
                message_str = message_str.replace(" score=: ", "score=").replace(" up=: ", "up=") + '\n'
                out_trans_f.write(message_str)
    out_trans_f.close()
trans_f.close()
#                print(" ")