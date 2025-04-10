"""
整理fa以及fr图像的工具
"""

from __future__ import print_function

import glob
import os
import re
import shutil
import random
import numpy as np
from PIL import Image
import cv2
import pandas as pd

FRRFLAG=1
SelRatio=1 #之前是15/5
Txtpre='Sim_Frr'

def get_simi_data_enhance(DPATH):
    for root, dirs, files in os.walk(DPATH):
        for fileName in files:
            # if fileName.endswith('.txt'):
            if fileName.startswith(Txtpre):
                filepath = os.path.join(root, fileName)
                file = open(filepath, mode = 'r')
                # newfilepath  = filepath.replace('Sim_','Sele_Sim_') 
                # filenew = open(newfilepath, mode = 'w') 
                lines = file.readlines()
                # lines = lines[::-1]
                index=0
                # thre=0
                # random.shuffle(lines)
                alllen=len(lines)
                for line in lines:
                    index+=1
                    # print(index)
                    if index%SelRatio==0:   #dk7_2频率15 其他频率5
                    # if 1:
                        # # line = line.replace('/home/zhangsn/simi_net/000_data/olddata/enhance/','/home/jianght/001_data/004_f_data_level16/DK7/')
                        
                        line = line.replace('Z:\\zhangsn\\simi_net\\000_data\\img_train','/home/zhangsn/simi_net/000_data/newdata/enhance')
                        # # line = line.replace('I:\\zhangsn\\6193_dk7','/home/zhangsn/simi_net/000_data/000_alldata/01enhance/DK7/train')
                        
                        # line = line.replace('F:\\zhangsn\\datasets\\6193\\dk4','/home/zhangsn/simi_net/000_data/000_alldata/01enhance/DK4_C3U/train')
                        # # line = line.replace('F:\\zhangsn\\datasets\\6193\\dk7','/home/zhangsn/simi_net/000_data/000_alldata/00ori/DK7/train')
                        line = line.replace('\\','/')
                        # # line = line.replace('/enhance/','/level16/')
                        # # 
                        # line = line.replace('/ori/','/enhance/')
                        #Z:\\zhangsn\\simi_net\\000_data\\img_train\\
                        temp = line.split(',')[0].split(':')[-1]
                        tempmask = temp.replace('/enhance/','/mask/')
                        # tempmask = tempmask.replace('01enhance','00ori')
                        # temp = temp.replace('.bmp','_LowLevel.bmp')
                        
                        print(index+1,'/',alllen,'********',temp)
                        samp = line.split(',')[1].split(':')[-1]   #left
                        sampmask = samp.replace('/enhance/','/mask/')
                        # sampmask = sampmask.replace('01enhance','00ori')
                        # samp = samp.replace('.bmp','_LowLevel.bmp')
                        # # frr
                        if FRRFLAG:
                            tempid = temp.split('/')[-1].split('.')[0]
                            sampid = samp.split('/')[-1].split('.')[0]
                        
                        #far
                        else:
                            tempid = temp.split('/')[-2]+temp.split('/')[-1].split('.')[0]
                            sampid = samp.split('/')[-2]+samp.split('/')[-1].split('.')[0]
                            
                        trans = line.split(',')[2].split(':')[-1]
                        transtmp=trans.split()
                        transtmp=np.float32(np.array(transtmp).reshape(2,3))
                        tt=np.array([0,0,1])
                        label_H=np.r_[transtmp,[tt]]

                        recogres = str(int(line.split(',')[3].split(':')[-1]))
                        Gridsim = str(int(line.split(',')[4].split(':')[-1]))
                        nScoreL = str(int(line.split(',')[5].split(':')[-1]))
                        nScoreH = str(int(line.split(',')[6].split(':')[-1]))
                        if int(recogres) == 1:
                            # newscore=Gridsim
                            if int(Gridsim)==127:
                                newscore=Gridsim
                            else:
                                newscore = str(max(int(Gridsim),int(nScoreL),int(nScoreH)))
                            
                            #连续按压数据直接给的0，这个label也用不到
                            newscore='000'
                            
                            savettt=temp.replace('/enhance/','/03transpair_enhance_label_0411/r0/')
                            na=savettt.split('/')[:]
                            save_data_path = na[0]
                            for i_na in range(1, len(na) - 1):
                                save_data_path = save_data_path + "/" + na[i_na]
                                if not os.path.exists(save_data_path):
                                    os.mkdir(save_data_path)
                            leftimg= Image.open(samp).convert('L')
                            rightimg=Image.open(temp).convert('L')
                            w,h=leftimg.size

                            leftimgmask= np.array(Image.open(sampmask).convert('L').resize((w,h)))
                            rightimgmask=np.array(Image.open(tempmask).convert('L').resize((w,h)))
                            leftimgmask = cv2.warpAffine(leftimgmask, label_H[:2], (w,h), flags=cv2.INTER_LINEAR, borderValue=0)  #INTER_CUBIC
                            
                            # 
                            saveimgpath=save_data_path+'/'+newscore+'_'+str(sampid)+'_'+str(tempid)+'_r'+recogres+'_g'+Gridsim+'_sl'+nScoreL+'_sh'+nScoreH+'.bmp'
                            saveimgpath = saveimgpath.replace('_LowLevel','')                

                            # print(sampid)
                            # filenew.write(line)
                            # print(line)
                            # newtranslist.append(line)
                            finger1=np.array(leftimg)
                            finger2=np.array(rightimg)
                            finger1m = cv2.warpAffine(finger1,label_H[:2],(w,h),flags=cv2.INTER_LINEAR, borderValue=0)
                            
                            
                            
                            # 融合相加,如果相加大于255，处理
                            ones = np.ones_like(finger1)
                            ones_warp = cv2.warpAffine(ones, label_H[:2], (w,h), flags=cv2.INTER_LINEAR, borderValue=0)  #INTER_CUBIC
                            ones_warp = ones_warp*255
                            ones_warp = np.where(ones_warp<128,0,255)
                            ones_warp = np.where(leftimgmask<10,0,ones_warp)
                            ones_warp = np.where(rightimgmask<10,0,ones_warp)
                            # 合并图像原本mask
                            finger1m = np.where(ones_warp==0,0,finger1m)
                            finger2 = np.where(ones_warp==0,0,finger2)
                        
                            zeros = np.zeros_like(finger1)   
                            dst = cv2.merge([zeros, finger1m, finger2])
                            
                            maskimg =np.repeat(ones_warp[..., np.newaxis], 3, -1)
                            leftimg = np.repeat(finger1[..., np.newaxis], 3, -1)
                            rightimg = np.repeat(finger2[..., np.newaxis], 3, -1)
                            warpimg =np.repeat(finger1m[..., np.newaxis], 3, -1)
                            
                            # img = np.hstack((leftimg,rightimg,maskimg,dst))
                            img = dst
                            # img = np.hstack((leftimg,rightimg,dst))
                            cv2.imwrite(saveimgpath, img)  
                            # np.save(saveimgpath[:-4]+'.npy',label_H)    
                            # thre = thre+int(Gridsim)           
                            # index+=1
                    # print(0)
                # filenew.close() 
                file.close()
                # print(filepath,index)
            
                                                   
if __name__ == '__main__':   
    # transpath ="/home/zhangsn/simi_net/000_data/000_alldata/02simitxtfar/simi_txt_dk7/6193-DK7-140-8-wet2_clear_test/"
    # transpath ="/home/zhangsn/simi_net/000_data/000_alldata/02simitxtfar/simi_txt_dk7/"
    transpath ="/home/zhangsn/simi_net/000_data/newdata/simi_res_1023"
    get_simi_data_enhance(transpath)