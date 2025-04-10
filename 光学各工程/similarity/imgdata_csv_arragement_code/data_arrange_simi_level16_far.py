import os 
import numpy as np
from PIL import Image
import cv2

from pathlib import Path
all_img=[str(f) for f in Path('root').rglob('*.bmp')]

def get_simi_data_level16(DPATH):
    for root, dirs, files in os.walk(DPATH):
        for fileName in files:
            if fileName.endswith('.txt'):
                filepath = os.path.join(root, fileName)
                file = open(filepath, mode = 'r')
                # newfilepath  = filepath.replace('.txt','_sele.txt') 
                # filenew = open(newfilepath, mode = 'w') 
                lines = file.readlines()
                # lines = lines[::-1]
                index=0
                # thre=0
                # random.shuffle(lines)
                for line in lines:
                    # if index%20==0:
                    # line = line.replace('F:\\zhangsn\\datasets\\6193\\traindata\\','/home/jianght/001_data/004_f_data_level16/DK7/')
                    temp = line.split(',')[0].split(':')[-1]
                    # print(temp)
                    samp = line.split(',')[1].split(':')[-1]   #left
                    
                    samp = samp.replace('.bmp','_LowLevel.bmp')
                    temp = temp.replace('.bmp','_LowLevel.bmp')
                    
                    tempid = temp.split('/')[-2]+temp.split('/')[-1].split('.')[0]
                    sampid = samp.split('/')[-2]+samp.split('/')[-1].split('.')[0]
                    trans = line.split(',')[2].split(':')[-1]
                    transtmp=trans.split()
                    transtmp=np.float32(np.array(transtmp).reshape(2,3))
                    tt=np.array([0,0,1])
                    label_H=np.r_[transtmp,[tt]]
                    '''变换图像,temple=>sample'''
                    Hs, Ws = 61,18
                    Homography_resize = np.zeros((3,3))
                    Homography_resize[0,0] = Ws / 36
                    Homography_resize[1,1] = Hs / 122
                    Homography_resize[2,2] = 1
                    Homography_level_16 = Homography_resize @ label_H @ np.linalg.inv(Homography_resize)
                    label_H = Homography_level_16
                    
                    recogres = str(int(line.split(',')[3].split(':')[-1]))
                    Gridsim = str(int(line.split(',')[4].split(':')[-1]))
                    nScoreL = str(int(line.split(',')[5].split(':')[-1]))
                    nScoreH = str(int(line.split(',')[6].split(':')[-1]))

                    # newscore=Gridsim
                    
                    savettt=temp.replace('/home/jianght/001_data/004_f_data_level16/DK7/','/home/zhangsn/simi_net/000_data/testdata_level16_far/')
                    na=savettt.split('/')[:]
                    save_data_path = na[0]
                    for i_na in range(1, len(na) - 1):
                        save_data_path = save_data_path + "/" + na[i_na]
                        if not os.path.exists(save_data_path):
                            os.mkdir(save_data_path)
                    leftimg= Image.open(samp).convert('L')
                    rightimg=Image.open(temp).convert('L')
                    w,h=leftimg.size
                
                    saveimgpath=save_data_path+'/'+str(sampid)+'_'+str(tempid)+'_r'+recogres+'_g'+Gridsim+'_sl'+nScoreL+'_sh'+nScoreH+'.bmp'
                    saveimgpath = saveimgpath.replace('_LowLevel','')                 

                    # print(sampid)
                    # filenew.write(line)
                    print(line)
                    # newtranslist.append(line)
                    finger1=np.array(leftimg)
                    finger2=np.array(rightimg)
                    finger1m = cv2.warpAffine(finger1,label_H[:2],(w,h),flags=cv2.INTER_CUBIC, borderValue=0)
                    zeros = np.zeros_like(finger1)   
                    dst = cv2.merge([zeros, finger1m, finger2])
                    leftimg = np.repeat(finger1[..., np.newaxis], 3, -1)
                    rightimg = np.repeat(finger2[..., np.newaxis], 3, -1)
                    warpimg =np.repeat(finger1m[..., np.newaxis], 3, -1)
                    
                    # 融合相加,如果相加大于255，处理
                    ones = np.ones_like(finger1)
                    ones_warp = cv2.warpAffine(ones, label_H[:2], (w,h), flags=cv2.INTER_CUBIC, borderValue=0)
                    ones_warp = ones_warp*255
                    ones_warp = np.where(ones_warp<128,0,255)
                    maskimg =np.repeat(ones_warp[..., np.newaxis], 3, -1)

                    img = np.hstack((leftimg,rightimg,maskimg,dst))
                    # img = np.hstack((leftimg,rightimg,dst))
                    cv2.imwrite(saveimgpath, img)  
                    np.save(saveimgpath[:-4]+'.npy',label_H)    
                    # thre = thre+int(Gridsim)           
                    index+=1
                # print(0)
                # filenew.close()    
                file.close()



if __name__ == '__main__':
    transpath ="/home/zhangsn/simi_net/000_data/simi_trans_far/00/"
    get_simi_data_level16(transpath)    