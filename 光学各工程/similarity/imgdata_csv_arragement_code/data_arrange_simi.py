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

OLDDATA=0
def get_simi_data_cuxi(DPATH):
    for root, dirs, files in os.walk(DPATH):
        for fileName in files:
            if fileName.endswith('.txt'):
                # print(dirs)
                filepath = os.path.join(root, fileName)
                file = open(filepath, mode = 'r')
                lines = file.readlines()
                newfilepath  = filepath.replace('.txt','_new.txt') 
                filenew = open(newfilepath, mode = 'w') 
                # lines = lines[::-1]
                index=0
                # thre=0
                # random.shuffle(lines)
                for line in lines:
                    # line = line.replace('F:\\zhangsn\\datasets\\6193\\traindata\\','/home/zhangsn/simi_net/000_data/olddata/enhance/')
                    # 
                    line = line.replace('Z:\\zhangsn\\simi_net\\000_data\\img_train\\','/home/zhangsn/simi_net/000_data/newdata/enhance/')
                    temp = line.split(',')[0].split(':')[-1]
                    # print(temp)
                    samp = line.split(',')[1].split(':')[-1]   #left
                    
                    tempid = temp.split('/')[-1].split('.')[0]
                    sampid = samp.split('/')[-1].split('.')[0]
                    trans = line.split(',')[2].split(':')[-1]
                    transtmp=trans.split()
                    transtmp=np.float32(np.array(transtmp).reshape(2,3))
                    tt=np.array([0,0,1])
                    label_H=np.r_[transtmp,[tt]]
                    
                    recogres = str(int(line.split(',')[3].split(':')[-1]))
                    Gridsim = str(int(line.split(',')[4].split(':')[-1]))
                    nScoreL = str(int(line.split(',')[5].split(':')[-1]))
                    nScoreH = str(int(line.split(',')[6].split(':')[-1]))
                    if not OLDDATA:
                        # if int(Gridsim)<200 or int(recogres)==0 :
                        if int(Gridsim)>=200 and int(recogres)==1:
                            filenew.write(line)
                            savettt=temp.replace('/enhance/','/testdata_r1/')
                            na=savettt.split('/')[:]
                            save_data_path = na[0]
                            for i_na in range(1, len(na) - 1):
                                save_data_path = save_data_path + "/" + na[i_na]
                                if not os.path.exists(save_data_path):
                                    os.mkdir(save_data_path)
                            leftimg= Image.open(samp).convert('L')
                            rightimg=Image.open(temp).convert('L')
                            w,h=leftimg.size

                            newscore = str(max(int(Gridsim),int(nScoreL),int(nScoreH)))
                            saveimgpath=save_data_path+'/'+newscore+'_'+str(sampid)+'_'+str(tempid)+'_r'+recogres+'_g'+Gridsim+'_sl'+nScoreL+'_sh'+nScoreH+'.bmp'
                                            
                            # img    
                            finger1=np.array(leftimg)
                            finger2=np.array(rightimg)
                            finger1m = cv2.warpAffine(finger1,label_H[:2],(w,h),flags=cv2.INTER_CUBIC, borderValue=0)
                            zeros = np.zeros_like(finger1)   
                            dst = cv2.merge([zeros, finger1m, finger2])
                            leftimg = np.repeat(finger1[..., np.newaxis], 3, -1)
                            rightimg = np.repeat(finger2[..., np.newaxis], 3, -1)
                            # warpimg =np.repeat(finger1m[..., np.newaxis], 3, -1)
                            # img = np.hstack((leftimg,rightimg,warpimg,dst))
                            
                            # 融合相加,如果相加大于255，处理
                            ones = np.ones_like(finger1)
                            ones_warp = cv2.warpAffine(ones, label_H[:2], (w,h), flags=cv2.INTER_CUBIC, borderValue=0)
                            ones_warp = ones_warp*255
                            ones_warp = np.where(ones_warp<128,0,255)
                            maskimg =np.repeat(ones_warp[..., np.newaxis], 3, -1)
                        
                            img = np.hstack((leftimg,rightimg,maskimg,dst))
                            cv2.imwrite(saveimgpath, img)                     
                        index+=1
                    
                    if OLDDATA:
                        # goodnum=0
                        if int(sampid)<20:
                            savettt=temp.replace('/enhance/','/traindata_use/')
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
                                            
                            # img   
                            # if int(sampid)%4==0: 
                                # print(sampid)
                            filenew.write(line)
                            finger1=np.array(leftimg)
                            finger2=np.array(rightimg)
                            finger1m = cv2.warpAffine(finger1,label_H[:2],(w,h),flags=cv2.INTER_CUBIC, borderValue=0)
                            zeros = np.zeros_like(finger1)   
                            dst = cv2.merge([zeros, finger1m, finger2])
                            leftimg = np.repeat(finger1[..., np.newaxis], 3, -1)
                            rightimg = np.repeat(finger2[..., np.newaxis], 3, -1)
                            warpimg =np.repeat(finger1m[..., np.newaxis], 3, -1)
                            img = np.hstack((leftimg,rightimg,warpimg,dst))
                            # img = np.hstack((leftimg,rightimg,dst))
                            # cv2.imwrite(saveimgpath, img)      
                            # thre = thre+int(Gridsim)
                        # if int(sampid)==19:
                        #     thre=int(thre*0.1)
                        if int(sampid)>=20:                   
                            if int(sampid)%6==0:
                                # print(thre)
                                # if thre-int(Gridsim)<10 or int(recogres)==0 :   #结果和前10个均值差大于10或者失败
                                #     newgsimi=str(thre+random.randint(-5,4))
                                # else:
                                    # newgsimi=Gridsim
                                filenew.write(line)
                                savettt=temp.replace('/enhance/','/traindata_use/')
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
                            
                                # img  
                                # print(leftimg.size)  
                                finger1=np.array(leftimg)
                                # print(finger1.shape)
                                finger2=np.array(rightimg)
                                finger1m = cv2.warpAffine(finger1,label_H[:2],(w,h),flags=cv2.INTER_CUBIC, borderValue=0)
                                zeros = np.zeros_like(finger1)   
                                dst = cv2.merge([zeros, finger1m, finger2])
                                leftimg = np.repeat(finger1[..., np.newaxis], 3, -1)
                                # print(leftimg.shape)
                                # print(sampid)
                                rightimg = np.repeat(finger2[..., np.newaxis], 3, -1)
                                warpimg =np.repeat(finger1m[..., np.newaxis], 3, -1)
                                img = np.hstack((leftimg,rightimg,warpimg,dst))
                                # cv2.imwrite(saveimgpath, img)                     
                        index+=1
                filenew.close()    
                file.close()

def get_simi_data(DPATH):
    for root, dirs, files in os.walk(DPATH):
        for fileName in files:
            # print(dirs)
            if fileName.endswith('.txt'):
                newtranslist=[]
                filepath = os.path.join(root, fileName)
                file = open(filepath, mode = 'r')
                # trans写入新的txt
                # newfilepath  = filepath.replace('.txt','_new.txt') 
                # filenew = open(newfilepath, mode = 'w') 
                lines = file.readlines()
                # lines = lines[::-1]
                index=0
                # thre=0
                # random.shuffle(lines)
                for line in lines:
                    # line = line.replace('F:\\zhangsn\\datasets\\6193\\traindata\\','/home/zhangsn/simi_net/000_data/olddata/enhance/')
                    line = line.replace('/home/zhangsn/simi_net/000_data/olddata/enhance/','/home/zhangsn/simi_net/000_data/olddata/enhance_nm530/')
                    #Z:\\zhangsn\\simi_net\\000_data\\img_train\\
                    temp = line.split(',')[0].split(':')[-1]
                    # print(temp)
                    samp = line.split(',')[1].split(':')[-1]   #left
                    
                    tempid = temp.split('/')[-1].split('.')[0]
                    sampid = samp.split('/')[-1].split('.')[0]
                    trans = line.split(',')[2].split(':')[-1]
                    transtmp=trans.split()
                    transtmp=np.float32(np.array(transtmp).reshape(2,3))
                    tt=np.array([0,0,1])
                    label_H=np.r_[transtmp,[tt]]
                    
                    recogres = str(int(line.split(',')[3].split(':')[-1]))
                    Gridsim = str(int(line.split(',')[4].split(':')[-1]))
                    nScoreL = str(int(line.split(',')[5].split(':')[-1]))
                    nScoreH = str(int(line.split(',')[6].split(':')[-1]))
                    # newscore =str(int((int(nScoreL)+int(nScoreH))*0.5))
                    if int(Gridsim)==127:
                        newscore=Gridsim
                    else:
                        newscore = str(max(int(Gridsim),int(nScoreL),int(nScoreH)))
                    
                    savettt=temp.replace('/olddata/enhance_nm530/','/traindata_use_1116/')
                    na=savettt.split('/')[:]
                    save_data_path = na[0]
                    for i_na in range(1, len(na) - 1):
                        save_data_path = save_data_path + "/" + na[i_na]
                        if not os.path.exists(save_data_path):
                            os.mkdir(save_data_path)
                    leftimg= Image.open(samp).convert('L')
                    rightimg=Image.open(temp).convert('L')
                    w,h=leftimg.size
                
                    saveimgpath=save_data_path+'/'+newscore+'_'+str(sampid)+'_'+str(tempid)+'_r'+recogres+'_g'+Gridsim+'_sl'+nScoreL+'_sh'+nScoreH+'.bmp'
                                    
                    # img   
                    if 1: #random.randint(0,3)==0: 
                        # print(sampid)
                        # filenew.write(line)
                        print(line)
                        # newtranslist.append(line)
                        finger1=np.array(leftimg)
                        finger2=np.array(rightimg)
                        finger1m = cv2.warpAffine(finger1,label_H[:2],(w,h),flags=cv2.INTER_CUBIC, borderValue=0)
                        zeros = np.zeros_like(finger1)   
                        
                        # 融合相加,如果相加大于255，处理
                        ones = np.ones_like(finger1)
                        ones_warp = cv2.warpAffine(ones, label_H[:2], (w,h), flags=cv2.INTER_CUBIC, borderValue=0)
                        ones_warp = ones_warp*255
                        ones_warp = np.where(ones_warp<128,0,255)
                        maskimg =np.repeat(ones_warp[..., np.newaxis], 3, -1)
                        
                        dst = cv2.merge([zeros, finger1m, finger2])
                        leftimg = np.repeat(finger1[..., np.newaxis], 3, -1)
                        rightimg = np.repeat(finger2[..., np.newaxis], 3, -1)
                        warpimg =np.repeat(finger1m[..., np.newaxis], 3, -1)
                        img = np.hstack((leftimg,rightimg,maskimg,dst))
                        # img = np.hstack((leftimg,rightimg,dst))
                        cv2.imwrite(saveimgpath, img)      
                        # thre = thre+int(Gridsim)           
                    index+=1
                # print(0)
                # filenew.close()    
                file.close()

            
def new_gridscore(DPATH):
    # print(DPATH)
    # file=sorted(os.listdir(root))
    for root, dirs, files in os.walk(DPATH):   
        if root.split('/')[-1] in ['L1','L2','L3','R1','R2','R3']:
            print(root)
            
            gridsum= 0
            allfiles=sorted(files)
            num=min(10,len(allfiles))
            nn=0
            for i in range(0,num):
                filepath = allfiles[i]
                # print(filepath)
                Gridsim = int((filepath.split('_')[3])[1:4])
                gridsum=gridsum+Gridsim
                nn=nn+1
            thregrid=int(gridsum/nn)
            print(thregrid)
            for file in allfiles:
                print(file)
                imgpathold=os.path.join(root,file)
                samp=file.split('_')[0]
                gridsim=(file.split('_')[3])
                if int(samp)>=20 and thregrid-int(gridsim[1:4])>10:                   
                # if thregrid-int(gridsim[1:4])>10:
                    newgridsim=str(thregrid+random.randint(-6,4))
                    # 'g'+
                    # print(gridsim)
                    # savefile = newgridsim+'_'+file.replace(gridsim,'g'+newgridsim)
                else:
                    # savefile=imgpathold
                    newgridsim=gridsim[1:4]
                savefile = os.path.join(root,newgridsim+'_'+file)
                savefile = savefile.replace('traindata_use','traindata_use_my')
                na=savefile.split('/')[:]
                save_data_path = na[0]
                for i_na in range(1, len(na) - 1):
                    save_data_path = save_data_path + "/" + na[i_na]
                    if not os.path.exists(save_data_path):
                        os.mkdir(save_data_path)
                shutil.copy(imgpathold,savefile)

def biaozhu_gridscore(DPATH):
    for root, dirs, files in os.walk(DPATH):   
        if root.split('/')[-1] in ['L1','L2','L3','R1','R2','R3']:
            print(root) 
            gridsum= 0
            allfiles=sorted(files)
            for file in allfiles:
                print(file)
                imgpathold=os.path.join(root,file)
                firstflag=file.split('_')[0]
                secondflag=file.split('_')[1]
                
                #标注以后两个相似度值 取新的
                if len(firstflag)==3:
                    print(imgpathold)
                    savefile = imgpathold.replace(firstflag+'_'+secondflag,secondflag)
                    os.rename(imgpathold,savefile)
                
                # # 把g255的网格相似度分数添加到文件名前面    
                # if len(firstflag)==4:
                #     gridsim=(file.split('_')[3])
                #     newgridsim=gridsim[1:4]
                #     savefile = os.path.join(root,newgridsim+'_'+file)
                #     os.rename(imgpathold,savefile)
                
                # 网格相似度为127的挪出来人工检查
                # gridsim=(file.split('_')[4])[1:4]
                # if(int(gridsim)==127):
                #     savefile=imgpathold.replace('change','change00')
                #     na=savefile.split('/')[:]
                #     save_data_path = na[0]
                #     for i_na in range(1, len(na) - 1):
                #         save_data_path = save_data_path + "/" + na[i_na]
                #         if not os.path.exists(save_data_path):
                #             os.mkdir(save_data_path)
                #     shutil.move(imgpathold,savefile)
                    

def re_merge(DPATH):
    for root, dirs, files in os.walk(DPATH):
        for file in sorted(files):
            if file.endswith('.bmp'):
                filepath = os.path.join(root,file)
                img = cv2.imread(filepath)  #122 108 3
                samp = img[:,0:36,:]
                temp = img[:,36:72,:]
                merge = img[:,108:144,:]
                # print(samp.shape,merge.shape,temp.shape)
                # samp_warp=merge[:,:,1]
                # warp=np.repeat(samp_warp[..., np.newaxis], 3, -1)
                img = np.hstack((samp,temp,merge))
                # img = np.hstack((leftimg,rightimg,dst))
                cv2.imwrite(filepath, img)
                

def csv_rename_transpair_my(DPATH):
    for root, dirs, files in os.walk(DPATH):
        for file in sorted(files):
            if file.endswith('1stversion.csv'):
            # if file.endswith('2ndversion_ht.csv'):
                print(file)
                filepath = os.path.join(root,file)
                df = pd.read_csv(
                        filepath,
                        header=0,
                        # encoding = "gb2312",
                        names=['root','name','simi_label',]
                        # names=['root','name','simi_label2','ham','ssim',]
                        )
                for i in range(len(df)):
                    # if len(df['name'][i]) != 33:
                    #     name = os.path.join(df['root'][i],df['name'][i])
                    #     print(name,'wrong',file)
                    name = os.path.join(df['root'][i],df['name'][i])
                    label=df['simi_label'][i]
                    newname=os.path.join(df['root'][i],str(label)+'_'+df['name'][i])
                    # os.rename(name,newname)
                    os.rename(name[:-4]+'.npy',newname[:-4]+'.npy')

def csv_rename_transpair(DPATH):
    for root, dirs, files in os.walk(DPATH):
        for file in sorted(files):
            if file.endswith('1stversion.csv'):
            # if file.endswith('2ndversion_ht.csv'):
            # if file.endswith('.csv'):
                print(file)
                filepath = os.path.join(root,file)
                df = pd.read_csv(
                        filepath,
                        header=0,
                        # encoding = "gb2312",
                        names=['root','name','simi_label',]
                        # names=['root','name','simi_label1','simi_label2','ham','ssim','gridscore']
                        )
                for i in range(len(df)):
                    # if len(df['name'][i]) != 33:
                    #     name = os.path.join(df['root'][i],df['name'][i])
                    #     print(name,'wrong',file)
                    tmp_root = df['root'][i]
                    tmp_name = df['name'][i]
                    label = str(df['simi_label'][i])
                    # label2 = str(df['simi_label2'][i])
                    # label1 = str(df['simi_label1'][i])
                    # ham=df['ham'][i]
                    # ssim=df['ssim'][i]
                    # ssim='ssim'+str(int(label2)-int(ham))
                    newroot = tmp_root.replace('traindata_use_my/','traindata_level16_1116/')              
                    newroot = newroot.replace('/new-lianxu-fail-review/','/new-lianxu-succes/fail_r0_200/')
                    for files0 in os.listdir(newroot):
                        if files0.endswith(tmp_name):
                        
                            # if len(files0)==33:
                            #     newname = label2+'_ham_'+ham+'_ssim_'+ssim+'_000_'+files0
                            # else:
                            #     newname = label2+'_ham_'+ham+'_ssim_'+ssim+'_'+files0
                            # newname = label2+'_ham'+str(ham)+'_ssim'+str(ssim)+'_'+str(label1)+'_'+tmp_name
                            if len(files0)==33:
                                newname = label+'_'+tmp_name
                                print(files0)         
                                print(newname)
                                oldpath=os.path.join(newroot,files0)
                                newpath=os.path.join(newroot,newname)
                                os.rename(oldpath,newpath)
                                os.rename(oldpath[:-4]+'.npy',newpath[:-4]+'.npy')
                    # name = os.path.join(df['root'][i],df['name'][i])
                    # label=df['simi_label'][i]
                    # newname=os.path.join(df['root'][i],str(label)+'_'+df['name'][i])
                    # # os.rename(name,newname)
                    # os.rename(name[:-4]+'.npy',newname[:-4]+'.npy')
                    
                    # newroot = tmp_root.replace('/home/jianght/001_data/003_all_label_img/test_lianxu-r1data-enhance_newlabel','/home/zhangsn/simi_net/000_data/test_lianxu-r1data/train') 
                    # newname = os.path.join(newroot,tmp_name)
                    # oldname = os.path.join(newroot,label2+'_ham_'+ham+'_ssim_'+ssim+'_'+tmp_name)
                    # os.rename(oldname,newname)
                    # os.rename(oldname[:-4]+'.npy',newname[:-4]+'.npy')
                    
                    
def createcsv(path):
    namelist=[]
    labellist=[]
    labellist2=[]
    hamlist=[]
    hamnormlist=[]
    ssimnormlist=[]
    labelnewlist=[]
    ssimlist=[]
    rootlist=[]
    gridscorelist=[]
    index = 0
    real_good_pair=0
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            # 
            if file.endswith('.bmp'):
                print(index,file) 
                # name = file[4:]
                # label = file[0:3]
                filepath = os.path.join(root,file)
                
                name = file[-33:]
                # name = file.split('_ori')[0]
                # name = name[-29:]+'.bmp'
                label = file[0:3]
                ttt=file.split('_')
                label2=int(ttt[0])
                ham=int((ttt[1])[3:])
                ssim=int((ttt[2])[4:])
                label1=ttt[3]
                gridscore = int((ttt[7])[1:])
                root00 = root
                # if label2>280:
                #     simi_new = (label2-280)*0.4+280
                # else:
                #     simi_new = label2
                
                # ham = ham-128
                # if ham>64:
                #     ham_norm = (ham-60)*0.1+60
                # else:
                #     ham_norm = ham
                # ssim_norm = ssim*0.64
                # simi_new = ham_norm+ssim_norm+127
                # if ham>150:
                #     ham_norm = (150+(ham-150)*0.5)   #大于190的压缩一下范围，然后求和，压缩到200以内，然后直接norm
                # else:
                #     ham_norm = ham
                # ssim_norm = ssim #*0.01*64   #理论值100
                # simi_new=ham_norm+ssim_norm #+127
        
                img = Image.open(filepath).convert('L')
                w ,h= img.size
                mask = np.array(img.crop([w/4*2,0,w/4*3,h]))
                black = (len(np.where(mask==0)[1]))/(w*h)
                
                if int(label2) > 300 and int(ham) >75 and black < 0.02:
                    real_good_pair+=1
                namelist.append(name)
                labellist.append(label1)
                labellist2.append(label2)
                hamlist.append(ham)
                ssimlist.append(ssim)
                # hamnormlist.append(ham_norm)
                # ssimnormlist.append(ssim_norm)
                # labelnewlist.append(simi_new)
                rootlist.append(root00)
                gridscorelist.append(gridscore)
                index+=1
    data={'root':rootlist,'name':namelist,'simi_label1':labellist,'simi_label2':labellist2,'ham':hamlist,'ssim':ssimlist,'gridscore':gridscorelist}  #'siminorm':ssimnormlist,'hamnorm':hamnormlist,'simi_labelnew':labelnewlist
    df=pd.DataFrame(data)
    # df.to_csv(path+'/out_label_1stversion'+'.csv',index=False)
    df.to_csv('out_label_new_ht'+'.csv',index=False)
    # print('ham',max(hamlist),min(hamlist))
    # print('ssim',max(ssimlist),min(ssimlist))
    # print('best matching pairs:',real_good_pair,'all matching pairs:',index)

def createcsv_test(path):
    namelist=[]
    labellist=[]
    reslist=[]
    hamlist=[]
    ssimlist=[]
    rootlist=[]
    gridscorelist=[]
    index = 0
    real_good_pair=0
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            # 
            if file.endswith('.bmp'):
                print(index,file) 
                # name = file[4:]
                # label = file[0:3]
                filepath = os.path.join(root,file)
                
                name = file[-33:]
                # name = file.split('_ori')[0]
                # name = name[-29:]+'.bmp'
                # label = file[0:3]
                ttt=file.split('_')
                res=ttt[0]
                label=ttt[1]
                ham=ttt[3]
                ssim=ttt[4]
                gridscore=int((ttt[9])[1:])
                root00 = root
               
                rootlist.append(root00) 
                namelist.append(name)
                reslist.append(res)
                labellist.append(label)
                hamlist.append(int(ham[3:]))
                ssimlist.append(int(ssim[4:]))
                gridscorelist.append(gridscore)
                index+=1
    data={'root':rootlist,'name':namelist,'res':reslist,'label':labellist,'ham':hamlist,'ssim':ssimlist,'gridscore':gridscorelist}  #
    df=pd.DataFrame(data)
    df.to_csv(path+'/out_test_simi132'+'.csv',index=False)
    # df.to_csv('out_test'+'.csv',index=False)
    print('ham',max(hamlist),min(hamlist))
    print('ssim',max(ssimlist),min(ssimlist))
    
def randomsele(path):
    index = 0
    real_good_pair=0
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            # 
            if file.endswith('.bmp'):
                print(index,file)
                filepath = os.path.join(root,file)
                savepath = filepath.replace('/train/','/trainsele/')
                name = file[-33:]
                label = file[0:3]
                ttt=file.split('_')
                label2=ttt[0]
                ham=ttt[1]
                ssim=ttt[2]
                label1=ttt[3]
                gridscore = int((ttt[7])[1:])
                root00 = root
                
                img = Image.open(filepath).convert('L')
                w ,h= img.size
                mask = np.array(img.crop([w/4*2,0,w/4*3,h]))
                black = (len(np.where(mask==0)[1]))/(w*h)
                
                if int(label2) > 300 and int(ham[3:]) >200 and black < 0.02:
                    real_good_pair+=1
                    per = random.randint(0,100)
                    if per > 85:
                        na=savepath.split('/')[:]
                        save_data_path = na[0]
                        for i_na in range(1, len(na) - 1):
                            save_data_path = save_data_path + "/" + na[i_na]
                            if not os.path.exists(save_data_path):
                                os.mkdir(save_data_path)
                        shutil.move(filepath,savepath)
                        shutil.move(filepath[:-4]+'.npy',savepath[:-4]+'.npy')
                else:
                    per = random.randint(0,100)
                    if per > 70:
                        na=savepath.split('/')[:]
                        save_data_path = na[0]
                        for i_na in range(1, len(na) - 1):
                            save_data_path = save_data_path + "/" + na[i_na]
                            if not os.path.exists(save_data_path):
                                os.mkdir(save_data_path)
                        shutil.move(filepath,savepath)
                        shutil.move(filepath[:-4]+'.npy',savepath[:-4]+'.npy')

                index+=1

    print('best matching pairs:',real_good_pair,'all matching pairs:',index)

def checkbmplen(DPATH):
    i=0
    for root, dirs, files in os.walk(DPATH):
        for fileName in files:
            if fileName.endswith('.bmp'):
                if len(fileName)!=37:
                    print(root,fileName)
                    newfilename = '000_'+fileName
                    oldpath=os.path.join(root,fileName)
                    newpath=os.path.join(root,newfilename)
                    # os.rename(oldpath,newpath)
                    # os.rename(oldpath[:-4]+'.npy',newpath[:-4]+'.npy')
                    i+=1
                    # na=newpath.split('/')[:]
                    # save_data_path = na[0]
                    # for i_na in range(1, len(na) - 1):
                    #     save_data_path = save_data_path + "/" + na[i_na]
                    #     if not os.path.exists(save_data_path):
                    #         os.mkdir(save_data_path)
                    # shutil.move(oldpath,newpath)
                    # shutil.move(oldpath[:-4]+'.npy',newpath[:-4]+'.npy')
    print(i)
                                                      
if __name__ == '__main__':
    # transpath ="/home/zhangsn/simi_net/000_data/traindata_use_my/train/normal-pair-sele/"
    # get_simi_data(transpath)
    # # get_simi_data_cuxi(transpath)
    
    # # transpath ="/home/zhangsn/simi_net/000_data/traindata_use_my/train/old-lianxu-identity"
    # get_simi_data_level16(transpath)
    
    
    # datapath='/home/zhangsn/simi_net/000_data/olddata/traindata_use'
    # new_gridscore(datapath)
    
    # #标注以的数据重命名
    # datapath='/home/zhangsn/simi_net/000_data/traindata_level16/train/'
    # biaozhu_gridscore(datapath)

    # # 把原来的三张拼接数据改为四张拼接
    # datapath='/home/zhangsn/simi_net/000_data/traindata_use_my'
    # re_merge(datapath)
    
    # 根据图像名找原来的trans
    datapath='/home/zhangsn/simi_net/000_data/traindata_level16/train/'
    # datapath='/home/zhangsn/simi_net/quality_6193/results/6193simi_132/valid_300/images/valid/new-lianxu-succes'
    # createcsv_test(datapath)
    # datapath='/home/jianght/001_data/003_all_label_img/test_lianxu-r1data-enhance_newlabel'
    createcsv(datapath)
    # # 随机筛选部分数据训练
    # # randomsele(datapath)
    
    # # 根据csv文件修改名字，先label2再label1
    # datapath='/home/zhangsn/simi_net/000_data/traindata_level16_1116'            
    # # csv_rename_transpair(datapath)
    # checkbmplen(datapath)
    
    # datapath='/home/jianght/001_data/all_label_img/test_lianxu-r1data-enhance'
    # for root, dirs, files in os.walk(datapath):
    #     for file in sorted(files):
    #         # filepath = os.path.join(root,file)
    #         if file.endswith('.bmp'):
    #             print(file)
    #             oldfile=file[18:]
    #             myroot= root.replace('/home/jianght/001_data/all_label_img/test_lianxu-r1data-enhance','/home/zhangsn/simi_net/000_data/test_lianxu-r1data/valid')
    #             oldpath = os.path.join(myroot,oldfile)
    #             newpath = os.path.join(myroot,file)
    #             os.rename(oldpath,newpath)
    #             os.rename(oldpath[:-4]+'.npy',newpath[:-4]+'.npy')