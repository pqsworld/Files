'''
读取csv文件跑结果
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch
from models.networks import MNV3_bufen_new5
import torch.nn as nn
import functools

BASEROOT='/home/zhangsn/simi_net/quality_6193'
CHEROOT='/home/zhangsn/simi_net/quality_6193/checkpoints/'
RESROOT='/home/zhangsn/simi_net/quality_6193/results/'
if __name__ == '__main__':
    netname='6193simi_3_1201'
    epoch='300'
    phase='valid_1130'
    model_path = CHEROOT+netname+'/'+epoch+'_net_G.pth'
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # '/home/zhangsn/simi_net/quality_6193/checkpoints/6193simi_3_1201/300_net_G.pth'
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    quality_model = MNV3_bufen_new5(2, 1, 4, norm_layer=norm_layer, use_dropout=True, n_blocks=9).to(device)
    quality_model.load_state_dict(torch.load(model_path))
    quality_model.eval()
    #create a website
    web_dir=os.path.join(RESROOT, netname, '{}_{}'.format(phase, epoch))
    print('creating web directory', web_dir)
    na = web_dir.split("/")[:]
    save_data_path = na[0]
    # print(image_dir)
    for i_na in range(0, len(na)):
        save_data_path = save_data_path + "/" + na[i_na]
        if not os.path.isdir(save_data_path):
            os.mkdir(save_data_path)

    cvs_path = '/home/zhangsn/simi_net/000_data/traindata_level16_1116/valid_1130/all_data_change_newlabel_changewet.csv'
    df = pd.read_csv(
            cvs_path,
            header=0,
            # index_col=0,
            # encoding = "gb2312",
            names=['img_path','him','ori','ssim','raw_label','change_label','flag','new_label'],
            )
 
    new_label_list=[]
    imglist_fr = []
    gtlist_fr = []
    reslist_fr = []
    
    imglist_fa = []
    gtlist_fa = []
    reslist_fa = []
    alltest=len(df)
    for index in range(len(df)):
        img_path=df['img_path'][index]
        if index % 100 == 0: 
            print('Test******%d/%d: %s' % (index, alltest,img_path))
        img = cv2.imread(img_path)
        h,w = img.shape[0],img.shape[1]
        A_level_16_img = img[:,:,1]
        B_level_16_img = img[:,:,2]

        A=Image.fromarray(A_level_16_img)
        B=Image.fromarray(B_level_16_img)     
        # print(AB_path,simi)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        AB = torch.cat([A,B],0).unsqueeze(0).to(device)
        # print(AB.shape)
        gt=df['new_label'][index]
        flag = df['flag'][index]
        classify_c = quality_model(AB).to('cpu')
        nerres = (classify_c[0][0].detach().numpy()*128+127).astype(int)
        if flag=='fr':
            imglist_fr.append(img_path)
            reslist_fr.append(nerres)
            gtlist_fr.append(gt)
        else:
            imglist_fa.append(img_path)
            reslist_fa.append(nerres)
            gtlist_fa.append(gt)
    data={'root':imglist_fr,'gridscore':gtlist_fr,'netres':reslist_fr}  #,'diff':alldiff 'siminorm':ssimnormlist,'hamnorm':hamnormlist,'simi_labelnew':labelnewlist
    df=pd.DataFrame(data)
    df.to_csv(web_dir+'/out_'+phase+'_fr.csv',index=False)
    data={'root':imglist_fa,'gridscore':gtlist_fa,'netres':reslist_fa}  #,'diff':alldiff 'siminorm':ssimnormlist,'hamnorm':hamnormlist,'simi_labelnew':labelnewlist
    df=pd.DataFrame(data)
    df.to_csv(web_dir+'/out_'+phase+'_fa.csv',index=False)   
    df.to_csv('_fa.csv',index=False)  

    
    # plot
    frrnumpy=np.array(reslist_fr)
    farnumpy=np.array(reslist_fa)
    # alldata=np.c_[resnumpy,gtnumpy]
    # print(alldata.shape)
    plt.figure()
    plt.title('res')
    plt.hist(frrnumpy,bins=140, rwidth=0.8, range=(120,260),label='Frr', align='left',alpha=0.5)
    plt.hist(farnumpy,bins=140, rwidth=0.8, range=(120,260), label='Far', align='left',alpha=0.5)
    # plt.hist(alldata,bins=65,color=['c','r'], rwidth=0.5, range=(55,315), label=['net_res','gt_label'], align='left',alpha=0.5,stacked=False)
    plt.legend()
    plt.savefig(web_dir+'/frr_far_res.png')
    
    frrnumpy=np.array(gtlist_fr)
    farnumpy=np.array(gtlist_fa)
    plt.figure()
    plt.title('gt')
    plt.hist(frrnumpy,bins=140, rwidth=0.8, range=(120,260),label='Frr', align='left',alpha=0.5)
    plt.hist(farnumpy,bins=140, rwidth=0.8, range=(120,260), label='Far', align='left',alpha=0.5)
    # plt.hist(alldata,bins=65,color=['c','r'], rwidth=0.5, range=(55,315), label=['net_res','gt_label'], align='left',alpha=0.5,stacked=False)
    plt.legend()
    plt.savefig(web_dir+'/frr_far_gt.png')