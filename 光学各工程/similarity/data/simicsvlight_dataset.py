import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset,make_dataset_sele
from PIL import Image
import torch
import random
import torchvision.transforms.functional as FT
import torchvision.transforms as transforms
import numpy as np
import cv2
import pandas as pd


class SimicsvlightDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot)  # get the image directory
        opt.phase=['0','1']

        datasets_use = [
            "1_2",
            "3_4",
            "45",
            "belly",
            "tip",
            "dust",
            "dw",
            "lgm",
            "msm",
            "wet",
                        
            "nova-cdw",
            "nova_dw",
            "nova_dw-bad",
            "nova_lgm",
            "nova_msm",
            "nova_sun-part",
            "nova_wet",
            "nova_dust",
            "nova_snm",
            "nova_sun-all",
            
            #small ttl
            #加了以后效果不好(st32)
            "TM-CWBF",
            "TM-DW",
            "TM-GSZ",
            "TM-QG",
            "TM-QGBF",
            "TM-SSZ",
            "TM-XS",
            
        ]


        self.AB_paths = []
        self.ori = []
        self.him = []
        self.ssim = []
        self.frr_flag=[] 
        data_list=[]
        for phase in opt.phase:
            for dataset_t in datasets_use:
            # for phase in opt.phase:
                self.csvpath = os.path.join(self.dir_AB,phase,dataset_t+'_train1219.csv')  #trans_choose.txt
                # print(self.csvpath)
            # for dataset_name in datasets_csv:
            #     self.csvpath = os.path.join(self.dir_AB,dataset_name)
            #     print(self.csvpath)
                # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
                df = pd.read_csv(
                            self.csvpath,
                            header=0,
                            # index_col=0,
                            encoding = "gb18030", #"gb2312",
                            # names=['img_path', 'ori','ham','ham_thre','ssim','label','grid_score','temp','samp','trans'],
                            names=['img_path', 'ori','ham','ham_thre','ssim','label_new1219','label','grid_score','temp','samp','trans'],
                            index_col=0,
                            )    # 'gb2312'
                
                print(self.csvpath,len(df),df.shape[1],min(df['label_new1219']),max(df['label_new1219'])) 
                # 删除部分nw-fa数据,按照1/2比例
                # if int(phase)==0 and 'nova' in dataset_t:
                #     # print(len(df))
                #     rate=6#4
                #     if 'nova_sun-all' in dataset_t:
                #         rate=20#15
                #     df=df.reset_index()
                #     savenum=int(len(df)/rate)
                #     # print(savenum)
                #     df = df.sample(n=savenum)
                #     # print(len(df))
                # if int(phase)==1 and 'nova' in dataset_t:
                #     df=df.reset_index()
                #     # delin=list(range(0,len(df)-1,2))
                #     # df = df.drop(delin)
                #     rate = 4#3
                #     if 'nova_sun-all' in dataset_t:
                #         rate=15#10
                #     savenum=int(len(df)/rate)
                #     df = df.sample(n=savenum)

                # 数据筛选，fr删除相似度很低的数据
                # df=df.reset_index()
                if int(phase)==1:
                    df = df.drop(df[df['label']<135].index) #simi1_1210
                    # df = df.drop(df[df['label_new1219']<142].index)  #simi21:用新版本去除frr中的一些差图
                # if int(phase)==0:
                #     # df = df.drop(df[df['label']<135].index)
                #     df = df.drop(df[df['label']>180].index)

                df.insert(0,'flag',int(phase))
                
                data_list.append(df)
            df_t = pd.concat(data_list,axis=0)
            print(phase,len(df_t))    
        df_all = pd.concat(data_list,axis=0)
        # print(len(df_all))    
        self.AB_paths = df_all['img_path'].to_list()
        # self.AB_paths0 = df['img_path'].to_list()
        self.sim=df_all['label'].to_list()
        self.sim_new=df_all['label_new1219'].to_list()
        # print(len(self.AB_paths))
        self.ori=df_all['ori'].to_list()
        self.him=df_all['ham'].to_list()
        self.ssim=df_all['ssim'].to_list()
        self.frr_flag=df_all['flag'].to_list()

        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        transformlist=[transforms.Grayscale(1),
                        transforms.ToTensor(),]
        self.transform = transforms.Compose(transformlist)

    def random_enhance(self,img1,img2):
        ww,hh=img1.shape
        # print(ww,hh)
        brightflag=random.uniform(0,10)
        if brightflag>6:
            brightness_factor = torch.tensor(1.0).uniform_(0.95, 1.05).item()
            if random.choice([0,1]):
                img1=Image.fromarray(img1)
                img1 = np.array(FT.adjust_brightness(img1, brightness_factor))
            else:
                img2 = Image.fromarray(img2)
                img2 = np.array(FT.adjust_brightness(img2, brightness_factor))

        rotateflag=random.uniform(0,10)
        if rotateflag>6:
            angle = random.choice([0, 90, 180, 270])
            scale = 1
            matRotate = cv2.getRotationMatrix2D((ww // 2, hh // 2), angle,scale)  # mat rotate 1 center 2 angle 3 缩放系数
            mask_color = 255  # 255: white, 0:black
            img1 = cv2.warpAffine(img1, matRotate, (ww, hh), flags=cv2.INTER_LINEAR, borderValue=mask_color)
            img2 = cv2.warpAffine(img2, matRotate, (ww, hh), flags=cv2.INTER_LINEAR, borderValue=mask_color)
        return img1,img2
                    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        # print(self.all_data)['img_path']

        AB_path = self.AB_paths[index]
        # print(AB_path)
        AB_path = AB_path.replace('/hdd/file-input/zhangsn/data/000_light_simi_1127add_nvwa/','/data/guest/zsn/simi_data_light/001_traindata/')
        # AB_path = AB_path.replace('/data/guest/zsn/simi_data_light/0/','/data/guest/zsn/simi_data_light/001_traindata/0/')
        
        # print(AB_path)
        if os.path.exists(AB_path):
            # AB = Image.open(AB_path).convert('RGB') 
            img = cv2.imread(AB_path)  #'BGR'
            h,w = img.shape[0],img.shape[1]  #128,128
            # print(img.shape)

            A_level_16_img = img[:,:,1]
            B_level_16_img = img[:,:,2]
            A_level_16_img = cv2.resize(A_level_16_img,(self.opt.load_size_h, self.opt.load_size_w))
            B_level_16_img = cv2.resize(B_level_16_img,(self.opt.load_size_h, self.opt.load_size_w))
            
            
            A_level_16_img,B_level_16_img=self.random_enhance(A_level_16_img,B_level_16_img)

            A=Image.fromarray(A_level_16_img)
            B=Image.fromarray(B_level_16_img)
            # print(A.size,B.size)
            # print(AB_path,simi)
            A = self.transform(A)  # 保持A,B两图处理方式一致
            B = self.transform(B)
            AB = torch.cat([A,B],0)
            # print(AB.shape)
            zero = torch.zeros_like(A)
            ABC= torch.cat([zero,AB],0)
            # print('111',ABC.shape)

            if self.opt.teacher:
                A_level_16_img_t = cv2.resize(A_level_16_img,(96, 96))
                B_level_16_img_t = cv2.resize(B_level_16_img,(96, 96))
                A=Image.fromarray(A_level_16_img_t)
                B=Image.fromarray(B_level_16_img_t)
                A_t = self.transform(A)  # 保持A,B两图处理方式一致
                B_t = self.transform(B)
                AB_t = torch.cat([A_t,B_t],0)
                # # print(AB.shape)
                # zero = torch.zeros_like(A_t)
                # ABC_t= torch.cat([zero,AB_t],0)
                # # print('111',ABC.shape)

            # # label2=ttt[0]
            simi = self.sim[index]
            flag = int(self.frr_flag[index])
            #微调计算的label
            if flag==1 and simi>135 and simi<155:#170:
                simi = simi+5#15
            if flag==0 and simi>135:
                simi = np.where(simi>55, simi-5, simi)
                
            if 1:#using_1210_flag:
                simi_new = self.sim_new[index]             
                if flag==1:
                    simi=max(simi,simi_new) #fr选大的
                else:
                    simi=min(simi,simi_new) #fa选小的
            if simi>320:
                simi=320
            if simi<55:
                simi=55
            # print(flag)
            
            simi = (simi-55)/265   #59 309
            #norm
            # simi = (simi-127)/128   #127~255          
            
            if self.opt.teacher:
                return {'A': AB, 'B': ABC,'A_tea':AB_t,'gt': simi, 'A_paths': AB_path, 'B_paths': AB_path,'flag':flag}
            else:
                return {'A': AB, 'B': ABC, 'gt': simi, 'A_paths': AB_path, 'B_paths': AB_path,'flag':flag}



    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
