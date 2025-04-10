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


class SimicsvDataset(BaseDataset):
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
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        datasets_csv=[
            # 'train_move_badceses_newlabel.csv',    
            # 'train_move_badceses_newlabel_changewet.csv',
            'train_move_badceses_newlabel_changewet_nolianxu.csv',
            'fake_worse_case_newlabel.csv',  
            'badcase.csv',   
            # 'all_data_change_newlabel_changewet.csv',
        ]


        self.AB_paths = []
        self.ori = []
        self.him = []
        self.ssim = []
        self.frr_flag=[] 
        data_list=[]
        # for dataset_t in datasets_use:
        #     self.csvpath = os.path.join(self.dir_AB,dataset_t,'train_1101.csv')  #trans_choose.txt
        #     print(self.csvpath)
        for dataset_name in datasets_csv:
            self.csvpath = os.path.join(self.dir_AB,dataset_name)
            print(self.csvpath)
            # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
            df = pd.read_csv(
                        self.csvpath,
                        header=0,
                        # index_col=0,
                        encoding = "gb18030", #"gb2312",
                        # names=['img_path','him','ssim','ori','old_label','new_label']
                        # names=['img_path','him','ssim','ori','old_label','new_label','img_path_1','change_label','img_path_2']
                        # names=['img_path','him','ssim','ori','old_label','new_label','img_path_1']
                        # names = ['img_path','zsn_img_path','enhance_img_path','him','ori','ssim','label'],
                        # names = ['zsn_img_path','enhance_img_path','him','ori','ssim','label','label_new'],
                        # names = ['img_path','him','ori','ssim','raw_label','change_label','flag','ori0','him0','label_zsn','zsn_label'],
                        names = ['img_path','him','ori','ssim','raw_label','change_label','flag','new_label']
                        # names=['img_path', 'ori','ham','ham_thre','ssim','label','grid_score','temp','samp','trans']
                        # index_col=None,
                        )    # 'gb2312'
            data_list.append(df)
        df_all = pd.concat(data_list,axis=0)
        # print(len(df_all))    
        self.AB_paths = df_all['img_path'].to_list()
        # self.AB_paths0 = df['img_path'].to_list()
        self.sim=df_all['label'].to_list()
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
        AB_path = AB_path.replace('/simi_data/train/','/simi_data/train_en_0410/')
        # print(AB_path)
        if os.path.exists(AB_path):
            # AB = Image.open(AB_path).convert('RGB') 
            img = cv2.imread(AB_path)  #'BGR'
            h,w = img.shape[0],img.shape[1]
            # print(img.shape)
            # if h == 122:
            #     img = cv2.resize(img,(18, 61, 3))
            level16=0
            if level16:
                if w!=18:    #level16 18 enhance 36
                    w2 = int(w / 4)
                    ABmerge = img[0:h,w2*3:w]
                    mask = img[0:h,w2*2:w2*3,0]
                    A_level_16_img = np.where(mask<10,0,ABmerge[:,:,1])
                    B_level_16_img = np.where(mask<10,0,ABmerge[:,:,2])
                else:
                    A_level_16_img = img[:,:,1]
                    B_level_16_img = img[:,:,2]
            else:
                A_level_16_img = img[:,:,1]
                B_level_16_img = img[:,:,2]
                if w==36:    #level16 18 enhance 36
                    A_level_16_img = cv2.resize(A_level_16_img,(18, 61))
                    B_level_16_img = cv2.resize(B_level_16_img,(18, 61))
            A=Image.fromarray(A_level_16_img)
            B=Image.fromarray(B_level_16_img)
            # print(A.size)

            # # label2=ttt[0]
            simi = self.sim[index]

            simi = (simi-55)/265   #59 309
            #norm
            # simi = (simi-127)/128   #127~255
            
            flag = self.frr_flag[index]
            # print(AB_path,simi)
            A = self.transform(A)  # 保持A,B两图处理方式一致
            B = self.transform(B)
            AB = torch.cat([A,B],0)
            # print(AB.shape)
            zero = torch.zeros_like(A)
            ABC= torch.cat([zero,AB],0)
            # print('111',ABC.shape)
            return {'A': AB, 'B': ABC, 'gt': simi, 'A_paths': AB_path, 'B_paths': AB_path,'flag':flag}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
