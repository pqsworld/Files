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


class SimiDataset(BaseDataset):
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
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        # self.AB_paths = sorted(make_dataset_sele(self.dir_AB, opt.max_dataset_size))
        #random.shuffle(self.AB_paths)
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
        AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        img = cv2.imread(AB_path)
        # print(img.shape)
        A = img[:,:,1]
        B = img[:,:,2]
        
        A = cv2.resize(A, (18,61))
        B = cv2.resize(B, (18,61))
        
        # load 4 imgs :crop
        # h,w = img.shape[0],img.shape[1]
        # # print(img.shape)
        # w2 = int(w / 4)
        # # A = AB.crop((w2*2,0, w2*3, h))
        # # B = AB.crop((w2, 0, w2*2, h))
        # # ABmerge = AB.crop((w2*3,0,w, h))
        # ABmerge = img[0:h,w2*3:w]
        # mask = img[0:h,w2*2:w2*3,0]
        # mask = cv2.resize(mask, (w//8,h//2))
        # print(ABmerge.shape)
        # A = ABmerge[:,:,1]
        # B = ABmerge[:,:,2]

        
        A_level_16_img = A
        B_level_16_img = B
        
        # A_level_16_img = np.where(mask<10,0,A_level_16_img)
        # B_level_16_img = np.where(mask<10,0,B_level_16_img)
        
        A=Image.fromarray(A_level_16_img)
        B=Image.fromarray(B_level_16_img)
        # print(A.size)
        #print(AB_path)
        # simi = int((AB_path.split('/')[-1]).split('_')[1])
        
        ttt= (AB_path.split('/')[-1]).split('_')
        # # label2=ttt[0]
        simi = int(ttt[0])
        # # 归一化ssim+ham
        # ham=int((ttt[1])[3:])   #[128,231] 扩大[125,235]
        # ssim=int((ttt[2])[4:])  #[13,99]   扩大[10,100]
        
        #norm1
        # ham_norm = 64*(ham-125)/(235-125)   #[0,64]
        # ssim_norm =64*(ssim-10)/(100-10)
        # simi_new=ham_norm+ssim_norm+127   #[127,127+128] 即[127，255]
        
        # ham_norm = ham/4    #理论值0~256，转换到0~64
        # #大于190的压缩一下范围，然后求和，压缩到200以内，然后直接norm
        # if ham>190:
        #     ham_norm = (190+(ham-190)*0.15)*0.32   #大于190的压缩一下范围，然后求和，压缩到(0,200)以内，然后直接norm
        # else:
        #     ham_norm = ham*0.32    
        # ssim_norm = ssim*0.01*64   #理论值100
        # simi_new=ham_norm+ssim_norm+127
        # # 取新label和之前网格相似度的最大值
        # gridscore = int((ttt[7])[1:])
        # simi = max(gridscore,simi)
        
        # simi = (simi_new-127)/128

        
        # print(gridscore)
        # exit()
        # simi = simi/355   #ht
        simi = (simi-55)/260
        # print(ttt)
        # print(ttt[0])
        # simi = int(ttt(0)) 
        # simi = (simi-100)/255   # 范围【128，330】
        
        # simi = int(ttt[3])     #[标注 127，255]
        # simi = (simi-127)/128 #[标注 127，255]
        
        # print(simi)
        # AB_transform = get_transform(self.opt)
        # print(A.size)
        A = self.transform(A)  # 保持A,B两图处理方式一致
        B = self.transform(B)
        AB = torch.cat([A,B],0)
        # print(AB.shape)
        zero = torch.zeros_like(A)
        ABC= torch.cat([zero,AB],0)
        # print(AB.shape)
        return {'A': AB, 'B': ABC, 'gt': simi, 'A_paths': AB_path, 'B_paths': AB_path}




    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
