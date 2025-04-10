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


class SimitestDataset(BaseDataset):
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
        h,w = img.shape[0],img.shape[1]
        ABmerge = img
        A = ABmerge[:,:,1]
        B = ABmerge[:,:,2]
        
        A = cv2.resize(A, (18,61))
        B = cv2.resize(B, (18,61))
        
        A_level_16_img = A
        B_level_16_img = B
        
        
        A=Image.fromarray(A_level_16_img)
        B=Image.fromarray(B_level_16_img)
        
        ttt= (AB_path.split('/')[-1]).split('_')
        # # # label2=ttt[0]
        simi = int(ttt[0])
        # simi=0
        # simi = (simi-55)/260
        simi = (simi-127)/128
        
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
