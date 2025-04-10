import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image,ImageFilter
import random
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from util.matlab_imresize import *

import torch.nn.functional as F

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        
        self.dir_A = opt.datarootA  # create a path '/path/to/data/trainA'
        self.dir_B = opt.datarootB
   
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        # print(self.A_paths[:10])
        # print(self.B_paths[:10])
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        # self.transform_A = get_transform(self.opt, grayscale=(output_nc == 1))

        self.transA = transforms.Compose([
            transforms.Grayscale(),
           # transforms.Lambda(lambda img: random_blur(img), ),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.transB = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.transC = transforms.Compose([

            transforms.ToTensor(),
            #TensorResize([128,128]),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        index_B = index % self.B_size
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('L')
        B_img = Image.open(B_path).convert('L')#.resize((128,128),Image.BILINEAR)

        imgsrc = np.array(A_img, dtype=np.float32)

        Bimg = imresize(imgsrc, method='bilinear', output_shape=[self.opt.load_sizeh, self.opt.load_sizeh], mode="vec")
        Bimg = (Bimg - 127.5) / 127.5

        # img0 = transform(imgA)
        img0 = torch.from_numpy(Bimg)
        img0 = img0.float()

        img = torch.unsqueeze(img0, 0)
        # img = torch.unsqueeze(img, 0)

        A = img
        B = img
        mask = np.ones_like(Bimg) * 0
        mask = self.transC(mask)
 
        return {'A': A, 'B': B, 'A_paths': A_path,'C':B, 'B_paths': B_path,'mask':mask}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)