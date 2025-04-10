import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import random
import torchvision.transforms.functional as FT
import torchvision.transforms as transforms
import numpy as np


class AlignedDataset(BaseDataset):
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
        #random.shuffle(self.AB_paths)
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

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
        AB = Image.open(AB_path).convert('RGB')
        simi = int(AB_path.split('_')[0])
        #print(AB_path)
        # quality =torch.tensor(0.0)
        # #quality = torch.tensor(float(AB_path.split("/")[-1].split("_")[0].split("q")[-1].split("bmp")[0])/100)
        # # split AB image into A and B
        # # w, h = AB.size
        # # w2 = int(w / 2)
        # # A = AB.crop((0, 0, w2, h))
        # # B = AB.crop((w2, 0, w, h))

        # # apply the same transform to both A and B
        # # transform_params = get_params(self.opt, A.size)
        # # rotate_angle = float(torch.empty(1).uniform_(float(-180), float(180)).item())
        # # rotate_angle = 0
        # #if "iftest" not in self.opt:
        # #    iw,ih = AB.size
        # #    h = ih
        # #    w = iw // 2
        # #    n=0
        # #    img = AB.crop([n, n, w - n, h - n])
        # #    B = AB.crop([w + n, n, w*2 - n, h - n])
        # #
        # #    #A, B = AB.chunk(2, dim=2)
        # #    contrast_factor = torch.tensor(1.0).uniform_(0.8, 1.2).item()
        # #    img = FT.adjust_contrast(img, contrast_factor)
        # #    img = np.array(img)
        # #    B = np.array(B)
        # #    img3 = np.hstack((img,B))
        # #    AB = Image.fromarray(img3)
        # #    quality = quality * contrast_factor
        # #res = transforms.ColorJitter(contrast=(0.8,1.2))
        # #x = random.random()
        # #if(x>0.5 and quality<0.33):
        #     #print("quality:",quality)
        #     #AB = res(AB)
            
        AB_transform = get_transform(self.opt)
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        AB = AB_transform(AB)  # 保持A,B两图处理方式一致
        #A, B, mask = AB.chunk(3, dim=2) # 等分出A,B
        #if "iftest" not in self.opt:
        #    A, B = AB.chunk(2, dim=2)
        #    return {'A': A, 'B': B, 'gt': quality, 'A_paths': AB_path, 'B_paths': AB_path}
        #else:
        #    return {'A': AB, 'B': AB, 'gt': quality, 'A_paths': AB_path, 'B_paths': AB_path}
        return {'A': AB, 'B': AB, 'gt': simi, 'A_paths': AB_path, 'B_paths': AB_path}




    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
