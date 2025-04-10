from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import torchvision.transforms as transforms
from data.base_dataset import *#BaseDataset, get_params, get_transform
from PIL import Image
import random
import numpy as np
import os
def gauss_noise(img,mean,var):
    # mean = 0
    # var = random.uniform(0.12,0.2)
    img = np.array(img)
    img = img/255
    noise = np.random.normal(mean, var, img.shape)
    gaussian_out = img + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)
    gaussian_out = Image.fromarray(np.uint8(gaussian_out*255)) 
    return gaussian_out

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
       

      
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        
        # input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.transform = get_transform(opt, grayscale=(input_nc == 1))


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        self.transformA = transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize([self.opt.load_sizeh, self.opt.load_sizew]),  #
                transforms.Lambda(lambda img: gauss_noise(img, 0, random.uniform(0.05,0.20)),),
        # read a image given a random integer index
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ])
        self.transformB = transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize([self.opt.load_sizeh, self.opt.load_sizew]),  #
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ])
        A_path = self.A_paths[index]
        B_path = A_path
        A_img = Image.open(A_path).convert('RGB')
       
        B_img = A_img
        
 
      
        A = self.transformA(A_img)
        B = self.transformB(B_img)
            

        return {'A': A, 'A_paths': A_path,'B': B, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)


    def resizepil(self,img):
        imgsrc = np.array(img, dtype=np.float32)
        Aimg = imresize(imgsrc, method='bilinear', output_shape=[self.opt.load_sizeh, self.opt.load_sizew], mode="vec")
        img1 = Image.fromarray(Aimg)
        img1 = img1.convert("L")
        return img1
