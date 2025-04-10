"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import torch.nn.functional as F
from .image_folder import make_dataset
import os
import math



class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, convert=True):
    #path_m = "/hdd/file-input/wangb/mask/6193/pycode/mask/datasets/img_process/dd2_mask"
    #mask_paths = sorted(make_dataset(path_m, 100))
    transform_list = []
    if convert:
        transform_list.append(transforms.Grayscale(1))
        # transform_list.append(transforms.RandomVerticalFlip(0.5))
        if "iftest" not in opt:  # the following transform only takes effect when testing
            transform_list += [transforms.ColorJitter(brightness=(0.8,1.2))]#,contrast=(0.8,1.2),contrast=(0.8,0.8)
            #transform_list.append(angle_RandomRotation(degrees=180))
            #transform_list.append(transforms.Lambda(lambda img: block_mask(img,mask_paths)))
            transform_list.append(transforms.Lambda(lambda img: __crop__(img)))
            #transform_list.append(__RandomColorJitter(brightness=(0.8,1.2)))   #contrast   brightness0.8-1.0   0.5-1.2     用过brightness0.8-1.2  ,contrast=(0.8,1.2)
            transform_list.append(angle_RandomRotation_translation(degrees=20))
            transform_list += [transforms.ToTensor()]
            transform_list.append(transforms.Lambda(lambda img: __resize__tensor1(img, opt.load_size_w, opt.load_size_h)))
            #transform_list.append(transforms.Lambda(lambda img: __trans__tensor2(img, opt.load_size_w, opt.load_size_h, 3 , 6 , 0.6)))
        else:
            #transform_list.append(transforms.Lambda(lambda img: __crop3__(img)))
            transform_list += [transforms.ToTensor()]
            transform_list.append(transforms.Lambda(lambda img: __resize__tensor1(img, opt.load_size_w, opt.load_size_h)))
            
            

    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

import torch

def __crop__(image_):
    iw,ih = image_.size
    h = ih
    w = iw
    n=1
    img = image_.crop([n, 2, w - n, h - 2])
    #B = image_.crop([w + n, n, w*2 - n, h - n])
    img = np.array(img)
    #B = np.array(B)
    #img3 = np.hstack((img,B))
    img = Image.fromarray(img)
    return img
def __crop3__(image_):
    iw,ih = image_.size
    h = ih
    w = iw // 3
    n=0
    img = image_.crop([n, n, w - n, h - n])
    B = image_.crop([w*2 + n, n, w*3 - n, h - n])
    img = np.array(img)
    B = np.array(B)
    img3 = np.hstack((img,B))
    img = Image.fromarray(img3)
    return img
def __crop2__(image_):
    iw,ih = image_.size
    h = ih
    w = iw
    n=0
    img = image_.crop([n, n, w - n, h - n])
    B = image_.crop([n, n, w - n, h - n])
    img = np.array(img)
    B = np.array(B)
    img3 = np.hstack((img,B))
    img = Image.fromarray(img3)
    return img
def __resize__tensor1(img, ow, oh):
    img = img.unsqueeze(dim=0)
    img = F.interpolate(img, size=[oh, ow], mode="bilinear", align_corners=False)
    return img.squeeze(dim=0)
def __resize__tensor(img, ow, oh):
    img = img.unsqueeze(dim=0)
    A, B, mask = img.chunk(3, dim=3)  # 等分出A,B
    A = F.interpolate(A, size=[oh, ow], mode="bilinear", align_corners=False)
    B = F.interpolate(B, size=[oh, ow], mode="bilinear", align_corners=False)
    #B[B>0.5]=1
    #B[B<=0.5]=0
    mask = F.interpolate(mask, size=[oh, ow], mode="bilinear", align_corners=False)
    #B = F.interpolate(B, size=[oh, ow], mode="nearest")#NEAREST
    #B = cro(B)
    img = torch.cat([A,B,mask],dim=3)
    return img.squeeze(dim=0)
def __resize__tensor2(img, ow, oh):
    img = img.unsqueeze(dim=0)
    A, B = img.chunk(2, dim=3)  # 等分出A,B
    A = F.interpolate(A, size=[oh, ow], mode="bilinear", align_corners=False)
    B = F.interpolate(B, size=[oh, ow], mode="bilinear", align_corners=False)
    img = torch.cat([A,B],dim=3)
    return img.squeeze(dim=0)
def __resize__tensor_test(img, ow, oh):
    img = img.unsqueeze(dim=0)
    A, B= img.chunk(2, dim=3)  # 等分出A,B
    A = F.interpolate(A, size=[oh, ow], mode="bilinear", align_corners=False)
    B = F.interpolate(B, size=[oh, ow], mode="bilinear", align_corners=False)
    mask = B
    #B = F.interpolate(B, size=[oh, ow], mode="nearest")#NEAREST
    #B = cro(B)
    img = torch.cat([A,B,mask],dim=3)
    return img.squeeze(dim=0)

def __trans__tensor(img,w,h,w_p,h_p,p):
    x = random.random()
    if(x>p):
        return img
    img = img.unsqueeze(dim=0)
    #print(img.shape)
    #sss
    A, B, mask = img.chunk(3, dim=3)
    A = F.pad(A, (w_p, w_p,h_p,h_p), value=1)
    B = F.pad(B, (w_p, w_p,h_p,h_p), value=0)
    mask = F.pad(mask, (w_p, w_p,h_p,h_p), value=1)
    w_index = random.randint(0,2*w_p)
    h_index = random.randint(0,2*h_p)
    A = A[:,:,h_index:h+h_index,w_index:w+w_index]
    B = B[:,:,h_index:h+h_index,w_index:w+w_index]
    mask = mask[:,:,h_index:h+h_index,w_index:w+w_index]
    img = torch.cat([A,B,mask],dim=3)
    return img.squeeze(dim=0)
def __trans__tensor2(img,w,h,w_p,h_p,p):
    x = random.random()
    if(x>p):
        return img
    img = img.unsqueeze(dim=0)
    #print(img.shape)
    #sss
    A, B = img.chunk(2, dim=3)
    A = F.pad(A, (w_p, w_p,h_p,h_p), value=1)
    B = F.pad(B, (w_p, w_p,h_p,h_p), value=0)
    w_index = random.randint(0,2*w_p)
    h_index = random.randint(0,2*h_p)
    A = A[:,:,h_index:h+h_index,w_index:w+w_index]
    B = B[:,:,h_index:h+h_index,w_index:w+w_index]
    img = torch.cat([A,B],dim=3)
    return img.squeeze(dim=0)

def __trans__tensor3(img,w,h,w_p,h_p,p):
    x = random.random()
    if(x>p):
        return img
    img = img.unsqueeze(dim=0)
    #print(img.shape)
    #sss
    A, B, mask = img.chunk(3, dim=3)
    A = F.pad(A, (w_p, w_p,h_p,h_p), value=1)
    B = F.pad(B, (w_p, w_p,h_p,h_p), value=0)
    mask = F.pad(mask, (w_p, w_p,h_p,h_p), value=0)
    #w_index = random.randint(2,2*w_p)
    x = random.random()
    if(x>0.5):
        w_index = random.randint(0,w_p-2)
    else:
        w_index = random.randint(w_p+2,2*w_p)
    #h_index = random.randint(2,2*h_p)
    x = random.random()
    if(x>0.5):
        h_index = random.randint(0,h_p-2)
    else:
        h_index = random.randint(h_p+2,2*h_p)
    x = random.random()
    if(x>0.8):
        w_index = w_p
    elif(x>0.6):
        h_index = h_p
    A = A[:,:,h_index:h+h_index,w_index:w+w_index]
    B = B[:,:,h_index:h+h_index,w_index:w+w_index]
    mask = mask[:,:,h_index:h+h_index,w_index:w+w_index]
    img = torch.cat([A,B,mask],dim=3)
    return img.squeeze(dim=0)
    

import PIL
import torchvision.transforms.functional as FT






class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: # 按概率进行
            # 把img转化成ndarry的形式
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # 原始图像的概率（这里为0.9）
            signal_pct = self.snr
            # 噪声概率共0.1
            noise_pct = (1 - self.snr)
            # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            # 将mask按列复制c遍
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255 # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB') # 转化为PIL的形式
        else:
            return img

#添加高斯噪声
class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img
class AddGaussianNoise_gray(object):

    def __init__(self, mean=0.0, variance=10, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            #img = np.array(img)
            h, w = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img[img < 0] = 0
            img = img.astype('uint8')
            return img
        else:
            return img
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
 
    def __init__(self, probability=0.5, sl=0.02, sh=0.2, r1=0.3, mean=[0, 0, 0]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
 
    def __call__(self, img):
 
        if random.uniform(0, 1) > self.probability:
            return img
        
        #img=np.asarray(img)
        #img.flags.writeable = True
        img = np.array(img)
        #img = Image.fromarray(img.astype('uint8'))
        for attempt in range(100):
            area = img.shape[0] * img.shape[1]
 
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
 
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
 
            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                #if img.size()[0] == 3:
                    #img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    #img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    #img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                #else:
                img[ x1:x1 + h, y1:y1 + w] = 255
                img = Image.fromarray(np.uint8(img))
                return img
        img = Image.fromarray(np.uint8(img))
        return img
def noise(image_):
    iw,ih = image_.size
    h = ih
    w = iw // 2
    n=0
    img = image_.crop([n, n, w - n, h - n])
    B = image_.crop([w + n, n, w*2 - n, h - n])
    img = np.array(img)
    B = np.array(B)
    B = 0
    img = random.randint(90,145)
    PepperNoise = AddPepperNoise()
    img = PepperNoise(img)







from torchvision.transforms.functional import rotate as Frotate
from torchvision.transforms import InterpolationMode
class angle_RandomRotation(transforms.RandomRotation):
    def __call__(self, image):
        n = 0
        A = image.crop([n, n, 188 - n, 188 - n])
        B = image.crop([188 + n, n, 376 - n, 188 - n])
        angle = self.get_params(self.degrees)
        # angle = random.choice([0,90,180,270])
        A = np.array(Frotate(A, angle, self.resample, self.expand, self.center, self.fill))
        B = np.array(Frotate(B, angle, self.resample, self.expand, self.center, self.fill))

        image = np.hstack((A, B))

        # inverse_phase = random.choice([0, 1])  # 随机反相
        # if inverse_phase:
        #     image = 255 - image

        return image

class angle_RandomRotation_translation(transforms.RandomRotation):
    def __call__(self, image):
        n = 0
        iw,ih = image.size
        w = iw
        h = ih
        A = image.crop([n, n, w - n, ih - n])
        height = h
        weight = w
        #gary_v = 255
        #xz_mask = Image.new('L', (weight, height), gary_v)
        #x = random.random()
        #if(x>0.6):
        #    gary_v = 255
        #else:
        #    gary_v = random.randint(0,255)
        #水平翻转
        x = random.random()
        if(x>0.5):
            res = transforms.RandomHorizontalFlip(p=1)
            A = res(A)
            #B = res(B)
            #C = res(C)
        #竖直翻转
        x = random.random()
        if(x>0.5):
            res = transforms.RandomVerticalFlip(p=1)
            A = res(A)
            #B = res(B)
            #C = res(C)
        
        
        #随机放大
        
        #x = random.random()
        #if(x>0.9):#0.9
        #    scale = 9/8
        #    #if(x>0.95):
        #        #scale = 5/4
        #    res = transforms.Resize([int(height*scale+0.5),int(weight*scale+0.5)])
        #    A = res(A)
        #    #cro = transforms.Resize([int(height*scale+0.5),int(weight*scale+0.5)])
        #    #cro = transforms.Resize([int(height*scale+0.5),int(weight*scale+0.5)],interpolation=InterpolationMode.NEAREST)#最邻近插值
        #    #Image.BICUBIC，PIL.Image.LANCZOS，PIL.Image.BILINEAR，PIL.Image.NEAREST
        #    
        #    #B = cro(B)
        #    #C = cro(C)
        #    i, j, h, w = transforms.RandomCrop.get_params(A, (ih,w))
        #    A = FT.crop(A, i, j, h, w)
            #B = FT.crop(B, i, j, h, w)
            #C = FT.crop(C, i, j, h, w)
        
        #随机缩小
        #elif(x>0.8):
            #pad = (4,16,4,16)
            #gary_v = random.randint(0,255)
            #gary_v = 255
            #A = FT.pad(A, pad, gary_v)
            #B = FT.pad(B, pad, 0)
            #xz_mask = FT.pad(xz_mask, pad, 0)
            #res = transforms.Resize([height,weight])
            #A = res(A)
            #B = res(B)
            #xz_mask = res(xz_mask)
        
        #平移
        #elif random.choice([0, 1]):
        #x = random.random()
        #if(x>0.4):
        #    pad = (4,16,4,16)
        #    ##gary_v = random.randint(0,255)
        #    ##gary_v = 255
        #    A = FT.pad(A, pad, gary_v)
        #    B = FT.pad(B, pad, 0)
        #    xz_mask = FT.pad(xz_mask, pad, 0)
        #
        #    i, j, h, w = transforms.RandomCrop.get_params(A, (ih,iw//2))
        #
        #    A = FT.crop(A, i, j, h, w)
        #    B = FT.crop(B, i, j, h, w)
        #    xz_mask = FT.crop(xz_mask, i, j, h, w)
        
        
        

        #旋转
        #x = random.random()
        #if(x>0.2):
            #angle = self.get_params(self.degrees)
            ## angle = random.choice([0,90,180,270])
            ##A = np.array(Frotate(A, angle, self.resample, self.expand, self.center, self.fill))
            ##B = np.array(Frotate(B, angle, self.resample, self.expand, self.center, self.fill))
                
            ##gary_v = 255
            #A = np.array(Frotate(A, angle, self.resample, self.expand, self.center, gary_v))
            #B = np.array(Frotate(B, angle, self.resample, self.expand, self.center, 0))
            #xz_mask = np.array(Frotate(xz_mask, angle, self.resample, self.expand, self.center, 0))

        #A = np.array(A)
        #B = np.array(B)
        #C = np.array(C)
        #xz_mask = np.array(xz_mask)
        #if np.sum(B<128) > (weight*height/32):
        #    mean = 0
        #    variance = random.randint(1,10)
        #    GaussianN = AddGaussianNoise_gray(mean=mean, variance=variance, amplitude=1.0,p=0.1)
        #    image = np.copy(A)
        #    A = GaussianN(A)
        #    A[B>128]=image[B>128]   


        #image = np.hstack((A, B, C))
        #image = PIL.Image.fromarray(image)

        # inverse_phase = random.choice([0, 1])  # 随机反相
        # if inverse_phase:
        #     image = 255 - image

        return A

