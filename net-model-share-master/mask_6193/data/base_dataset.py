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
from data.image_folder import make_dataset
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
            #transform_list.append(angle_RandomRotation(degrees=180))
            #transform_list.append(transforms.Lambda(lambda img: block_mask(img,mask_paths)))
            #transform_list.append(transforms.Lambda(lambda img: __crop__(img)))
            transform_list.append(__RandomColorJitter(brightness=(0.8,1.2),contrast=(0.6,1.2)))   #contrast   brightness0.8-1.0   0.5-1.2     用过brightness0.8-1.2
            transform_list.append(angle_RandomRotation_translation(degrees=20))
            transform_list += [transforms.ToTensor()]
            transform_list.append(transforms.Lambda(lambda img: __resize__tensor(img, opt.load_size_w, opt.load_size_h)))
            transform_list.append(transforms.Lambda(lambda img: __trans__tensor(img, opt.load_size_w, opt.load_size_h, 3 , 6 , 0.6)))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop1_3__(img)))
            transform_list += [transforms.ToTensor()]
            transform_list.append(transforms.Lambda(lambda img: __resize__tensor_test(img, opt.load_size_w, opt.load_size_h)))
            
            

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
    w = iw // 2
    n=2
    img = image_.crop([n, n, w - n, h - n])
    B = image_.crop([w + n, n, w*2 - n, h - n])
    img = np.array(img)
    B = np.array(B)
    img3 = np.hstack((img,B))
    img = Image.fromarray(img3)
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
def __crop1_3__(image_):
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
def quan(image_):
    #image_ = Image.open(path1).convert('L')
    iw,ih = image_.size
    h = ih
    w = iw // 3
    n=0
    img = image_.crop([n, n, w - n, h - n])
    B = image_.crop([w + n, n, w*2 - n, h - n])
    C = image_.crop([w*2 + n, n, w*3 - n, h - n])
    B = np.array(B)
    C = np.array(C)
    #res = transforms.ColorJitter(brightness=(1.5,1.6),contrast=(0.4,1.0))
    
    
    x = random.random()
    #if(x>0.4):
    res = transforms.ColorJitter(contrast=(0.2,0.5))#0.4  0.50.2-0.5
    res2 = transforms.ColorJitter(brightness=(1.1,1.6))#  1.5  1.6  1.1-1.4
    
    #else:
        #res = transforms.ColorJitter(contrast=(0.12,0.2))#0.4  0.5
        #res2 = transforms.ColorJitter(brightness=(1.1,1.3))#  1.5  1.6  1.1-1.4
    
    
    
    img2 = res(img)
    img2 = res2(img2)
    img = np.array(img)
    img2 = np.array(img2)
    img3 = np.copy(img)
    #ssss
    #start_h = 90
    start_h = random.randint(50,80)#50-90
    k = random.randint(20,30)
    m = random.randint(10,25)#15 25
    #k=30
    #m=20
    for i in range(k):
        #print(i)
        img3[start_h+i,:]=img2[start_h+i,:]*(i/k)+img[start_h+i,:]*((k-i)/k)
    img3[start_h+k:,:]=img2[start_h+k:,:]
    x = random.random()
    if(x>0.8):#0.7
        for i in range(m):
        #print(i)
            img3[start_h:,i]=img3[start_h:,i]*(i/m)+img[start_h:,i]*((m-i)/m)
    #img4[70:,k:]=img2[70:,k:]
    #img = img3/2+img4/2
    
    x = random.random()
    if(x>0.8):
        path_m = "../datasets/6193/process_im/mask_by"
        mask_paths = sorted(make_dataset(path_m, 100))
        mask_dir = random.choice(mask_paths)
        mask = Image.open(mask_dir).convert('L')
        iw_m,ih_m = mask.size
        bili = [8/8,17/16]
        bi = random.choice(bili)
        res_m = transforms.Resize([int(ih_m*bi),int(iw_m*bi)],interpolation=InterpolationMode.NEAREST)
        cor = transforms.RandomCrop([ih_m,iw_m])
        mask = res_m(mask)
        mask = cor(mask)
        mask = np.array(mask)
        img3[mask>128]=255
        B[mask>128]=0
        C[mask>128]=255
        mean = random.randint(-5,0)
        variance = random.randint(1,3)
        GaussianN = AddGaussianNoise_gray(mean=mean, variance=variance, amplitude=1.0,p=0.4)
        image = np.copy(img3)
        image = GaussianN(image)
        img3[mask>128]=image[mask>128]
    
    
    #img = 255-img
    img3 = np.hstack((img3,B,C))
    img = Image.fromarray(img3)
    return img

def quan_white(image_):
    #image_ = Image.open(path1).convert('L')
    iw,ih = image_.size
    h = ih
    w = iw // 3
    n=0
    img = image_.crop([n, n, w - n, h - n])
    B = image_.crop([w + n, n, w*2 - n, h - n])
    C = image_.crop([w*2 + n, n, w*3 - n, h - n])
    B = np.array(B)
    C = np.array(C)
    C_COPY = np.copy(C)
    #res = transforms.ColorJitter(brightness=(1.5,1.6),contrast=(0.4,1.0))
    
    
    #x = random.random()
    #if(x>0.4):
    #res = transforms.ColorJitter(contrast=(0.2,0.5))#0.4  0.50.2-0.5
    #res2 = transforms.ColorJitter(brightness=(1.1,1.6))#  1.5  1.6  1.1-1.4
    
    #else:
        #res = transforms.ColorJitter(contrast=(0.12,0.2))#0.4  0.5
        #res2 = transforms.ColorJitter(brightness=(1.1,1.3))#  1.5  1.6  1.1-1.4
    
    
    
    #img2 = res(img)
    #img2 = res2(img2)
    img = np.array(img)
    img2 = np.copy(img)
    img2[:] = 255
    img3 = np.copy(img)
    #ssss
    #start_h = 90
    start_h = random.randint(50,80)#50-90
    k = random.randint(20,30)#20 35
    m = random.randint(15,25)#15 29
    #k=30
    #m=20
    for i in range(k):
        #print(i)
        img3[start_h+i,:]=img2[start_h+i,:]*(i/k)+img[start_h+i,:]*((k-i)/k)
    img3[start_h+k:,:]=img2[start_h+k:,:]
    B[start_h+k:,:]=0
    C[start_h+k:,:]=255
    x = random.random()
    if(x>0.8):#0.7
        for i in range(m):
        #print(i)
            img3[start_h:,i]=img3[start_h:,i]*(i/m)+img[start_h:,i]*((m-i)/m)
        B[start_h+k:,:m]=255
        C[start_h+k:,:m]=C_COPY[start_h+k:,:m]
    #img4[70:,k:]=img2[70:,k:]
    #img = img3/2+img4/2
    
    
    
    #img = 255-img
    img3 = np.hstack((img3,B,C))
    img = Image.fromarray(img3)
    return img


def quan2(image_):
    #image_ = Image.open(path1).convert('L')
    iw,ih = image_.size
    h = ih
    w = iw // 2
    n=0
    img = image_.crop([n, n, w - n, h - n])
    B = image_.crop([w + n, n, w*2 - n, h - n])
    B = np.array(B)
    #res = transforms.ColorJitter(brightness=(1.5,1.6),contrast=(0.4,1.0))
    res = transforms.ColorJitter(contrast=(0.5,0.8))
    res2 = transforms.ColorJitter(brightness=(1.1,1.5))
    img2 = res(img)
    img2 = res2(img2)
    img = np.array(img)
    img2 = np.array(img2)
    img3 = np.copy(img)
    #ssss
    #start_h = 90
    start_h = random.randint(50,70)
    k = random.randint(15,25)
    m = random.randint(15,25)
    #k=30
    #m=20
    for i in range(k):
        #print(i)
        img3[start_h+i,:]=img2[start_h+i,:]*(i/k)+img[start_h+i,:]*((k-i)/k)
    img3[start_h+k:,:]=img2[start_h+k:,:]
    #x = random.random()
    #if(x>0.7):
        #for i in range(m):
        #print(i)
            #img3[start_h:,i]=img3[start_h:,i]*(i/m)+img[start_h:,i]*((m-i)/m)
    #img4[70:,k:]=img2[70:,k:]
    #img = img3/2+img4/2
    
    #img = 255-img
    img3 = np.hstack((img3,B))
    img = Image.fromarray(img3)
    return img
def find_min_d(img,h,w,i,j):
    down = -1
    up = -1
    left = -1
    right = -1
    down_sx = 0
    up_sx = 0
    left_sx = 0
    right_sx = 0
    for i_d in range(i,h):
        if img[i_d,j] == 0:
            down += 1
        else:
            down_sx = 1
            break
    for i_u in range(i):
        if img[i-i_u,j] == 0:
            up += 1
        else:
            up_sx =1
            break
    for j_r in range(j,w):
        if img[i,j_r] == 0:
            right += 1
        else:
            right_sx = 1
            break
    for j_l in range(j):
        if img[i,j-j_l] == 0:
            left += 1
        else:
            left_sx = 1
            break
    #if down == -1:
    #    down = 200
    #if up == -1:
    #    up = 200
    #if left == -1:
    #    left = 200
    #if right == -1:
    #    right = 200
    min_v = 200
    if down < min_v and down_sx:
        min_v = down
    #min_v = down
    if up < min_v and up_sx:
        min_v = up
    if left < min_v and left_sx:
        min_v = left
    if right < min_v and right_sx:
        min_v = right
    #print(min_v)
    
    return min_v
def bufen(image_):
    path_b = r'/hdd/file-input/wangb/mask/6193/pycode/mask/datasets/img_process/dd4_result'
    base_name_list=os.listdir(path_b)
    base_name = random.choice(base_name_list)
    base_path = path_b + "/" + base_name
    image_base = Image.open(base_path).convert('L')
    iw,ih = image_base.size
    h = ih
    w = iw // 2
    n=0
    img_b = image_base.crop([n, n, w - n, h - n])
    B_b = image_base.crop([w + n, n, w*2 - n, h - n])
    res = transforms.ColorJitter(brightness=(0.8,1.2),contrast=(0.8,1.2))
    img_b = res(img_b)
    
    bili = [8/8,9/8,10/8,11/8]
    bi = random.choice(bili)
    res = transforms.Resize([int(h*bi),int(w*bi)])
    res_m = transforms.Resize([int(h*bi),int(w*bi)],interpolation=InterpolationMode.NEAREST)
    #cor = transforms.RandomCrop([h,w])
    
    img_b = res(img_b)
    B_b = res_m(B_b)
    i, j, h, w = transforms.RandomCrop.get_params(img_b, (h,w))
    img_b = FT.crop(img_b, i, j, h, w)
    B_b = FT.crop(B_b, i, j, h, w)
    
    
    #翻转
    cro1 = transforms.RandomHorizontalFlip(p=1)
    cro2 = transforms.RandomVerticalFlip(p=1)
    x = random.random()
    if(x>0.5):
        img_b = cro1(img_b)
        B_b = cro1(B_b)
    x = random.random()
    if(x>0.5):
        img_b = cro2(img_b)
        B_b = cro2(B_b)
    
    #img_b = cor(img_b)
    #B_b = cor(B_b)
    #print(image_base)
    img_b = np.array(img_b)
    B_b = np.array(B_b)

    iw,ih = image_.size
    h = ih
    w = iw // 2
    n=0
    img = image_.crop([n, n, w - n, h - n])
    B = image_.crop([w + n, n, w*2 - n, h - n])
    img = np.array(img)
    B = np.array(B)
    
    #img[B_b<128]=img_b[B_b<128]
    #B = B_b
    
    
    img2 = np.copy(img)
    img2[B_b<128]=img_b[B_b<128]
    B_b2 = np.copy(B_b)
    
    rh = 8
    rh_m = 6
    
    for i in range(h):
        for j in range(w):
            if B_b[i,j] == 0:
                d = find_min_d(B_b,h,w,i,j)
                if d > rh:
                    continue
                else:
                    img2[i,j] = img[i,j]*(rh-d)/rh + img_b[i,j]*d/rh
                    if d < rh_m:
                        B_b2[i,j] = 255
    img2 = np.hstack((img2,B_b2))
    img = Image.fromarray(img2)
    return img
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

def block_mask(image):
    path_m = "/hdd/file-input/wangb/mask/6193/pycode/mask/datasets/img_process/dd2_mask_118"
    path_k = "/hdd/file-input/wangb/mask/6193/pycode/mask/datasets/img_process/dd5_mask_118"
    mask_paths = sorted(make_dataset(path_m, 100))
    mask_paths_k = sorted(make_dataset(path_k, 100))
    mask_f = 0
    #x = random.random()
    #if(x>0.8):
        #return image
    iw,ih = image.size
    h = ih
    w = iw // 2
    n=0
    img = image.crop([n, n, w - n, h - n])
    B = image.crop([w + n, n, w*2 - n, h - n])
    mask_dir = random.choice(mask_paths)
    mask_dir_k = random.choice(mask_paths_k)
    res1 = transforms.ColorJitter(contrast=(0.3,0.5))
    res2 = transforms.ColorJitter(brightness=(1.0,1.5))
    x = random.random()
    if(x>0.5):
        mask_f = 1
        mask_dir = mask_dir_k
        res1 = transforms.ColorJitter(contrast=(0.4,0.8))
        res2 = transforms.ColorJitter(brightness=(0.9,1.4))
    mask = Image.open(mask_dir).convert('L')
    x = random.random()
    if(x>0.5):
        mask = np.array(mask)
        mask[:] = 0
        mask = PIL.Image.fromarray(mask)
        random_mask = RandomErasing()
        mask = random_mask(mask)
    #res = transforms.ColorJitter(brightness=(1.0,1.4),contrast=(0.3,0.8))#0.6-1.4#0.3,1.8
    #0.6-1.4#0.3,1.8
    
    #0.6-1.4#0.3,1.8
    cro1 = transforms.RandomHorizontalFlip(p=0.5)
    cro2 = transforms.RandomVerticalFlip(p=0.5)
    cro3 = transforms.RandomRotation([-10,10])
    cro4 = transforms.RandomAffine(0,(0,0.4),(0.8,1.5),shear=45)#0-0.2  0.7-1.3
    mask = cro1(mask)
    mask = cro2(mask)
    if(mask_f == 0):
        x = random.random()
        if(x>0.5):
            mask = cro3(mask)
        mask = cro4(mask)
    img2 = res1(img)
    img2 = res2(img2)
    ima_np = np.array(img)
    B = np.array(B)
    img2_np = np.array(img2)
    mask_np = np.array(mask)
    #mask_np[mask_np>128]=1
    #img2_np = img2_np*mask_np
    #print(ima_np.shape)
    #print(mask_np.shape)
    #print(ima_np[mask_np>128])
    #print(img2_np[mask_np>128])
    ima_np[mask_np>128]=img2_np[mask_np>128]
    ima_np = np.hstack((ima_np, B))
    image_c = PIL.Image.fromarray(ima_np)
    #name2=name.replace(".bmp", "_msk1.bmp")
    #name3=name.replace("_msk1.bmp", ".bmp")
    #path1=os.path.join(root,name)
    #path2=os.path.join(root,name2)
    image_c = image_c.convert('L')
    return image_c  

def block_mask_gary(image):#湿手指
    path_m = "/hdd/file-input/wangb/mask/6193/pycode/mask/datasets/img_process/ssz_mask"
    mask_paths = sorted(make_dataset(path_m, 500))
    mask_f = 0
    #x = random.random()
    #if(x>0.8):
        #return image
    iw,ih = image.size
    h = ih
    w = iw // 2
    n=0
    img = image.crop([n, n, w - n, h - n])
    B = image.crop([w + n, n, w*2 - n, h - n])
    mask_dir = random.choice(mask_paths)
    mask = Image.open(mask_dir).convert('L')
    mask = np.array(mask)
    sct = np.copy(mask)
    sct[:] = random.choice([120,121,122,123,124,125,126,127,128])
    variance = random.randint(1,5)
    GaussianN = AddGaussianNoise_gray(mean=0, variance=variance, amplitude=1.0,p=0.95)
    sct = GaussianN(sct)
    #sct = PIL.Image.fromarray(sct)
    mask = PIL.Image.fromarray(mask)

    #res = transforms.ColorJitter(brightness=(1.0,1.4),contrast=(0.3,0.8))#0.6-1.4#0.3,1.8
    #0.6-1.4#0.3,1.8
    
    #0.6-1.4#0.3,1.8
    cro1 = transforms.RandomHorizontalFlip(p=0.5)
    cro2 = transforms.RandomVerticalFlip(p=0.5)
    mask = cro1(mask)
    mask = cro2(mask)

    ima_np = np.array(img)
    B = np.array(B)
    
    img2_np = sct
    mask_np = np.array(mask)
    #mask_np[mask_np>128]=1
    #img2_np = img2_np*mask_np
    #print(ima_np.shape)
    #print(mask_np.shape)
    #print(ima_np[mask_np>128])
    #print(img2_np[mask_np>128])
    B[mask_np>128] = 0
    ima_np[mask_np>128]=img2_np[mask_np>128]
    ima_np = np.hstack((ima_np, B))
    image_c = PIL.Image.fromarray(ima_np)
    #name2=name.replace(".bmp", "_msk1.bmp")
    #name3=name.replace("_msk1.bmp", ".bmp")
    #path1=os.path.join(root,name)
    #path2=os.path.join(root,name2)
    image_c = image_c.convert('L')
    return image_c  

def cebian_mask(image):
    path_m = "../datasets/6193/process_im/mask_rd"
    #path_k = "/hdd/file-input/wangb/mask/6193/pycode/mask/datasets/img_process/linshi5_mask2"
    mask_paths = sorted(make_dataset(path_m, 50))
    #mask_paths_k = sorted(make_dataset(path_k, 50))
    #mask_f = 0
    #x = random.random()
    #if(x>0.8):
        #return image
    iw,ih = image.size
    h = ih
    w = iw // 3
    n=0
    img = image.crop([n, n, w - n, h - n])
    B = image.crop([w + n, n, w*2 - n, h - n])
    C = image.crop([w*2 + n, n, w*3 - n, h - n])
    #x = random.random()
    #if(x>0.5):
    mask_dir = random.choice(mask_paths)
    #else:
        #mask_dir = random.choice(mask_paths_k)
        #mask_f = 1
    mask = Image.open(mask_dir).convert('L')
    mask = np.array(mask)
    sct = np.copy(mask)
    #x = random.random()
    #if(x>0.5):
    sct[:] = random.randint(200,255)
    
    #else:
        #white_max_index = random.randint(6,26)
        #white_max = random.randint(240,255)
        #cha_l = random.randint(0,4)
        #cha_r = random.randint(0,4)
        #sct[white_max_index,:] = white_max
        #for i in range(white_max_index):
        #    sct[:,white_max_index - i - 1] =  sct[:,white_max_index - i] - cha_l
        #for i in range(white_max_index+1,w):
        #    sct[:,i] =  sct[:,i-1] - cha_r
    
    variance = random.randint(1,5)
    GaussianN = AddGaussianNoise_gray(mean=0, variance=variance, amplitude=1.0,p=0.95)
    sct = GaussianN(sct)
    #sct = PIL.Image.fromarray(sct)
    mask = PIL.Image.fromarray(mask)

    #res = transforms.ColorJitter(brightness=(1.0,1.4),contrast=(0.3,0.8))#0.6-1.4#0.3,1.8
    #0.6-1.4#0.3,1.8
    
    #0.6-1.4#0.3,1.8
    cro1 = transforms.RandomHorizontalFlip(p=0.5)
    cro2 = transforms.RandomVerticalFlip(p=0.5)
    mask = cro1(mask)
    mask = cro2(mask)
    #if(mask_f == 0):
    x = random.random()
    if(x>0.2):
        pad = (0,16,0,16)
        mask = FT.pad(mask, pad, 0)
    
        i, j, h, w = transforms.RandomCrop.get_params(mask, (ih,w))
    
        mask = FT.crop(mask, i, j, h, w)

    ima_np = np.array(img)
    B = np.array(B)
    C = np.array(C)
    
    img2_np = sct
    mask_np = np.array(mask)
    #mask_np[mask_np>128]=1
    #img2_np = img2_np*mask_np
    #print(ima_np.shape)
    #print(mask_np.shape)
    #print(ima_np[mask_np>128])
    #print(img2_np[mask_np>128])
    B[mask_np>128] = 0
    C[mask_np>128] = 255
    ima_np[mask_np>128]=img2_np[mask_np>128]
    ima_np = np.hstack((ima_np, B,C))
    image_c = PIL.Image.fromarray(ima_np)
    #name2=name.replace(".bmp", "_msk1.bmp")
    #name3=name.replace("_msk1.bmp", ".bmp")
    #path1=os.path.join(root,name)
    #path2=os.path.join(root,name2)
    image_c = image_c.convert('L')
    return image_c  
class __RandomColorJitter(transforms.ColorJitter):
    def forward(self, image):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        #w,h = image.size
        x = random.random()
        if(x>0.8):
            return image
        iw,ih = image.size
        w = iw//3
        h = ih
        n = 0
        #image = PIL.Image.fromarray(image)
        
        img = image.crop([n, n, w - n, ih - n])
        B = image.crop([w + n, n, 2*w - n, ih - n])
        C = image.crop([2*w + n, n, 3* - n, ih - n])
        
        x = random.random()
        if(x>0.96):
            B1 = np.array(B)
            if np.sum(B1>128) > (w*h-2):
                image = quan(image)
                return image
        #elif(x>0.94):
        #    B1 = np.array(B)
        #    if np.sum(B1>128) == (w*h-2):
        #        image = block_mask(image)
        #        return image
        
        #elif(x>0.85):
            #B1 = np.array(B)
            #if np.sum(B1>128) == (w*h-2):
                #image = block_mask_gary(image)
                #return image
        
        
        #elif(x>0.96):
            #B1 = np.array(B)
            #if np.sum(B1>128) == (w*h):
                #image = bufen(image)
                #return image
        #elif(x>0.9)
            #B1 = np.array(B)
            #image = noise(image)
            #return image
        elif(x>0.94):
            B1 = np.array(B)
            if np.sum(B1>128) == (w*h):
                image = cebian_mask(image)
                return image
        
        elif(x>0.92):
            B1 = np.array(B)
            if np.sum(B1>128) == (w*h):
                image = quan_white(image)
                #return image
        
        
        
        img = image.crop([n, n, w - n, h - n])
        B = image.crop([w + n, n, w*2 - n, h - n])
        C = image.crop([w*2 + n, n, w*3 - n, h - n])
        
        
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = FT.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = FT.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = FT.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = FT.adjust_hue(img, hue_factor)
        img = np.array(img)
        B = np.array(B)
        C = np.array(C)
        
        
        #添加噪声
        #if np.sum(B<128) > (w*h/16):
        #    mean = random.randint(-100,0)
        #    variance = random.randint(1,10)
        #    GaussianN = AddGaussianNoise_gray(mean=mean, variance=variance, amplitude=1.0,p=0.02)
        #    image = np.copy(img)
        #    img = GaussianN(img)
        #    img[B>128]=image[B>128]
        
        
        image = np.hstack((img, B,C))
        image = PIL.Image.fromarray(image)
        return image



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
        w = iw // 3
        h = ih
        A = image.crop([n, n, w - n, ih - n])
        B = image.crop([w + n, n, 2*w - n, ih - n])
        C = image.crop([w*2 + n, n, 3*w - n, ih - n])
        height = h
        weight = w
        gary_v = 255
        xz_mask = Image.new('L', (weight, height), gary_v)
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
            B = res(B)
            C = res(C)
        #竖直翻转
        x = random.random()
        if(x>0.5):
            res = transforms.RandomVerticalFlip(p=1)
            A = res(A)
            B = res(B)
            C = res(C)
        
        
        #随机放大
        
        x = random.random()
        if(x>0.9):#0.9
            scale = 9/8
            #if(x>0.95):
                #scale = 5/4
            res = transforms.Resize([int(height*scale+0.5),int(weight*scale+0.5)])
            A = res(A)
            cro = transforms.Resize([int(height*scale+0.5),int(weight*scale+0.5)])
            #cro = transforms.Resize([int(height*scale+0.5),int(weight*scale+0.5)],interpolation=InterpolationMode.NEAREST)#最邻近插值
            #Image.BICUBIC，PIL.Image.LANCZOS，PIL.Image.BILINEAR，PIL.Image.NEAREST
            
            B = cro(B)
            C = cro(C)
            i, j, h, w = transforms.RandomCrop.get_params(A, (ih,w))
            A = FT.crop(A, i, j, h, w)
            B = FT.crop(B, i, j, h, w)
            C = FT.crop(C, i, j, h, w)
        
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

        A = np.array(A)
        B = np.array(B)
        C = np.array(C)
        #xz_mask = np.array(xz_mask)
        #if np.sum(B<128) > (weight*height/32):
        #    mean = 0
        #    variance = random.randint(1,10)
        #    GaussianN = AddGaussianNoise_gray(mean=mean, variance=variance, amplitude=1.0,p=0.1)
        #    image = np.copy(A)
        #    A = GaussianN(A)
        #    A[B>128]=image[B>128]   


        image = np.hstack((A, B, C))
        image = PIL.Image.fromarray(image)

        # inverse_phase = random.choice([0, 1])  # 随机反相
        # if inverse_phase:
        #     image = 255 - image

        return image

