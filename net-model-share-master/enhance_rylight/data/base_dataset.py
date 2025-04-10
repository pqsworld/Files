"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import cv2
from data.image_folder import *
import torchvision
from PIL import ImageFilter
# from util.matlab_imresizefloat import *
from util.matlab_imresize import *

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

def random_blurThree(img, outsize=128,thre=6):
    """
        :param img: 输入的图像
        :param pos: 图像截取的位置,类型为元组，包含(x, y)
        :param size: 图像截取的大小
        :return: 返回截取后的图像
    """
    blur_flag = random.randint(1, 10)
    if blur_flag > thre:#6:
        n = 0
        img = PIL.Image.fromarray(img)
        iw, ih = img.size

        sw = int(iw // 3)

        if sw < outsize:

            A = img.crop([n, n, int(iw // 2) - n, ih - n])
            B = img.crop([int(iw // 2) + n, n, 2 * int(iw // 2), ih - n])
            C = np.ones_like(A) * 0
            C = Image.fromarray(C)
            C = C.convert("L")

        else:
            A = img.crop([n, n, sw - n, ih - n])
            B = img.crop([sw + n, n, 2 * sw, ih - n])
            C = img.crop([2 * sw + n, n, 3 * sw, ih - n])

        w, h = A.size

        wi_random = torch.randint(1, outsize // 2, size=(1,)).item()
        hi_random = torch.randint(1, outsize // 2, size=(1,)).item()

        hi = torch.randint(0, h - hi_random + 1, size=(1,)).item()
        wj = torch.randint(0, w - wi_random + 1, size=(1,)).item()

        tmp = A.crop((hi, wj, hi + hi_random, wj + wi_random))
        blur_img = cv2.GaussianBlur(np.uint8(tmp), ksize=(7, 7), sigmaX=random.uniform(0.1, 2))
        blur_img = Image.fromarray(blur_img)
        A.paste(blur_img, (hi, wj, hi + hi_random, wj + wi_random))
        img = np.hstack((A, B, C))
    # print('random_blurThree')
    # print(img.shape)
    return img


def random_paddThree(img, outsize=112):
    """
        :param img: 输入的图像
        :param pos: 图像截取的位置,类型为元组，包含(x, y)
        :param size: 图像截取的大小
        :return: 返回截取后的图像
    """
    blur_flag = random.randint(1, 10)
    if blur_flag > 4:
        n = 0
        img = np.array(img)
        img = PIL.Image.fromarray(img)
        iw, ih = img.size

        sw = int(iw // 3)

        if sw < outsize:

            A = img.crop([n, n, int(iw // 2) - n, ih - n])
            B = img.crop([int(iw // 2) + n, n, 2 * int(iw // 2), ih - n])
            C = np.ones_like(A) * 0
            C = Image.fromarray(C)
            C = C.convert("L")

        else:
            A = img.crop([n, n, sw - n, ih - n])
            B = img.crop([sw + n, n, 2 * sw, ih - n])
            C = img.crop([2 * sw + n, n, 3*sw, ih - n])

        w, h = A.size

        wi_random = torch.randint(2, 22, size=(1,)).item()  # w//4
        hi_random = torch.randint(2, 22, size=(1,)).item()  # h//4

        value = 255  # torch.randint(60, 255, size=(1,)).item()

        imarray = np.array(A)
        flag = random.randint(1, 5)
        ratio = random.randint(1, 10) * 0.05
        if flag == 1:
            imarray[:hi_random, :] = value  # *0.9+imarray[:hi_random, :]*ratio
        elif flag == 2:
            imarray[h - hi_random:, :] = value  # *0.9+imarray[h - hi_random:, :]*ratio
        elif flag == 3:
            imarray[:, :wi_random] = value  # *0.9+imarray[:, :wi_random]*ratio
        else:
            imarray[:, w - wi_random:] = value  # *0.9+imarray[:, w - wi_random:]*ratio
        # k = random.randint(1, 4)
        # imarray = imarray / k * k

        img = np.hstack((imarray, B, C))
    return img


def random_gauss_noiseThree(img, outsize=112,thre=6, mean=0, ratio=0.001):
    """
        :param img: 输入的图像
        :param pos: 图像截取的位置,类型为元组，包含(x, y)
        :param size: 图像截取的大小
        :return: 返回截取后的图像
    """
    blur_flag = random.randint(1, 10)
    ratio = random.randint(1, 3) * 0.0003
    if blur_flag > thre:#6:
        n = 0
        img = PIL.Image.fromarray(img)
        iw, ih = img.size
        sw = int(iw // 3)
        # print(iw)

        if sw < outsize:

            A = img.crop([n, n, int(iw // 2) - n, ih - n])
            B = img.crop([int(iw // 2) + n, n, 2 * int(iw // 2), ih - n])
            C = np.ones_like(A) * 0
            C = Image.fromarray(C)
            C = C.convert("L")

        else:
            A = img.crop([n, n, sw - n, ih - n])
            B = img.crop([sw + n, n, 2 * sw, ih - n])
            C = img.crop([2 * sw + n, n, 3*sw, ih - n])

        w, h = A.size

        wi_random = torch.randint(w // 4, w // 2, size=(1,)).item()
        hi_random = torch.randint(h // 4, h // 2, size=(1,)).item()

        hi = torch.randint(0, h - hi_random + 1, size=(1,)).item()
        wj = torch.randint(0, w - wi_random + 1, size=(1,)).item()

        tmp = A.crop((hi, wj, hi + hi_random, wj + wi_random))

        tmp = np.array(tmp)
        noise = 100 * np.random.normal(mean, ratio ** 0.5, [wi_random, hi_random])  # ratio ** 0.005
        cv2.normalize(noise, noise, 0.3, 0.7, cv2.NORM_MINMAX)
        tmp = tmp + noise
        out_after = np.clip(tmp, 0.0, 255.0)
        out_after = np.uint8(out_after)

        blur_img = Image.fromarray(out_after)
        A.paste(blur_img, (hi, wj, hi + hi_random, wj + wi_random))

        img = np.hstack((A, B, C))
    # print('random_blurThree')
    # print(img.shape)
    return img


# part_paths = sorted(make_dataset('/data/guest/liugq/04_RY/part/00trainM'))
# partmask_paths = sorted(make_dataset('/data/guest/liugq/04_RY/part/00trainM2'))
# lenpart = len(part_paths)

# def random_addpartpress(img, outsize=112):
#     part_flag = random.randint(1, 10)
#     if part_flag >= 6:
#         n = 0
#         img = PIL.Image.fromarray(img)
#         iw, ih = img.size

#         sw = int(iw // 3)

#         if sw < outsize:
#             A = img.crop([n, n, int(iw // 2) - n, ih - n])
#             B = img.crop([int(iw // 2) + n, n, 2 * int(iw // 2), ih - n])
#             C = np.ones_like(A) * 0
#             C = Image.fromarray(C)
#             C = C.convert("L")

#         else:
#             A = img.crop([n, n, sw - n, ih - n])
#             B = img.crop([sw + n, n, 2 * sw, ih - n])
#             C = img.crop([2 * sw + n, n, iw - n, ih - n])

#         w, h = A.size
        
#         id_random = torch.randint(1, lenpart, size=(1,)).item()
#         partimg = Image.open(part_paths[id_random]).convert('L')
#         partimg = partimg.resize((outsize, outsize), Image.BILINEAR)
#         partim = np.array(partimg)
#         Aa = np.array(A)

#         partmask = Image.open(partmask_paths[id_random]).convert('L')
#         partmask = partmask.resize((outsize, outsize), Image.BILINEAR)
#         partmask = np.array(partmask)
#         C = np.array(C)
#         ratio = 0.01 * random.randint(80, 100)
#         tmp = (1-ratio)*Aa + ratio*partim
#         out_after = np.clip(tmp, 0.0, 255.0)
#         out_after = np.uint8(out_after)

#         Ares = np.where(partmask > 220, out_after, Aa)

#         img = np.hstack((Ares, B, C))
#     return img

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
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



def get_transformMask(opt, convert=True):
    transform_list = []
    if convert:
        transform_list.append(transforms.Grayscale(1))
        # transform_list.append(transforms.RandomVerticalFlip(0.5))
        rotate_flag = 1

        if "iftest" not in opt:  # the following transform only takes effect when testing
                transform_list.append(__RandomColorJitterThree(brightness=(0.2, 1.2), contrast=(0.8, 1.2)))

                transform_list.append(transforms.Lambda(lambda img: random_blurThree(img, opt.load_sizeh,thre=6), ), )
               
                if opt.gaussnoise:
                    transform_list.append(
                        transforms.Lambda(lambda img: random_gauss_noiseThree(img, opt.load_sizeh,thre=6), ), )
                if opt.partflag:
                    transform_list.append(
                        transforms.Lambda(lambda img: random_addpartpress(img, opt.load_sizeh), ), )
                if opt.whiteflag:
                    transform_list.append(transforms.Lambda(lambda img: random_paddThree(img, opt.load_sizeh), ), )

                if opt.pro_slim:
                    transform_list.append(transforms.Lambda(lambda img: slim_flatMask(img, opt.load_sizeh), ), )
                
                transform_list.append(angle_RandomRotation_translationThree(degrees=20, outsize=opt.load_sizeh))

        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    return transforms.Compose(transform_list)

import torch
# import torch.nn.functional as F
import PIL
import torchvision.transforms.functional as FT

class __RandomColorJitterThree(transforms.ColorJitter):
    def forward(self, image):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        outsize=128
        if random.randint(0, 10) > 6:
            n = 0
            iw, ih = image.size
            sw = int(iw // 3)
            if sw < outsize:
                A = image.crop([n, n, int(iw // 2) - n, ih - n])
                B = image.crop([int(iw // 2) + n, n, 2 * int(iw // 2), ih - n])
                C = np.ones_like(A) * 0
                C = Image.fromarray(C)
                C = C.convert("L")

            else:
                A = image.crop([n, n, sw - n, ih - n])
                B = image.crop([sw + n, n, 2 * sw - n, ih - n])
                C = image.crop([2 * sw + n, n, 3 * sw, ih - n])
                

            fn_idx = torch.randperm(4)
            for fn_id in fn_idx:
                if fn_id == 0 and self.brightness is not None:
                    brightness = self.brightness
                    brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                    A = FT.adjust_brightness(A, brightness_factor)

                if fn_id == 1 and self.contrast is not None:
                    contrast = self.contrast
                    contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                    A = FT.adjust_contrast(A, contrast_factor)

                if fn_id == 2 and self.saturation is not None:
                    saturation = self.saturation
                    saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                    A = FT.adjust_saturation(A, saturation_factor)

                if fn_id == 3 and self.hue is not None:
                    hue = self.hue
                    hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                    A = FT.adjust_hue(A, hue_factor)
            A = np.array(A)
            B = np.array(B)
            C = np.array(C)
            image = np.hstack((A, B, C))
            # print(img.shape)
            # print(B.shape)
        else:
            image = np.array(image)

        return image

from torchvision.transforms.functional import rotate as Frotate


class angle_RandomRotation_translationThree():
    def __init__(self, degrees=10, outsize=128):
        super(angle_RandomRotation_translationThree)
        self.degree = degrees
        self.outsize = outsize

    def __call__(self, image):
        n = 0
        outsize = self.outsize
        # print(type(image))
        image = np.array(image)
        image = PIL.Image.fromarray(image)

        iw, ih = image.size
        sw = int(iw // 3)
        if sw < self.outsize:

            A = image.crop([n, n, int(iw // 2) - n, ih - n])
            B = image.crop([int(iw // 2) + n, n, 2 * int(iw // 2), ih - n])
            C = np.ones_like(A) * 0
            C = Image.fromarray(C)
            C = C.convert("L")

        else:
            A = image.crop([n, n, sw - n, ih - n])
            B = image.crop([sw + n, n, 2 * sw, ih - n])
            C = image.crop([2 * sw + n, n, 3 * sw, ih - n])
        # A = image.crop([n, n, sw - n, ih - n])
        # B = image.crop([sw + n, n, 2 * sw - n, ih - n])
        # C = image.crop([2 * sw + n, n, iw - n, ih - n])

        A = np.array(A)
        B = np.array(B)
        C = np.array(C)

        if random.choice([0, 1]):
            # if random.choice([0, 1]):
            if random.choice([0, 1]):
                angle = random.choice([0, 90, 180, 270])
            else:
                angle = random.randint(-self.degree, self.degree)

            matRotate = cv2.getRotationMatrix2D((outsize // 2, ih // 2), angle,
                                                1.0)  # mat rotate 1 center 2 angle 3 缩放系数
            mask_color = 255  # 255: white, 0:black
            A = cv2.warpAffine(A, matRotate, (outsize, ih), flags=cv2.INTER_LINEAR, borderValue=mask_color)
            B = cv2.warpAffine(B, matRotate, (outsize, ih), flags=cv2.INTER_LINEAR, borderValue=mask_color)
            C = cv2.warpAffine(C, matRotate, (outsize, ih), flags=cv2.INTER_LINEAR, borderValue=mask_color)

            mask_tmp = np.ones_like(A) * 0
            mask_rotate = cv2.warpAffine(mask_tmp, matRotate, (outsize, ih), flags=cv2.INTER_LINEAR, borderValue=255)
            A = np.where(mask_rotate == 0, A, mask_color)  # mask_color
            B = np.where(mask_rotate == 0, B, mask_color)
            C = np.where(mask_rotate == 0, C, mask_color)

        image = np.hstack((A, B, C))

        return image

def slim_flatMask(img, outsize=128):
    n = 0
    img = PIL.Image.fromarray(img)
    iw, ih = img.size

    sw = int(iw // 3)

    if sw < outsize:

        A = img.crop([n, n, int(iw // 2) - n, ih - n])
        B = img.crop([int(iw // 2) + n, n, 2 * int(iw // 2), ih - n])
        C = np.ones_like(A) * 0
        C = Image.fromarray(C)
        C = C.convert("L")

    else:
        A = img.crop([n, n, sw - n, ih - n])
        B = img.crop([sw + n, n, 2 * sw, ih - n])
        C = img.crop([2 * sw + n, n, 3 * sw, ih - n])

    if random.choice([-1, 0, 1]):
        asw = random.randint(40, outsize // 2)
        asw = 2 * asw
        A = torchvision.transforms.functional.resize(A, [asw, asw])
        B = torchvision.transforms.functional.resize(B, [asw, asw])
        C = torchvision.transforms.functional.resize(C, [asw, asw])

        #
        padd = int((outsize - asw) / 2)
        #
        A = torchvision.transforms.functional.pad(A, padd, 255, "constant")  # 255
        B = torchvision.transforms.functional.pad(B, padd, 255, "constant")
        C = torchvision.transforms.functional.pad(C, padd, 255, "constant")
    image = np.hstack((A, B, C))
    # print(image.shape)
    return image
