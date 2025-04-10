""" JIT scripting/tracing utils

Hacked together by / Copyright 2020 Ross Wightman
"""
from __future__ import division
import os
import glob
import matplotlib as plt
import cv2
from tqdm import tqdm
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
import skimage.io
from skimage.morphology import disk
from skimage.filters import rank
from scipy.ndimage import gaussian_filter
from skimage.morphology import reconstruction
import math
import cv2
import sklearn.linear_model

# path_old = r"/home/panq/dataset/spoof/6193/train-q2r0csp-0817-qua2"
# path_new = r"/home/panq/dataset/spoof/6193/train-q2r0csp-0817-qua2-lce"
# path_new_g = r"/home/panq/dataset/spoof/6193/train-q2r0csp-0817-qua2-lceg"
path_old = r"/home/panq/dataset/spoof/6193/data/test-10-qua"
path_new_1 = r"/home/panq/dataset/spoof/6193/data/test-10-qua-1lce"
path_new_2 = r"/home/panq/dataset/spoof/6193/data/test-10-qua-2lceg"
path_new_3 = r"/home/panq/dataset/spoof/6193/data/test-10-qua-3stft"
import past


def local_constrast_enhancement(img):
    """
    局部对比度增强
    """
    h, w = img.shape
    img = img.astype(np.float32)

    meanV = cv2.blur(img, (15, 15))
    normalized = img - meanV
    var = abs(normalized)

    var = cv2.blur(var, (15, 15))

    normalized = normalized / (var + 10) * 0.75
    normalized = np.clip(normalized, -1, 1)
    normalized = (normalized + 1) * 127.5
    return normalized


def local_constrast_enhancement_gaussian(img, sigma=15):
    """
    局部对比度增强(高斯)
    """
    h, w = img.shape
    img = img.astype(np.float32)

    meanV = cv2.GaussianBlur(img, (sigma, sigma), 0)
    normalized = img - meanV
    var = abs(normalized)
    var = cv2.GaussianBlur(var, (sigma, sigma), 0)

    normalized = normalized / (var + 10) * 0.75
    normalized = np.clip(normalized, -1, 1)
    normalized = (normalized + 1) * 127.5
    return normalized


def STFT(img, R=100):
    patch_size = 64
    block_size = 16
    ovp_size = (patch_size - block_size) // 2
    h0, w0 = img.shape
    img = cv2.copyMakeBorder(
        img, ovp_size, ovp_size, ovp_size, ovp_size, cv2.BORDER_CONSTANT, value=0
    )
    h, w = img.shape
    blkH = (h - patch_size) // block_size
    blkW = (w - patch_size) // block_size

    # -------------------------
    # Bandpass filter
    # -------------------------
    RMIN = 3  # min allowable ridge spacing
    RMAX = 18  # maximum allowable ridge spacing
    FLOW = patch_size / RMAX
    FHIGH = patch_size / RMIN
    patch_size2 = int(patch_size / 2)
    x, y = np.meshgrid(
        range(-patch_size2, patch_size2), range(-patch_size2, patch_size2)
    )
    r = np.sqrt(x * x + y * y) + 0.0001

    dRLow = 1.0 / (1 + (r / FHIGH) ** 4)  # low pass     butterworth     filter
    dRHigh = 1.0 / (1 + (FLOW / r) ** 4)  # high    pass     butterworth     filter
    dBPass = dRLow * dRHigh  # bandpass

    sigma = patch_size / 3
    weight = np.exp(-(x * x + y * y) / (sigma * sigma))
    rec_img = np.zeros((h, w))
    for i in range(0, blkH):
        for j in range(0, blkW):
            patch = img[
                i * block_size : i * block_size + patch_size,
                j * block_size : j * block_size + patch_size,
            ].copy()
            patch = patch - np.median(patch)
            f = np.fft.fft2(patch)
            fshift = np.fft.fftshift(f)

            filtered = dBPass * fshift
            norm = np.linalg.norm(filtered)
            filtered = filtered / (norm + 0.0001)
            f_ifft = np.fft.ifftshift(filtered)
            rec_patch = np.real(np.fft.ifft2(f_ifft))
            rec_img[
                i * block_size : i * block_size + patch_size,
                j * block_size : j * block_size + patch_size,
            ] += (
                rec_patch * weight
            )

    rec_img = rec_img[ovp_size : ovp_size + h0, ovp_size : ovp_size + w0]
    img = (rec_img - np.median(rec_img)) / (np.std(rec_img) + 0.000001)
    img = img * 14 + 127
    img[img < 0] = 0
    img[img > 255] = 255
    rec_img = (rec_img - np.min(rec_img)) / (np.max(rec_img) - np.min(rec_img) + 0.0001)

    return img


def list_bmp_directory(dir_imagefolder, type="*.bmp"):
    """
    递归取得文件夹下所有匹配路径
    Args:
        dir_imagefolder (_type_): 文件夹路径
        type (str, optional): 匹配字符串.   Defaults to "*.bmp".
    Returns:
        _type_: list
    """
    list_image_dir = glob.glob(
        os.path.join(dir_imagefolder, "**", type), recursive=True
    )
    return list_image_dir


def handle_bmps(
    list_src,type
):
    """对两个路径list的图像进行基于权值的图像混合

    Args:
        list_src_1 (_type_):
        list_src_2 (_type_):
        name_folder_save (str, optional):     若保存，保存路径. Defaults to "".
        alpha (float, optional):     addWeighted的加权参数. Defaults to 0.5.
    """
    for i in tqdm(range(len(list_src)), position=0, ncols=80):
        img_src = cv2.imread(list_src[i], cv2.IMREAD_GRAYSCALE)

        if type == 1:
            img_dst = STFT(img_src)
            path_save = list_src[i].replace(path_old, path_new_1)
        if type == 2:
            img_dst = local_constrast_enhancement(img_src)
            path_save = list_src[i].replace(path_old, path_new_2)
        if type == 3:
            img_dst = local_constrast_enhancement_gaussian(img_src)
            path_save = list_src[i].replace(path_old, path_new_3)

        if not os.path.exists(path_save):
            if not os.path.exists(os.path.dirname(path_save)):
                os.makedirs(os.path.dirname(path_save))
        cv2.imwrite(path_save, img_dst)
        # if not os.path.exists(path_save):
        #     if not os.path.exists(os.path.dirname(path_save)):
        #         os.makedirs(os.path.dirname(path_save))
        #     cv2.imwrite(path_save, img_dst)
        #     print("Num  " + str(i))
        # else:
        #     print("SKIP " + str(path_save)
        # input()
    print("works done! handle bmps {}".format(len(list_src)))


def main():
    list_src = list_bmp_directory(path_old, "*.bmp")
    handle_bmps(list_src,1)
    handle_bmps(list_src,2)
    handle_bmps(list_src,3)


if __name__ == "__main__":
    main()
