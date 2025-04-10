"""util functions for visualization

"""

import argparse
import time
import csv
import yaml
import os
import logging
from pathlib import Path

import numpy as np
import torch
from utils.utils import inv_warp_image
from tqdm import tqdm

from tensorboardX import SummaryWriter
import cv2
import matplotlib.pyplot as plt


def plot_imgs(imgs, titles=None, cmap='brg', ylabel='', normalize=False, ax=None, dpi=100):
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap]*n
    if ax is None:
        fig, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
        if n == 1:
            ax = [ax]
    else:
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) == len(imgs)
    for i in range(n):
        if imgs[i].shape[-1] == 3:
            imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmap[i]),
                     vmin=None if normalize else 0,
                     vmax=None if normalize else 1)
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    ax[0].set_ylabel(ylabel)
    plt.tight_layout()


# from utils.draw import img_overlap
def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img

def draw_keypoints(img, corners, label_corners=None, color=(0, 255, 0), radius=3, s=3):
    '''

    :param img:
        image:
        numpy [H, W]
    :param corners:
        Points
        numpy [N, 2]
    :param color:
    :param radius:
    :param s:
    :return:
        overlaying image
        numpy [H, W]
    '''
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners).T:
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)

    if label_corners is not None:
        for c in np.stack(label_corners):
            # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
            cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)
            # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    '''绘制网格线'''
    # for i in range(0, 384, 24):
    #     cv2.line(img, (0, i), (383, i), (192, 192, 192), 1, 1)
    #     cv2.line(img, (i, 0), (i, 383), (192, 192, 192), 1, 1)
    
    return img

def draw_keypoints_pair_test(input_img, input_pts, label_pts, color=(0, 255, 0), radius=3, s=3):
    '''

    :param img:
        image:
        numpy [H, W]
    :param corners:
        Points
        numpy [N, 2]
    :param color:
    :param radius:
    :param s:
    :return:
        overlaying image
        numpy [H, W]
    '''
    img = np.hstack((input_img['img_1'], input_img['img_2'])) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(input_pts).T:
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    for c in np.stack(label_pts):
        c[0] += 128
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    flag = (label_pts[:, 1]<128) & (label_pts[:, 0]<128) & (label_pts[:, 0]>0) & (label_pts[:, 1]>0)
    label_pts = label_pts[flag, :]
    for c in np.stack(label_pts):
        c[0] += 128
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)
    
    '''绘制网格线'''
    # for i in range(0, 384, 24):
    #     cv2.line(img, (0, i), (383, i), (192, 192, 192), 1, 1)
    #     cv2.line(img, (i, 0), (i, 383), (192, 192, 192), 1, 1)
    
    return img

def draw_keypoints_pair(input_img, input_pts, label_pts, color=(0, 255, 0), radius=3, s=3):
    '''

    :param img:
        image:
        numpy [H, W]
    :param corners:
        Points
        numpy [N, 2]
    :param color:
    :param radius:
    :param s:
    :return:
        overlaying image
        numpy [H, W]
    '''
    anchor = int(input_img['img_1'].shape[1])
    img = np.hstack((input_img['img_1'], input_img['img_2'])) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(input_pts['pts']):
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)    # 图像坐标系，先列后行
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    for c in np.stack(input_pts['pts_B']):
        c[0] += anchor
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=-1)
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    # flag = (label_pts[:, 1]<anchor) & (label_pts[:, 0]<anchor) & (label_pts[:, 0]>0) & (label_pts[:, 1]>0)
    # label_pts = label_pts[flag, :]
    if label_pts.size == 0:
        return img

    for c in np.stack(label_pts):
        if c.size == 1:
            break
        c[0] += anchor
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 255, 0), thickness=1)
    
    '''绘制网格线'''
    # for i in range(0, 384, 24):
    #     cv2.line(img, (0, i), (383, i), (192, 192, 192), 1, 1)
    #     cv2.line(img, (i, 0), (i, 383), (192, 192, 192), 1, 1)
    
    return img

def draw_keypoints_pair_tradition(input_img, input_pts, label_pts, color=(0, 255, 0), radius=3, s=3):
    '''
    :param img:
        image:
        numpy [H, W]
    :param corners:
        Points
        numpy [N, 2]
    :param color:
    :param radius:
    :param s:
    :return:
        overlaying image
        numpy [H, W]
    '''
    anchor = int(input_img['img_1'].shape[1])
    img = np.hstack((input_img['img_1'], input_img['img_2'])) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(input_pts['tra_pts']):
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)    # 图像坐标系，先列后行
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    for c in np.stack(input_pts['tra_pts_B']):
        c[0] += anchor
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=-1)
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    # flag = (label_pts[:, 1]<anchor) & (label_pts[:, 0]<anchor) & (label_pts[:, 0]>0) & (label_pts[:, 1]>0)
    # label_pts = label_pts[flag, :]
    if label_pts.size == 0:
        return img

    for c in np.stack(label_pts):
        if c.size == 1:
            break
        c[0] += anchor
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 255, 0), thickness=1)
    
    return img


def draw_keypoints_pair_train(input_img, input_pts, label_pts, color=(0, 255, 0), radius=3, s=3, H=None):
    '''

    :param img:
        image:
        numpy [H, W]
    :param corners:
        Points
        numpy [N, 2]
    :param color:
    :param radius:
    :param s:
    :return:
        overlaying image
        numpy [H, W]
    '''
    anchor = int(input_img['img'].shape[1])
    # img = input_img['img'] * 255
    img = np.hstack((input_img['img'], input_img['img_H'])) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(input_pts['pts']):
        if c.size == 1:
            break
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)    # 图像坐标系，先列后行
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    if input_pts['lab'].size == 0:
        return img
        
    # for c in np.stack(input_pts['lab']):
    #     if c.size == 1:
    #         break
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)

    for c in np.stack(input_pts['pts_H']):
        if c.size == 1:
            break
        c[0] += anchor
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=-1)
    # for c in np.stack(input_pts['lab_H']):
    #     if c.size == 1:
    #         break
    #     c[0] += anchor
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)
    for c in np.stack(input_pts['pts_TH']):
        if c.size == 1:
            break
        c[0] += anchor
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 0, 255), thickness=-1)

    image_warp = inv_warp_image(torch.from_numpy(input_img['img']), H)
    image_warp = np.array(image_warp * 255).astype(np.uint8)
    imageB = (input_img['img_H'] * 255).astype(np.uint8)
    b = np.zeros_like(imageB)
    g = image_warp
    r = imageB
    img_match = cv2.merge([b, g, r])
    img = np.hstack((img, img_match))

    return img

def draw_orientation(input_img, input_pts, color=(255, 0, 0), radius=3, s=3,):
    img = np.array(input_img['img']) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    # print(input_pts['pts'], input_pts['angles'])
    for c, angle in zip(np.stack(input_pts['pts']), np.stack(input_pts['angles'])):
        if c.size == 1:
            break
        c_new = np.zeros_like(c)
        # print(np.cos(angle*0.017453292))
        c_new[0] = c[0] + 5*np.cos(angle*0.017453292)
        c_new[1] = c[1] - 5*np.sin(angle*0.017453292)
        # print(c_new, c, np.sin(angle*0.017453292))
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)    # 图像坐标系，先列后行
        # cv2.circle(img, tuple((s * c_new[:2]).astype(int)), radius, color, thickness=-1)
        cv2.line(img, tuple((s * c[:2]).astype(int)),tuple((s * c_new[:2]).astype(int)), (0, 0, 255), 1, shift=0)
    
    return img


def draw_match_pair_train(input_img, input_pts, color=(255, 0, 0), radius=3, s=3,):
    anchor = int(input_img['img'].shape[1])
    # img = input_img['img'] * 255
    img = np.hstack((input_img['img'], input_img['img_H'])) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(input_pts['pts']):
        if c.size == 1:
            break
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)    # 图像坐标系，先列后行
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    if input_pts['lab'].size == 0:
        return img
        
    # for c in np.stack(input_pts['lab']):
    #     if c.size == 1:
    #         break
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)

    for c in np.stack(input_pts['pts_H']):
        if c.size == 1:
            break
        c[0] += anchor
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=-1)
    # for c in np.stack(input_pts['lab_H']):
    #     if c.size == 1:
    #         break
    #     c[0] += anchor
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)
    # for c in np.stack(input_pts['pts_TH']):
    #     if c.size == 1:
    #         break
    #     c[0] += anchor
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 0, 255), thickness=-1)

    for cA, cB in zip(np.stack(input_pts['pts_repeatA']), np.stack(input_pts['pts_repeatB'])):
        # c = c[[1,0,3,2]]
        if c.size == 1:
            break
        cB[0] += anchor
        # cb = np.random.randint(0, 256)
        # cg = np.random.randint(0, 256)
        # cr = np.random.randint(0, 256)
        cv2.circle(img, tuple((s * cA[:2]).astype(int)), radius, (0, 0, 255), -1)
        cv2.circle(img, tuple((s * cB[:2]).astype(int)), radius, (0, 0, 255), -1)
        cv2.line(img, tuple((s * cA[:2]).astype(int)),tuple((s * cB[:2]).astype(int)), (0, 0, 255), 1, shift=0)

    # for cA, cB in zip(np.stack(input_pts['pts_nncandA']), np.stack(input_pts['pts_nncandB'])):
    #     # c = c[[1,0,3,2]]
    #     if c.size == 1:
    #         break
    #     cB[0] += anchor
    #     # cb = np.random.randint(0, 256)
    #     # cg = np.random.randint(0, 256)
    #     # cr = np.random.randint(0, 256)
    #     cv2.circle(img, tuple((s * cA[:2]).astype(int)), radius, (0, 255, 255), -1)
    #     cv2.circle(img, tuple((s * cB[:2]).astype(int)), radius, (0, 255, 255), -1)
    #     cv2.line(img, tuple((s * cA[:2]).astype(int)),tuple((s * cB[:2]).astype(int)), (0, 255, 255), 1, shift=0)

    try:
        assert input_pts['pts_nnA'].shape[0] > 0 and input_pts['pts_nnB'].shape[0] > 0
    except:
        print('Hanming distance is greater than threshold all!')
    else:

        for cA, cB in zip(np.stack(input_pts['pts_nnA']), np.stack(input_pts['pts_nnB'])):
            # c = c[[1,0,3,2]]
            if c.size == 1:
                break
            cB[0] += anchor
            # cb = np.random.randint(0, 256)
            # cg = np.random.randint(0, 256)
            # cr = np.random.randint(0, 256)
            cv2.circle(img, tuple((s * cA[:2]).astype(int)), radius, (0, 255, 0), -1)
            cv2.circle(img, tuple((s * cB[:2]).astype(int)), radius, (0, 255, 0), -1)
            cv2.line(img, tuple((s * cA[:2]).astype(int)),tuple((s * cB[:2]).astype(int)), (0, 255, 0), 1, shift=0)

    return img

def draw_match_pair_degree_train(input_img, input_pts, color=(255, 0, 0), radius=3, s=3,):
    anchor = int(input_img['img'].shape[1])
    # img = input_img['img'] * 255
    img = np.hstack((input_img['img'], input_img['img_H'])) * 255
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    # for c in np.stack(input_pts['pts']):
    #     if c.size == 1:
    #         break
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)    # 图像坐标系，先列后行
    #     cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    img = draw_orientation_degree(img, input_pts['pts'], input_pts['pts_degree'], input_pts['pts_degree_label'], color=(0, 255, 0), radius=radius, s=s)
    
    
    if input_pts['lab'].size == 0:
        return img
        
    # for c in np.stack(input_pts['lab']):
    #     if c.size == 1:
    #         break
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)

    # for c in np.stack(input_pts['pts_H']):
    #     if c.size == 1:
    #         break
    #     c[0] += anchor
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=-1)
    img = draw_orientation_degree(img, input_pts['pts_H'], input_pts['pts_H_degree'], input_pts['pts_H_degree_label'], color=(0, 255, 0), radius=radius, s=s, offset=anchor)


    # for c in np.stack(input_pts['lab_H']):
    #     if c.size == 1:
    #         break
    #     c[0] += anchor
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=1)
    # for c in np.stack(input_pts['pts_TH']):
    #     if c.size == 1:
    #         break
    #     c[0] += anchor
    #     cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 0, 255), thickness=-1)

    # for cA, cB in zip(np.stack(input_pts['pts_repeatA']), np.stack(input_pts['pts_repeatB'])):
    #     # c = c[[1,0,3,2]]
    #     if c.size == 1:
    #         break
    #     cB[0] += anchor
    #     # cb = np.random.randint(0, 256)
    #     # cg = np.random.randint(0, 256)
    #     # cr = np.random.randint(0, 256)
    #     cv2.circle(img, tuple((s * cA[:2]).astype(int)), radius, (0, 0, 255), -1)
    #     cv2.circle(img, tuple((s * cB[:2]).astype(int)), radius, (0, 0, 255), -1)
    #     cv2.line(img, tuple((s * cA[:2]).astype(int)),tuple((s * cB[:2]).astype(int)), (0, 0, 255), 1, shift=0)

    # for cA, cB in zip(np.stack(input_pts['pts_nncandA']), np.stack(input_pts['pts_nncandB'])):
    #     # c = c[[1,0,3,2]]
    #     if c.size == 1:
    #         break
    #     cB[0] += anchor
    #     # cb = np.random.randint(0, 256)
    #     # cg = np.random.randint(0, 256)
    #     # cr = np.random.randint(0, 256)
    #     cv2.circle(img, tuple((s * cA[:2]).astype(int)), radius, (0, 255, 255), -1)
    #     cv2.circle(img, tuple((s * cB[:2]).astype(int)), radius, (0, 255, 255), -1)
    #     cv2.line(img, tuple((s * cA[:2]).astype(int)),tuple((s * cB[:2]).astype(int)), (0, 255, 255), 1, shift=0)

    # try:
    #     assert input_pts['pts_nnA'].shape[0] > 0 and input_pts['pts_nnB'].shape[0] > 0
    # except:
    #     print('Hanming distance is greater than threshold all!')
    # else:

    #     for cA, cB in zip(np.stack(input_pts['pts_nnA']), np.stack(input_pts['pts_nnB'])):
    #         # c = c[[1,0,3,2]]
    #         if c.size == 1:
    #             break
    #         cB[0] += anchor
    #         # cb = np.random.randint(0, 256)
    #         # cg = np.random.randint(0, 256)
    #         # cr = np.random.randint(0, 256)
    #         cv2.circle(img, tuple((s * cA[:2]).astype(int)), radius, (0, 255, 0), -1)
    #         cv2.circle(img, tuple((s * cB[:2]).astype(int)), radius, (0, 255, 0), -1)
    #         cv2.line(img, tuple((s * cA[:2]).astype(int)),tuple((s * cB[:2]).astype(int)), (0, 255, 0), 1, shift=0)

    return img


def draw_orientation_degree(img, corners, degrees, degrees_label, color=(0, 255, 0), radius=3, s=3, offset=0):
    # img = image
    # img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    
    deltaX = np.cos(degrees) * 2
    deltaY = np.sin(degrees) * 2

    corners_new = corners.copy()
    corners_new[:, 0] += offset

    start_XY = corners_new.copy()
    end_XY = corners_new.copy()

    start_XY[:,0] -= deltaX
    start_XY[:,1] += deltaY

    end_XY[:,0] += deltaX
    end_XY[:,1] -= deltaY

    deltaX_label = np.cos(degrees_label) * 2
    deltaY_label = np.sin(degrees_label) * 2

    start_XY_label = corners_new.copy()
    end_XY_label = corners_new.copy()

    start_XY_label[:,0] -= deltaX_label
    start_XY_label[:,1] += deltaY_label

    end_XY_label[:,0] += deltaX_label
    end_XY_label[:,1] -= deltaY_label

    
    for c, d, a, e, f in zip(start_XY, end_XY, corners_new, start_XY_label, end_XY_label):
        if a.size == 1:
            break
        cv2.circle(img, tuple((s * a).astype(int)), radius, (255,0,0), thickness=-1)
        # cv2.line(img,tuple((s * c).astype(int)),tuple((s * d).astype(int)), (0,0,255),1,shift=0)
        cv2.arrowedLine(img,tuple((s * c).astype(int)),tuple((s * d).astype(int)), (0,0,255),2,0,0,0.2)
        cv2.arrowedLine(img,tuple((s * e).astype(int)),tuple((s * f).astype(int)), (0,255,0),2,0,0,0.2)
        # cv2.putText(img, str(degree), tuple((s * c).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (255, 255, 255), 1)
    return img


# def draw_keypoints(img, corners, color=(0, 255, 0), radius=3, s=3):
#     '''

#     :param img:
#         np (H, W)
#     :param corners:
#         np (3, N)
#     :param color:
#     :param radius:
#     :param s:
#     :return:
#     '''
#     img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
#     for c in np.stack(corners).T:
#         # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
#         cv2.circle(img, tuple((s*c[:2]).astype(int)), radius, color, thickness=-1)
#     return img

def draw_matches(rgb1, rgb2, match_pairs, lw = 0.5, color='g', if_fig=True,
                filename='matches.png', show=False):
    '''

    :param rgb1:
        image1
        numpy (H, W)
    :param rgb2:
        image2
        numpy (H, W)
    :param match_pairs:
        numpy (keypoiny1 x, keypoint1 y, keypoint2 x, keypoint 2 y)
    :return:
        None
    '''
    from matplotlib import pyplot as plt

    h1, w1 = rgb1.shape[:2]
    h2, w2 = rgb2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=rgb1.dtype)
    canvas[:h1, :w1] = rgb1[:,:,np.newaxis]
    canvas[:h2, w1:] = rgb2[:,:,np.newaxis]
    # fig = plt.figure(frameon=False)
    if if_fig:
        fig = plt.figure(figsize=(15,5))
    plt.axis("off")
    plt.imshow(canvas, zorder=1)

    xs = match_pairs[:, [0, 2]]
    xs[:, 1] += w1
    ys = match_pairs[:, [1, 3]]

    alpha = 1
    sf = 5
    # lw = 0.5
    # markersize = 1
    markersize = 2

    plt.plot(
        xs.T, ys.T,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        marker='o',
        markersize=markersize,
        fillstyle='none',
        color=color,
        zorder=2,
        # color=[0.0, 0.8, 0.0],
    );
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    print('#Matches = {}'.format(len(match_pairs)))
    if show:
        plt.show()



# from utils.draw import draw_matches_cv
def draw_matches_cv(data):
    keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints1']]
    keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints2']]
    inliers = data['inliers'].astype(bool)
    matches = np.array(data['matches'])[inliers].tolist()
    def to3dim(img):
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        return img
    img1 = to3dim(data['image1'])
    img2 = to3dim(data['image2'])
    img1 = np.concatenate([img1, img1, img1], axis=2)
    img2 = np.concatenate([img2, img2, img2], axis=2)
    return cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                           None, matchColor=(0,255,0), singlePointColor=(0, 0, 255))


def drawBox(points, img, offset=np.array([0,0]), color=(0,255,0)):
#     print("origin", points)
    offset = offset[::-1]
    points = points + offset    
    points = points.astype(int)
    for i in range(len(points)):
        img = img + cv2.line(np.zeros_like(img),tuple(points[-1+i]), tuple(points[i]), color,5)
    return img

