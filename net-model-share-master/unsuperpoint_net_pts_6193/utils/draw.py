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

def draw_keypoints(img, corners, color=(0, 255, 0), radius=1, s=1):
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
    return img

def draw_keypoints_AB(img, corners, corners_T, color=(0, 255, 0), radius=1, s=1, thick=1):
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
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0,255,0), thickness=-1)
    for c in np.stack(corners_T).T:
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0,0,255), thickness=thick)
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    return img

def draw_keypoints_match(data, color=(0, 255, 0), radius=3, s=3, show_label=False):
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
    img = np.hstack((data['image1'],data['image2']))
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    (img_h, img_w) = data['image1'].shape

    for c in np.stack(data['keypoints1']):
        # c = c[[1,0]]
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 255, 0), -1)
    for c in np.stack(data['keypoints2']):
        # c = c[[1,0]]
        c[0] += img_w
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 255, 0), -1)
    
    if not show_label:
        if data['matches'].size != 0:
            for c in np.stack(data['matches']):
                # c = c[[1,0,3,2]]
                c[2] += img_w

                cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 255, 0), -1)
                cv2.circle(img, tuple((s * c[2:4]).astype(int)), radius, (0, 255, 0), -1)
                cv2.line(img,tuple((s * c[:2]).astype(int)),tuple((s * c[2:4]).astype(int)), (0, 255, 0),1,shift=0)
        
        if data['unmatches'].size != 0:
            for c in np.stack(data['unmatches']):
                # c = c[[1,0,3,2]]
                c[2] += img_w
                cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 0, 255), -1)
                cv2.circle(img, tuple((s * c[2:4]).astype(int)), radius, (0, 0, 255), -1)
                cv2.line(img,tuple((s * c[:2]).astype(int)),tuple((s * c[2:4]).astype(int)), (0, 0, 255),1,shift=0)

    else:
        if data['matches'].size != 0:
            for c in np.stack(data['matches']):
                # c = c[[1,0,3,2]]
                c[2] += img_w
                ca = np.random.randint(0,256)
                cb = np.random.randint(0,256)
                cc = np.random.randint(0,256)
                cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (ca, cb, cc), -1)
                cv2.circle(img, tuple((s * c[2:4]).astype(int)), radius, (ca, cb, cc), -1)
                cv2.line(img,tuple((s * c[:2]).astype(int)),tuple((s * c[2:4]).astype(int)), (ca, cb, cc),1,shift=0)

            
    return img


def draw_orientation_degree(image, corners, degrees, color=(0, 255, 0), radius=3, s=3):
    img = image
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    
    deltaX = np.cos(np.radians(degrees.cpu().numpy())) * 2
    deltaY = np.sin(np.radians(degrees.cpu().numpy())) * 2

    start_XY = corners.copy()
    end_XY = corners.copy()

    start_XY[:,0] -= deltaX
    start_XY[:,1] += deltaY

    end_XY[:,0] += deltaX
    end_XY[:,1] -= deltaY

    
    for c, d, a in zip(start_XY, end_XY, corners):
        cv2.circle(img, tuple((s * a).astype(int)), radius, (255,0,0), thickness=-1)
        # cv2.line(img,tuple((s * c).astype(int)),tuple((s * d).astype(int)), (0,0,255),1,shift=0)
        cv2.arrowedLine(img,tuple((s * c).astype(int)),tuple((s * d).astype(int)), (0,0,255),2,0,0,0.2)
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

def draw_matches(data, lw = 0.5, color='g', if_fig=True,
                filename='matches.png', show=False):
    '''

    :param rgb1:
        image1
        numpy (H, W)
    :param rgb2:
        image2
        numpy (H, W)
    :param match_pairs:
        numpy (keypoint1 x, keypoint1 y, keypoint2 x, keypoint 2 y)
    :return:
        None
    '''
    from matplotlib import pyplot as plt
    rgb1 = data['image1']
    rgb2 = data['image2']
    
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
    
    inliers = data['inliers'].astype(bool)
    match_pairs = np.array(data['matches'])[inliers]
    
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
    )
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

