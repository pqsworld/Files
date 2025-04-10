from __future__ import print_function
import numpy as np
from math import ceil, floor
from PIL import Image
import os

def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale

# bilinear kernel
def triangle(x):
    x = np.array(x).astype(np.float64)
    # 逻辑与
    lessthanzero = np.logical_and((x >= -1), x < 0)  # -1~0
    greaterthanzero = np.logical_and((x <= 1), x >= 0) # 0~1
    f = np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)
    return f


def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + np.multiply(-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2,
                                                                            (1 < absx) & (absx <= 2))
    return f


def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = np.sum(weights * ((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = np.sum(weights * ((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out


def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method == 'bicubic':
        kernel = cubic
    elif method == 'bilinear':
        kernel = triangle
    else:
        raise ValueError('unidentified kernel method supplied')

    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None and output_shape is not None:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    # scale与缩放顺序（缩放比例小的dim先）
    scale_np = np.array(scale)
    order = np.argsort(scale_np)   # 元素从小到大排序的索引
    weights = []
    indices = []
    # 权重确定
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I)
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B


def convertDouble2Byte(I):
    B = np.clip(I, 0.0, 1.0)
    B = 255 * B
    return np.around(B).astype(np.uint8)

if __name__=='__main__':
    imgroot = '/home/zhangsn/enhance/datasets/7015sp/00'
    for root, dirs, files in os.walk(imgroot):
        for file in files:
            imgpath = os.path.join(root, file)
            img = Image.open(imgpath)#.convert('L')
            imgout = imresize(np.array(img), method='bilinear', output_shape=(128,128), mode="vec")
            imgout = Image.fromarray(imgout)
            
        
        # imgsrc = np.array(A_img, dtype=np.float32)
        # Bimg = imresize(imgsrc, method='bilinear', output_shape=[128, 128], mode="vec")
        # Bimg = (Bimg - 127.5) / 127.5

        # # img0 = transform(imgA)
        # img0 = torch.from_numpy(Bimg)
        # img0 = img0.float()


        # # img0 =img0.cuda()
        # # print(img0.shape)
        # img = torch.unsqueeze(img0, 0)
        # # img = torch.unsqueeze(img, 0)

        # A = img
        # B = img