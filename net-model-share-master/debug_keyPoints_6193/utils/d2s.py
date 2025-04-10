'''
Author: your name
Date: 2021-06-23 05:33:49
LastEditTime: 2021-07-07 02:05:09
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /superpoint/utils/d2s.py
'''
"""Module used to change 2D labels to 3D labels and vise versa.
Mimic function from tensorflow.

"""

import torch
import torch.nn as nn
class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)    # [64,16,16,64,1]
        spl = t_1.split(self.block_size, 3)     # 在第3维分割成每块包含block_size   len(spl) : 8     spl[0].shape : [64,16,16,8,1]
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]    # stack is a list, stack[0].shape : [64,16,128,1]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)     # [64,128,128,1]
        output = output.permute(0, 3, 1, 2)     # [64,1,128,128]

        return output

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size        # 8
        self.block_size_sq = block_size*block_size

    def forward(self, input):   # 将labels标签(128*128大小的尺寸)转换为特征图的尺寸(16*16*64)
        output = input.permute(0, 2, 3, 1)      # 将tensor的维度换位; [64,1,128,128] => [64,128,128,1]
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)  # split_size=block_size,dim=2，沿第2轴进行拆分,每个划分大小为block_size
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]     # len(stack) : 16  , stack[0].shape : [64,16,64]
        output = torch.stack(stack, 1)          # [64,16,16,64]
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)     # [64,64,16,16]
        return output
