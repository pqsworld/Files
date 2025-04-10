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
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)

        return output

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

class SpaceToPatch_score(nn.Module):
    def __init__(self, block_size):
        super(SpaceToPatch_score, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        output = output.sum(dim=1).unsqueeze(dim=1)
        ones_tensor = torch.ones_like(output)
        output = torch.where(output > 1, ones_tensor, output)
        return output


def correct_position(input, direction, block_size):  # input:[B,1,H,W]
        correct_tensor = torch.ones_like(input)
        assert direction in [1,2]
        if direction == 1:
            for index in range(input.size(2)):
                correct_tensor[:,:,index,:] = index % block_size
        else:
            for index in range(input.size(3)):
                correct_tensor[:,:,:,index] = index % block_size
        correct_tensor = input + correct_tensor
        output = torch.where(input < 0, correct_tensor, input)
        return output

class SpaceToPatch_position(nn.Module):
    def __init__(self, block_size):
        super(SpaceToPatch_position, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input, direction=1):
        output = correct_position(input,direction,self.block_size)
        output = output.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        ones_tensor = torch.ones_like(output)
        ones_div = torch.where(output > 0, ones_tensor, output)
        output = output.sum(dim=1).unsqueeze(dim=1)
        ones_div = ones_div.sum(dim=1).unsqueeze(dim=1)
        ones_div = ones_div + 1
        output = output / ones_div
        
        return output
