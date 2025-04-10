"""latest version of SuperpointNet. Use it!

"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from models.unet_parts import *
import numpy as np

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out

# from models.SubpixelNet import SubpixelNet
class UnSuperPointNet_small(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, subpixel_channel=1):
        super(UnSuperPointNet_small, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 1
        model = [  # nn.ReflectionPad2d(3),
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True)]
        model += [  # nn.ReflectionPad2d(3),
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True)]

        model += [nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(8),
                  nn.ReLU(True)]
        model += [nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(16),
                  nn.ReLU(True)]
        model += [nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(32),
                  nn.ReLU(True)]
        for i in range(6):  # add ResNet blocks
            model += [
                Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1)]

        self.model = nn.Sequential(*model)


        # self.relu = torch.nn.ReLU(inplace=True)
        self.nonlinear = hsigmoid()
        # self.outc = outconv(64, n_classes)

        # Detector Head Score.
        self.block_score = Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1)
        # self.bnPa_score = nn.BatchNorm2d(c5)
        self.convPb_score = torch.nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        # self.bnPb_score = nn.BatchNorm2d(1)
        # Detector Head Position.
        self.block_position = Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
                     nolinear=hswish(), semodule=SeModule(int(32)), stride=1)
        # self.bnPa_position = nn.BatchNorm2d(c5)
        self.convPb_position = torch.nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)
        self.bnPb_position = nn.BatchNorm2d(2)

        # Descriptor Head.
        # self.block_descriptor = Block(kernel_size=3, in_size=int(32), expand_size=int(128), out_size=int(64),
        #              nolinear=hswish(), semodule=SeModule(int(64)), stride=1)
        # self.block_descriptor = nn.Sequential(
        #             Block(kernel_size=3, in_size=int(32), expand_size=int(128), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        #             Block(kernel_size=3, in_size=int(32), expand_size=int(128), out_size=int(32),
        #              nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
        # )
        # self.block_descriptor = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True)
        # )

        # self.convPb_descriptor = torch.nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0, bias=False)        
        # self.reshape_descriptor = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)   
        # self.bnDb_descriptor = nn.BatchNorm2d(128)
        self.output = None



    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu
        x4 = self.model(x)  # encoder output

        # Score Head.
        cPa_score = self.block_score(x4)
        # cPa_score = self.bnPa_score(self.convPa_score(x4))
        # semi_score = self.nonlinear(self.bnPb_score(self.convPb_score(cPa_score)))
        semi_score = self.nonlinear(self.convPb_score(cPa_score))

        # Position Head.
        cPa_position = self.block_position(x4)
        # semi_position = self.nonlinear(self.convPb_position(cPa_position))
        semi_position = self.nonlinear(self.bnPb_position(self.convPb_position(cPa_position)))

        semi = torch.cat((semi_score, semi_position), dim=1)

        desc = None
        # Descriptor Head.
        # cPa_descriptor = self.block_descriptor(x4)
        # cPa_descriptor = x4
        # desc = self.convPb_descriptor(cPa_descriptor)
        # desc = self.bnDb_descriptor(desc)
        # desc = self.reshape_descriptor(desc)

        # desc = torch.nn.functional.normalize(input=desc, p=2, dim=1, eps=1e-12)  # -1 - 1
       
        # dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        output = {'semi': semi, 'desc': desc, 'encoder': x4}
        self.output = output

        return output

    def process_output(self, sp_processer):
        """
        input:
          N: number of points
        return: -- type: tensorFloat
          pts: tensor [batch, N, 2] (no grad)  (x, y)
          pts_offset: tensor [batch, N, 2] (grad) (x, y)
          pts_desc: tensor [batch, N, 256] (grad)
        """
        from utils.utils import flattenDetection
        # from models.model_utils import pred_soft_argmax, sample_desc_from_points
        output = self.output
        semi = output['semi']
        desc = output['desc']
        # flatten
        heatmap = flattenDetection(semi) # [batch_size, 1, H, W] decoder，将深度图转为heatmap
        # nms
        heatmap_nms_batch = sp_processer.heatmap_to_nms(heatmap, tensor=True)
        # extract offsets
        outs = sp_processer.pred_soft_argmax(heatmap_nms_batch, heatmap)
        residual = outs['pred']
        # extract points
        outs = sp_processer.batch_extract_features(desc, heatmap_nms_batch, residual)

        # output.update({'heatmap': heatmap, 'heatmap_nms': heatmap_nms, 'descriptors': descriptors})
        output.update(outs)
        self.output = output
        return output


def get_matches(deses_SP):
    from models.model_wrap import PointTracker
    tracker = PointTracker(max_length=2, nn_thresh=1.2)
    f = lambda x: x.cpu().detach().numpy()
    # tracker = PointTracker(max_length=2, nn_thresh=1.2)
    # print("deses_SP[1]: ", deses_SP[1].shape)
    matching_mask = tracker.nn_match_two_way(f(deses_SP[0]).T, f(deses_SP[1]).T, nn_thresh=1.2)
    return matching_mask

    # print("matching_mask: ", matching_mask.shape)
    # f_mask = lambda pts, maks: pts[]
    # pts_m = []
    # pts_m_res = []
    # for i in range(2):
    #     idx = xs_SP[i][matching_mask[i, :].astype(int), :]
    #     res = reses_SP[i][matching_mask[i, :].astype(int), :]
    #     print("idx: ", idx.shape)
    #     print("res: ", idx.shape)
    #     pts_m.append(idx)
    #     pts_m_res.append(res)
    #     pass

    # pts_m = torch.cat((pts_m[0], pts_m[1]), dim=1)
    # matches_test = toNumpy(pts_m)
    # print("pts_m: ", pts_m.shape)

    # pts_m_res = torch.cat((pts_m_res[0], pts_m_res[1]), dim=1)
    # # pts_m_res = toNumpy(pts_m_res)
    # print("pts_m_res: ", pts_m_res.shape)
    # # print("pts_m_res: ", pts_m_res)
        
    # pts_idx_res = torch.cat((pts_m, pts_m_res), dim=1)
    # print("pts_idx_res: ", pts_idx_res.shape)

def main():
    device = torch.device('cuda:%s' % args.gpu_ids)
    model = SuperPointNet_gauss2()
    model = model.to(device)


    # check keras-like model summary using torchsummary
    from torchsummary import summary
    summary(model, input_size=(1, 240, 320))

    ## test
    image = torch.zeros((2,1,120, 160))
    outs = model(image.to(device))
    print("outs: ", list(outs))

    from utils.print_tool import print_dict_attr
    print_dict_attr(outs, 'shape')

    from models.model_utils import SuperPointNet_process 
    params = {
        'out_num_points': 500,
        'patch_size': 5,
        'device': device,
        'nms_dist': 4,
        'conf_thresh': 0.015
    }

    sp_processer = SuperPointNet_process(**params)
    outs = model.process_output(sp_processer)
    print("outs: ", list(outs))
    print_dict_attr(outs, 'shape')

    # timer
    import time
    from tqdm import tqdm
    iter_max = 50

    start = time.time()
    print("Start timer!")
    for i in tqdm(range(iter_max)):
        outs = model(image.to(device))
    end = time.time()
    print("forward only: ", iter_max/(end - start), " iter/s")

    start = time.time()
    print("Start timer!")
    xs_SP, deses_SP, reses_SP = [], [], []
    for i in tqdm(range(iter_max)):
        outs = model(image.to(device))
        outs = model.process_output(sp_processer)
        xs_SP.append(outs['pts_int'].squeeze())
        deses_SP.append(outs['pts_desc'].squeeze())
        reses_SP.append(outs['pts_offset'].squeeze())
    end = time.time()
    print("forward + process output: ", iter_max/(end - start), " iter/s")

    start = time.time()
    print("Start timer!")
    for i in tqdm(range(len(xs_SP))):
        get_matches([deses_SP[i][0], deses_SP[i][1]])
    end = time.time()
    print("nn matches: ", iter_max/(end - start), " iters/s")


if __name__ == '__main__':
    model = UnSuperPointNet_small()
    total = 0
    for name, parameters in model.named_parameters():
        total += parameters.nelement()
        print(name, ":", parameters.size())
    print("Number of parameter: %.5fM" % (total / 1e6))





