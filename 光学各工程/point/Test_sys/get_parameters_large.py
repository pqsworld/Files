from models.SuperPointNet_large import *
from pathlib import Path
from collections import deque, OrderedDict
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import copy
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# pthpath = 'logs/7_30_magicpoint_finally_Focal_35/checkpoints/superPointNet_18000_checkpoint.pth.tar'
pthpath = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/6159_large/superPointNet_33000_checkpoint.pth.tar'


inconv3x3 = ['conv3x3s1_in1', 'conv3x3s1_in2']

dsconv3x3_d1 = ['conv3x3s1_di_1', 'conv3x3s1_1']   # down sample 1          dw:depthwise
dsconv3x3_d2 = ['conv3x3s1_di_2', 'conv3x3s1_2']   # 2
dsconv3x3_d3 = ['conv3x3s1_di_3', 'conv3x3s1_3']   # 3

semiconv3x3 = ['conv3x3s1_semi', 'conv1x1s1_semi']
# positionconv3x3 = ['conv3x3s1_di_pos', 'conv1x1s1_pos']


block = [inconv3x3, dsconv3x3_d1, dsconv3x3_d2, dsconv3x3_d3, semiconv3x3]
name = []
for i in block:
    name.extend(i)


net = SuperPointNet_large()
checkpoint = torch.load(pthpath, map_location=lambda storage, loc: storage.cuda(5))    # model_state_dict
# checkpoint = torch.load(pthpath)

# ''' remove "model_" '''
# new_checkpoint = OrderedDict()
# for k, v in checkpoint.items():
#     new_checkpoint[k.replace('model_','')] = v

net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

# net.load_state_dict(torch.load(pthpath).state_dict)
net.to('cpu')

def list_layers(layer):
    layers = []
    if isinstance(layer, (inconv, down, double_conv)) or isinstance(layer, nn.Sequential) or isinstance(layer, SeModule):
        for i in layer.children():
            layers.extend(list_layers(i))
    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.ConvTranspose2d):
        layers.append(layer)
    return layers


def get_parameters_layer(net):
    layers = []
    queue = deque()
    for i in net.children():
        queue.append(i)
    while len(queue):
        root = queue.popleft()
        layers.extend(list_layers(root))
    return layers


print(net)
layers = get_parameters_layer(net)
count = 0
res = '#if 1\n'
for i in range(len(layers)):
    conv = None
    if i + 1 < len(layers) and isinstance(layers[i], nn.Conv2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
        conv = nn.utils.fusion.fuse_conv_bn_eval(layers[i], layers[i + 1])
    elif i + 1 < len(layers) and isinstance(layers[i], nn.ConvTranspose2d) and isinstance(layers[i + 1],
                                                                                          nn.BatchNorm2d):
        # print(layers[i].weight.size())
        # print(layers[i].weight.transpose(0,1)[0,2,:,:])
        fused_deconv = copy.deepcopy(layers[i])
        fused_deconv.weight = torch.nn.Parameter(torch.transpose(layers[i].weight, 0, 1))
        # print(fused_deconv.weight.size())
        # print(fused_deconv.weight[0,2,:,:])
        # exit()
        conv = nn.utils.fusion.fuse_conv_bn_eval(fused_deconv, layers[i + 1])
    elif isinstance(layers[i], nn.Conv2d):
        conv = layers[i]
    if conv is not None:
        convw, convb = conv.weight.detach().numpy(
        ).flatten(), conv.bias.detach().numpy().flatten()
        res += 'static float ' + \
               name[count] + \
               '_weight[{0}] = '.format(
                   convw.flatten().shape[0]) + '{ \n'
        if convw.shape[0] % 8 == 0:
            np.savetxt('param.txt', convw.reshape(-1, 8), fmt='%1.10f',
                       delimiter=', ', newline=',\n')
        else:
            np.savetxt('param.txt', convw, fmt='%1.10f',
                       delimiter=', ', newline=',\n')
        with open('param.txt', 'r') as f:
            res += f.read()
        res += '};\n\n'

        res += 'static float ' + \
               name[count] + \
               '_bias[{0}] = '.format(
                   convb.flatten().shape[0]) + '{ \n'
        if convb.shape[0] % 8 == 0:
            np.savetxt('param.txt', convb.reshape(-1, 8), fmt='%1.10f',
                       delimiter=', ', newline=',\n')
        elif convb.shape[0] % 4 == 0:
            np.savetxt('param.txt', convb.reshape(-1, 4), fmt='%1.10f',
                       delimiter=', ', newline=',\n')
        else:
            np.savetxt('param.txt', convb, fmt='%1.10f',
                       delimiter=', ', newline=',\n')
        with open('param.txt', 'r') as f:
            res += f.read()
        res += '};\n\n'
        count += 1
print(count)
res += '#endif\n'
with open('result_test.h', 'w') as f:
    f.write(res)
Path(r'param.txt').unlink()
