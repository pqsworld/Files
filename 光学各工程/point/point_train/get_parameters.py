from models.SuperPointNet_small_128 import *
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

pthpath = 'logs/1202_super/1202_train_/checkpoints/superPointNet_200000_checkpoint.pth.tar'

inconv3x3 = ['conv3x3s1_di_1' , 'conv3x3s1_2' , 'conv3x3s2_ds_1' , 'conv3x3s2_ds_2', 'conv3x3s2_ds_3']

block_se_1 = ['conv1x1s1_1_1', 'conv3x3s1_1_2', 'conv1x1s1_1_3', 'conv1x1s1_1_se_1', 'conv1x1s1_1_se_2']
block_se_2 = ['conv1x1s1_2_1', 'conv3x3s1_2_2', 'conv1x1s1_2_3', 'conv1x1s1_2_se_1', 'conv1x1s1_2_se_2']
block_se_3 = ['conv1x1s1_3_1', 'conv3x3s1_3_2', 'conv1x1s1_3_3', 'conv1x1s1_3_se_1', 'conv1x1s1_3_se_2']
block_se_4 = ['conv1x1s1_4_1', 'conv3x3s1_4_2', 'conv1x1s1_4_3', 'conv1x1s1_4_se_1', 'conv1x1s1_4_se_2']
block_se_5 = ['conv1x1s1_5_1', 'conv3x3s1_5_2', 'conv1x1s1_5_3', 'conv1x1s1_5_se_1', 'conv1x1s1_5_se_2']
block_se_6 = ['conv1x1s1_6_1', 'conv3x3s1_6_2', 'conv1x1s1_6_3', 'conv1x1s1_6_se_1', 'conv1x1s1_6_se_2']
# block_se_7 = ['conv1x1s1_7_1', 'conv3x3s1_7_2', 'conv1x1s1_7_3', 'conv1x1s1_7_se_1', 'conv1x1s1_7_se_2']
# block_se_8 = ['conv1x1s1_8_1', 'conv3x3s1_8_2', 'conv1x1s1_8_3', 'conv1x1s1_8_se_1', 'conv1x1s1_8_se_2']

detectorconv3x3 = ['conv3x3s1_detec', 'conv1x1s1_detec']
# descriptorconv3x3 = ['']


block = [
        inconv3x3, \
        block_se_1, block_se_2, block_se_3, \
        block_se_4, block_se_5, block_se_6, \
        detectorconv3x3
        ]
name = []
for i in block:
    name.extend(i)


net = SuperPointNet_small_128()
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
    if isinstance(layer, (Block)) or isinstance(layer, nn.Sequential) or isinstance(layer, SeModule):
        for i in layer.children():
            layers.extend(list_layers(i))
    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.ConvTranspose2d):
        layers.append(layer)
    return layers


def get_parameters_layer(net):
    layers = []
    queue = deque()
    for i in net.children():
        '''去掉descriptor部分，暂时不用'''
        if i == net.model_descriptor:
            break
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
with open('result.h', 'w') as f:
    f.write(res)
Path(r'param.txt').unlink()

print('# net parameters:', sum(param.numel() for param in net.parameters()))