from desc.hardnet_model import *
from pathlib import Path
from collections import deque
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import copy

pthpath = r'./logs/0810_correct_16_28_noise20_sift180/checkpoints/superPointNet_12200_desc.pth.tar'

preconv3x3 = ['conv3x3_pre']

block1 = ['conv1x1s1_di_1', 'convdw3x3s1_1', 'conv1x1s1_dd_1',
          'conv1x1s1_dd_se_1', 'conv1x1s1_di_se_1']
block2 = ['conv1x1s1_di_2', 'convdw3x3s2_2', 'conv1x1s1_dd_2',
          'conv1x1s1_dd_se_2', 'conv1x1s1_di_se_2']
block3 = ['conv1x1s1_di_3', 'convdw3x3s2_3', 'conv1x1s1_dd_3',
          'conv1x1s1_dd_se_3', 'conv1x1s1_di_se_3']

postconv1x1 = ['conv1x1_post']

block = [preconv3x3, block1, block2, block3, postconv1x1]

name = []
for i in block:
    name.extend(i)

# net = MNV3_bufen(1, 1, 16, n_blocks=12)

net = HardNet_small()
net.load_state_dict(torch.load(pthpath)['model_state_dict'])
net.eval()

# net.load_state_dict(torch.load(pthpath).state_dict)
net.to('cpu')


def list_layers(layer):
    layers = []
    if isinstance(layer, Block) or isinstance(layer, nn.Sequential) or isinstance(layer, SeModule):
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
params_num = 0
count = 0
res = '#if 1\n'
for i in range(len(layers)):
    conv = None
    if i + 1 < len(layers) and isinstance(layers[i], nn.Conv2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
        conv = nn.utils.fusion.fuse_conv_bn_eval(layers[i], layers[i + 1])
    elif i + 1 < len(layers) and isinstance(layers[i], nn.ConvTranspose2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
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
        params_num += convw.flatten().shape[0]
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

        params_num += convb.flatten().shape[0]
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
res += '//desc_parameters: {:d}\n'.format(params_num)
with open('0811_desc_result.h', 'w') as f:
    f.write(res)
Path(r'param.txt').unlink()

