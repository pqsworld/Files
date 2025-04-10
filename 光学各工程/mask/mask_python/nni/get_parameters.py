from models.networks import *
from pathlib import Path
from collections import deque
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import copy

pthpath = r'./checkpoints/fast_bottleneck6_d8_g8_F_pix2pix/499_net_G.pth'

preconv3x3_1 = ['conv3x3_pre1']
preconv3x3_2 = ['conv3x3_pre2']

dsconv3x3_1 = ['conv3x3_ds1']
dsconv3x3_2 = ['conv3x3_ds2']

block1 = ['conv1x1s1_di_1', 'convdw3x3s2_1', 'conv1x1s1_dd_1',
          'conv1x1s1_dd_se_1', 'conv1x1s1_di_se_1']
block2 = ['conv1x1s1_di_2', 'convdw3x3s1_2', 'conv1x1s1_dd_2',
          'conv1x1s1_dd_se_2', 'conv1x1s1_di_se_2']
block3 = ['conv1x1s1_di_3', 'convdw3x3s2_3', 'conv1x1s1_dd_3',
          'conv1x1s1_dd_se_3', 'conv1x1s1_di_se_3']
block4 = ['conv1x1s1_di_4', 'convdw3x3s1_4', 'conv1x1s1_dd_4',
          'conv1x1s1_dd_se_4', 'conv1x1s1_di_se_4']
block5 = ['conv1x1s1_di_5', 'convdw3x3s2_5', 'conv1x1s1_dd_5',
          'conv1x1s1_dd_se_5', 'conv1x1s1_di_se_5']
block6 = ['conv1x1s1_di_6', 'convdw3x3s1_6', 'conv1x1s1_dd_6',
          'conv1x1s1_dd_se_6', 'conv1x1s1_di_se_6']
block7 = ['conv1x1s1_di_7', 'convdw3x3s1_7', 'conv1x1s1_dd_7',
          'conv1x1s1_dd_se_7', 'conv1x1s1_di_se_7']
block8 = ['conv1x1s1_di_8', 'convdw3x3s1_8', 'conv1x1s1_dd_8',
          'conv1x1s1_dd_se_8', 'conv1x1s1_di_se_8']
block9 = ['conv1x1s1_di_9', 'convdw3x3s1_9', 'conv1x1s1_dd_9',
          'conv1x1s1_dd_se_9', 'conv1x1s1_di_se_9']

usdecon3x3_1 = ['deconv3x3_us1']
usdecon3x3_2 = ['deconv3x3_us2']

postconv3x3 = ['conv3x3_post']

generateconv3x3 = ['conv3x3_generate']

block = [preconv3x3_1, preconv3x3_2, dsconv3x3_1, dsconv3x3_2, block1, block2, block3, block4, block5,
         block6, block7, block8, block9, usdecon3x3_1, usdecon3x3_2, postconv3x3, generateconv3x3]
name = []
for i in block:
    name.extend(i)

net = define_G(1, 1, 4, 'self')

net.load_state_dict(torch.load(pthpath))
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


layers = get_parameters_layer(net)
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
