# from models.SuperPointNet_small_128 import *
from models.SuperPointNet_small_128_fulldesc import *
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



    #    model_1 = [  # nn.ReflectionPad2d(3),
    #         nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
    #         norm_layer(64),
    #         nn.ReLU(True)]
    #     model_1 += [nn.Conv2d(64, output_nc, kernel_size=1, padding=0)]
    #     # model += [hsigmoid()]



# pthpath = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/superPointNet_100800_checkpoint.pth.tar'
# pthpath = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0425_train_bs128_120pointlabels_6blocks_notnorm_fulldesc/superPointNet_47000_checkpoint.pth.tar'
# pthpath = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0425_train_bs128_120pointlabels_6blocks_notnorm_fulldesc_addcosloss/superPointNet_49800_checkpoint.pth.tar'
pthpath = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0519_train6159_bs64_120pointlabels_6blocks_256desc_selfsup/superPointNet_113200_checkpoint.pth.tar'



mdconv3x3 = ['conv3x3_pre1' , 'conv3x3_pre2' , 'conv3x3_ds1' , 'conv3x3_ds2', 'conv3x3_ds3']

block_se_1 = ['conv1x1_di_1', 'convdw3x3_1', 'conv1x1_dd_1', 'conv1x1_dd_se_1', 'conv1x1_di_se_1']
block_se_2 = ['conv1x1_di_2', 'convdw3x3_2', 'conv1x1_dd_2', 'conv1x1_dd_se_2', 'conv1x1_di_se_2']
block_se_3 = ['conv1x1_di_3', 'convdw3x3_3', 'conv1x1_dd_3', 'conv1x1_dd_se_3', 'conv1x1_di_se_3']
block_se_4 = ['conv1x1_di_4', 'convdw3x3_4', 'conv1x1_dd_4', 'conv1x1_dd_se_4', 'conv1x1_di_se_4']
block_se_5 = ['conv1x1_di_5', 'convdw3x3_5', 'conv1x1_dd_5', 'conv1x1_dd_se_5', 'conv1x1_di_se_5']
block_se_6 = ['conv1x1_di_6', 'convdw3x3_6', 'conv1x1_dd_6', 'conv1x1_dd_se_6', 'conv1x1_di_se_6']


semiconv3x3 = ['conv3x3s1_semi', 'conv3x3s2_semi']
descconv3x3 = ['conv3x3s1_desc', 'conv3x3s2_desc']


block = [mdconv3x3, block_se_1, block_se_2, block_se_3, block_se_4, block_se_5, block_se_6, semiconv3x3, descconv3x3]
name = []
for i in block:
    name.extend(i)



net = SuperPointNet_small_128_fulldesc()
checkpoint = torch.load(pthpath, map_location=lambda storage, loc: storage.cuda(5))    # model_state_dict


net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

# net.load_state_dict(torch.load(pthpath).state_dict)
net.to('cpu')


def list_layers(layer):
    layers = []
    if isinstance(layer, Block) or isinstance(layer, nn.Sequential)or isinstance(layer, SeModule):
        for i in layer.children():
            layers.extend(list_layers(i))
    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d)  or isinstance(layer, nn.ConvTranspose2d):
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

# print(net)

print(30 * "*****")
print(30 * "*****")


layers = get_parameters_layer(net)
count = 0
params_num = 0
res = '#if 1\n'

# for i in range(len(layers)): 
#     print(i, layers[i])


for i in range(len(layers)):
    if i == 77:
        break
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
        print(i, name[count])
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
        print(i, layers[i], count)

print(count)
print("params_num",params_num)
res += '#endif\n'
with open('result_256desc_selfsup_113200.h', 'w') as f:
    f.write(res)
Path(r'param.txt').unlink()
