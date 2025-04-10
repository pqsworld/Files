from desc.hardnet_model import *
from pathlib import Path
from collections import deque
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import copy
from desc.hardnet_model_short import HardNet_fast_short

# pthpath = r'./logs/0810_correct_16_28_noise20_sift180/checkpoints/superPointNet_12200_desc.pth.tar'

# preconv3x3 = ['conv3x3_pre']

# block1 = ['conv1x1s1_di_1', 'convdw3x3s1_1', 'conv1x1s1_dd_1',
#           'conv1x1s1_dd_se_1', 'conv1x1s1_di_se_1']
# block2 = ['conv1x1s1_di_2', 'convdw3x3s2_2', 'conv1x1s1_dd_2',
#           'conv1x1s1_dd_se_2', 'conv1x1s1_di_se_2']
# block3 = ['conv1x1s1_di_3', 'convdw3x3s2_3', 'conv1x1s1_dd_3',
#           'conv1x1s1_dd_se_3', 'conv1x1s1_di_se_3']

# postconv1x1 = ['conv1x1_post']

# block = [preconv3x3, block1, block2, block3, postconv1x1]

pthpath = r'/data/yey/work/6191/unsuperpoint/logs/0324_correct_noise0_sift180_14c2_t20-5_93_nonorm_144_mask_e4800_nomatch_extend_ori5dist_cos_wbmask_addFA_sample22cat32_siftori_256_9800/checkpoints/superPointNet_402800_desc.pth.tar'

# preconv3x3 = ['conv3x3_pre']

block1 = ['conv1x1s1_di_1', 'convdw3x3s2_1', 'conv1x1s1_dd_1',
          'conv1x1s1_dd_se_1', 'conv1x1s1_di_se_1']
block2 = ['conv1x1s1_di_2', 'convdw3x3s2_2', 'conv1x1s1_dd_2',
          'conv1x1s1_dd_se_2', 'conv1x1s1_di_se_2']
block3 = ['conv1x1s1_di_3', 'convdw3x3s1_3', 'conv1x1s1_dd_3',
          'conv1x1s1_dd_se_3', 'conv1x1s1_di_se_3']
block4 = ['conv1x1s1_di_4', 'convdw3x3s1_4', 'conv1x1s1_dd_4',
          'conv1x1s1_dd_se_4', 'conv1x1s1_di_se_4']
block5 = ['conv1x1s1_di_5', 'convdw3x3s1_5', 'conv1x1s1_dd_5',
          'conv1x1s1_dd_se_5', 'conv1x1s1_di_se_5']
block6 = ['conv1x1s1_di_6', 'convdw3x3s1_6', 'conv1x1s1_dd_6',
          'conv1x1s1_dd_se_6', 'conv1x1s1_di_se_6']
          
postconv1x1 = ['conv1x1_post']

block = [block1, block2, block3, block4, block5, block6, postconv1x1]

name = []
prefix = "para_patch_"
for i in block:
    name.extend(i)
for i in range(len(name)):
    name[i] = prefix + name[i]

    


net = HardNet_fast()
net.load_state_dict(torch.load(pthpath)['model_state_dict'])
net.eval()

# net.load_state_dict(torch.load(pthpath).state_dict)
net.to('cpu')

net_short = HardNet_fast_short()
net_short.eval()

net_short.to('cpu')


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
params_short_num = 0
count = 0
conw_short_list = []
conb_short_list = []
maxceo = []
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
        maxincw = int(32760 / np.maximum(np.max(convw),-np.min(convw)) - 10)
        convw = (convw*maxincw).astype(np.int)

        maxceo.append(maxincw)
        #量化模型准备
        scale = (1.0 / maxincw)
        conw_short_list.append((conv.weight.detach()*maxincw).long() *scale)
        conb_short_list.append(conv.bias.detach())

        params_short_num += convw.flatten().shape[0]
        res += 'static short ' + \
               name[count] + \
               '_weight[{0}] = '.format(
                   convw.flatten().shape[0]) + '{ \n'
        if convw.shape[0] % 8 == 0:
            np.savetxt('param.txt', convw.reshape(-1, 8), fmt='%d',
                       delimiter=', ', newline=',\n')
        else:
            np.savetxt('param.txt', convw, fmt='%d',
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

#保存scale
params_num += len(maxceo)
res += 'static int ' + \
        'weight_scale[{0}] = '.format(
            len(maxceo)) + '{ \n'
maxceo_np = np.array(maxceo)
if maxceo_np.shape[0] % 5 == 0:
    np.savetxt('param.txt', maxceo_np.reshape(-1, 8), fmt='%d',
                delimiter=', ', newline=',\n')
else:
    np.savetxt('param.txt', maxceo_np, fmt='%d',
                delimiter=', ', newline=',\n')
with open('param.txt', 'r') as f:
    res += f.read()
res += '};\n\n'

print(count)
memory_total = (params_short_num*2+params_num*4)/1024
res += '#endif\n'
res += '//0410_93043_desc_parameters: float[{:d}] short[{:d}]\n'.format(params_num, params_short_num)
res += '//0410_93043_desc_memory: {:.4f} K\n'.format(memory_total)
with open('0410_93043_desc_result.h', 'w') as f:
    f.write(res)
Path(r'param.txt').unlink()

quant_pth_save_pth = "/data/yey/work/6191/unsuperpoint/93043_short.pth.tar"
net_short.model_save_quant_param(conw_short_list, conb_short_list, quant_pth_save_pth)

