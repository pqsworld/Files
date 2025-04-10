# from models.SuperPointNet_small_128 import *
from models.ReHardNet_fast_featuremap import *
from pathlib import Path
from collections import deque, OrderedDict
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import copy
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['CUDA_LAUNCH_BLOCKING'] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #    model_1 = [  # nn.ReflectionPad2d(3),
    #         nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
    #         norm_layer(64),
    #         nn.ReLU(True)]
    #     model_1 += [nn.Conv2d(64, output_nc, kernel_size=1, padding=0)]
    #     # model += [hsigmoid()]



# pthpath = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0516_train78400_bs128_120pointlabels_6blocks_256desc_addinnerloss/superPointNet_78400_checkpoint.pth.tar'
# pthpath = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0602_train6159_bs128_120pointlabels_6blocks_addsuperviseddesc/superPointNet_89400_checkpoint.pth.tar'
# pthpath = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0425_train_bs128_120pointlabels_6blocks_notnorm_fulldesc/superPointNet_47000_checkpoint.pth.tar'
# pthpath = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0425_train_bs128_120pointlabels_6blocks_notnorm_fulldesc_addcosloss/superPointNet_49800_checkpoint.pth.tar'
# pthpath = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0519_train6159_bs64_120pointlabels_6blocks_256desc_selfsup/superPointNet_113200_checkpoint.pth.tar'

pthpath = "/hdd/file-input/qint/6159_parallel/6159_hardnet_self/logs/1201_Re_dense_6193/1202_93_Re_dense_extim_midpad_37x16_ds2_softgrid_nohalf_C4_nofixp_split64_7b/checkpoints/superPointNet_52100_checkpoint.pth.tar"



# mdconv3x3 = ['econv3x3_pre1' , 'econv3x3_pre2' , 'econv3x3_ds1' , 'econv3x3_ds2', 'econv3x3_ds3']

block_1 = ['conv1x1s1_di_1', 'convdw3x3s2_1', 'conv1x1s1_dd_1']
block_2 = ['conv1x1s1_di_2', 'convdw3x3s2_2', 'conv1x1s1_dd_2']
block_3 = ['conv1x1s1_di_3', 'convdw3x3s1_3', 'conv1x1s1_dd_3']
block_4 = ['conv1x1s1_di_4', 'convdw3x3s1_4', 'conv1x1s1_dd_4', 'conv1x1_di_shortcut_4']
block_5 = ['conv1x1s1_di_5', 'convdw3x3s1_5', 'conv1x1s1_dd_5', 'conv1x1_di_shortcut_5']
block_6 = ['conv1x1s1_di_6', 'convdw3x3s1_6', 'conv1x1s1_dd_6']
block_7 = ['conv1x1s1_di_7', 'convdw3x3s1_7', 'conv1x1s1_dd_7', 'conv1x1_di_shortcut_7']


downconv3x3 = ['conv1x1_post']


block = [block_1, block_2, block_3, block_4, block_5, block_6, block_7, downconv3x3]
name = []
for i in block:
    name.extend(i)



# net = SuperPointNet_small_128().to(device)
net = ReHardNet_fast_featuremap_depth().to(device)
checkpoint = torch.load(pthpath, map_location=lambda storage, loc: storage)    # model_state_dict


net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

# net.load_state_dict(torch.load(pthpath).state_dict)
net.to('cpu')


def list_layers(layer):
    layers = []
    if isinstance(layer, ReBlock) or isinstance(layer, enn.SequentialModule):
        for i in layer.children():
            layers.extend(list_layers(i))
    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.ConvTranspose2d):
        layers.append(layer)
    elif isinstance(layer, enn.R2Conv) or isinstance(layer, enn.InnerBatchNorm)  or isinstance(layer, enn.R2ConvTransposed):
        layers.append(layer.export().eval())
    
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
    # if i == 77:
    #     break
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
    elif isinstance(layers[i], nn.ConvTranspose2d):
        deconv = copy.deepcopy(layers[i])
        deconv.weight = torch.nn.Parameter(torch.transpose(layers[i].weight, 0, 1))
        conv = deconv
    if conv is not None:
        if conv.bias != None:
            convw, convb = conv.weight.detach().numpy(
            ).flatten(), conv.bias.detach().numpy().flatten()
        else:
            convw = conv.weight.detach().numpy().flatten()
            convb = np.zeros((conv.weight.shape[0]), dtype=np.float32).flatten()

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
with open('result_densedesc_52100.h', 'w') as f:
    f.write(res)
Path(r'param.txt').unlink()
