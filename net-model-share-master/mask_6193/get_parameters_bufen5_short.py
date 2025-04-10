from models.MobileNet import *
from pathlib import Path
from collections import deque
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import copy

pthpath = r'./checkpoints/6193mask_10/256_net_G.pth'

preconv3x3_1 = ['conv3x3_pre1']
preconv3x3_2 = ['conv3x3_pre2']

dsconv3x3_1 = ['conv3x3_ds1']
dsconv3x3_2 = ['conv3x3_ds2']
dsconv3x3_4 = ['conv3x3_ds4']
dsconv1x1_4 = ['conv1x1_ds4']
ds_max = ['ds_max']
regress = ['reconv1x1']

block3 = ['conv1x1s1_di_1', 'convdw3x3s2_1', 'conv1x1s1_dd_1']
block4 = ['convdw3x3s1_2', 'conv1x1s1_dd_2','conv1x1s1_dd_se_2', 'conv1x1s1_di_se_2']
block2 = ['conv1x1s1_di_2', 'convdw3x3s1_2', 'conv1x1s1_dd_2',
          'conv1x1s1_dd_se_2', 'conv1x1s1_di_se_2']
block3_o = ['conv1x1s1_di_3', 'convdw3x3s2_3', 'conv1x1s1_dd_3',
          'conv1x1s1_dd_se_3', 'conv1x1s1_di_se_3']
block4_o = ['conv1x1s1_di_4', 'convdw3x3s1_4', 'conv1x1s1_dd_4',
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

usfzdecon3x3_1 = ['deconv3x3_us1']
usdecon1x1_1 = ['deconv1x1_us1']
usdecon3x3_2 = ['deconv3x3_us2']
usdecon3x3_3 = ['deconv3x3_us3']
usdecon3x3_4 = ['deconv3x3_us4']
us_upsample = ['us_upsample']

postconv3x3 = ['conv3x3_post']

generateconv3x3 = ['conv3x3_generate']

block = [preconv3x3_1, preconv3x3_2, dsconv3x3_1, dsconv3x3_2, block3, block4, regress, usfzdecon3x3_1,usdecon1x1_1,usdecon3x3_2, usdecon3x3_3, postconv3x3, generateconv3x3]
name = []
for i in block:
    name.extend(i)

net = MNV3_bufen_new5(1, 1, 4, n_blocks=1)

net.load_state_dict(torch.load(pthpath))
model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
net.eval()
# net.load_state_dict(torch.load(pthpath).state_dict)
net.to('cpu')


def list_layers(layer):
    layers = []
    if isinstance(layer, Block_4) or isinstance(layer, nn.Sequential) or isinstance(layer, SeModule):
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
num =0
beishu = []
for i in range(len(layers)):
    #print(i)
    #print(layers)
    conv = None
    if i + 1 < len(layers) and isinstance(layers[i], nn.Conv2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
        print("1111")
        print(layers[i].weight.size())
        conv = nn.utils.fusion.fuse_conv_bn_eval(layers[i], layers[i + 1])
    elif i + 1 < len(layers) and isinstance(layers[i], nn.ConvTranspose2d) and isinstance(layers[i + 1], nn.BatchNorm2d)  and  num == 0:
        print("2222")
        print(layers[i].weight.size())
        #print(layers[i].bias)
        #print(layers[i+1].weight.size())
        #print(layers[i+1].bias)
        #print(layers[i+1].weight)
        #print(layers[i+1].bias)
        #print(layers[i+1].running_mean)
        #print(layers[i+1].running_var)
        lay = nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,groups=1),


        lay = lay[0]
        nor = nn.BatchNorm2d(4)
        #nor = nor[0]
        lay.eval()
        nor.eval()
        A, B, C, D = layers[i].weight.chunk(4, dim=0)
        A1, B1, C1, D1 = layers[i+1].weight.chunk(4, dim=0)
        A2, B2, C2, D2 = layers[i+1].bias.chunk(4, dim=0)
        A3, B3, C3, D3 = layers[i+1].running_mean.chunk(4, dim=0)
        A4, B4, C4, D4 = layers[i+1].running_var.chunk(4, dim=0)
        #print(nor)
        #print(layers[i+1])
        ##print(lay.weight)
        #print(A)
        lay.weight.data = A
        nor.weight.data = A1
        nor.bias.data = A2
        nor.running_mean.data = A3
        nor.running_var.data = A4
        fused_deconv1 = copy.deepcopy(lay)
        fused_deconv1.weight = torch.nn.Parameter(torch.transpose(A, 0, 1))
        conv1 = nn.utils.fusion.fuse_conv_bn_eval(fused_deconv1, nor)

        lay.weight.data = B
        nor.weight.data = B1
        nor.bias.data = B2
        nor.running_mean.data = B3
        nor.running_var.data = B4
        fused_deconv1 = copy.deepcopy(lay)
        fused_deconv1.weight = torch.nn.Parameter(torch.transpose(B, 0, 1))
        conv2 = nn.utils.fusion.fuse_conv_bn_eval(fused_deconv1, nor)

        lay.weight.data = C
        nor.weight.data = C1
        nor.bias.data = C2
        nor.running_mean.data = C3
        nor.running_var.data = C4
        fused_deconv1 = copy.deepcopy(lay)
        fused_deconv1.weight = torch.nn.Parameter(torch.transpose(C, 0, 1))
        conv3 = nn.utils.fusion.fuse_conv_bn_eval(fused_deconv1, nor)

        lay.weight.data = D
        nor.weight.data = D1
        nor.bias.data = D2
        nor.running_mean.data = D3
        nor.running_var.data = D4
        fused_deconv1 = copy.deepcopy(lay)
        fused_deconv1.weight = torch.nn.Parameter(torch.transpose(D, 0, 1))
        conv4 = nn.utils.fusion.fuse_conv_bn_eval(fused_deconv1, nor)
        #print(conv1)
        #print(conv2.weight.size())
        #print(conv3.weight)
        #print(conv4.weight)
        #conv = torch.cat([conv1.weight,conv2.weight,conv3.weight,conv4.weight],dim=1)
        #conv_bias = torch.cat([conv1.bias,conv2.bias,conv3.bias,conv4.bias],dim=0)
        #fused_deconv = copy.deepcopy(layers[i])
        #fused_deconv.weight = torch.nn.Parameter(torch.transpose(layers[i].weight, 0, 1))
        #fused_deconv.weight.data = conv
        #fused_deconv.bias = torch.nn.Parameter(conv_bias)
        #print(fused_deconv.weight.size())
        #print(fused_deconv.bias.size())
        conv_weight = torch.cat([conv1.weight,conv2.weight,conv3.weight,conv4.weight],dim=0)
        conv_bias = torch.cat([conv1.bias,conv2.bias,conv3.bias,conv4.bias],dim=0)
        fused_deconv = copy.deepcopy(layers[i])
        fused_deconv.weight = torch.nn.Parameter(conv_weight)
        fused_deconv.bias = torch.nn.Parameter(conv_bias)
        print("fused_deconv_weight_size")
        print(fused_deconv.weight.size())
        conv = fused_deconv
        num=1
    elif i + 1 < len(layers) and isinstance(layers[i], nn.ConvTranspose2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
        print("3333")
        print(layers[i].weight.size())
        #print(layers[i+1].weight.size())
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
        print(name[count])
        convw_max = np.max(convw)
        convw_min = np.min(convw)
        convb_max = np.max(convb)
        convb_min = np.min(convb)
        if (convw_max > -convw_min):
            convw_max_zz = convw_max
        else:
            convw_max_zz = -convw_min
        if (convb_max > -convb_min):
            convb_max_zz = convb_max
        else:
            convb_max_zz = -convb_min
        convw_bs = np.int32(32760/convw_max_zz)
        convb_bs = np.int32(32760/convb_max_zz)
        print(convw_bs)
        print(convb_bs)
        beishu.append(convw_bs)
        beishu.append(convb_bs)
        convw[convw*convw_bs>0] = convw[convw*convw_bs>0]*convw_bs+0.5
        convw[convw*convw_bs<0] = convw[convw*convw_bs<0]*convw_bs-0.5
        convb[convb*convb_bs>0] = convb[convb*convb_bs>0]*convb_bs+0.5
        convb[convb*convb_bs<0] = convb[convb*convb_bs<0]*convb_bs-0.5
        convw = np.int16(convw)
        convb = np.int16(convb)
        res += 'static short para_mask_' + \
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

        res += 'static short para_mask_' + \
               name[count] + \
               '_bias[{0}] = '.format(
                   convb.flatten().shape[0]) + '{ \n'
        if convb.shape[0] % 8 == 0:
            np.savetxt('param.txt', convb.reshape(-1, 8), fmt='%d',
                       delimiter=', ', newline=',\n')
        elif convb.shape[0] % 4 == 0:
            np.savetxt('param.txt', convb.reshape(-1, 4), fmt='%d',
                       delimiter=', ', newline=',\n')
        else:
            np.savetxt('param.txt', convb, fmt='%d',
                       delimiter=', ', newline=',\n')
        with open('param.txt', 'r') as f:
            res += f.read()
        res += '};\n\n'
        count += 1


print(beishu)
beishu = np.array(beishu)
res += 'static int ' + \
       'para_mask_magnification_factor[{0}] = '.format(
           beishu.flatten().shape[0]) + '{ \n'
np.savetxt('param.txt', beishu, fmt='%d',
               delimiter=', ', newline=',\n')
with open('param.txt', 'r') as f:
    res += f.read()
res += '};\n\n'
print(count)
res += '#endif\n'
with open('6193mask_result_10_256_v5_shi_regress_short.h', 'w') as f:
    f.write(res)
Path(r'param.txt').unlink()
