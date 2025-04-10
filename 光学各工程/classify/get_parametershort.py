#from symbol.MobileNet_test import *
#from MobileNet_dropout_linear import *
from mobilenetv3 import *
from pathlib import Path
from collections import deque
import numpy as np
from pathlib import Path
import torch

pthpath=r'./checkpoints/boe_10_2/ckpt_mnv_small1_97_0.99449_0.97078.pth'

# optic_classify96_audi124x241 61850
pre = ['conv3x3']
#block1 = ['conv1x1s1_di_1', 'convdw3x3s1_1', 'conv1x1s1_dd_1']
block2 = ['conv1x1s1_di_2', 'convdw3x3s2_2', 'conv1x1s1_dd_2']
block3 = ['conv1x1s1_di_3', 'convdw3x3s1_3', 'conv1x1s1_dd_3']
block4 = ['conv1x1s1_di_4', 'convdw3x3s2_4', 'conv1x1s1_dd_4']
block5 = ['conv1x1s1_di_5', 'convdw3x3s1_5', 'conv1x1s1_dd_5']
block6 = ['conv1x1s1_di_6', 'convdw3x3s2_6', 'conv1x1s1_dd_6']
#block7 = ['conv1x1s1_di_7', 'convdw3x3s2_7', 'conv1x1s1_dd_7',
#          'conv1x1s1_dd_se_7', 'conv1x1s1_di_se_7']
#block8 = ['conv1x1s1_di_8', 'convdw3x3s1_8', 'conv1x1s1_dd_8',
#          'conv1x1s1_dd_se_8', 'conv1x1s1_di_se_8']
# block9 = ['conv1x1s1_di_9', 'convdw3x3s2_9', 'conv1x1s1_dd_9',
#           'conv1x1s1_dd_se_9', 'conv1x1s1_di_se_9']
# block10 = ['conv1x1s1_di_10', 'convdw3x3s1_10',
#            'conv1x1s1_dd_10', 'conv1x1s1_dd_se_10', 'conv1x1s1_di_se_10']
classifier = ['conv1x1s1']

#block = [pre, block1, block2, block3, block4, block5,
#         block6, classifier]
block = [pre, block2, block3, block4, block5,
         block6, classifier]
name = []
for i in block:
    name.extend(i)
#net = MNV3_large(2)
net = MNV30811_SMALL()
net.load_state_dict(torch.load(pthpath))#['net']
model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

net.eval()
# net.load_state_dict(torch.load(pthpath).state_dict)
net.to('cpu')
print(net)

def list_layers(layer):
    layers = []
    if isinstance(layer, Block) or isinstance(layer, nn.Sequential) or isinstance(layer, SeModule):
        for i in layer.children():
            layers.extend(list_layers(i))
    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
        layers.append(layer)
    return layers


def get_parameters_layer(net):
    layers = []
    queue = deque()
    for i in net.children():
        queue.append(i)
    #print(queue)
    while len(queue):
        root = queue.popleft()
        layers.extend(list_layers(root))
    return layers


def write_parameter(layers,savename):
    count = 0
    beishu = []
    res = '#if 1\n'
    num = 0
    for i in range(len(layers)):
        conv = None
        if i + 1 < len(layers) and isinstance(layers[i], nn.Conv2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
            conv = nn.utils.fusion.fuse_conv_bn_eval(layers[i], layers[i + 1])                  #conv和batch层内部融合，直接输出
        elif isinstance(layers[i], nn.Conv2d):
            conv = layers[i]
        if conv is not None:
            convw, convb = conv.weight.detach().numpy().flatten(), conv.bias.detach().numpy().flatten()#detach将某一层的参数单独拆出来，flatten将tensor展平
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
            #print(convw_bs)
            #print(convb_bs)
            large_thr = 100#800 #100
            if convw_bs>large_thr:
                beishu.append(convw_bs)
                
                convw[convw*convw_bs>0] = convw[convw*convw_bs>0]*convw_bs+0.5
                convw[convw*convw_bs<0] = convw[convw*convw_bs<0]*convw_bs-0.5
                convw = np.int16(convw)
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
            else:
                print(convw_bs)
                beishu.append(1)
                # convw = convw_max_zz
                res += 'static float ' + \
                name[count] + \
                '_weight[{0}] = '.format(
                    convw.flatten().shape[0]) + '{ \n'
                if convw.shape[0] % 8 == 0:
                    np.savetxt('param.txt', convw.reshape(-1, 8), fmt='%.7f',
                            delimiter=', ', newline=',\n')
                else:
                    np.savetxt('param.txt', convw, fmt='%.7f',
                            delimiter=', ', newline=',\n')
            with open('param.txt', 'r') as f:
                res += f.read()
            res += '};\n\n'
            #if convb_bs>100:
            if convb_bs>large_thr:
                beishu.append(convb_bs)
                convb[convb*convb_bs>0] = convb[convb*convb_bs>0]*convb_bs+0.5
                convb[convb*convb_bs<0] = convb[convb*convb_bs<0]*convb_bs-0.5
                convb = np.int16(convb)
                res += 'static short ' + \
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
            else:
                beishu.append(1)
                # convb = convb_max_zz
                res += 'static float ' + \
                name[count] + \
                '_bias[{0}] = '.format(
                    convb.flatten().shape[0]) + '{ \n'
                if convb.shape[0] % 8 == 0:
                    np.savetxt('param.txt', convb.reshape(-1, 8), fmt='%.7f',
                            delimiter=', ', newline=',\n')
                elif convb.shape[0] % 4 == 0:
                    np.savetxt('param.txt', convb.reshape(-1, 4), fmt='%.7f',
                            delimiter=', ', newline=',\n')
                else:
                    np.savetxt('param.txt', convb, fmt='%d',
                            delimiter=', ', newline=',\n')
            
            num = num+convw.flatten().shape[0]+convb.flatten().shape[0]
            with open('param.txt', 'r') as f:
                res += f.read()
            res += '};\n\n'
            count += 1
    print("param_num")
    print(num)
    print(beishu)
    beishu = np.array(beishu)
    res += 'static int ' + \
        'magnification_factor[{0}] = '.format(
            beishu.flatten().shape[0]) + '{ \n'
    np.savetxt('param.txt', beishu, fmt='%d',
                delimiter=', ', newline=',\n')
    with open('param.txt', 'r') as f:
        res += f.read()
    res += '};\n\n'
    print(count)
    res += '#endif\n'

    with open(savename, 'w') as f:
        f.write(res)
    Path(r'param.txt').unlink()

def write_parameter(layers,savename,preffix):
    count = 0
    beishu = []
    res = '#if 1\n'
    num = 0
    for i in range(len(layers)):
        conv = None
        if i + 1 < len(layers) and isinstance(layers[i], nn.Conv2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
            conv = nn.utils.fusion.fuse_conv_bn_eval(layers[i], layers[i + 1])                  #conv和batch层内部融合，直接输出
        elif isinstance(layers[i], nn.Conv2d):
            conv = layers[i]
        if conv is not None:
            convw, convb = conv.weight.detach().numpy().flatten(), conv.bias.detach().numpy().flatten()#detach将某一层的参数单独拆出来，flatten将tensor展平
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
            #print(convw_bs)
            #print(convb_bs)
            large_thr = 100#800 #100
            if convw_bs>large_thr:
                beishu.append(convw_bs)
                
                convw[convw*convw_bs>0] = convw[convw*convw_bs>0]*convw_bs+0.5
                convw[convw*convw_bs<0] = convw[convw*convw_bs<0]*convw_bs-0.5
                convw = np.int16(convw)
                res += 'static short ' + \
                name[count] + \
                '_weight{0}[{1}] = '.format(preffix,
                    convw.flatten().shape[0]) + '{ \n'
                if convw.shape[0] % 8 == 0:
                    np.savetxt('param.txt', convw.reshape(-1, 8), fmt='%d',
                            delimiter=', ', newline=',\n')
                else:
                    np.savetxt('param.txt', convw, fmt='%d',
                            delimiter=', ', newline=',\n')
            else:
                print(convw_bs)
                beishu.append(1)
                # convw = convw_max_zz
                res += 'static float ' + \
                name[count] + \
                '_weight{0}[{1}] = '.format(preffix,
                    convw.flatten().shape[0]) + '{ \n'
                if convw.shape[0] % 8 == 0:
                    np.savetxt('param.txt', convw.reshape(-1, 8), fmt='%.7f',
                            delimiter=', ', newline=',\n')
                else:
                    np.savetxt('param.txt', convw, fmt='%.7f',
                            delimiter=', ', newline=',\n')
            with open('param.txt', 'r') as f:
                res += f.read()
            res += '};\n\n'
            #if convb_bs>100:
            if convb_bs>large_thr:
                beishu.append(convb_bs)
                convb[convb*convb_bs>0] = convb[convb*convb_bs>0]*convb_bs+0.5
                convb[convb*convb_bs<0] = convb[convb*convb_bs<0]*convb_bs-0.5
                convb = np.int16(convb)
                res += 'static short ' + \
                name[count] + \
                '_bias{0}[{1}] = '.format(preffix,
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
            else:
                beishu.append(1)
                # convb = convb_max_zz
                res += 'static float ' + \
                name[count] + \
                '_bias{0}[{1}] = '.format(preffix,
                    convb.flatten().shape[0]) + '{ \n'
                if convb.shape[0] % 8 == 0:
                    np.savetxt('param.txt', convb.reshape(-1, 8), fmt='%.7f',
                            delimiter=', ', newline=',\n')
                elif convb.shape[0] % 4 == 0:
                    np.savetxt('param.txt', convb.reshape(-1, 4), fmt='%.7f',
                            delimiter=', ', newline=',\n')
                else:
                    np.savetxt('param.txt', convb, fmt='%d',
                            delimiter=', ', newline=',\n')
            
            num = num+convw.flatten().shape[0]+convb.flatten().shape[0]
            with open('param.txt', 'r') as f:
                res += f.read()
            res += '};\n\n'
            count += 1

        
    print("param_num")
    print(num)
    print(beishu)
    beishu = np.array(beishu)
    res += 'static int ' + \
        'magnification_factor{0}[{1}] = '.format(preffix,
            beishu.flatten().shape[0]) + '{ \n'
    np.savetxt('param.txt', beishu, fmt='%d',
                delimiter=', ', newline=',\n')
    with open('param.txt', 'r') as f:
        res += f.read()
    res += '};\n\n'
    print(count)
    res += '#endif\n'
    #with open('result_6193_spoof_model_panqi_v1_short.h', 'w') as f:
    # with open('result_6193_compare_93122x122_l_v3_new_short.h', 'w') as f:
    # with open('result_6193_compare_93122x66_l_v17_3.1_new_short.h', 'w') as f:
    # with open('result_6193_compare_93122x122_l_v3.2_short.h', 'w') as f:
    # with open('result_6193_compare_93122x66_l_wet014_short.h', 'w') as f:
    # with open('result_6193_compare_93122x122_l_v3.4_short.h', 'w') as f:
    with open(savename, 'w') as f:
        f.write(res)
    Path(r'param.txt').unlink()


layers = get_parameters_layer(net)
for i in layers:
    print(i)

para_name = './compare_boe102_Vshort.h'
write_parameter(layers,para_name,'_v7')

