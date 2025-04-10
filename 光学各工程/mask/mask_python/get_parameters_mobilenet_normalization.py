
from models.MobileNet import *
from pathlib import Path
from collections import deque
import numpy as np
from pathlib import Path
import copy

layer_ID = {
    'CONVBN_ID': -1, #带BN的卷积层
    'MNV3BLOCK_ID': -2, #monilenet
    'SEMODULE_ID': -3,
    'CLASSIFIER_ID': -4,
    'DECONVBN_ID': -5, #带BN的反卷积层
    'CONV_ID': -100,
}

pthpath = r'./checkpoints/24_1010_zd_qp_499/499_net_G.pth'
net = MNV3_bufen_new(1, 1, 4, n_blocks=9)
net.load_state_dict(torch.load(pthpath))
net.eval()
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


num = 0
s = 0
p = 0
out = 0
r = 0
count = 0
count1 = 0
layercount = 0


for name, module in net._modules.items():
    #print(module)
    if isinstance(module, nn.Sequential):
        for index in range(len(module)):
            conv = None
            if index + 1 < len(module) and isinstance(module[index], nn.Conv2d) and isinstance(module[index + 1],
                                                                                               nn.BatchNorm2d):
                conv = nn.utils.fusion.fuse_conv_bn_eval(module[index], module[index + 1])
                if conv is not None:
                    convw, convb = conv.weight.detach().numpy(
                    ).flatten(), conv.bias.detach().numpy().flatten()

                    count1 += convw.shape[0]
                    count1 += convb.shape[0]
                    count += 2
                count += 3
                layercount += 1
            elif index + 1 < len(module) and isinstance(module[index], nn.ConvTranspose2d) and isinstance(module[index + 1], nn.BatchNorm2d):
                fused_deconv = copy.deepcopy(module[index])
                fused_deconv.weight = torch.nn.Parameter(torch.transpose(module[index].weight, 0, 1))
                conv = nn.utils.fusion.fuse_conv_bn_eval(fused_deconv, module[index + 1])
                if conv is not None:
                    convw, convb = conv.weight.detach().numpy(
                    ).flatten(), conv.bias.detach().numpy().flatten()

                    count1 += convw.shape[0]
                    count1 += convb.shape[0]
                    count += 2
                count += 3
                layercount += 1

            elif isinstance(module[index], nn.Conv2d):
                conv = module[index]
                if conv is not None:
                    convw, convb = conv.weight.detach().numpy(
                    ).flatten(), conv.bias.detach().numpy().flatten()

                    count1 += convw.shape[0]
                    count1 += convb.shape[0]
                    count += 2
                count += 3
                layercount += 1
            elif isinstance(module[index], Block):
                layers = get_parameters_layer(module[index])
                # print(layers)
                for j in range(len(layers)):
                    # print(layers[j])
                    conv = None
                    if j + 1 < len(layers) and isinstance(layers[j], nn.Conv2d) and isinstance(layers[j + 1],
                                                                                               nn.BatchNorm2d):
                        conv = nn.utils.fusion.fuse_conv_bn_eval(layers[j], layers[j + 1])
                    elif isinstance(layers[j], nn.Conv2d):
                        conv = layers[j]
                    if conv is not None:
                        # print(conv)
                        convw, convb = conv.weight.detach().numpy(
                        ).flatten(), conv.bias.detach().numpy().flatten()
                        count1 += convw.shape[0]
                        count1 += convb.shape[0]
                        count += 2
                count += 4
                layercount += 1
            elif isinstance(module[index], SeModule):
                layers = get_parameters_layer(module[index])
                for j in range(len(layers)):
                    # print(layers[j])
                    conv = None
                    if j + 1 < len(layers) and isinstance(layers[j], nn.Conv2d) and isinstance(layers[j + 1],
                                                                                               nn.BatchNorm2d):
                        conv = nn.utils.fusion.fuse_conv_bn_eval(layers[j], layers[j + 1])
                    elif isinstance(layers[j], nn.Conv2d):
                        conv = layers[j]
                    if conv is not None:
                        # print(conv)
                        convw, convb = conv.weight.detach().numpy(
                        ).flatten(), conv.bias.detach().numpy().flatten()
                        count1 += convw.shape[0]
                        count1 += convb.shape[0]
                        count += 2
                count += 2
                layercount += 1
    elif isinstance(module, nn.Conv2d):  # 这里假定只有分类器满足此条件
        conv = module
        if conv is not None:
            convw, convb = conv.weight.detach().numpy(
            ).flatten(), conv.bias.detach().numpy().flatten()

            count1 += convw.shape[0]
            count1 += convb.shape[0]
            count += 2
        count += 1
        layercount += 1

res_int = '#if 1\n'
res_int += 'static int parameter[{}] = '.format(count + 3)
res_int += '{\n'
res_int += '{},\n'.format(count + 3)
res_int += "24,\n"  # 版本号
res_int += "{},\n".format(count1)
res_fp = 'static float parameter[{}] = '.format(count1 + 3)
res_fp += '{\n'
num = 0         
for name, module in net._modules.items():
    if isinstance(module, nn.Sequential):
        for index in range(len(module)):
            conv = None
            if index + 1 < len(module) and isinstance(module[index], nn.Conv2d) and isinstance(module[index + 1],
                                                                                               nn.BatchNorm2d):
                # res += '{},'.format(layer_ID['CONVBN_ID'])

                conv = nn.utils.fusion.fuse_conv_bn_eval(module[index], module[index + 1])
                if conv is not None:
                    convw, convb = conv.weight.detach().numpy(
                    ).flatten(), conv.bias.detach().numpy().flatten()

                    res_int += '{},'.format(num)        # weight pos
                    # res += '\n'
                    num += convw.shape[0]           # bias pos
                    res_int += '{},'.format(num)
                    # res += '\n'

                    num += convb.shape[0]           # next weigth pos

                    s = conv.stride[0]
                    # print(s)
                    p = conv.padding[0]
                    out = conv.out_channels
                    layerID = -1
                res_int += '{},{},{},\n'.format(s, p, out)
            elif index + 1 < len(module) and isinstance(module[index], nn.ConvTranspose2d) and isinstance(module[index + 1], nn.BatchNorm2d):
                
                fused_deconv = copy.deepcopy(module[index])
                fused_deconv.weight = torch.nn.Parameter(torch.transpose(module[index].weight, 0, 1))
                conv = nn.utils.fusion.fuse_conv_bn_eval(fused_deconv, module[index + 1])
                if conv is not None:
                    convw, convb = conv.weight.detach().numpy(
                    ).flatten(), conv.bias.detach().numpy().flatten()

                    res_int += '{},'.format(num)        # weight pos
                    # res += '\n'
                    num += convw.shape[0]           # bias pos
                    res_int += '{},'.format(num)
                    # res += '\n'

                    num += convb.shape[0]           # next weigth pos

                    s = conv.stride[0]
                    # print(s)
                    p = conv.padding[0]
                    out = conv.out_channels
                    layerID = -5
                res_int += '{},{},{},\n'.format(s, p, out)

            elif isinstance(module[index], nn.Conv2d):
                # res += '{},'.format(layer_ID['CONV_ID'])

                conv = module[index]
                if conv is not None:
                    convw, convb = conv.weight.detach().numpy(
                    ).flatten(), conv.bias.detach().numpy().flatten()

                    res_int += '{},'.format(num)
                    # res += '\n'
                    num += convw.shape[0]
                    res_int += '{},'.format(num)
                    # res += '\n'
                    num += convb.shape[0]

                    s = conv.stride[0]
                    # print(s)
                    p = conv.padding[0]
                    out = conv.out_channels

                res_int += '{},{},{},\n'.format(s, p, out)

            elif isinstance(module[index], Block):
                # res += '{},'.format(layer_ID['MNV3BLOCK_ID'])

                layers = get_parameters_layer(module[index])
                # print(layers)
                for j in range(len(layers)):
                    # print(layers[j])
                    conv = None
                    if j + 1 < len(layers) and isinstance(layers[j], nn.Conv2d) and isinstance(layers[j + 1],
                                                                                               nn.BatchNorm2d):
                        conv = nn.utils.fusion.fuse_conv_bn_eval(layers[j], layers[j + 1])
                    elif isinstance(layers[j], nn.Conv2d):
                        conv = layers[j]
                    if conv is not None:
                        # print(conv)
                        convw, convb = conv.weight.detach().numpy(
                        ).flatten(), conv.bias.detach().numpy().flatten()
                        res_int += '{},'.format(num)
                        # res += '\n'
                        num += convw.shape[0]
                        res_int += '{},'.format(num)
                        # res += '\n'
                        num += convb.shape[0]

                s = module[index].stride
                p = module[index].conv2.padding[0]
                out = module[index].conv3.out_channels
                r = int(module[index].conv1.out_channels / module[index].conv1.in_channels)

                res_int += '{},{},{},{},\n'.format(s, p, out, r)

            elif isinstance(module[index], SeModule):
                # res += '{},'.format(layer_ID['SEMODULE_ID'])

                layers = get_parameters_layer(module[index])
                for j in range(len(layers)):
                    conv = None
                    if j + 1 < len(layers) and isinstance(layers[j], nn.Conv2d) and isinstance(layers[j + 1],
                                                                                               nn.BatchNorm2d):
                        conv = nn.utils.fusion.fuse_conv_bn_eval(layers[j], layers[j + 1])
                    elif isinstance(layers[j], nn.Conv2d):
                        conv = layers[j]
                    if conv is not None:
                        # print(conv)
                        convw, convb = conv.weight.detach().numpy(
                        ).flatten(), conv.bias.detach().numpy().flatten()
                        res_int += '{},'.format(num)
                        # res += '\n'
                        num += convw.shape[0]
                        res_int += '{},'.format(num)
                        # res += '\n'
                        num += convb.shape[0]

                out = layers[0].in_channels
                r = int(layers[0].in_channels / layers[0].out_channels)

                res_int += '{},{},\n'.format(s, r)
    elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):  # 这里假定只有分类器满足此条件
        # res += '{},'.format(layer_ID['CLASSIFIER_ID'])

        conv = module
        if conv is not None:
            convw, convb = conv.weight.detach().numpy(
            ).flatten(), conv.bias.detach().numpy().flatten()
            res_int += '{},'.format(num)
            # res += '\n'
            num += convw.shape[0]
            res_int += '{},'.format(num)
            # res += '\n'
            num += convb.shape[0]
        #out = conv.out_features
        out = conv.out_channels

        res_int += '{}, \n'.format(out)
res_int += "}; \n\n"

layers = get_parameters_layer(net)
print(layers)
for i in range(len(layers)):
    # print(i)
    conv = None
    if i + 1 < len(layers) and isinstance(layers[i], nn.Conv2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
        conv = nn.utils.fusion.fuse_conv_bn_eval(layers[i], layers[i + 1])
    elif i + 1 < len(layers) and isinstance(layers[i], nn.ConvTranspose2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
        fused_deconv = copy.deepcopy(layers[i])
        fused_deconv.weight = torch.nn.Parameter(torch.transpose(layers[i].weight, 0, 1))
        conv = nn.utils.fusion.fuse_conv_bn_eval(fused_deconv, layers[i + 1])
    elif isinstance(layers[i], nn.Conv2d):
        conv = layers[i]

    if conv is not None:
        print(conv)
        convw, convb = conv.weight.detach().numpy(
        ).flatten(), conv.bias.detach().numpy().flatten()
        if convw.shape[0] % 8 == 0:
            np.savetxt('param.txt', convw.reshape(-1, 8), fmt='%1.10f',
                       delimiter=', ', newline=',\n')
        else:
            np.savetxt('param.txt', convw, fmt='%1.10f',
                       delimiter=', ', newline=',\n')
        with open('param.txt', 'r') as f:
            res_fp += f.read()
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
            res_fp += f.read()
# low/high/normal threshold
res_fp += '1, 3, 3,\n' 
res_fp += "}; \n"
# res+='{}'.format(count1)
res_fp += '#endif\n'
res_fp += '//24_0515_499\n'
with open(r'24_0515_499.h', 'w') as f:
    f.write(res_int + res_fp)
Path(r'param.txt').unlink()
