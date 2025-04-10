from collections import deque
from models.networks import *

import copy

def list_layers(layer):
    layers = []
    if isinstance(layer, nn.Sequential) or isinstance(layer,DepthwiseSeparableConvolution) or isinstance(layer,SAM)  :
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
        print(list_layers(root))
    return layers

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# netG.apply(weights_init)

def GetConvParam(layers):
    count = 0
    res = '#if 1\n'
    i = 0
    maxcoe=[]
    while i < len(layers):
        #print(i)
        # print(count)
        # print(net_names[count])
        conv = None
        if i + 1 < len(layers) and isinstance(layers[i], nn.Conv2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
            conv = nn.utils.fusion.fuse_conv_bn_eval(layers[i], layers[i + 1])
            i = i + 2
        elif isinstance(layers[i], nn.Conv2d):
            conv = layers[i]
            i = i + 1
        elif i + 1 < len(layers) and isinstance(layers[i], nn.ConvTranspose2d) and isinstance(layers[i + 1],
                                                                                              nn.BatchNorm2d):

            fused_deconv = copy.deepcopy(layers[i])
            fused_deconv.weight = torch.nn.Parameter(torch.transpose(layers[i].weight, 0, 1))
            conv = nn.utils.fusion.fuse_conv_bn_eval(fused_deconv, layers[i + 1])
            i = i + 2
        elif isinstance(layers[i], nn.ConvTranspose2d):
            conv = layers[i]
            i = i + 1
        else:
            i = i + 1

        if conv is not None:
            if conv.bias is not None:
                #print(conv.weight)
                convw, convb = conv.weight.detach().numpy().flatten(), conv.bias.detach().numpy().flatten()
            else:
                convw = conv.weight.detach().numpy().flatten()
            
            
            
            res += 'static short ' + net_names[count] + '_weight[{0}] = '.format(convw.flatten().shape[0]) + '{ \n'
            maxincw=int(32760/np.maximum(np.max(convw),-np.min(convw))-10)
            print(maxincw,",")
            maxcoe.append(maxincw)
            # np.array(maxcoe,[maxincw])
            convw = convw*maxincw
            if convw.shape[0] % 8 == 0:
                np.savetxt('param.txt', convw.reshape(-1, 8), fmt='%d',delimiter=', ', newline=',\n')
            elif convw.shape[0] % 4 == 0:
                np.savetxt('param.txt', convw.reshape(-1, 4), fmt='%d', delimiter=', ', newline=',\n')
            
            else:
                np.savetxt('param.txt', convw, fmt='%d',delimiter=', ', newline=',\n')
            with open('param.txt', 'r') as f:
                res += f.read()
            res += '};\n\n'

            res += 'static short ' + net_names[count] + '_bias[{0}] = '.format(convb.flatten().shape[0]) + '{ \n'
            maxincb=int(32760/np.maximum(np.max(convb),-np.min(convb))-10)
            print(maxincb,",")
            maxcoe.append(maxincb)
            # np.array(maxcoe,[maxincb])
            convb = convb*maxincb
            if convb.shape[0] % 8 == 0:
                np.savetxt('param.txt', convb.reshape(-1, 8), fmt='%d', delimiter=', ', newline=',\n')
            elif convb.shape[0] % 4 == 0:
                np.savetxt('param.txt', convb.reshape(-1, 4), fmt='%d', delimiter=', ', newline=',\n')
            else:
                np.savetxt('param.txt', convb, fmt='%d', delimiter=', ', newline=',\n')
            with open('param.txt', 'r') as f:
                res += f.read()
            res += '};\n\n'
            count = count + 1
            print("count:")
            print(count)
    res += 'static int maxcoe['+str(count*2)+']={ \n'
    maxcoe=np.array(maxcoe)
    if maxcoe.shape[0] % 8 == 0:
        np.savetxt('param.txt', maxcoe.reshape(-1, 8), fmt='%d', delimiter=', ', newline=',\n')
    elif maxcoe.shape[0] % 4 == 0:
        np.savetxt('param.txt', maxcoe.reshape(-1, 4), fmt='%d', delimiter=', ', newline=',\n')
    elif maxcoe.shape[0] % 5 == 0:
        np.savetxt('param.txt', maxcoe.reshape(-1, 5), fmt='%d', delimiter=', ', newline=',\n')
    elif maxcoe.shape[0] % 3 == 0:
        np.savetxt('param.txt', maxcoe.reshape(-1, 3), fmt='%d', delimiter=', ', newline=',\n')
    elif maxcoe.shape[0] % 2 == 0:
        np.savetxt('param.txt', maxcoe.reshape(-1, 2), fmt='%d', delimiter=', ', newline=',\n')
    else:
        np.savetxt('param.txt', maxcoe, fmt='%d', delimiter=', ', newline=',\n')
    with open('param.txt', 'r') as f:
        res += f.read()
    res += '};\n\n'
    res += '#endif\n'

    return res


device = torch.device('cpu')

# model = ResnetGenerator323_7_RSG_small(1,2)
model = ResnetGenerator323_7_RSG_small2_212(1,2)
# model = ResnetGenerator323_7_RSG_small2_41(1,2)
layers = get_parameters_layer(model)

net_names = [
    'featrues_deconv1',
    'featrues_deconv2_1',
    'featrues_deconv2_2',
    'featrues_deconv3',
    'featrues_deconv4_1',
    'featrues_deconv4_11',
    'featrues_deconv4_2',
    'featrues_deconv4_12',

    'featrues_resnet1_1',
    'featrues_resnet1_2',
    'featrues_resnet1_3',
    'featrues_resnet1_4',

    'featrues_upconv3_3x3',
    'featrues_upconv3_1x1',
    'featrues_upconv3_3x3_1',
    #'featrues_upconv3_1x1_1',
    
    'featrues_upconv2',
    'featrues_upconv2_1',
    'featrues_upconv2_2',
    'featrues_upconv1_3x3',
    'featrues_upconv1_1x1',
    'featrues_upconv0',
]

# checkpoint = torch.load(r'/home/zhangsn/enhance/checkpoints/neten6193/neten_e22_4/288_net_G.pth', map_location='cpu')
checkpoint = torch.load(r'/home/zhangsn/enhance/checkpoints/neten6193/neten_m51_2/275_net_G.pth', map_location='cpu')
# checkpoint = torch.load(r'/home/zhangsn/enhance/checkpoints/desc6193/desc_exp_use/229_net_G.pth', map_location='cpu')


# print(checkpoint)  #
# exit()
# netG = torch.nn.DataParallel(netG)
model.load_state_dict(checkpoint, strict=False)  #
model.eval()
# print(layers)
# exit()
param = GetConvParam(layers)
#
txt_p = open("enhance_param_6193nm512_275_short.h", "w")
txt_p.write(param)
txt_p.close()
