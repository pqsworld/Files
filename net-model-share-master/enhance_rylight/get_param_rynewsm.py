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
    while i < len(layers):
        #print(i)
        # print(count)
        print(net_names[count])
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
            res += 'static float ' + net_names[count] + '_weight[{0}] = '.format(
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

            res += 'static float ' + net_names[count] + '_bias[{0}] = '.format(convb.flatten().shape[0]) + '{ \n'
            if convb.shape[0] % 8 == 0:
                np.savetxt('param.txt', convb.reshape(-1, 8), fmt='%1.10f', delimiter=', ', newline=',\n')
            elif convb.shape[0] % 4 == 0:
                np.savetxt('param.txt', convb.reshape(-1, 4), fmt='%1.10f', delimiter=', ', newline=',\n')
            else:
                np.savetxt('param.txt', convb, fmt='%1.10f', delimiter=', ', newline=',\n')
            with open('param.txt', 'r') as f:
                res += f.read()
            res += '};\n\n'
            count = count + 1
            print("count:")
            print(count)
    res += '#endif\n'

    return res


device = torch.device('cpu')

model = ResnetGenerator_rynew_2_41(1,2)
layers = get_parameters_layer(model)

net_names = [
    'featrues_deconv1',
    'featrues_deconv2_1',
    'featrues_deconv2_2',
    'featrues_deconv2_3',
    'featrues_deconv2_4',
    'featrues_deconv3',
    'featrues_deconv4_1',
    'featrues_deconv4_2',
    'featrues_deconv5',
    'featrues_deconv51x1',
    'featrues_deconv51x12',
    # 'featrues_deconv51x13',
    'featrues_resnet1_1',
    'featrues_resnet1_2',
    'featrues_resnet1_3',
    'featrues_resnet1_4',
    'featrues_deconv6',
    'featrues_deconv61x1',
    'featrues_deconv7',
    'featrues_deconv71x1',
    'featrues_resnet2_1',
    'featrues_resnet2_2',
    'featrues_resnet2_3',
    'featrues_resnet2_4',
    'featrues_resnet3_1',
    'featrues_resnet3_2',
    'featrues_resnet3_3',
    'featrues_resnet3_4',
    # 'features_fc',
    'featrues_upconv5',
    'featrues_upconv51x1',
    'featrues_upconv4',
    'featrues_upconv41x1',   
    'featrues_upconv4_1',
    'featrues_upconv4_2', 
    'featrues_upconv3_3x3',
    'featrues_upconv3_1x1',
    'featrues_upconv3_3x3_1',
    'featrues_upconv3_1x1_1',
    'featrues_upconv2',
    'featrues_upconv2_1',
    'featrues_upconv2_2',
    'featrues_upconv2_3',
    'featrues_upconv2_4',
    'featrues_upconv1_3x3',
    'featrues_upconv1_1x1',
    'featrues_upconv0',
    'featrues_sam1',
    'featrues_sam2',
]

checkpoint = torch.load(r'/home/zhangsn/enhance/02proj/checkpoints/netenlight/newGT_sm_241_15/280_net_G.pth', map_location='cpu')
# print(checkpoint)  #
# exit()
# netG = torch.nn.DataParallel(netG)
model.load_state_dict(checkpoint, strict=False)  #
model.eval()
# print(layers)
# exit()
param = GetConvParam(layers)
#
txt_p = open("enhance_param_sm24115_280.h", "w")
txt_p.write(param)
txt_p.close()
