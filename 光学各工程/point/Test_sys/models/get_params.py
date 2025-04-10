import torch
import copy
from pathlib import Path
from collections import deque
from hardnet_model import *
# from models.hardnet_model import HardNet_fast_Ecnn
from collections import OrderedDict

NoneDP_param = OrderedDict()
checkpoint_desc = torch.load('/hdd/file-input/linwc/Descriptor/code/Test_sys/checkpoints/Des/0309a_ALikeWithHard_patch_group_inner_re_smallc4_ext_rt_nocut_nort_ne_140000_checkpoint.pth.tar', map_location=lambda storage, loc: storage)
for k in checkpoint_desc['model_state_dict'].keys():
    if 'descriptor_net' in k:
        NoneDP_param[k.replace('.module', '').replace('descriptor_net.', '')] = checkpoint_desc['model_state_dict'][k]

net = HardNet_fast_Ecnn(train_flag=False)

net.load_state_dict(NoneDP_param)
net.eval()
net.to('cpu')

# block1 = ['conv1x1s1_di_1', 'convdw3x3s2_1', 'conv1x1s1_dd_1']
# block2 = ['conv1x1s1_di_2', 'convdw3x3s2_2', 'conv1x1s1_dd_2']
# # block3 = ['conv1x1s1_di_3', 'convdw3x3s1_3', 'conv1x1s1_dd_3']
# # block4 = ['conv1x1s1_di_4', 'convdw3x3s1_4', 'conv1x1s1_dd_4', 'conv1x1_di_shortcut_4']
# # block5 = ['conv1x1s1_di_5', 'convdw3x3s1_5', 'conv1x1s1_dd_5']
# block5 = ['conv1x1s1_di_5', 'convdw3x3s1_5', 'conv1x1s1_dd_5', 'conv1x1_di_shortcut_5']
# block6 = ['conv1x1s1_di_6', 'convdw3x3s1_6', 'conv1x1s1_dd_6']
# block5 = ['conv1x1s1_di_7', 'convdw3x3s1_7', 'conv1x1s1_dd_7', 'conv1x1_di_shortcut_7']

# block1 = ['conv1x1s1_di_1', 'convdw3x3s2_1', 'conv1x1s1_dd_1']
# block2 = ['conv1x1s1_di_2', 'convdw3x3s2_2', 'conv1x1s1_dd_2']
# block3 = ['conv1x1s1_di_3', 'convdw3x3s1_3', 'conv1x1s1_dd_3']
# block4 = ['conv1x1s1_di_4', 'convdw3x3s1_4', 'conv1x1s1_dd_4', 'conv1x1_di_shortcut_4']
# # block5 = ['conv1x1s1_di_5', 'convdw3x3s1_5', 'conv1x1s1_dd_5']
# block5 = ['conv1x1s1_di_5', 'convdw3x3s1_5', 'conv1x1s1_dd_5', 'conv1x1_di_shortcut_5']
# block6 = ['conv1x1s1_di_6', 'convdw3x3s1_6', 'conv1x1s1_dd_6']
# block7 = ['conv1x1s1_di_7', 'convdw3x3s1_7', 'conv1x1s1_dd_7', 'conv1x1_di_shortcut_7']


block1 = ['conv1x1s1_di_1', 'convdw3x3s2_1', 'conv1x1s1_dd_1']
block2 = ['conv1x1s1_di_2', 'convdw3x3s2_2', 'conv1x1s1_dd_2']
block3 = ['conv1x1s1_di_3', 'convdw3x3s1_3', 'conv1x1s1_dd_3']
block4 = ['conv1x1s1_di_4', 'convdw3x3s1_4', 'conv1x1s1_dd_4', 'conv1x1_di_shortcut_4']
block5 = ['conv1x1s1_di_5', 'convdw3x3s1_5', 'conv1x1s1_dd_5', 'conv1x1_di_shortcut_5']
block6 = ['conv1x1s1_di_6', 'convdw3x3s1_6', 'conv1x1s1_dd_6']
block7 = ['conv1x1s1_di_7', 'convdw3x3s1_7', 'conv1x1s1_dd_7', 'conv1x1_di_shortcut_7']

postconv1x1 = ['conv1x1_post']

block = [block1, block2, block3, block4, block5, block6, block7, postconv1x1]

name = []
for i in block:
    name.extend(i)
def list_layers(layer):
    layers = []
    if isinstance(layer, EBlock) or isinstance(layer, enn.SequentialModule) or isinstance(layer, ESeModule):
        for i in layer.children():
            layers.extend(list_layers(i))
    elif isinstance(layer, enn.R2Conv) or isinstance(layer, enn.InnerBatchNorm) or isinstance(layer, enn.R2ConvTransposed):
        layers.append(layer.export().eval())
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

# print(net)
layers = get_parameters_layer(net)
params_num = 0
count = 0
res = '#if 1\n'
for i in range(len(layers)):
    conv = None
    if i + 1 < len(layers) and isinstance(layers[i], nn.Conv2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
        conv = nn.utils.fusion.fuse_conv_bn_eval(layers[i], layers[i + 1])
        # conv = layers[i]
        # conv.bias = torch.nn.Parameter(torch.zeros(conv.weight.shape[0]))
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
output_dir = '/hdd/file-input/linwc/Descriptor/code/Test_sys/logs/param/'
with open(output_dir + '0309a_ALikeWithHard_patch_group_inner_re_smallc4_ext_rt_nocut_nort_ne_140000.h', 'w') as f:
    f.write(res)
Path(r'param.txt').unlink()
