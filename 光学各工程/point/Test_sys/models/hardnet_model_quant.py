from this import d
from simplejson import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
# PTQ
from models.PTQ.layers import QConv2d, QAct, QLinear, QSequential, QBlock, QSeModule
from models.PTQ import ptq_config

import numpy as np

# # Group Convolution
# from models.splitgconv2d import P4ConvZ2, P4MConvZ2, P4ConvP4, P4MConvP4M

# # E(2) Group Convolution
# from typing import Tuple
# from e2cnn import gspaces
# from e2cnn import nn as enn
# from .utils import *

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # nn.init.orthogonal(m.weight.data, gain=0.6)
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant_(m.bias.data, 0.01)
        except:
            pass
    return

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class HardNet_fast_quant(nn.Module):
    """HardNet model definition
    """
    def __init__(self, cfg: ptq_config.Config, train_flag=False):
        super(HardNet_fast_quant, self).__init__()
        self.train_flag = train_flag
        self.config = cfg

        self.input_act = QAct(
            quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
        )
        self.features = QSequential(
            QBlock( 
                kernel_size=3, 
                in_size=int(1), 
                expand_size=int(8), 
                out_size=int(8),
                nolinear=hswish(), 
                semodule=QSeModule(int(8), cfg), 
                stride=2, 
                cfg=cfg
            ),
            QBlock( 
                kernel_size=3, 
                in_size=int(8), 
                expand_size=int(16), 
                out_size=int(16),
                nolinear=hswish(), 
                semodule=QSeModule(int(16), cfg), 
                stride=2, 
                cfg=cfg
            ),
            QBlock( 
                kernel_size=3, 
                in_size=int(16), 
                expand_size=int(32), 
                out_size=int(32),
                nolinear=hswish(), 
                semodule=QSeModule(int(32), cfg), 
                stride=1, 
                cfg=cfg
            ),
            QBlock( 
                kernel_size=3, 
                in_size=int(32), 
                expand_size=int(32), 
                out_size=int(32),
                nolinear=hswish(), 
                semodule=QSeModule(int(32), cfg), 
                stride=1, 
                cfg=cfg
            ),
            QBlock( 
                kernel_size=3, 
                in_size=int(32), 
                expand_size=int(32), 
                out_size=int(32),
                nolinear=hswish(), 
                semodule=QSeModule(int(32), cfg), 
                stride=1, 
                cfg=cfg
            ),
            QBlock( 
                kernel_size=3, 
                in_size=int(32), 
                expand_size=int(64), 
                out_size=int(64),
                nolinear=hswish(), 
                semodule=QSeModule(int(64), cfg), 
                stride=1, 
                cfg=cfg
            ),
            QConv2d(
                64,
                8,
                kernel_size=1,
                bias=False,
                quant=False,
                padding=0,
                bit_type=cfg.BIT_TYPE_W,
                calibration_mode=cfg.CALIBRATION_MODE_W,
                observer_str=cfg.OBSERVER_W,
                quantizer_str=cfg.QUANTIZER_W
            )
            
            # Block(kernel_size=3, in_size=int(1), expand_size=int(8), out_size=int(8),
            #          nolinear=hswish(), semodule=SeModule(int(8)), stride=2),
            # Block(kernel_size=3, in_size=int(8), expand_size=int(16), out_size=int(16),
            #          nolinear=hswish(), semodule=SeModule(int(16)), stride=2),
            # Block(kernel_size=3, in_size=int(16), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block(kernel_size=3, in_size=int(32), expand_size=int(32), out_size=int(32),
            #          nolinear=hswish(), semodule=SeModule(int(32)), stride=1),
            # Block(kernel_size=3, in_size=int(32), expand_size=int(64), out_size=int(64),
            #          nolinear=hswish(), semodule=SeModule(int(64)), stride=1),
            # nn.Conv2d(64, 8, kernel_size=1, padding=0)
        )
                
        if self.train_flag:
            self.features.apply(weights_init)

        self.l2norm_act =  QAct(
            quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
        )

        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    # 将非量化结构模型参数加载到量化结构上(需修改)
    def load_param_from_origin_pth(self, pth: str):
        origin_dict = torch.load(pth)['model_state_dict']
        origin_dict_keys = list(origin_dict.keys())
        new_dict = OrderedDict()
        count = 0
        idx = 0
        new_count = -4
        new_modules = list(self.modules())

        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                pass

            try:
                import unicodedata
                unicodedata.numeric(s)
                return True
            except (TypeError, ValueError):
                pass

            return False
        while idx < len(new_modules):
            m = new_modules[idx]
            if type(m) == QConv2d:
                middle_num = origin_dict_keys[count].split('.')[1]
                if is_number(middle_num):
                    new_key = origin_dict_keys[count].replace(middle_num, str(new_count))
                else:
                    new_key = origin_dict_keys[count]
                new_dict[new_key] = origin_dict[origin_dict_keys[count]]
                if m.bias != None:
                    count = count + 1
                    new_dict[new_key.replace('weight', 'bias')] = origin_dict[origin_dict_keys[count]]
                count = count + 1
                idx = idx + 1
                continue
            elif type(m) == nn.BatchNorm2d:
                for i in range(3):  # BN layer conisits
                    new_key = origin_dict_keys[count].replace(origin_dict_keys[count].split('.')[1], str(new_count))
                    new_dict[new_key] = origin_dict[origin_dict_keys[count]]
                    count = count + 1
            elif type(m) == QBlock:
                for i in range(30):
                    middle_num = origin_dict_keys[count].split('.')[1]                    
                    if is_number(middle_num):
                        new_key = origin_dict_keys[count].replace(middle_num, str(new_count)).replace('conv', 'qconv')
                    else:
                        new_key = origin_dict_keys[count].replace('conv', 'qconv')
                    # semodule
                    new_key = new_key.replace('se.2', 'se.3')
                    new_key = new_key.replace('se.1', 'se.2')
                    new_key = new_key.replace('se.5', 'se.7')
                    new_key = new_key.replace('se.4', 'se.6')
                    new_dict[new_key] = origin_dict[origin_dict_keys[count]]
                    count = count + 1
                idx = idx + 44
            elif type(m) == nn.ReLU:
                idx = idx + 1
                continue    
            new_count = new_count + 1
            idx = idx + 1

        try:
            self.load_state_dict(new_dict)
            print("[QHardNet_small]Translate old->quant model success!")
        except:
            print("[QHardNet_small]Translate old->quant model failed!")

    def model_quant(self):
        count = 0
        modules_name = self.get_quant_modules_name()
        
        def is_ban_module(m_name):
            # extra calculation
            ban_modules = [
                'l2norm_act',
                'conv1x1s1_block_dd_act_',  # 后面semodule的AGP
                'conv3x3s1_block_dw_act_',  # 后面semodule的AGP
                'conv1x1_se_di_act_',       # 后面hsigmoid
                'mul_act_',                 # 后面是不同scale的"+"操作
                'conv1x1s1_sc_act_'         # 后面是不同scale的"+"操作
            ]
            for m_prefix in ban_modules:
                if m_prefix in m_name:
                    return True
            return False
            
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                if not is_ban_module(modules_name[count]):
                    m.quant = True
                count = count + 1

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QAct]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QAct]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QAct]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QAct]:
                m.calibrate = False
                m.last_calibrate = False
                m.hist_mode = False
    
    def model_open_hist_mode(self):
        for m in self.modules():
            if type(m) in [QConv2d, QAct] and m.observer_str == 'kldiv':
                m.hist_mode = True

    def get_quant_modules_name(self):
        # module name
        modules_name = [
            'hardnet_input',
        ]
        block_temp = [
            'conv1x1s1_block_dd_', 
            'conv1x1s1_block_dd_act_',
            'conv1x1s1_block_dd_nolinear_act_',
            'conv3x3s1_block_dw_',
            'conv3x3s1_block_dw_act_',
            'conv3x3s1_block_dw_nolinear_act_',
            'conv1x1s1_block_di_',
            'conv1x1s1_block_di_act_',
            'semodule_pool_act_',
            'conv1x1_se_dd_',
            'conv1x1_se_dd_act_',
            'conv1x1_se_di_',
            'conv1x1_se_di_act_',
            'hsigmoid_',
            'mul_act_'
        ] 
        shortcut_temp_none = [
            'add_act_'
        ]
        shortcut_temp = [
            'conv1x1s1_sc_',
            'conv1x1s1_sc_act_',
            'add_act_'
        ]
        
        i = 0
        modules_name.extend([prefix.replace('3x3s1', '3x3s2') + str(i) for prefix in block_temp + shortcut_temp_none]) 
        i += 1
        modules_name.extend([prefix.replace('3x3s1', '3x3s2') + str(i) for prefix in block_temp + shortcut_temp_none]) 
        i += 1
        modules_name.extend([prefix + str(i) for prefix in block_temp + shortcut_temp_none]) 
        i += 1
        modules_name.extend([prefix + str(i) for prefix in block_temp + shortcut_temp]) 
        i += 1
        modules_name.extend([prefix + str(i) for prefix in block_temp + shortcut_temp]) 
        i += 1
        modules_name.extend([prefix + str(i) for prefix in block_temp + shortcut_temp_none]) 
        
        modules_name.extend([
            'conv1x1s1_dd',
            # 'conv1x1s1_dd_act_2',
            'l2norm_act'
        ])
        return modules_name

    # 提参函数（须修改）
    def get_parameters(self, param_path, begin_flag=True, end_flag=True):
        count = 0
        if begin_flag:
            res = '#if 1\n'
        else:
            res = ''
        modules_name = self.get_quant_modules_name()
        all_modules = list(self.modules())  
        for idx in range(len(all_modules)):
            m = all_modules[idx]        
            if type(m) in [QConv2d, QLinear, QAct]:
                if not m.quant:
                    count += 1
                    continue
                if type(m) == QConv2d:
                    if idx + 2 < len(all_modules) and type(all_modules[idx + 2]) == nn.BatchNorm2d:
                        conv = nn.utils.fusion.fuse_conv_bn_eval(m, all_modules[idx + 2])
                        weight, bias = conv.weight, conv.bias.detach().cpu().numpy().flatten()
                        bn_alpha = (conv.weight / m.weight).reshape(weight.shape[0], -1)[:, 0]
                        # print(bn_alpha, bn_alpha.shape)
                    else:
                        weight, bias = m.weight, m.bias.detach().cpu().numpy().flatten()
                        bn_alpha = None   

                    qweight = m.quantizer(weight, False, bn_alpha).detach().cpu().numpy().flatten()
                    res += 'static signed char ' + modules_name[count] + '_weight_int8[{0}] ='.format(qweight.shape[0]) + '{\n'
                    if qweight.shape[0] % 8 == 0:
                        np.savetxt('param.txt', qweight.reshape(-1, 8), fmt='%d',
                                    delimiter=', ', newline=',\n')
                    else:
                        np.savetxt('param.txt', qweight, fmt='%d',
                                    delimiter=', ', newline=',\n')
                    with open('param.txt', 'r') as f:
                        res += f.read()
                    res += "}; \n"

                    res += 'static float ' + modules_name[count] + '_bias[{0}] ='.format(bias.shape[0]) + '{\n'
                    if bias.shape[0] % 8 == 0:
                        np.savetxt('param.txt', bias.reshape(-1, 8), fmt='%1.10f',
                                    delimiter=', ', newline=',\n')
                    else:
                        np.savetxt('param.txt', bias, fmt='%1.10f',
                                    delimiter=', ', newline=',\n')
                    with open('param.txt', 'r') as f:
                        res += f.read()
                    res += "}; \n"

                    dequant_scale = m.quantizer.dequant_scale
                    dequant_scale = dequant_scale * bn_alpha if not bn_alpha is None else dequant_scale
                    dequant_scale = dequant_scale.detach().cpu().numpy().flatten()
                    res += 'static float ' + modules_name[count] + '_dqscale[{0}] ='.format(dequant_scale.shape[0]) + '{\n'
                    if dequant_scale.shape[0] % 8 == 0:
                        np.savetxt('param.txt', dequant_scale.reshape(-1, 8), fmt='%1.10f',
                                    delimiter=', ', newline=',\n')
                    else:
                        np.savetxt('param.txt', dequant_scale, fmt='%1.10f',
                                    delimiter=', ', newline=',\n')
                    with open('param.txt', 'r') as f:
                        res += f.read()
                    res += "}; \n"
                    
                else:
                    scale, zero_point = m.quantizer.scale.detach().cpu().numpy().flatten(), m.quantizer.zero_point.detach().cpu().numpy().flatten()
                    res += 'static float ' + modules_name[count] + '_qscale[{0}] ='.format(scale.shape[0]) + '{\n'
                    if scale.shape[0] % 8 == 0:
                        np.savetxt('param.txt', scale.reshape(-1, 8), fmt='%1.10f',
                                    delimiter=', ', newline=',\n')
                    else:
                        np.savetxt('param.txt', scale, fmt='%1.10f',
                                    delimiter=', ', newline=',\n')
                    with open('param.txt', 'r') as f:
                        res += f.read()
                    res += "}; \n"

                    # res += modules_name[count] + '_qzero_point[] = {\n'
                    # if zero_point.shape[0] % 8 == 0:
                    #     np.savetxt('param.txt', zero_point.reshape(-1, 8), fmt='%1.10f',
                    #                 delimiter=', ', newline=',\n')
                    # else:
                    #     np.savetxt('param.txt', zero_point, fmt='%1.10f',
                    #                 delimiter=', ', newline=',\n')
                    # with open('param.txt', 'r') as f:
                    #     res += f.read()

                    # res += "}; \n"
                count = count + 1
                # last_m = m
        if end_flag:
            res += '#endif\n'
        if begin_flag:    
            with open(param_path, 'w') as f:
                f.write(res)
        else:
            with open(param_path, 'a') as f:
                f.write(res)

    def forward(self, input, angle=0):
        input_n = self.input_norm(input)
        input_n = self.input_act(input_n)
        x_features = self.features(input_n, self.input_act.quantizer.scale)      # bx8x4x4
        b, o, _, _ = x_features.shape
        # print('x_features.shape: ', x_features.shape
        x_features_trans = x_features.permute(0, 2, 3, 1).contiguous().view(b, -1, o)
        if angle == 1:
            x_features_expand = torch.flip(x_features_trans, dims=[1])
            x = x_features_expand.view(x_features_expand.size(0), -1)
        else:
            x = x_features_trans.view(x_features_trans.size(0), -1)

        x_l2norm = L2Norm()(x)
        x_l2norm = self.l2norm_act(x_l2norm)
        # x_features = self.input_norm(input)
        # # print('x_features.shape: ', x_features.shape)
        # x = x_features.view(x_features.size(0), -1)
        return x_l2norm