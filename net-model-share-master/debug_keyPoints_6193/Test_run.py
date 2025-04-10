#!/usr/bin/python3 -utt
# -*- coding: utf-8 -*-
import os
import yaml
import torch
import random
# import pynvml
import logging
from models import FPDT
from Test_component import get_module

EXPER_PATH = 'logs'
'''
切patch: HardNet_fast, net_patch->True is256->False(double 128) e2cnn_switch->False
密集型:  ReHardNet_fast_featuremap.ReHardNet_fast_featuremap_depth, net_patch->False is256->True e2cnn_switch->True
'''
default_config = {
    # 'testclass': 'Test_Real_Repeat_Enhance',    # 替换Test_component.py中的类即可
    # 'testclass': 'Test_Real_Repeat_Enhance_V2',    # 真实复现率: 2224对跑库数据(136*36)：6159_p21s600_data--Sift_transSucc
    # 'testclass': 'Test_Net_AccuracyandRepeat_Enhance',
    # 'testclass': 'My_Test_e2cnn',
    # 'testclass': 'C_Test',
    'testclass': 'Descriptor_Hamming_Test_V6',  # 测试网络描述子汉明距，输出汉明百分比/命中率
    # 'testclass': 'FAFR_common_data_Test',  # FA/FR trans csv文件读取并分析

    # 'testclass': 'Output_trans_from_h',   # 读取从跑库工具输出的trans信息,规整为测试/训练格式 -> .csv, len33ROC
    # 'testclass': 'Merge_Files_and_LRtrain',     # 合并fafrlog并训练分类器
    # 'model': 'SuperPointNet_small_128_desc',    # 100800
    # 'model': 'SuperPointNet_small_128_CBAM',
    # 'model': 'SuperPointNet_small_128',
    # 'model': 'HardNet',   
    # 'model': 'ReHardNet_fast_featuremap',   
    # 'model': 'ReHardNet_fast_featuremap.ReHardNet_fast_featuremap_ghost',   # 深度卷积5blocks + 1ghost
    # 'model': 'ReHardNet_fast_featuremap.ReHardNet_fast_featuremap_depth',   # 深度卷积blocks
    # 'model': 'HardNet_fast',      # 切patch
    'model': 'HardNet_fast_256.HardNet_fast_256',      # 256维描述子
    # 'model': 'HardNet_fast_256.HardNet_fast_256_Overparam',      # 重参数化
    # 'model': 'HardNet_fast_256.HardNet_fast_256_double',      # 大教师模型 double channel
    # 'model': 'HardNet_fast_256.HardNet_fast_128_double_rect',      # 大教师模型 double channel 128 长条形
    # 'model': 'HardNet_fast_256.HardNet_fast_256_double_MSAD',      # 大教师模型 CFF
    # 'model': 'HardNet_fast_256.HardNet_fast_256_double_MSAD_2T',      # 大教师模型 单独训练第三个teacher
    # 'model': 'HardNet_fast_Pconv',      # 切patch

    # 'model': 'SuperPointNet_small_128_fulldesc',
    # 'resize': [136, 36],        # DOG 传统增强32->36是扩边，非resize
    # 'resize': [160, 36],        # enhance160*36 传统增强32->36是扩边，非resize，复现率测量数据
    # 'resize': [136, 36],        # sift_enhance136*36 传统增强32->36是扩边，非resize，精度测量数据
    'isDilation': False,        # True: 图像送入网络的尺寸136*40 False: 136*32
    'newLabels': False,
    'output_images': False,
    'output_ratio': True,

    'isRectangle': False,       # <<< 是否是长条形patch描述子 长条形只需要128维描述子
    'is256': True,              # <<< 256dim desc
    'net_patch': True,          # True: patch输入网络 False: 全尺寸的密集描述子网络
    'set_144x52': False,        # 136x36 -> 144x52 切换到扩边尺寸
    'patch_size': 16,
    'sample_size': 22,          # <<<

    'v9800': True,      # 采用9800识别版本 增强融合图+点, trans基于9300旧数据

    'im6159_rot2_p10s300': False,   # 6159旋转库
    'im6159_p21s600': False,        # 小位移
    'im6191_DK7_rot': False,         # True:6191_7201图
    'im6191_DK7_rot_7204': False,           # 7204增强版本的rot测试数据(6191_DK7-140_Mul_8_rot_p6_7024)
    'im6191_DK7_rot_7204_extend': False,     # 7204增强版本的rot测试数据(描述子扩边)(6191_DK7-140_Mul_8_rot_p6_7024)
    'im91_DK4_cd_p8s120_extend': False,     # 7204增强版本的正常手指测试数据
    'im6193_extend': True,                 # 6193版本，多个数据库（128x52）
    'internal_images': False,   # True: 内部测试库db_811_p10s200

    'e2cnn_switch': False,       # True: 模型为群等变卷积 False: patch图输入网络
    'top_k': 130,
    'detec_thre': 0.01,
    'nms': 2,
    'dis_thr': 2,               # 点对距离满足<self.dis_thr认为是匹配对
    'augmentation':{
        'photometric':{
            'enable': True,
            'primitives': [
                'random_brightness', 'random_contrast', 'additive_speckle_noise',
                'additive_gaussian_noise', 'additive_shade', 'motion_blur' ],
            'params':{
                'random_brightness': {'max_abs_change': 12},
                'random_contrast': {'strength_range': [0.6, 1.4]},
                'additive_gaussian_noise': {'stddev_range': [0, 10]},
                'additive_speckle_noise': {'prob_range': [0, 0.0035]},
                'additive_shade':{
                    'transparency_range': [-0.5, 0.8],
                    'kernel_size_range': [50, 100]
                },
                'motion_blur': {'max_kernel_size': 7}
            }
        },
        'homo':{
            'enable': True,
            'params':{
                'max_angle': 30,
                'n_angles': 25,
            }
        }
    }
}

def processing(FPDT, img_path=None, info_path=None, device="cpu"):

    Getclass = get_module(default_config['testclass'])
    TestComponent = Getclass(img_path, info_path, device, **default_config)  # 初始化测试类
    TestComponent.test_process(FPDT)
    # TestComponent.test_process_hardnet(FPDT)

    pass


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES']='5'
    # GPU_ids = 6
    lucky_gpu = random.choice([7])
    device = torch.device('cuda:' + str(lucky_gpu))
    print("==> Use GPU: {}".format(lucky_gpu))
    
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    root_path= '/hdd/file-input/qint/6159_parallel/6159_hardnet_self/'
    info_path = '/hdd/file-input/qint/data/6193Test/'
    image_path = None
    model_weights_detector = None
    output_dir = "test"

    # image_path = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/C_test/'    # C代码测试
    # image_path = '/home/qint/data/toQt_Jy/DescribeToJy_Qt/'    # 点&描述子标签
    # image_path = '/hdd/file-input/qint/data/toQt_Jy/newLabelLog/'  # 测量精度,新标签

    '''##6193##''' 
    '''CReLU oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230418_patch_6193/0418_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_Crelu/46000'''
    # model_weights_detector = root_path + "logs/20230418_patch_6193/0418_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_Crelu/checkpoints/superPointNet_46000_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230418_patch_6193/0418_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_Crelu_TEST46000_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''CReLU-V2 oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230418_patch_6193/0418_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_CreluV2/41200/58800/102400/118000/167400/176400/180600'''
    # model_weights_detector = root_path + "logs/20230418_patch_6193/0418_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_CreluV2/checkpoints/superPointNet_180600_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230418_patch_6193/0418_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_CreluV2_TEST180600_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''b126CReLU-V2 oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230418_patch_6193/0419_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_b126CreluV2/55000/79600/84800/89800/130600/152000/177800/186200/206200'''
    # model_weights_detector = root_path + "logs/20230418_patch_6193/0419_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_b126CreluV2/checkpoints/superPointNet_206200_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230418_patch_6193/0419_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_b126CreluV2_TEST206200_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''b1236CReLU-V2 oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230418_patch_6193/0421_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_b1236CreluV2/79600/99800/113600/132200/148400/167400/177800/186200/193400'''
    # model_weights_detector = root_path + "logs/20230418_patch_6193/0421_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_b1236CreluV2/checkpoints/superPointNet_193400_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230418_patch_6193/0421_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_b1236CreluV2_TEST193400_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''All-CReLU-V2 oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230418_patch_6193/0426_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_AllCreluV2/41200/50400/136000/160600/188600/202400'''
    # model_weights_detector = root_path + "logs/20230418_patch_6193/0426_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_AllCreluV2/checkpoints/superPointNet_160600_checkpoint.pth.tar"
    # # output_dir = root_path + "logs/20230418_patch_6193/0426_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_AllCreluV2_TEST160600_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    # output_dir = root_path + "logs/20230418_patch_6193/0426_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_AllCreluV2_TEST160600_Ctest"
    
    '''Half-b123CReLU-V2 oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230512_patch_6193/0512_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-b123CReLU/130600/141800/159800/169800/176400/186200/201200'''
    # model_weights_detector = root_path + "logs/20230512_patch_6193/0512_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-b123CReLU/checkpoints/superPointNet_201200_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230512_patch_6193/0512_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-b123CReLU_TEST201200_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''Half-b12345CReLU-V2 oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230512_patch_6193/0512_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-b12345CReLU/136000/145200/157000/169800'''
    # model_weights_detector = root_path + "logs/20230512_patch_6193/0512_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-b12345CReLU/checkpoints/superPointNet_169800_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230512_patch_6193/0512_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-b12345CReLU_TEST169800_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''Half oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230512_patch_6193/0515_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half/42400/109800/113600/127200/141800/145200/160600/176400/186200/188600'''
    # model_weights_detector = root_path + "logs/20230512_patch_6193/0515_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half/checkpoints/superPointNet_145200_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230512_patch_6193/0515_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half_TEST145200_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''Half-b12CReLU-V2 oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230512_patch_6193/0515_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-b12CReLU/45800'''
    # model_weights_detector = root_path + "logs/20230512_patch_6193/0515_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-b12CReLU/checkpoints/superPointNet_45800_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230512_patch_6193/0515_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-b12CReLU_TEST45800_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''Over-2param  Half oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230512_patch_6193/0518_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-2preparam/38800/50200/121600/137800/141800/159800/186200'''
    # model_weights_detector = root_path + "logs/20230512_patch_6193/0518_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-2preparam/checkpoints/superPointNet_186200_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230512_patch_6193/0518_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-2preparam_TEST186200_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''Over-4param  Half oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230512_patch_6193/0518_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-4preparam/127200/147200/149000/152000/169800/176400/177600/186200/188600/201200/208800'''
    # model_weights_detector = root_path + "logs/20230512_patch_6193/0518_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-4preparam/checkpoints/superPointNet_201200_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230512_patch_6193/0518_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-4preparam_TEST201200_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''Over-6param  Half oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230512_patch_6193/0518_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-6preparam/79600/99800/112200/127200/160600'''
    # model_weights_detector = root_path + "logs/20230512_patch_6193/0518_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-6preparam/checkpoints/superPointNet_160600_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230512_patch_6193/0518_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_half-6preparam_TEST160600_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''teacher model T=20  double channel oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230524_patch_6193/0524_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_doubleC/160600/176400/184800/188600/202600'''
    # model_weights_detector = root_path + "logs/20230524_patch_6193/0524_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_doubleC/checkpoints/superPointNet_176400_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230524_patch_6193/0524_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_doubleC_TEST176400_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''repvgg  oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230524_patch_6193/0530_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_repvgg/38800/76400/167400'''
    # model_weights_detector = root_path + "logs/20230524_patch_6193/0530_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_repvgg/checkpoints/superPointNet_167400_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230524_patch_6193/0530_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_repvgg_TEST167400_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''DK  oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230524_patch_6193/0601_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DK/41200/55400/145200/166600/186200'''
    # model_weights_detector = root_path + "logs/20230524_patch_6193/0601_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DK/checkpoints/superPointNet_186200_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230524_patch_6193/0601_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DK_TEST186200_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''DKx2  oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230524_patch_6193/0601_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DK_x2/36400/127200/160600/177800'''
    # model_weights_detector = root_path + "logs/20230524_patch_6193/0601_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DK_x2/checkpoints/superPointNet_177800_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230524_patch_6193/0601_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DK_x2_TEST177800_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''DKx2_disAx2  oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230524_patch_6193/0602_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DKx2_disAx2/41200/156400/160400/170400'''
    # model_weights_detector = root_path + "logs/20230524_patch_6193/0602_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DKx2_disAx2/checkpoints/superPointNet_170400_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230524_patch_6193/0602_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DKx2_disAx2_TEST170400_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''DKx8  oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230524_patch_6193/0602_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DKx8/41200/157200/160000'''
    # model_weights_detector = root_path + "logs/20230524_patch_6193/0602_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DKx8/checkpoints/superPointNet_160000_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230524_patch_6193/0602_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DKx8_TEST160000_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''DKx2_stu-2p-reparam  oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230524_patch_6193/0602_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DKx2_stu-2p-reparam/38800/128600/170000'''
    # model_weights_detector = root_path + "logs/20230524_patch_6193/0602_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DKx2_stu-2p-reparam/checkpoints/superPointNet_170000_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230524_patch_6193/0602_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DKx2_stu-2p-reparam_TEST170000_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''DK_softmax T=20  oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230524_patch_6193/0606_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_contraDK1d5/41200/55400/61400/136000/141800/160600/177800/186200'''
    # model_weights_detector = root_path + "logs/20230524_patch_6193/0606_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_contraDK1d5/checkpoints/superPointNet_186200_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230524_patch_6193/0606_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_contraDK1d5_TEST186200_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''DKx8_disAx20 T=20  oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230524_patch_6193/0608_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DKx8_disAx20/45800/136000/176400'''
    # model_weights_detector = root_path + "logs/20230524_patch_6193/0608_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DKx8_disAx20/checkpoints/superPointNet_136000_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230524_patch_6193/0608_bs32_t20_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_DKx8_disAx20_TEST136000_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''teacher model sample22 T=25  double channel oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230609_patch_6193/0609_bs32_t25_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_doubleC/139600/147200'''
    # model_weights_detector = root_path + "logs/20230609_patch_6193/0609_bs32_t25_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_doubleC/checkpoints/superPointNet_147200_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230609_patch_6193/0609_bs32_t25_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_doubleC_TEST147200_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''teacher model sample22 T=30  double channel oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230609_patch_6193/0612_bs32_t30_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_doubleC/141800/166000/188800'''
    # model_weights_detector = root_path + "logs/20230609_patch_6193/0612_bs32_t30_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_doubleC/checkpoints/superPointNet_188800_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230609_patch_6193/0612_bs32_t30_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_doubleC_TEST188800_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''DK_softmax T=25 oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230609_patch_6193/0614_bs32_t25_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_Tnew_contraDKx4/127200/137800//141800/157000/169800/177800/188600'''
    # model_weights_detector = root_path + "logs/20230609_patch_6193/0614_bs32_t25_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_Tnew_contraDKx4/checkpoints/superPointNet_188600_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230609_patch_6193/0614_bs32_t25_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_Tnew_contraDKx4_TEST188600_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''teacher model T=25  noiseXY1 double channel oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230609_patch_6193/0620_bs32_t25_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_doubleC_noisexy1/143000/172000/186200'''
    # model_weights_detector = root_path + "logs/20230609_patch_6193/0620_bs32_t25_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_doubleC_noisexy1/checkpoints/superPointNet_186200_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230609_patch_6193/0620_bs32_t25_nms1d5_noimgenh_bw0d85_diffori5_binmean_3x3_v9800_d256_doubleC_noisexy1_TEST186200_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''Rect256 teacher model T=25   double channel oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230609_patch_6193/0626_bs32_t25_nms1d5_noimgenh_noFA_binmean_3x3_v9800_d256_doubleC_rect/130600/175600'''
    # model_weights_detector = root_path + "logs/20230609_patch_6193/0626_bs32_t25_nms1d5_noimgenh_noFA_binmean_3x3_v9800_d256_doubleC_rect/checkpoints/superPointNet_175600_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230609_patch_6193/0626_bs32_t25_nms1d5_noimgenh_noFA_binmean_3x3_v9800_d256_doubleC_rect_TEST175600_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''cross-cat-soft doubleC model T=25   oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230609_patch_6193/0627_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d256_doubleC_crosssoft/99800/135200'''
    # model_weights_detector = root_path + "logs/20230609_patch_6193/0627_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d256_doubleC_crosssoft/checkpoints/superPointNet_135200_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230609_patch_6193/0627_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d256_doubleC_crosssoftt_TEST135200_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''Rect-128 doubleC  T=25   oldH v9800增强图   No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230609_patch_6193/0628_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d128_doubleC_rect/69800/156200/158600/166000'''
    # model_weights_detector = root_path + "logs/20230609_patch_6193/0628_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d128_doubleC_rect/checkpoints/superPointNet_166000_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230609_patch_6193/0628_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d128_doubleC_rect_TEST166000_2x128_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''Rect-256 doubleC fa_bwcoefloss T=25   oldH v9800增强图   No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230609_patch_6193/0705_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d128_doubleC_rect_fa_bwcoef/46000/55400'''
    # model_weights_detector = root_path + "logs/20230609_patch_6193/0705_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d128_doubleC_rect_fa_bwcoef/checkpoints/superPointNet_55400_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230609_patch_6193/0705_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d128_doubleC_rect_fa_bwcoef_TEST55400_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''DK random±2_samplesize T=25   oldH v9800增强图   No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230609_patch_6193/0703_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d256_DK_rand2size/53800/137800/145200'''
    # model_weights_detector = root_path + "logs/20230609_patch_6193/0703_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d256_DK_rand2size/checkpoints/superPointNet_145200_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230609_patch_6193/0703_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d256_DK_rand2size_TEST145200_256_n2d2_extend_V6_test1_9800_floatorient_oldH"
    
    '''teacher model sample-28 T=25  double channel oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 20230609_patch_6193/0704_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d256_doubleC_samp28//84800/93000'''
    # model_weights_detector = root_path + "logs/20230609_patch_6193/0704_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d256_doubleC_samp28/checkpoints/superPointNet_93000_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230609_patch_6193/0704_bs32_t25_nms1d5_noimgenh_binmean_3x3_v9800_d256_doubleC_samp28_TEST93000_256_n2d2_p16s28_V6_test1_9800_floatorient_oldH"
    
    '''###### MSAD ######'''
    '''teacher model CFF sample-XL:28 L:22 T=25  double channel oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 /158200/199400/254400/257400'''
    # root_path= '/hdd/file-input/qint/6159_parallel/6193_net_DK/'
    # model_weights_detector = root_path + "logs/20230609_patch_6193/0707_bs16_t25_nms1d5_3x3_v9800_d256_doubleC_h28l22_CFF/checkpoints/superPointNet_257400_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230609_patch_6193/0707_bs16_t25_nms1d5_3x3_v9800_d256_doubleC_h28l22_CFF_TEST257400_256_n2d2_p16s22_V6_test1_9800_floatorient_oldH"
    
    '''teacher model CFF by two-teacher(xx_2T) T=25  double channel oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 /137800/177800'''
    # root_path= '/hdd/file-input/qint/6159_parallel/6193_net_DK/'
    # model_weights_detector = root_path + "logs/20230609_patch_6193/0706_bs32_t25_nms1d5_bys22s28_3x3_v9800_d256_doubleC/checkpoints/superPointNet_177800_checkpoint.pth.tar"
    # output_dir = root_path + "logs/20230609_patch_6193/0706_bs32_t25_nms1d5_bys22s28_3x3_v9800_d256_doubleC_TEST177800_256_n2d2_p16s22_V6_test1_9800_floatorient_oldH"
    
    '''DK Stu-model from CFF-h28l22-net254400 T=25  double channel oldH v9800增强图   d256 + No-imgA-enh + 3x3mesh-points + fa_frdrop patch-HardNet_fast_256 /49600/127200/136000/137800/141800'''
    root_path= '/hdd/file-input/qint/6159_parallel/6193_net_DK/'
    model_weights_detector = root_path + "logs/20230609_patch_6193/0710_bs32_t25_nms1d5_3x3_v9800_d256_teach-h28l22-CFF_DKx4/checkpoints/superPointNet_141800_checkpoint.pth.tar"
    output_dir = root_path + "logs/20230609_patch_6193/0710_bs32_t25_nms1d5_3x3_v9800_d256_teach-h28l22-CFF_DKx4_TEST141800_256_n2d2_p16s22_V6_test1_9800_floatorient_oldH"
    

    output_dir = os.path.join(EXPER_PATH, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print('Output-> ', output_dir)

    default_config.update({
        'model_weights': model_weights_detector,
        'data': image_path,
        'output_dir': output_dir})
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)

    FPDT = FPDT(default_config['model'], model_weights_detector, device=device)

    processing(FPDT, image_path, info_path, device=device)
