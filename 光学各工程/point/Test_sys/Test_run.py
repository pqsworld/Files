#!/usr/bin/python3 -utt
# -*- coding: utf-8 -*-
import os
import yaml
import torch
import logging
from models import FPDT
from Test_component import get_module

EXPER_PATH = 'logs'

default_config = {
    # 'testclass': 'Test_Real_Repeat_Enhance',    # 替换Test_component.py中的类即可
    # 'testclass': 'Test_Net_AccuracyandRepeat_Enhance',
    # 'testclass': 'Output_label',
    # 'testclass': 'Descriptor_Verify',
    # 'testclass': 'C_Test',
    'testclass': 'Get_Keypoint_Parameter_Short_7F',  
    # Test_Real_Repeat_Enhance_V2 Test_Real_Repeat_Enhance_6195_V2      
    # Get_Descriptor_Parameter_Ecnn Get_Descriptor_Parameter Get_Pts_Descriptor_From_Head Get_Descriptor_Parameter_Half Get_Descriptor_Parameter_Short Get_Keypoint_Parameter_Short
    # Get_Pts_Descriptor_From_Head2 Get_Descriptor_Parameter_Ecnn Get_Pts_Descriptor_From_HeadFA Head2Log Head2Log93 Generate_Enhance_Img 
    # Get_Descriptor_Code Get_Pnt_Code
    # Distribution_Candidate_Hit_Ratio_oridiff_ALikeWithHard_Ext93 Distribution_Candidate_Hit_Ratio_oridiff_ALikeWithHard_Ext93_Pnt Distribution_Candidate_Hit_Ratio_oridiff_ALikeWithHard_Ext95_Pnt
    # Distribution_Candidate_Hit_Ratio_oridiff_ALikeWithHard_Ext93_Pnt_Match Distribution_Candidate_Hit_Ratio_oridiff_ALikeWithHard_Ext93_Pnt_Match_wbSimi Distribution_Candidate_Hit_Ratio_oridiff_ALikeWithHard_Ext93_Pnt_Match_wbSimi_score
    # Output_Pts_Desc_Tool_Ext93 Output_Pts_Tool_Ext93 Output_Pts_Desc_Tool
    # Plot_Spoof_Match_Img

    # 'model': 'SuperPointNet_small_128_desc',
    'model': 'ALikeWithHardPnt',        # SuperPointNet_small_128_fulldesc  KeyNet ALikeWithHard ALikeKpt ALikeWithHardPnt ALikeWithHardPntMatch
    # 'model': 'SuperPointNet_small_128',
    # 'model': 'SuperPointNet_small_2',   # 2blocks
    # 'model': 'SuperPointNet_large',
    # 'resize': [136, 36],     # DOG 传统增强32->36是扩边，非resize
    # 'resize': [160, 36],     # enhance160*36 传统增强32->36是扩边，非resize，复现率测量数据
    'resize': [122, 30], # [136, 36],     # sift_enhance136*36 传统增强32->36是扩边，非resize，精度测量数据
    'isDilation': True, 
    'newLabels': False,
    'output_images': True,
    'output_ratio': True,
    'top_k': 130,
    'detec_thre': 0.2,
    'nms': 1,
    'w_size': 8,
    'augmentation':{
        'photometric':{
            'enable': False,
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

    pass


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5'
    GPU_ids = 0
    device = torch.device('cuda:' + str(GPU_ids))
    print("==> Use GPU: {}".format(GPU_ids))
    
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    info_path, image_path = None, None

    # image_path = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/C_test/'    # C代码测试

    # image_path = '/hdd/file-input/linwc/Descriptor/Test_Tool/v_1_0/enhance/'    # 描述子图
    # image_path = '/hdd/file-input/linwc/Descriptor/Test_Tool/v_1_0/pnt/'    # 点图
    # image_path = '/hdd/file-input/linwc/Descriptor/Test_Tool/v_1_0/pnt_debase/'    # 点图debase

    # image_path = '/hdd/file-input/linwc/Descriptor/data/repeat/'  # 测量精度,新标签
    # image_path = '/hdd/file-input/linwc/Descriptor/data/'
    # image_path = '/hdd/file-input/linwc/Descriptor/data/tmp/6191test_19p2_X8/'
    # image_path = '/hdd/file-input/liugq/6191/DB_X888rest/'
    # image_path = '/hdd/file-input/linwc/Descriptor/data/6191/process/'
    # image_path = '/hdd/file-input/linwc/Descriptor/data/6191/DB_X888rest/tmp_new/'
    # image_path = '/hdd/file-input/linwc/Descriptor/data/tmp/'
    # image_path = '/hdd/file-input/linwc/Descriptor/data/6193Test/6193_DK7_stitch/'
    image_path = '/home/linwc/match/data/6195Test/'

    # info_path = "/hdd/file-input/qint/data/toQt_Jy/DOG_01/trans.csv"
    # image_path = '/hdd/file-input/qint/data/toQt_Jy/sift/6159_cd_p11s400/val/pic/'  # 测量精度
    
    # image_path = '/hdd/file-input/qint/data/toQt_Jy/03_大数据库/images/val/' # 输出标签
    # model_weights_detector = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/superPointNet_158400_checkpoint.pth.tar'  # 增强数据训练网络（100800目前跑库性能最好）
    # model_weights_detector = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/0217_desc/superPointNet_31000_checkpoint.pth.tar'
    # model_weights_detector = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/superPointNet_100800_checkpoint.pth.tar'
    # model_weights_detector = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/superPointNet_180800_checkpoint.pth.tar'

    # model_weights_detector = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/newLabels/0324_train_bs128_afterselect_120pointlabels/superPointNet_74400_checkpoint.pth.tar'
    
    # model_weights_detector = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0425_train_bs128_120pointlabels_2blocks_notnorm_fulldesc/superPointNet_70200_checkpoint.pth.tar'
    # model_weights_detector = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0425_train_bs128_120pointlabels_2blocks_notnorm_fulldesc_Upsample/superPointNet_65400_checkpoint.pth.tar'
    # model_weights_detector = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0425_train_bs128_120pointlabels_2blocks_notnorm_fulldesc_addcosloss/superPointNet_67000_checkpoint.pth.tar'
    
    model_weights_detector = '/home/linwc/match/code/Test_sys/checkpoints/Des/95009_short.pth.tar' # 93061 95004 95005 95007 95008 95009
    # model_weights_detector = '/hdd/file-input/linwc/Descriptor/code/Test_sys/checkpoints/Pnt/0927a_600000_checkpoint.pth.tar'

    
    # model_weights_detector = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0425_train_bs128_120pointlabels_6blocks_notnorm_fulldesc_addcosloss/superPointNet_49800_checkpoint.pth.tar'
    # model_weights_detector = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0516_train_bs128_120pointlabels_6blocks_256desc_addinnerloss/superPointNet_78400_checkpoint.pth.tar'
    # model_weights_detector = '/hdd/file-input/qint/6159_parallel/Test_sys/checkpoints/Desc/0519_train6159_bs64_120pointlabels_6blocks_256desc_selfsup/superPointNet_113200_checkpoint.pth.tar'

    # output_dir = "enhance_train94000_smallnet_test/0128_repeatability_enhance94000_rotate_150pts"
    # output_dir = "0217_match/test"
    # output_dir = "0316_newLabels_test/0331_120pointlabels_focal_train190000_net150_repeat_rotate"
    # output_dir = "0407_newLabels_test/0411_120labels_129000_net150_S1_2_acc"
    # output_dir = "soft_NMS_compare/0224_train100800_repeat"
    # output_dir = "Desc/0428_120pointlabels_2blocks_fulldesc_Upsample_65400_repeat_rotate"
    # rdis: AtoB匹配对判断阈值; desc: 点置信度阈值; nms: nms筛点半径; dis: 真实重复率点对距离阈值
    # output_dir = "output/desc/0709n_alikewithhard_group_C4_c2_rdis2_nms1_dis2_Distribution_Candidate_Hit_Ratio_oridiff_ALikeWithHard_Ext93_248000_0a45_siftori_160t_5k_ne_short_new"
    # output_dir = "output/desc/95009_short_Pnt_0906a_222200_0.01_bord2_nms2_netori_float_netp_nn_615_normal_NTrans"
    # output_dir = "output/pnt/95009_short_Pnt_0906a_222200_0.01_bord2_nms2_netori_float_netp_615_DK7wet"
    # output_dir = "output/pnt/0927a_600000_thr0.01_nms2_t1_bord2_mask_netori_float"
    output_dir = "param/Pnt/"
    
    # output_dir = '/hdd/file-input/linwc/Descriptor/Test_Tool/v_1_0/sift_net/fast_patch_248000_16_int_project_0_mask96_3.5c_93_5k_0725_siftori_kp9800/'
    # output_dir = '/hdd/file-input/linwc/Descriptor/Test_Tool/v_1_0/net_net/alike_0927a_600000_91100_130_mask_-3500_0.01_bord2_nms2_siftori_float_netp/'

    # output_dir = '/hdd/file-input/linwc/Descriptor/data/6159_rot2_p10s300/desc_smaller_patch_98400_28_hadama_0000_thr/' # '/hdd/file-input/linwc/Descriptor/data/repeat/desc_small_patch_98400_28_hadama_0002/' 
    # output_dir = '/hdd/file-input/linwc/Descriptor/code/Test_sys/logs/param/square256/'
    # output_dir = '/hdd/file-input/linwc/Descriptor/code/Test_sys/logs/param/square256/'
    # output_dir = '/hdd/file-input/linwc/Descriptor/code/Test_sys/logs/param/PntAngle/'      # Pnt PntAngle
    # output_dir = '/hdd/file-input/linwc/Descriptor/code/Test_sys/logs/ccode/pnt/'
    # output_dir = '/hdd/file-input/linwc/Descriptor/data/6191/DB_X888rest/'

    output_dir = os.path.join(EXPER_PATH, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    default_config.update({
        'model_weights': model_weights_detector,
        'data': image_path,
        'output_dir': output_dir})
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)

    FPDT = FPDT(default_config['model'], model_weights_detector, device=device)

    processing(FPDT, image_path, info_path, device=device)
