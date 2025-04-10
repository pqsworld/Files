## quick script to run export and evaluation

# export_folder='superpoint_coco_heat2_0_170k_nms4_det0.015'
# export_folder='superpoint_kitti_heat2_0'
# echo $export_folder

# output_folder="7_30_magicpoint_finally_Focal_35"
# output_log=${output_folder}_logs.txt

# python3 test.py export_detector configs/magicpoint_finger_test.yaml 08_30_magicpoint_finally_score-CBAM_Focal_60 --gpu_ids 1 --outputImg --PR 
# python3 train4.py train_base configs/magicpoint_shapes_pair.yaml 08_30_magicpoint_synth_score-CBAM_Focal_60 --gpu_ids 4


# python3 test.py export_detector configs/magicpoint_finger_test.yaml 09_01_magicpoint_finally_256C_Focal_60 --gpu_ids 1 --outputImg --PR 

# step1
# lambda_loss: 0,  warped_pair:enable:true,  homographic:enable:false
# python3 train4.py train_base configs/magicpoint_shapes_pair.yaml 1105_super/1105_magicpoint_synth_base_small_128


# step2
# [export.py] homography_adaptation:enable:true,num:100  export_folder:'val_out'(Depends on your folder name)   detection_threshold: 0.01(claim attention)   pretrained:'xxxx'
# [test.py]   
# python3 export.py export_detector_homoAdapt configs/magicpoint_finger_export.yaml 1126_supervised/1126_ronghefinger_checkpoint8000_output --gpu_ids 2  --outputImg
# python3 test.py export_detector configs/magicpoint_finger_test.yaml 1202_super/1208_test --gpu_ids 2 --outputImg

# step3
# lambda_loss: 10
# python3 train4.py train_base configs/train_external_data_enhance.yaml 1208_super/1208_train_2blocks --gpu_ids 3


# step4
# test repeatable & repeatable_tongshouzhi
# python3 test_repeatability.py export_detector configs/magicpoint_finger_test.yaml 1202_super/1203_repeatable --gpu_ids 6 --outputImg --PR
# python3 test_repeatability_tongshouzhi.py export_detector configs/repeatablity_finger_test_AHB.yaml 1117_super/1117_repeatable_tongshouzhi__check23000 --gpu_ids 5 --outputImg --PR

#step5
# test  warped match
# python3 test_warped_match.py export_detector configs/magicpoint_finger_test.yaml 1105_super/1105_warped_match --gpu_ids 6 



#========= 6159 step ==========

# python3 export_6159.py export_detector_homoAdapt configs/magicpoint_finger_export.yaml 1217_6159out/1216_export_sec38000 --gpu_ids 7 #--outputImg

# python3 train4.py train_base configs/magicpoint_shapes_pair.yaml 0420_train/0506_train_bs128_120pointlabels_6blocks_fulldescmethod2_addinnerloss --gpu_ids 4
# python3 train4.py train_base configs/magicpoint_shapes_pair.yaml 0609n_train_Ecnn_ext_patch_ne_th1.5bw_wg/0609n_train_bs32_nms1_thr0.2_lr0.001_correspond1.5_patch16_C4_odd_ext --gpu_ids 1

nohup python3 train4.py train_base configs/magicpoint_shapes_pair.yaml 0713a_train_Ecnn_ext_patch_ne_th1.5bw_wg/0713a_train_bs32_nms1_thr0.2_lr0.001_correspond1.5_patch16_C4_odd_ext --gpu_ids 3 &
nohup python3 train4.py train_base configs/magicpoint_shapes_pair.yaml 1008a_train_patch_debase_big/1008a_train_bs32_nms1_thr0.2_lr0.001_correspond1.5_patch16_C4_odd_ext --gpu_ids 6 &

nohup python3 train4.py train_base configs/magicpoint_shapes_pair_point.yaml 0826m_train_point/0826m_train_bs32 --gpu_ids 3 &
nohup python3 train4.py train_base configs/magicpoint_shapes_pair_angle.yaml 0903a_train_angle/0903a_train_bs32 --gpu_ids 4 &


# 单张图重复率测试
# python3 test_repeatability_cv2.py export_detector configs/magicpoint_finger_test.yaml 0105_out/0105_repeatability_DOG40000_singleimg_addnoise --gpu_ids 7 --outputImg --PR

#DOG 同手指重复率测试
# python3 test_repeatability_real_DOG.py export_detector configs/repeatablity_finger_test_AHB.yaml DOG_train_optimal_ratio/0105_real_repeatability_DOG40000_nms1_150pts --gpu_ids 2 --outputImg --PR

#displacement位移
# python3 test_repeatability_real.py export_detector configs/repeatablity_finger_test_AHB.yaml self_train/1231_real_repeatability_enhance1202train200000__200pts --gpu_ids 2 --outputImg --PR

# 测试网络在sift标签的精确度和召回率
# python3 test.py export_detector configs/magicpoint_finger_test.yaml FJY_onDOG/0112_prec_recall_DOG100000_firstDOGlayer --gpu_ids 5 --outputImg --PR
# python3 test.py export_detector configs/magicpoint_finger_test.yaml 0107_enhanceout/0112_prec_recall_DOG39800_firstDOGlayer --gpu_ids 5 --outputImg --PR