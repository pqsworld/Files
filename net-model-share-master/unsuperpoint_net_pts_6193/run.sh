#python3 train4.py train_base configs/train_RL.yaml 0118_correct_noise0_sift180_14_t20-5_93_nonorm_144_mask_e4800_nomatch_extend_ori5dist_cos_addFA_addnormalFA  --gpu_ids 7
python3 train4.py train_base configs/train_RL.yaml 0203_correct_noise0_sift180_14_GNN_t20-5_93_nonorm_144_mask_e4800_nomatch_extend_ori5dist_cos_addFA_wbmask_addnormalFA --gpu_ids 4


nohup python3 train4.py train_base configs/train_RL.yaml 08_24/unsuperpoint_160_48_0824 --gpu_ids 5 > unsuperpoint_160_48_0824.log 2>&1 &
