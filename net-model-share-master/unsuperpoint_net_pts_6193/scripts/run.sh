
step1:
python3 train4.py train_base configs/train.yaml magicpoint_synth --gpu_ids 1

finally:
python3 test_pair.py export_detector configs/magicpoint_finger_test.yaml 0728_magicpoint_synth_0.68_addtionloss --outputImg --PR

python3 test.py export_detector configs/magicpoint_finger_test_PR.yaml 0728_magicpoint_synth_0.68_addtionloss --outputImg --PR