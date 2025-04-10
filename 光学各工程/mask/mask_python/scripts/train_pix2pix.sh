set -ex
# --dataroot  训练数据路径
# --name      实验代号
# --model     pix2pix 这个无需改动，指定训练方式
# --direction AtoB 这个无需改动，指定训练方向
# --load_size --crop_size 写一样的值即可，图像缩放大小
# --input_nc --output_nc 写1即可，输入输出通道数
# --netG      模型名称，选择对应的模型
python3 train.py --dataroot /ssd/share/mask_train/datasets/bufen_danbei/ --name 21_oppoyg --model pix2pix --direction AtoB --batch_size 256  --load_size 96 --crop_size 96 --input_nc 1 --output_nc 1 --netG bufen --init_type kaiming --ndf 4 --ngf 4 --gpu_ids=2