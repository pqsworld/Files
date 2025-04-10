
# --dataroot  测试图库根目录
# --phase     测试图库名称，图像路径：根目录/图库名称 
# --name      实验代号，将要进行测试的模型
# --epoch     测试模型的第几个epoch
# --model     pix2pix 这个无需改动，指定训练方式
# --direction AtoB 这个无需改动，指定训练方向
# --load_size --crop_size 写一样的值即可，图像缩放大小
# --input_nc --output_nc 写1即可，输入输出通道数
# --netG      模型名称，选择对应的模型
# --area      测试mask的面积，不加则测试DICE
# --one       测试图库图像分为两种形式，一种是[原图|mask图]，一种是只输入原图，--one代表此时只输入原图，测试时大多情况没有mask
python3 test.py --dataroot /hdd/file-input/yey/IMG/20241115-DF-v9.2-阳光-raw --phase=241118_1_522210213 --name 24_1010_zd_qp_499  --epoch=499  --model pix2pix --direction AtoB --batch_size 1 --load_size 96 --crop_size 96 --input_nc 1 --output_nc 1 --netG bufen --ndf 4 --ngf 4 --gpu_ids=-1 --area --one
