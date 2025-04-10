set -ex
# python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
python3 train_nni.py --dataroot ../datasets/vivo --name fast_se_bottleneck9_d4_g4_pix2pix --model pix2pix --direction AtoB --batch_size 256  --load_size 128 --crop_size 128 --input_nc 1 --output_nc 1 --netG self --init_type kaiming --ndf 4 --ngf 4 --gpu_ids=7
