set -ex
python test.py --dataroot ./datasets/scratch188x188/test/ --name scratch188x188 --model test --netG unet_256 --direction AtoB --dataset_mode single --norm batch --gpu_ids=-1
