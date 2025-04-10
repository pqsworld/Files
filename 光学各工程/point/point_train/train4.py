"""Training script
This is the training script for superpoint detector and descriptor.

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import argparse
import yaml
import os
import logging
from logging import handlers

import torch
import torch.optim
import torch.utils.data

from tensorboardX import SummaryWriter

# from utils.utils import tensor2array, save_checkpoint, load_checkpoint, save_path_formatter
from utils.utils import getWriterPath
from settings import EXPER_PATH

## loaders: data, model, pretrained model
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.logging import *
# from models.model_wrap import SuperPointFrontend_torch, PointTracker
# torch.backends.cudnn.enable =True
# torch.backends.cudnn.benchmark = True

###### util functions ######
def datasize(train_loader, config, tag='train'):
    logging.info('== %s split size %d in %d batches'%\
    (tag, len(train_loader)*config['model']['batch_size'], len(train_loader)))
    pass

from utils.loader import get_save_path

###### util functions end ######


###### train script ######
def train_base(config, output_dir, args):
    return train_joint(config, output_dir, args)
    pass

# def train_joint_dsac():
#     pass

def train_joint(config, output_dir, args):
    assert 'train_iter' in config

    # config
    # from utils.utils import pltImshow
    # from utils.utils import saveImg
    torch.set_default_tensor_type(torch.FloatTensor)
    task = config['data']['dataset']
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = args.gpu_ids
    device = torch.device('cuda:%s'%args.gpu_ids)
    # device = torch.device('cuda:%s'% args.gpu_ids)
    logging.info('train on device: %s', device)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    # writer = SummaryWriter(getWriterPath(task=args.command, date=True))
    writer = SummaryWriter(getWriterPath(task=args.command, 
        exper_name=args.exper_name, date=True))     # create an event file（in ./runs/train_base）
    ## save data
    save_path = get_save_path(output_dir)   # 保存checkpoints(in ./logs/magicpoint_synth/checkpoints)

    # data loading
    # data = dataLoader(config, dataset='syn', warp_input=True)
    data = dataLoader(config, dataset=task, warp_input=True)        # data中包含train&val数据
    train_loader, val_loader = data['train_loader'], data['val_loader']

    datasize(train_loader, config, tag='train')
    datasize(val_loader, config, tag='val')
    # init the training agent using config file
    # from train_model_frontend import Train_model_frontend
    from utils.loader import get_module
    train_model_frontend = get_module('', config['front_end_model'])                    # 加载的是Train_model_heatmap.py

    train_agent = train_model_frontend(config, save_path=save_path, device=device)      # Train_model_heatmap类初始化（__init__）

    # writer from tensorboard
    train_agent.writer = writer

    # feed the data into the agent
    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader

    # load model initiates the model and load the pretrained model (if any)
    train_agent.loadModel()     # train_model_frontend.py 初始化模型结构、设置训练方式（retrain or pretrained（.pth））
    train_agent.dataParallel()

    try:
        # train function takes care of training and evaluation
        train_agent.train()     # Train_model_frontend.py
    except KeyboardInterrupt:
        print ("press ctrl + c, save model!")
        train_agent.saveModel()
        pass

if __name__ == '__main__':
    # global var
    #os.environ['CUDA_VISIBLE_DEVICES']='5'
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # add parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train_base')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    p_train.set_defaults(func=train_base)

    # Training command
    p_train = subparsers.add_parser('train_joint')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.add_argument('--gpu_ids', type=str, default='2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    p_train.add_argument('--n_epochs', type=int, default=200, help='number of epochs with the initial learning rate')
    p_train.set_defaults(func=train_joint)

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    # EXPER_PATH from settings.py
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    # with capture_outputs(os.path.join(output_dir, 'log')):
    logging.info('Running command {}'.format(args.command.upper()))
    # sh = logging.StreamHandler()
    # fh = logging.FileHandler()
    # logging = logging.addHandler(sh)
    # logging = logging.addHandler(fh)
    args.func(config, output_dir, args)


