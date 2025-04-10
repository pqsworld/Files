"""many loaders
# loader for model, dataset, testing dataset
"""

import os
import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim
import torch.utils.data


from utils.utils import tensor2array, save_checkpoint, load_checkpoint, save_path_formatter
# from settings import EXPER_PATH

# from utils.loader import get_save_path
def get_save_path(output_dir):
    """
    This func
    :param output_dir:
    :return:
    """
    save_path = Path(output_dir)
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path

def worker_init_fn(worker_id):
   """The function is designed for pytorch multi-process dataloader.
   Note that we use the pytorch random generator to generate a base_seed.
   Please try to be consistent.

   References:
       https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

   """
   base_seed = torch.IntTensor(1).random_().item()
   # print(worker_id, base_seed)
   np.random.seed(base_seed + worker_id)


def dataLoader(config, dataset='syn', warp_input=False, train=True, val=True):
    import torchvision.transforms as transforms
    training_params = config.get('training', {})
    workers_train = training_params.get('workers_train', 0) # 16    get相当于一条if...else...语句。如果参数workers_train在字典training_params中，字典将返回training_params[workers_train];否则返回参数1。
    workers_val   = training_params.get('workers_val', 0) # 16
        
    logging.info(f"workers_train: {workers_train}, workers_val: {workers_val}") # 以 f开头表示在字符串内支持大括号内的python 表达式

    sizer = config['data']['preprocessing']['resize']
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((sizer[0], sizer[1])),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((sizer[0], sizer[1])),
        ]),
    }
    # if dataset == 'syn':
    #     from datasets.SyntheticDataset_gaussian import SyntheticDataset as Dataset
    # else:
    Dataset = get_module('datasets', dataset)   # 加载模块''datasets/SyntheticDataset_gaussian.py',此时Dataset和类SyntheticDataset_gaussian等价
    print(f"dataset: {dataset}")

    train_set = Dataset(
        transform=data_transforms['train'],
        task = 'train',
        **config['data'],
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config['model']['batch_size'], shuffle=True,
        pin_memory=True,
        num_workers=workers_train,
        worker_init_fn=worker_init_fn
    )
    val_set = Dataset(
        transform=data_transforms['train'],
        task = 'val',
        **config['data'],
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=config['model']['eval_batch_size'], shuffle=True,
        pin_memory=True,
        num_workers=workers_val,
        worker_init_fn=worker_init_fn
    )
    # val_set, val_loader = None, None
    return {'train_loader': train_loader, 'val_loader': val_loader,
            'train_set': train_set, 'val_set': val_set}

def dataLoader_test(config, dataset='syn', warp_input=False, export_task='train'):
    import torchvision.transforms as transforms
    training_params = config.get('training', {})
    workers_test = training_params.get('workers_test', 1) # 2
    logging.info(f"workers_test: {workers_test}")

    sizer = config['data']['preprocessing']['resize']
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((sizer[0], sizer[1])),
        ])
    }
    test_loader = None
    if dataset == 'syn':
        from datasets.SyntheticDataset import SyntheticDataset
        test_set = SyntheticDataset(
            transform=data_transforms['test'],
            train=False,
            warp_input=warp_input,
            getPts=True,
            seed=1,
            **config['data'],
        )
    elif dataset == 'hpatches':
        from datasets.patches_dataset import PatchesDataset
        if config['data']['preprocessing']['resize']:
            size = config['data']['preprocessing']['resize']
        test_set = PatchesDataset(
            transform=data_transforms['test'],
            **config['data'],
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False,
            pin_memory=True,
            num_workers=workers_test,
            worker_init_fn=worker_init_fn
        )
    # elif dataset == 'Coco' or 'Kitti' or 'Tum':
    else:
        # from datasets.Kitti import Kitti
        logging.info(f"load dataset from : {dataset}")
        Dataset = get_module('datasets', dataset)   # 打开的是Finger.py
        test_set = Dataset(
            export=True,
            transform=data_transforms['test'],
            task=export_task,
            **config['data'],
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False,
            pin_memory=True,
            num_workers=workers_test,
            worker_init_fn=worker_init_fn

        )
    return {'test_set': test_set, 'test_loader': test_loader}

def get_module(path, name):
    import importlib
    if path == '':
        mod = importlib.import_module(name)
    else:
        mod = importlib.import_module('{}.{}'.format(path, name))
    print(path, name)
    return getattr(mod, name)

def get_model(name):
    mod = __import__('models.{}'.format(name), fromlist=[''])
    return getattr(mod, name)

def modelLoader(model='SuperPointNet', **options):
    # create model
    logging.info("=> creating model: %s", model)
    net = get_model(model)
    net = net(**options)
    return net


# mode: 'full' means the formats include the optimizer and epoch
# full_path: if not full path, we need to go through another helper function
def pretrainedLoader(net, optimizer, epoch, path, mode='full', full_path=False):
    # load checkpoint
    if full_path == True:
        checkpoint = torch.load(path, map_location={'cuda:0':'cuda:7'})
    else:
        checkpoint = load_checkpoint(path)
    # apply checkpoint
    if mode == 'full':
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         epoch = checkpoint['epoch']
        epoch = checkpoint['n_iter']
#         epoch = 0
    else:
        net.load_state_dict(checkpoint)
        # net.load_state_dict(torch.load(path,map_location=lambda storage, loc: storage))
    return net, optimizer, epoch

if __name__ == '__main__':
    net = modelLoader(model='SuperPointNet')

