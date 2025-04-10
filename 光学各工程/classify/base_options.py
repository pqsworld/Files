import argparse
import os
import torch
import models
import config


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--data_train', type=str,default='/hdd/file-input/liugq/datasets/6195/train_wet',help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--data_test', type=str, default='/hdd/file-input/liugq/datasets/6197/frr_6197_chuanyin_cut95_test_crop',help='path to images')
        parser.add_argument('--name', type=str, default='smallttl_debase0', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')

        parser.add_argument('--lr_policy', type=str, default='step',
                            help='learning rate policy. [linear | step | plateau | cosine]')

        parser.add_argument('--lr', default=0.005, type=float, help='start learning rate')
        parser.add_argument('--epoch', default=200, type=int, help='number of epochs')
        parser.add_argument('--batchsize', default=1024, type=int, help='batch size in each context')
        # parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda exist')
        parser.add_argument('--checkpoints_dir', default='./checkpoints', type=str, help='model path')
        parser.add_argument('--model', default="mnv_small1", type=str, help='model path')
        parser.add_argument('--optim_choose', default="Adam", type=str, help='Adam / SGD')
        parser.add_argument('--lr_decay_iters', type=int, default=2)
        parser.add_argument('--testmodel', type=str, default='')
        parser.add_argument('--save_res', type=str, default='./results')
        parser.add_argument('--test_datas', type=str, default='name0')
        parser.add_argument('--inputchannels', type=int, default=2)
        
        parser.add_argument('--demo_flag', type=bool, default=False)
        parser.add_argument('--loadmodel_flag', type=bool, default=False)
        parser.add_argument('--zerolabel_loss', type=bool, default=False)
        parser.add_argument('--threeflag', type=bool, default=False)
        parser.add_argument('--three_path', type=str, default='/hdd/file-input/liugq/datasets/6197/saveimgs',help='path to images')
        parser.add_argument('--trainroot2', type=str, default='')
        parser.add_argument('--enlarge_flag', type=bool, default=True)
        parser.add_argument('--teacher_flag', type=bool, default=False)
        parser.add_argument('--transfer_flag', type=bool, default=False)
        parser.add_argument('--random_data', type=bool, default=False)
        parser.add_argument('--batch_crop', type=bool, default=False)
        parser.add_argument('--add_smallarea', type=bool, default=False)
        parser.add_argument('--smallroot', type=str, default='/ssd/share/liugq/datas/train620',help='path to images')
        parser.add_argument('--add_threeroot', type=bool, default=False)
        parser.add_argument('--extraroot', type=str, default='',help='path to images')
        parser.add_argument('--save_epoch', type=bool, default=True)
        parser.add_argument('--crop_flag', type=bool, default=True)
        parser.add_argument('--width', type=int, default=66)
        parser.add_argument('--weight_flag', type=bool, default=False)
        parser.add_argument('--imsize',type=int,default=66)
        parser.add_argument('--rm_num',type=int,default=1)
        parser.add_argument('--rm_num2',type=int,default=1)
        parser.add_argument('--area_flag',type=bool,default=False)
        parser.add_argument('--jitter',type=bool,default=False)
        parser.add_argument('--debaseflag',type=bool,default=False)
        parser.add_argument('--honor96flag',type=bool,default=False)
        parser.add_argument('--resizeflag',type=bool,default=False)
        parser.add_argument('--envir_choose',type=bool,default=False)
        parser.add_argument('--txtflag',type=bool,default=False)

        parser.add_argument('--dataflag', type=int, default=1) #1:all datas 2:small area all datas 3:large areas 4: small areas
   
        self.initialized = False
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

    def mkdirs(self,paths):
        """create empty directories if they don't exist

        Parameters:
            paths (str list) -- a list of directory paths
        """
        if isinstance(paths, list) and not isinstance(paths, str):
            for path in paths:
                if not os.path.exists(path):
                    os.makedirs(path)
        else:
            if not os.path.exists(paths):
                os.makedirs(paths)
