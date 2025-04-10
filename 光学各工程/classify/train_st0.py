'''Fingerprint ASP training with PyTorch.'''

## 1.由于取数不均衡，这里重新设置取数方式，保证每个类每个batch中都能取到
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import nni
import torchvision.transforms.functional as FT

import torchvision
from torchvision import transforms as transforms
from repvit import *

import os

import numpy as np
import argparse
import logging
# import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torch.autograd import Variable
# import config
import torch
import datetime
from base_options import *
from get_images import *
import random
import config
from mobilenetv3 import *

# import win32api
logger = logging.getLogger('mnist_AutoML')

from shutil import copyfile

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def log_print(log, content):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    print(time, content)
    log.append(time)
    log.append("  " + content + "\n")


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def inv_warp_image_batch_cv2(img, mat_homo_inv, device='cpu', mode='bilinear',size=None):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    '''
    # compute inverse warped points
    if len(img.shape) == 2:
        img = img.view(1, 1, img.shape[0], img.shape[1])
    if len(img.shape) == 3:
        img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)

    _, _, H, W = img.shape
	
    if size == None:
        size = (W, H)
    #print(size)

    warped_img = cv2.warpPerspective(img.squeeze().numpy(), mat_homo_inv.squeeze().numpy(), size)
    warped_img = torch.tensor(warped_img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)

    # warped_img = cv2.warpAffine(img.squeeze().numpy(), mat_homo_inv[0, :2, :].squeeze().numpy(), (W, H))
    # warped_img = torch.tensor(warped_img, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    return warped_img

def inv_warp_image(img, mat_homo_inv, device='cpu', mode='bilinear',size = None):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [H, W]
    '''
    warped_img = inv_warp_image_batch_cv2(img, mat_homo_inv, device, mode,size)
    return warped_img.squeeze()

def position_disturbance(img):
    x=random.random()
    if(x<0.3):
        return img
    img = np.array(img)
    # print('---start--')
    # print(img.shape) #61 61 3
    img_g = img[:,:,1]
    # print(img_g.shape)
    x1 = (random.random()-0.5)/100
    x2 = (random.random()-0.5)/5
    x3 = (random.random()-0.5)/100
    x4 = (random.random()-0.5)/5
    Htrans = torch.tensor([[1,0,0],[0,1,0],[0,0,1]],dtype=torch.float32)
    Htrans[0,1] = Htrans[0,1] + x1
    Htrans[0,2] = Htrans[0,2] + x2
    Htrans[1,0] = Htrans[1,0] + x3
    Htrans[1,2] = Htrans[1,2] + x4
    img_g = inv_warp_image(torch.from_numpy(img_g), Htrans,size=(img.shape[1],img.shape[0]))
    img_g = np.array(img_g).astype(np.uint8)
    # print(img_g.shape)
    # print(img[:,:,1].shape)
    # print('---end--')
    img[:,:,1] = img_g
    img = Image.fromarray(np.uint8(img))
    return img

def position_disturbance_25(img):
    x=random.random()
    if(x<0.4):
        return img
    img = np.array(img)
    # print('---start--')
    # print(img.shape) #61 61 3
    img_g = img[:,:,1]
    img_r = img[:,:,0]
    # # print(img_g.shape)
    # x1 = (random.random()-0.5)/100
    # x2 = (random.random()-0.5)/5
    # x3 = (random.random()-0.5)/100
    # x4 = (random.random()-0.5)/5
    x2 = random.randint(-100,100)*0.01
    x4 = random.randint(-100,100)*0.01
    Htrans = torch.tensor([[1,0,0],[0,1,0],[0,0,1]],dtype=torch.float32)
    # Htrans[0,1] = Htrans[0,1] + x1
    Htrans[0,2] = Htrans[0,2] + x2
    # Htrans[1,0] = Htrans[1,0] + x3
    Htrans[1,2] = Htrans[1,2] + x4

    level_256_B_M = np.ones_like(img_g)
    level_256_B_M[level_256_B_M > 0] = 255

    img_g = inv_warp_image(torch.from_numpy(img_g), Htrans,size=(img.shape[1],img.shape[0]))
    img_r = inv_warp_image(torch.from_numpy(img_r), Htrans,size=(img.shape[1],img.shape[0]))
    img_g = np.array(img_g).astype(np.uint8)
    img_r = np.array(img_r).astype(np.uint8)

    image_warp_B_M = inv_warp_image(torch.from_numpy(level_256_B_M), Htrans,size=(img.shape[1],img.shape[0]))
    image_warp_B_M[image_warp_B_M < 255] = 0
    image_warp_B_M[image_warp_B_M >= 255] = 1
    image_warp_B_M = np.array(image_warp_B_M).astype(np.uint8)
    img_g = img_g * image_warp_B_M
    img_r = img_r * image_warp_B_M
    # print(img_g.shape)
    # print(img[:,:,1].shape)
    # print('---end--')
    img[:,:,1] = img_g
    img[:,:,0] = img_r
    img = Image.fromarray(np.uint8(img))
    return img


def enlarge_main(ori_img0,width,jitter_flag=False):
    #ori_img0 = cv2.imread(imgpth)
    # b,g,r = cv2.split(ori_img)
    h,w,c = ori_img0.shape
    b,g,r = cv2.split(ori_img0)

    flagcol = random.randint(1,8)
    if flagcol<4:
        ratio = random.randint(90,100)*0.01
        flagt = random.randint(0,1)
        if flagt>0:
            g = g*ratio
            g = np.uint8(g)
            g = np.where(g>255,255,g)
            g = np.where(g<0,0,g)
        else:
            r = r*ratio
            r = np.uint8(r)
            r = np.where(r>255,255,r)
            r= np.where(r<0,0,r)
    flagcol = random.randint(1,8)
    if flagcol<3:
        delt = random.randint(-30,30)
        flagt = random.randint(0,1)
        if flagt>0:
            g = g+delt
            g = np.where(g>255,255,g)
            g = np.where(g<0,0,g)
        else:
            r = r+delt
            r = np.where(r>255,255,r)
            r= np.where(r<0,0,r)

    r = np.uint8(r)
    g = np.uint8(g)
    b = np.uint8(b)
    imm = cv2.merge([r,g,b])

    sample = Image.fromarray(imm).convert('RGB')
    if w>width:
       sample = sample.resize([width,width],Image.BILINEAR)
    
    if jitter_flag:
        #flagtrans = random.randint(1,10)
       # if flagtrans<3:
           sample = position_disturbance(sample)
           sample = position_disturbance_25(sample)
    #    print(sample.size)
    # sample.save(pthsave+'/'+str(i)+'.bmp')
    # print(pthsave+'/'+str(i)+'.bmp')
    return sample


def ColorJitterOne(img):
    #res = transforms.ColorJitter(brightness=(0.9,1.1),contrast=(0.8,1.2))
    #res = transforms.ColorJitter(brightness=(0.9,1.1),contrast=(0.9,1.1))
    #res2 = transforms.ColorJitter(brightness=(0.9,1.1),contrast=(0.9,1.1))
    res = transforms.ColorJitter(brightness=(0.9,1.1))
    res2 = transforms.ColorJitter(brightness=(0.9,1.1))
    #img = np.array(img)
    img2 = img.copy()
    #print(img.shape)
    x = random.random()
    #x1 = random.random()
    if (x > 0.8):
        img = res(img)
    elif (x > 0.6):
        img2 = res2(img2)
    img = np.array(img)
    img2 = np.array(img2)
    #print(img.shape)
    img[:,:,1] = img2[:,:,1]
    img = Image.fromarray(img)
    return img


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, transform=None,random_crop = False,width=122,enlarge_flag = False,removeimg=1,area_flag=False,jitter_flag=False,resizeflag=False,envir_choose =False):
        self.img_path = img_path
        self.imgs = store_dataset(img_path)
        self.imgs = self.imgs[::removeimg]
        self.transform = transform
        self.datalen = len(self.imgs)
        self.random_crop = random_crop
        self.width = width
        self.enlarge = enlarge_flag
        self.area_flag = area_flag
        self.jitter_flag=jitter_flag
        self.resizeflag = resizeflag

    def __getitem__(self, index):
        path = self.imgs[index]
        lend = len(self.img_path.split('/'))

        ppoint = path.split('/')[lend:lend+1]

        label = int(ppoint[0])
        

        # print(ppoint)
        ori_img0 = cv2.imread(path)
        # w,h,c = ori_img0.shape
        # print(path)
        if ori_img0 is None:
            path = self.imgs[(index+1)%self.datalen]
            ori_img0 = cv2.imread(path)

        if self.enlarge:
      
            sample = enlarge_main(ori_img0,self.width,jitter_flag=self.jitter_flag,resizeflag=self.resizeflag)
        else:
            
            h,w,c = ori_img0.shape
            
            b,g,r = cv2.split(ori_img0)
            if w>self.width:
                leftw = (w-self.width)//2
                ori_img = ori_img0[:,leftw:leftw+self.width,:]
                b,g,r = b[leftw:leftw+self.width,:],g[leftw:leftw+self.width,:],r[leftw:leftw+self.width,:]
                sample = Image.fromarray(ori_img).convert('RGB')
            else:
                sample = Image.fromarray(ori_img0).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
      
        return sample, label
    
    def __len__(self) :
        return self.datalen 

class MyDataset_twopath2(torch.utils.data.Dataset):
    def __init__(self, img_path,img_path2, transform=None,random_crop = False,width=122,enlarge_flag = False):
        self.img_path = img_path
        self.imgs1 = store_dataset(img_path)
        self.imgs2 = store_dataset(img_path2)
        # self.imgs00 = self.imgs1.extend(self.imgs2)
        self.imgs =  self.imgs1+self.imgs2
        print(len(self.imgs1),len(self.imgs2),len(self.imgs))
        self.transform = transform
        self.datalen = len(self.imgs)
        self.random_crop = random_crop
        self.width = width
        self.enlarge = enlarge_flag

    def __getitem__(self, index):
        path = self.imgs[index]
        lend = len(self.img_path.split('/'))
        ppoint = path.split('/')[lend:]
       
        label = int(ppoint[0])
        # print(ppoint)
        # print(label)

        # print(ppoint)
        ori_img0 = cv2.imread(path)
        if self.enlarge:

            sample = enlarge_main(ori_img0,self.width)
        else:
            
            h,w,c = ori_img0.shape
            
            b,g,r = cv2.split(ori_img0)
            if w>self.width:
                leftw = (w - self.width)//2
                ori_img = ori_img0[:,leftw:leftw+self.width,:]
                b,g,r = b[leftw:leftw+self.width,:],g[leftw:leftw+self.width,:],r[leftw:leftw+self.width,:]
                sample = Image.fromarray(ori_img).convert('RGB')
            else:
                sample = Image.fromarray(ori_img0).convert('RGB')
            

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label
    
    def __len__(self) :
        return self.datalen 


class Net(object):
    def __init__(self, args):
        self.opt = args
        self.gpu_ids = args.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(args.checkpoints_dir, args.name)  # save all the checkpoints to save_dir
        self.loss_names = []
        self.visual_names = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.trainroot = args.data_train
        self.testroot = args.data_test
        self.lr = args.lr
        self.epochs = args.epoch
        self.batch_size = args.batchsize
        self.log = []
        self.argsize = args.imsize
        self.loadmodel_flag = args.loadmodel_flag
        self.model_name = args.model
        self.zerolabel = args.zerolabel_loss
        self.teacher_flag = args.teacher_flag
        self.random = args.random_data
        self.crop_flag = args.crop_flag
        self.width = args.width
        self.batch_flag = args.batch_crop
        self.save_epoch = args.save_epoch
        self.enlarge_flag = args.enlarge_flag
        self.addsmallarea_flag = args.add_smallarea
        self.trainsmallroot = args.smallroot
        self.inputchannel = args.inputchannels
        self.debaseflag = args.debaseflag
        self.jitter_flag = args.jitter
        self.honor96flag = args.honor96flag
        self.resizeflag = args.resizeflag

        ## init models
        self.init_models()
 
        # trainimgswet,trainimgsnonwet = wet_split_dataset(config.traindir)
        # testimgswet,testimgsnonwet = wet_split_dataset(config.tstdir)
        # valimgswet,valimgsnonwet = testimgswet,testimgsnonwet

        copyfile("./config.py", os.path.join(self.save_dir, 'config.py'))
        copyfile("./train_st0.py", os.path.join(self.save_dir, 'train_st0.py'))
        copyfile("./mobilenetv3.py",  os.path.join(self.save_dir, 'mobilenetv3.py'))

        if self.honor96flag:
            train_transform = transforms.Compose([#transforms.CenterCrop(config.input_size),
                                              #transforms.Lambda(lambda img:Rotation180(img)),
                                              #transforms.RandomRotation([-5,5]),
                                              #transforms.RandomCrop([128,32]),
                                              transforms.Resize([96,96]),
                                              #transforms.Grayscale(),
                                              #transforms.ColorJitter(brightness=(0.9,1.1),contrast=(0.9,1.1)),
                                              #transforms.Lambda(lambda img:junheng(img)),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomVerticalFlip(p=0.5),
                                            #   transforms.ColorJitter(brightness=(0.7,1.4),contrast=(0.6,1.4)),
                                              transforms.ColorJitter(brightness=(0.8,1.2),contrast=(0.8,1.2)),
                                              #transforms.ColorJitter(brightness=(0.9,1.1)),
                                            #   transforms.Lambda(lambda img:ColorJitterOne(img)),
                                            #   transforms.Lambda(lambda img:expand_move(img)),
                                            # transforms.Lambda(lambda img:position_disturbance(img)),
                                            
                                            #   transforms.Lambda(lambda img:expand_cut(img)),
                                            #   transforms.Lambda(lambda img:expand_wet2(img)),
                                            #   transforms.Lambda(lambda img:expand_wet(img)),
                                            #   transforms.Lambda(lambda img:expand_contrast(img)),
                                            #   transforms.Lambda(lambda img:expand_color(img)),
                                              #transforms.Lambda(lambda img:amplification_reduce118x32(img)),
                                              #transforms.CenterCrop([132,132]),
                                              #transforms.RandomCrop(config.input_size),
                                              transforms.RandomAffine(degrees = 0,translate=(1/16,1/16)),#仿射变换
                                              transforms.ToTensor(),
                                              #transforms.RandomErasing(p=0.3,scale=(0.02,0.2),ratio=(0.3,3.3),value=0),
                                              #transforms.Normalize()
                                              ])
        else:
            train_transform = transforms.Compose([#transforms.CenterCrop(config.input_size),
                                              #transforms.Lambda(lambda img:Rotation180(img)),
                                              #transforms.RandomRotation([-5,5]),
                                              #transforms.RandomCrop([128,32]),
                                              #transforms.Resize(config.input_size),
                                              #transforms.Grayscale(),
                                              transforms.Resize([124,124]),
                                              #transforms.ColorJitter(brightness=(0.9,1.1),contrast=(0.9,1.1)),
                                              #transforms.Lambda(lambda img:junheng(img)),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomVerticalFlip(p=0.5),
                                            #   transforms.ColorJitter(brightness=(0.7,1.4),contrast=(0.6,1.4)),
                                              transforms.ColorJitter(brightness=(0.8,1.2),contrast=(0.8,1.2)),
                                              #transforms.ColorJitter(brightness=(0.9,1.1)),
                                            #   transforms.Lambda(lambda img:ColorJitterOne(img)),
                                            #   transforms.Lambda(lambda img:expand_move(img)),
                                            # transforms.Lambda(lambda img:position_disturbance(img)),
                                            
                                            #   transforms.Lambda(lambda img:expand_cut(img)),
                                            #   transforms.Lambda(lambda img:expand_wet2(img)),
                                            #   transforms.Lambda(lambda img:expand_wet(img)),
                                            #   transforms.Lambda(lambda img:expand_contrast(img)),
                                            #   transforms.Lambda(lambda img:expand_color(img)),
                                              #transforms.Lambda(lambda img:amplification_reduce118x32(img)),
                                              #transforms.CenterCrop([132,132]),
                                              #transforms.RandomCrop(config.input_size),
                                              transforms.RandomAffine(degrees = 0,translate=(1/16,1/16)),#仿射变换
                                              transforms.ToTensor(),
                                              #transforms.RandomErasing(p=0.3,scale=(0.02,0.2),ratio=(0.3,3.3),value=0),
                                              #transforms.Normalize()
                                              ])


      
        if self.honor96flag:
            test_transform = transforms.Compose([  # transforms.Resize(config.input_size),
                            # transforms.Grayscale(),
                            transforms.Resize([96,96]),
                            # transforms.CenterCrop([176, 176]),
                            transforms.ToTensor(),
                            # transforms.Normalize()
                            #transforms.Normalize(mean=0.5, std=0.5)
                        ])  
        else:
            test_transform = transforms.Compose([  # transforms.Resize(config.input_size),
                # transforms.Grayscale(),
                # transforms.Resize([self.argsize,self.argsize]),
                # transforms.CenterCrop([176, 176]),
                transforms.ToTensor(),
                # transforms.Normalize()
                #transforms.Normalize(mean=0.5, std=0.5)
            ])  # (0.34347,), (0.0479,)   vivo数据集
        # if self.crop_flag:
        if self.addsmallarea_flag:
            train_set1 = MyDataset_twopath2(self.trainroot,self.trainsmallroot,train_transform,random_crop=self.crop_flag,width=self.width,enlarge_flag=self.enlarge_flag,jitter_flag=self.jitter_flag,resizeflag=self.resizeflag)
        else:
               train_set1 = MyDataset(self.trainroot,train_transform,random_crop=self.crop_flag,width=self.width,enlarge_flag=self.enlarge_flag,jitter_flag=self.jitter_flag,resizeflag = self.resizeflag)
        # else:
        #     if self.enlarge_flag:
        #         train_set1 = MyDataset_simi(self.trainroot,train_transform)
        #     elif self.addsmallarea_flag:
        #         train_set1 = MyDataset1(self.trainroot,self.trainsmallroot,train_transform)
        #     else:
        #         train_set1 =MyDataset2(self.trainroot,train_transform,random_crop = False,width=122)
        #         # train_set1 = torchvision.datasets.ImageFolder(self.trainroot,train_transform)
        #train_set1 = torchvision.datasets.ImageFolder(self.trainroot,transforms.Compose( [self.pre,train_transform1, self.post]))

        self.train_loader1 = torch.utils.data.DataLoader(train_set1, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=16, pin_memory=True)  #

        # train_set2 = torchvision.datasets.ImageFolder(self.trainroot + '/data3',
        #                                               transforms.Compose([self.pre, train_transform1, self.post]))
        # self.train_loader2 = torch.utils.data.DataLoader(train_set2, batch_size=self.batch_size//2, shuffle=True,
        #                                                 num_workers=48, pin_memory=True)  #
        if self.crop_flag:
           test_set = MyDataset(self.trainroot,train_transform,random_crop=self.crop_flag,width=self.width,enlarge_flag=self.enlarge_flag)
        else:
            test_set = torchvision.datasets.ImageFolder(self.testroot, test_transform)
        #test_set = torchvision.datasets.ImageFolder(self.testroot, test_transform)
        ''' 
        classes_idx = test_set.class_to_idx
        appear_times = Variable(torch.zeros(len(classes_idx), 1))
        for label in train_set.targets:
            appear_times[label] += 1
        self.classes_weight = 1./(appear_times / 256089)
        '''
        self.len_imgs_train = len(train_set1)
        self.len_imgs_test = len(test_set)
        if self.random:
            maxnum =  self.len_imgs_train//self.batch_size
            self.choosenum = min(4200000//self.batch_size,maxnum) 
            self.batchss = [i for i in range(maxnum)]

            maxnum1 =  self.len_imgs_test//self.batch_size
            self.test_batchss = [i for i in range(maxnum1)]
            self.choosenum1 = min(50000//self.batch_size,maxnum1) 
        self.val_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=True,
                                                      num_workers=16, pin_memory=True)  #

        self.test_imgs = test_set.imgs
        print(self.len_imgs_train)
        print(self.len_imgs_test)

        # checkpoint = torch.load('./model/ckpt_8.pth')
        # self.symbol.load_state_dict(checkpoint['net'])
        # self.symbol = torch.nn.DataParallel(self.symbol)
        torch.backends.cudnn.benchmark = True

        if args.optim_choose == 'SGD':
           self.optimizer = torch.optim.SGD(self.symbol.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        elif args.optim_choose =='rmsprop':
            self.optimizer = torch.optim.RMSprop(self.symbol.parameters(),lr=self.lr, alpha=0.9)
        else:
           self.optimizer = torch.optim.Adam(self.symbol.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=mile, gamma=0.5)
        # self.criterion = nn.CrossEntropyLoss().to(self.device)
        # self.criterion2 = nn.CrossEntropyLoss(torch.Tensor([3, 1])).to(self.device)
        if args.weight_flag:
               self.criterion = nn.CrossEntropyLoss(torch.Tensor([4,1])).to(self.device) #[1.5, 1]
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        self.criterL1 =nn.SmoothL1Loss().to(self.device)
     
        self.schedulers = get_scheduler(self.optimizer, self.opt)
        # self.schedulers = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min',factor = 0.5, patience = 5, verbose = True)
        # self.criterion = FocalLoss(6, alpha=self.classes_weight.view(-1)).to(self.device)
        print_network(self.symbol)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def init_models(self,init_type='normal', init_gain=0.02):
        # Model
        print('==> Building model..')
        if self.model_name == "mnv_small":
            self.symbol = MNV30811_SMALL(numclasses=2).to(self.device)
       
        if self.loadmodel_flag:
            # checkpoint = torch.load('./MODEL_6193/model_6195_all1/ckpt1_0.9968_0.9916.pth', map_location=self.device) #['net']
            # new_state_dict={}
            # for k,v in checkpoint['net'].items():
            #     new_state_dict[k[7:]]=v
          
            if self.model_name =='mnv_small':  #class6195_3_c4_8_2/ckpt_mnv_small1_63_0.00000_0.98662.pth
                
                checkpoint=torch.load('./checkpoints//boe_10/ckpt_mnv_small1_40_0.99484_0.96972.pth', map_location=self.device)
               
                self.symbol.load_state_dict(checkpoint, strict=True)# 
            return
        else:
            return init_net(self.symbol , init_type, init_gain, self.gpu_ids)

    def load_networks(self):
        # device = torch.device('cpu') #   './models_loss/good_model_new/ckpt_SGD_0.020000_500_100_66_0.996784.pth'
        # checkpoint=torch.load('./model_2020_07_24_09/ckpt_170.pth') # ./model/ckpt_ASPNETV_6_24_0.992537.pth
        checkpoint = torch.load(self.model_path, map_location=self.device)
        new_state_dict={}
        for k,v in checkpoint['net'].items():
            new_state_dict[k]=v
        # self.symbol = torch.nn.DataParallel(self.symbol)
        self.symbol.load_state_dict(new_state_dict, strict=True)
        self.symbol.eval()

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for param_group in self.optimizer.param_groups:
            lr_datas = param_group['lr']

        if self.opt.lr_policy == 'plateau':
            self.schedulers.step(self.metric)
        else:
            self.schedulers.step()
        return lr_datas

    def save(self, epoch, trainacc,testacc):
        model_out_path = "%s/ckpt_%s_%d_%.3f_%.3f.pth" % (self.save_dir, self.model_name, epoch,testacc, trainacc)
        torch.save(self.model, model_out_path)
        '''
        print("Checkpoint saved to {}".format(model_out_path))
        self.log.append("Checkpoint saved to {} \n".format(model_out_path))
        '''
        log_print(self.log, "Checkpoint saved to {}".format(model_out_path))

    def save_networks(self, epoch,trainacc,testacc):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if isinstance(self.model_name, str):
            save_filename = "ckpt_%s_%d_%.5f_%.5f.pth" % (self.model_name, epoch, testacc,trainacc)
            save_path = os.path.join(self.save_dir, save_filename)
            net = self.symbol

            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.cpu().state_dict(), save_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)
        log_print(self.log, "Checkpoint saved to {}".format(save_filename))
        return

 
    def tensor2img(self,img,pthsave):
        image_numpy = torch.squeeze(img)
        # print(image_numpy.shape)
        image_numpy = image_numpy.transpose(0,2).transpose(0,1)
        # print(image_numpy.shape)
        image_numpy = image_numpy.cpu().detach().numpy()

        image_numpy = image_numpy * 255.0

        image_numpy = np.maximum(image_numpy, 0)
        image_numpy = np.minimum(image_numpy, 255)

        # image_numpy = 255-image_numpy
        image_numpy = np.uint8(image_numpy)
        # print(image_numpy.size)
        imgALL = Image.fromarray(image_numpy)
        imgALL = imgALL.convert("RGB")
        imgALL.save(pthsave)
        return


    def train(self):
        print("train fingerprint asp:")
        self.symbol.train()

        loss = torch.tensor([0], dtype=torch.float32).cuda()
        # maxs = torch.tensor([1], dtype=torch.float32).cuda()
        # zeroscore = torch.tensor([1], dtype=torch.float32).cuda()

        total = loss.clone().cuda()
        train_correct = loss.clone().cuda()
        train_loss = loss.clone().cuda()
        train_loss_ret = loss.clone().cuda()
  
        if self.random:
            self.tain_batchs = sorted(random.sample(self.batchss,self.choosenum))#
        # print(self.tain_batchs)
       # self.len_imgs_test
        if self.save_epoch:
            pathsave=self.save_dir+'/images/'
            if not os.path.exists(pathsave):
                os.makedirs(pathsave)

        for batch_num, (data, target) in enumerate(self.train_loader1):
            if self.random:
                if batch_num  not in  self.tain_batchs:
                    # print(batch_num)
                    continue
            # print(data.shape)
            data, target = data.to(self.device), target.to(self.device)
            # print(data.shape,target.shape)

            self.optimizer.zero_grad()
            
            # print(target.shape)
            if self.batch_flag:
                b,c,h,w = data.shape
                crop_wl = torch.randint(0,6, size=(1,)).item()
                crop_wr = torch.randint(w-6,w, size=(1,)).item()
                crop_hu = torch.randint(0,2, size=(1,)).item()
                crop_hd = torch.randint(h-2,h, size=(1,)).item()
                data = data[:,:,crop_hu:crop_hd,crop_wl:crop_wr]
            if self.save_epoch:
                pathsave0 = pathsave+str(batch_num)+str(target.cpu().numpy()[0])+'.bmp'
                # print(pathsave0)
                self.tensor2img(data[0,:,:,:],pathsave0)
            # print(data.shape)
            feature,output = self.symbol(data[:,:self.inputchannel,:,:])
            # print(output.shape)

        
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += torch.sum(prediction[1] == target)

            loss = self.criterion(output, target)
            loss = loss.float()
            if self.thr65535_flag:  
                out = F.softmax(output,dim = -1)
                score = out[:,1]*65536
                t1 = torch.where(score  > 65535,1,0)
                # print(t1)
                # print(t1.shape)
                # print(target)
                t1 = t1.float()
                thr_loss = self.criterL1(t1, target)
                loss = loss+thr_loss

            if self.zerolabel:
                outscore = F.softmax(output,dim = -1)
                outscore = outscore[:,1]
                
                tid = torch.where(target==0)
                s = torch.sum(torch.where(target==0,1,0))
                if s>0:
                    outscore = outscore[tid]
                    maxs = torch.max(outscore)
                    loss = loss+maxs
                    errid = torch.where(outscore>0.5)
                    s1 = torch.sum(torch.where(outscore>0.5,1,0))
                    if s1>0:
                        zeroscore = torch.mean(outscore[errid])
        
                        # print(outscore[tid])
                        # print(zeroscore)
                        # print(loss)
                        loss = loss+zeroscore

            # print(loss)

            loss.backward()
            self.optimizer.step()
            train_loss +=loss
            train_loss_ret += train_loss/ (batch_num + 1)
        
            if self.thr65535_flag:  
                return train_loss_ret, train_correct / total,train_correct,total,thr_loss
            else:
                return train_loss_ret, train_correct / total,train_correct,total

    def test(self):
        print("test:")

        self.symbol.eval()
        loss = torch.tensor([0], dtype=torch.float32).cuda()
        total = loss.clone().cuda()
        test_loss = loss.clone().cuda()
        test_correct = loss.clone().cuda()
        test_loss_ret = loss.clone().cuda()
        with torch.no_grad():
            if self.random:
                self.test_batchs = sorted(random.sample(self.test_batchss,self.choosenum1))
            
            for batch_num, (data, target) in enumerate(self.val_loader):
                if self.random:
                    if batch_num  not in  self.test_batchs:
                        continue
                data, target = data.to(self.device), target.to(self.device)
                # print(data.shape)
                data= F.interpolate(data, size=[96, 96], mode="bilinear", align_corners=False)
                feature,output = self.symbol(data[:,:self.inputchannel,:,:])
                
                loss = self.criterion(output, target)
                test_loss += loss
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += torch.sum(prediction[1] == target)
                test_loss_ret = test_loss / (batch_num + 1)
        log_print(self.log, "test loss: %1.5f, test acc：%1.5f" % (test_loss_ret, test_correct / total))
        return test_loss_ret, test_correct / total

    def start(self):
        # self.resume()
        # 记录acc最高，lost最低的轮数
        accuracy = torch.tensor([0], dtype=torch.float32).cuda()
        accuracyT = torch.tensor([0], dtype=torch.float32).cuda()
        lostvalueT = torch.tensor([1], dtype=torch.float32).cuda()
        scoreT = torch.tensor([0], dtype=torch.float32).cuda()
        bestone = 0

        train_accuracy = torch.tensor([0], dtype=torch.float32).cuda()
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        test_result=torch.tensor([0.0,0.0], dtype=torch.float32).cuda()

        maxva1 = torch.tensor([0.970], dtype=torch.float32).cuda()
        maxva2 = torch.tensor([0.986], dtype=torch.float32).cuda()
        flag_val=True

        # for epoch in range(self.model['epoch']+1, self.epochs + 1):
        for epoch in range(1, self.epochs + 1):
            train_result = self.train()
            # self.scheduler.step(epoch)
            '''
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            print(time, "Epoch[%03d]: %03d/%d    acc=%1.5f    lossvalue=%1.5f" % (epoch,epoch,self.epochs,train_result[1],train_result[0]))
            self.log.append(time)
            self.log.append("  Epoch[%03d]: %03d/%d    acc=%1.5f    lossvalue=%1.5f \tlearning_rate=%1.7f\n" % (epoch,epoch,self.epochs,train_result[1],train_result[0], get_lr(self.optimizer)))
            '''
            if self.thr65535_flag: 
                log_print(self.log, "\nEpoch[%03d]: %03d/%d    acc=%1.5f  acc_num=%d  total_num=%d  lossvalue=%1.5f  thr65535loss=%.5f  teacherloss=%.5f \tlearning_rate=%1.7f" % (
                    epoch, epoch, self.epochs, train_result[1], train_result[2], train_result[3],train_result[0], train_result[4],train_result[5],get_lr(self.optimizer)))      

            else:
                log_print(self.log, "\nEpoch[%03d]: %03d/%d    acc=%1.5f  acc_num=%d  total_num=%d  lossvalue=%1.5f \tlearning_rate=%1.7f" % (
                    epoch, epoch, self.epochs, train_result[1], train_result[2], train_result[3],train_result[0], get_lr(self.optimizer)))
            train_loss.append(train_result[0])

            train_acc.append(train_result[1])
            if (train_result[1] > train_accuracy and train_result[1] >maxva1) or (train_result[1] > maxva2) or (
                    epoch == self.epochs):
                state = {
                    'net': self.symbol.state_dict(),
                    'train_acc': train_result[1],
                    'test_acc': test_result[1],
                    'epoch': epoch,
                }
                self.model = state
                self.save_networks(epoch, train_result[1],test_result[1])
            train_accuracy = max(train_accuracy, train_result[1])

            if flag_val:
                # nni.report_intermediate_result(train_accuracy)
                test_term = 20
                if train_accuracy > maxva1 or epoch>30:
                    test_term = 10
                if epoch % test_term == 0:
                    test_result = self.test()
                    val_loss.append(test_result[0])
                    val_acc.append(test_result[1])
                    if test_result[1] > accuracy:
                        state = {
                            'net': self.symbol.state_dict(),
                            'train_acc': train_result[1],
                            'test_acc': test_result[1],
                            'epoch': epoch,
                        }
                        self.model = state
                        self.save_networks(epoch, train_result[1],test_result[1])
                    accuracy = max(accuracy, test_result[1])
                else:
                    val_loss.append(test_result[0])
                    val_acc.append(test_result[1])

            f = open(self.save_dir + "/log.txt", 'a')
            f.writelines(self.log)
            f.close()
            self.log = []

            lr_now = self.update_learning_rate()

            if lr_now < 1e-5:
                state = {
                    'net': self.symbol.state_dict(),
                    'train_acc': train_result[1],
                    'test_acc': test_result[1],
                    'epoch': epoch,
                }
                self.model = state
                self.save_networks(epoch, train_result[1],test_result[1])
                break

    '''            
    def plotacc(self, train_loss, train_acc, val_loss, val_acc, epoch):
        plt.close()
        epochx = np.arange(0, epoch)
        plt.plot(epochx, train_loss, label = "train loss", marker = 'o')
        plt.plot(epochx, train_acc, label = "train acc", marker = '*')
        plt.plot(epochx, val_loss, label = "val loss", marker = 'o')
        plt.plot(epochx, val_acc, label = "val acc", marker = '*') 
        for m in (train_loss, train_acc, val_loss, val_acc):
            num = 0
            for xy in zip(epochx, m):
                plt.annotate("%.3f" % m[num], xy = xy, xytext = (-20, 5), textcoords='offset points')
                num = num + 1
        plt.legend(frameon = False)
        plt.show()  # %matplotlib qt5
     '''

from pathlib import Path
def main():
    # global args
    # global time
    # global model_path
   # parser = argparse.ArgumentParser(description="ASP Train Network")
    #par = BaseOptions.initialize(parser)
    params = BaseOptions().parse()
    #params = BaseOptions.parse(par)

    # time = datetime.datetime.now().strftime('_%Y_%m_%d_%H_11')
    # model_path = "./model%s" % time
    # if not os.path.isdir(model_path):
    #     os.mkdir(model_path)
    


    net = Net(params)
    net.start()

    print('the train work is over!')


if __name__ == '__main__':
    main()
