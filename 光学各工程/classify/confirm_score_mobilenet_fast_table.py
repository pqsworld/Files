import torch
import numpy as np
import torchvision
from torchvision import transforms as transforms
import config

from torch.autograd import Variable

# from MobileNet_dropout_linear import MNV30811
from mobilenetv3 import *
from tqdm import tqdm
import datetime
import cv2

import itertools
# import matplotlib.pyplot as plt
import os
from shutil import copyfile
from shutil import move
from math import exp
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image
from torchvision import transforms
# 绘制混淆矩阵

class Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, is_valid_file=None,width=66):
        super(Dataset, self).__init__(root, transform=transform, target_transform=target_transform,
                                      is_valid_file=is_valid_file)
        self.width = width

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        ori_img0 = cv2.imread(path)
        h,w,c = ori_img0.shape
        b,g,r = cv2.split(ori_img0)
        if w>self.width:
            leftw = get_crop_imgsize1(r,self.width)
            ori_img = ori_img0[:,leftw:leftw+self.width,:]
            sample = Image.fromarray(ori_img).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


from get_images import *

class MyDatasetOptic(torch.utils.data.Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.imgs = store_dataset(img_path)
        self.transform = transform
        self.datalen = len(self.imgs)

    def __getitem__(self, index):
        path = self.imgs[index]
        lend = len(self.img_path.split('/'))
        ppoint = path.split('/')[lend:]
       
        label = int(ppoint[0])
        try:
            sample = Image.open(path).convert('RGB')
        except:
            print(path)
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label,path
    
    def __len__(self) :
        return self.datalen 


PTH_EXTENSIONS = [
    '.pth', '.PTH',
]
def is_pth_file(filename):
    return any(filename.endswith(extension) for extension in PTH_EXTENSIONS)

def allpth_dataset(dir):
    all_pathA = []
    for root, _, fnames in sorted(os.walk(dir)):#
        for fname in fnames:
            if is_pth_file(fname):
                pathA = os.path.join(root, fname)
                all_pathA.append(pathA)

    return all_pathA


def get_fa_thresholds():
    time = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M')
    imgsort = 1
    logprint = 1
    # thr = 64800#340
    device = torch.device('cuda:5')

    test_dir = r'/hdd/file-input/liugq/datasets/ry/phd_imgs'
    # test_dir = r'/hdd/file-input/liugq/datasets/ry/testsnorm_txt'
    
  
    # test_dir = r'/ssd/share/liugq/ttl/audi_142_bmp_for_train/128size/sm24df_test'
    # test_dir = r'/ssd/share/liugq/datas/test620'

    # test_dir = r'/hdd/file-input/liugq/datasets/6197/frr_6197_chuanyin_cut95_test_crop'
    # test_dir = r'/hdd/file-input/liugq/datasets/6197/test97'
    # test_dir = r'/hdd/file-input/liugq/datasets/6197/small_area'
    # test_dir = r'/ssd/share/liugq/datas/test620'
    # test_dir = r'/hdd/file-input/liugq/datasets/6195/test_model/src'
    # test_dir = r'/hdd/file-input/liugq/datasets/6195/test_enh612'
    # test_dir = r'/hdd/file-input/liugq/datasets/6195/FRR/frr_far1'
    # test_dir = r'/hdd/file-input/wangb/classify/compare/data/test_wet_140_2'
    # test_dir = r'/hdd/file-input/wangb/classify/compare/data/test'
    teacher_flag = False
    pth_dir = './checkpoints/optic_classify96_tmp'
    # pth_dir = '/hdd/file-input/liugq/classify/readtxt/oppo_audi_nova_96/model_resize96_0__2024_12_05_04_11'
    # pth_dir = '/hdd/file-input/liugq/compare/checkpoints/models'
    confusion_out_path = "./confusion_%s_%s_%s" % (test_dir.split('/')[-1],pth_dir.split('/')[-1],time)
    if not os.path.isdir(confusion_out_path):
        os.mkdir(confusion_out_path)
    # print(test_dir)
    test_transform = transforms.Compose([  # transforms.Resize(config.input_size),
        #transforms.CenterCrop(config.input_size),  #
        #transforms.Grayscale(),
        transforms.Resize([96,96]),
        # transforms.CenterCrop(config.input_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.34347,), (0.0479,))
        # transforms.Normalize()
    ])
    # test_set = torchvision.datasets.ImageFolder(test_dir, test_transform)
    test_set = MyDatasetOptic(test_dir, test_transform)
    # test_set = MyDataset(test_dir,test_transform,random_crop=True,width=66,enlarge_flag=False)

    if  teacher_flag:
        val_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=False,
                                                num_workers=8)
    else:
        # val_loader = torch.utils.data.DataLoader(test_set, batch_size=2048, shuffle=False,
        #                                         num_workers=8)  # , pin_memory=True
        val_loader = torch.utils.data.DataLoader(test_set, batch_size=12000, shuffle=False,
                                                num_workers=8)  # , pin_memory=True
        #/checkpoints/class6195_3_c4_10_1/ckpt_mnv_llarge_99_0.00000_0.99114.pth
   

    best_step0=0
    best_step0_score = 1
    best_step1=0
    best_step1_score=1
    best_step3=0
    best_step3_score=1
    best_step130=0
    best_step130_score=1
    best_step30 = 0
    best_step30_score=1
    log_best = []


    pthsss = allpth_dataset(pth_dir)
    for ipth  in range(len(pthsss)):   
        name = pthsss[ipth].split('/')
        # if int(name[-1].split('_')[3])<47:
        #     continue
        # if int(name[-1].split('_')[1][:-4])<=21:
        #     continue
        ttmpname = name[-2]+'_'+name[-1]
        checkpoint = torch.load(pthsss[ipth], map_location=device)

        # net=ASP0809_tpc().to(device)
        if  teacher_flag:
            net  = MNV30811_LLarg(2).to(device)
        else:
            # net = MNV30811_SMALL(2).to(device)
            net = MNV30811_SMALL1(2).to(device)
        # net = MNV30811_LLarg_down(2).to(device)
        # net.load_state_dict(checkpoint['net'],strict=True)
        if name[-1][0]=='w':
           net.load_state_dict(checkpoint['net'],strict=True)#
        else:
            net.load_state_dict(checkpoint,strict=True)
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(params)
        # ddd
        softmax = nn.Softmax(dim=1)

        net.eval()
        print('example')
        num = 0
        log = []

        img_num = 0
        finger_num = 0
        miss_num = 0
        omit_num = 0
        error_class_1=0
        error_class_0=0
        class_1_num=0
        class_0_num=0

        total_score = []
        total_flag = []
        class_0_scores=[]

        with torch.no_grad():
            with tqdm(total=len(val_loader),position=0,ncols=80) as pbar:
                # for thr in range(32768,65536,10):
                for batch_num, (data, target, path) in enumerate(val_loader):
                    data, target = data.to(device), target.to(device)
                    pat = np.array(list(range(len(target))))

                    data = data[:, :2, : ,:]
                #print(data.shape)
                #sss
                # print(data[0] * 255)
                    features, output = net(data)
                    
                    if len(output.shape) == 1:
                        #pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                        out1= output*65536
                    else:
                        out = F.softmax(output,dim=-1)
                        out1 = out[:,1] * 65536

                    total_score.extend(out1)
                    total_flag.extend(target)

                    id = torch.where(target==0)

                    class_0_scores.extend(out1[id[0]].cpu().detach().numpy())

                    class_1_num += len(target[target==1])
                    class_0_num += len(target[target==0])

                    pbar.update(1)
        class_0_scores = sorted(np.array(class_0_scores))

        total_flag = torch.Tensor(total_flag) 
        total_score = torch.Tensor(total_score) 
        
        for thr in range(1000,65535,5):

            out_class_0_target = total_flag[total_score<=thr]
            out_class_1_target = total_flag[total_score>thr]
                
            error_class_1 = len(out_class_0_target[out_class_0_target==1])
            error_class_0 = len(out_class_1_target[out_class_1_target==0])
  
            data_len = class_0_num + class_1_num
            data_correct = data_len - error_class_1 - error_class_0


            if error_class_0==0:
                error_1_acc = error_class_1/class_1_num
                if best_step0_score>error_1_acc:
                    best_step0_score = error_1_acc
                    best_step0 = pthsss[ipth]
            
            if error_class_0==1:
                error_1_acc = error_class_1/class_1_num
                if best_step1_score>error_1_acc:
                    best_step1_score = error_1_acc
                    best_step1 = pthsss[ipth]

            if error_class_0==3:
                error_1_acc = error_class_1/class_1_num
                if best_step3_score>error_1_acc:
                    best_step3_score = error_1_acc
                    best_step3 = pthsss[ipth]

            
            if error_class_0==30:
                error_1_acc = error_class_1/class_1_num
                if best_step30_score>error_1_acc:
                    best_step30_score = error_1_acc
                    best_step30 = pthsss[ipth]

            if error_class_0==130:
                error_1_acc = error_class_1/class_1_num
                if best_step130_score>error_1_acc:
                    best_step130_score = error_1_acc
                    best_step130 = pthsss[ipth]            

            #log.append("log_path:%s"%(confusion_out_path))
            log.append("thresh:%d "%(thr))
            log.append("accuracy:%f(%d/%d) "%(data_correct/data_len,data_correct,data_len))
            log.append("error_0:%f(%d/%d) "%(error_class_0/class_0_num,error_class_0,class_0_num))
            log.append("error_1:%f(%d/%d) "%(error_class_1/class_1_num,error_class_1,class_1_num))
            log.append("\n")

        if logprint:
            f = open(confusion_out_path + "/"+ttmpname+"log.txt", 'a')
            f.writelines(log)
            f.close()
    
    log_best.append("best0 pth:%s  , frr: %f   "%(best_step0,best_step0_score))
    log_best.append("best1 pth:%s  , frr: %f   "%(best_step1,best_step1_score))
    log_best.append("best3 pth:%s  , frr: %f   "%(best_step3,best_step3_score))
    log_best.append("best30 pth:%s  , frr: %f   "%(best_step30,best_step30_score))
    log_best.append("best130 pth:%s  , frr: %f   "%(best_step130,best_step130_score))
    if logprint:
        f = open(confusion_out_path + "/"+"log_best.txt", 'a')
        f.writelines(log_best)
        f.close()


if __name__ == '__main__':
   get_fa_thresholds()
