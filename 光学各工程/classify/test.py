'''Fingerprint ASP training with PyTorch.'''
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix

import torchvision
from torchvision import transforms as transforms
from math import exp
import os
import cv2

import numpy as np
import argparse
import logging
# import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torch.autograd import Variable
# import config
import torch
from models import *
import datetime
from test_options import *
import random
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
        #self.trainroot = args.data_train
        self.testroot = args.data_test

        self.demo_flag = args.demo_flag
        # print(self.demo_flag)

        self.checkpoints = args.checkpoints_dir
        self.model_path = self.checkpoints+'/'+args.name+'/'+args.testmodel

        self.model_name = args.model
        ## init models
        
        #self.lr = args.lr
        #self.epochs = args.epoch
        self.batch_size = args.batchsize
        self.log = []
        self.argsize = args.imsize

        test_transform = transforms.Compose([  # transforms.Resize(config.input_size),
            # transforms.Grayscale(),
            # transforms.Resize([self.argsize, self.argsize]),
            # transforms.CenterCrop([176, 176]),
            transforms.ToTensor(),
            # transforms.Normalize()
            #transforms.Normalize(mean=0.5, std=0.5)
        ])  # (0.34347,), (0.0479,)   vivo数据集

        if self.demo_flag:
            self.demo_transform = transforms.Compose([  # transforms.Resize(config.input_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])

        test_set = torchvision.datasets.ImageFolder(self.testroot, test_transform)
        print(len(test_set))

        self.val_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False,
                                                      num_workers=16, pin_memory=True)  #

        self.test_imgs = test_set.imgs

        torch.backends.cudnn.benchmark = True

        self.save_dir = args.save_res+'/'+args.name

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.init_models()
        # time = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M')
        # self.save_dir = self.save_dir+'/'+args.test_datas
        # if not os.path.isdir(self.save_dir):
        #     os.mkdir(self.save_dir)
        # self.save_dir0 = self.save_dir + '/0'
        # self.save_dir1 = self.save_dir + '/1'

        # if not os.path.isdir(self.save_dir0):
        #     os.mkdir(self.save_dir0)
        # if not os.path.isdir(self.save_dir1):
        #     os.mkdir(self.save_dir1)

    def init_models(self,init_type='normal', init_gain=0.02):
        # Model
        print('==> Building model..')
        if self.model_name == "ASPNETV_1":
            self.symbol = ASPNETV_1().to(self.device)
        if self.model_name == "ASPNETV_2":
            self.symbol = ASPNETV_2().to(self.device)
        if self.model_name == "ASPNETV_3":
            self.symbol = ASPNETV_3().to(self.device)
        if self.model_name == "ASPNETV_4":
            self.symbol = ASPNETV_4().to(self.device)
        if self.model_name == "ASPNETV_5":
            self.symbol = ASPNETV_5().to(self.device)
        if self.model_name == "ASPNETV_6":
            self.symbol = ASPNETV_6().to(self.device)
        if self.model_name == "ASPNETV_7":
            self.symbol = ASPNETV_7().to(self.device)
        if self.model_name == "ASPNETV_8":
            self.symbol = ASPNETV_8().to(self.device)
        if self.model_name == "ASPNETV_9":
            self.symbol = ASPNETV_9().to(self.device)
        if self.model_name == "ASPNETV_10":
            self.symbol = ASPNETV_10().to(self.device)
        if self.model_name == "mobilenet":
            self.symbol = MNV3_large2(2).to(self.device)
        if self.model_name == "ShuffleNetV_1":
            self.symbol = ShuffleNetV_1().to(self.device)
        if self.model_name == "mnv_small":
            self.symbol = MNV30811_SMALL(2).to(self.device)
        return #init_net(self.symbol , init_type, init_gain, self.gpu_ids)

    def load_networks(self):
        # device = torch.device('cpu') #   './models_loss/good_model_new/ckpt_SGD_0.020000_500_100_66_0.996784.pth'
        # checkpoint=torch.load('./model_2020_07_24_09/ckpt_170.pth') # ./model/ckpt_ASPNETV_6_24_0.992537.pth
        checkpoint = torch.load(self.model_path, map_location=self.device)
        new_state_dict={}
        for k,v in checkpoint['net'].items():
            # new_state_dict[k[7:]]=v
            new_state_dict[k]=v
        # self.symbol = torch.nn.DataParallel(self.symbol)
        self.symbol.load_state_dict(new_state_dict, strict=True)
        self.symbol.eval()
    
    def save_badimgs(self,datas,scores):
        b,c,h,w = datas.shape
        for i in range(b):
            img = datas[i,:,:,:]
            image_numpy = torch.squeeze(img)
            # print(image_numpy.shape)
        
            image_numpy = image_numpy.cpu().detach().numpy()
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) ) * 255.0 
            # image_numpy = image_numpy  * 255.0
            image_numpy = np.maximum(image_numpy, 0)
            image_numpy = np.minimum(image_numpy, 255)
            
            # imgALL = Image.fromarray(image_numpy).convert('RGB')
             
            if not os.path.exists(self.save_dir+'/bad_test/'):
                os.mkdir(self.save_dir+'/bad_test/')
            ss = str(scores[i].cpu().numpy())
            num = self.st
            num = num+i
            idd = ss.find('.')
            sc = ss[:idd+2]
            finame = str(num)+'_'+sc+'.bmp'
            # print(finame)
            # imgALL.save(self.save_dir+'/bad_test/'+finame)
            # tmp = image_numpy[:,:,0] 
            # image_numpy[:,:,0] = image_numpy[:,:,2]
            # image_numpy[:,:,2] = tmp
            mg = cv2.merge([image_numpy[:,:,2],image_numpy[:,:,1],image_numpy[:,:,0]])
            cv2.imwrite(self.save_dir+'/bad_test/'+finame, mg)


    def test(self):
        # print("test:")
        self.load_networks()
        # print_network(self.symbol)

        total = torch.tensor([0], dtype=torch.float32).cuda()
        test_correct = torch.tensor([0], dtype=torch.float32).cuda()

        y_pred = []
        y_true = []
        scores=[]
        maxscores =0
        minscores = 65536
        self.st = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                if self.demo_flag:
                    f,output = self.symbol.demo_forward(data[:,:2,:,:])
                else:
                    f,output = self.symbol(data[:,:2,:,:])
                out = F.softmax(output,dim = -1)
                score = out[:,1]*65536
                tid = torch.where(target==0)
                if len(tid)>0:
                    # print(tid)
                    scc = score[tid]
                    # print(scc.shape)
                    dcc = data[tid[0],:,:,:]
                    # print(dcc.shape)
                    sid = torch.where(scc>32768)
                    s = torch.sum(torch.where(scc>32678,1,0))
                    if s>0:
                        badscore = scc[sid]
                        bad_data = dcc[sid[0],:,:,:]
                        scores.append(scc)
                        # print(bad_data.shape)
                        # print(self.st)
         
                        self.save_badimgs(bad_data,badscore)
                        self.st = self.st+s.cpu().numpy()
                    if len(scc) >0 :
                        maxx = torch.max(scc)
                        minx = torch.min(scc)
                        if maxscores<maxx:
                            maxscores = maxx
                        if minscores>minx:
                            minscores = minx
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += torch.sum(prediction[1] == target)
                y_pred.extend(prediction[1].cpu().detach().numpy())
                y_true.extend(target.cpu().detach().numpy())

        print('\n****************************************\n')
        cm = confusion_matrix(y_true,y_pred)

        ## non threshold mode ======end=====
        self.plot_confusion_matrix(cm, True)
        print("---------------------------------------")
        self.plot_confusion_matrix(cm, False)
        print('\n****************************************\n')
        # self.plot_confusion_matrix(confusionmap1, False)
        print(maxscores)
        print(minscores)
       
        return

    def test0(self):
        print("test0:")

        self.symbol.eval()
        loss = torch.tensor([0], dtype=torch.float32).cuda()
        total = loss.clone().cuda()
        test_loss = loss.clone().cuda()
        test_correct = loss.clone().cuda()
        test_loss_ret = loss.clone().cuda()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                # print(data.shape)
                feature,output = self.symbol(data[:,:2,:,:])
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += torch.sum(prediction[1] == target)
                test_loss_ret = test_loss / (batch_num + 1)
                y_pred.extend(prediction[1].cpu().detach().numpy())
                y_true.extend(target.cpu().detach().numpy())
        log_print(self.log, "test loss: %1.5f, test acc：%1.5f" % (test_loss_ret, test_correct / total))

        print('\n****************************************\n')
        cm = confusion_matrix(y_true,y_pred)

        ## non threshold mode ======end=====
        self.plot_confusion_matrix(cm, True)
        print("---------------------------------------")
        self.plot_confusion_matrix(cm, False)
        print('\n****************************************\n')
        return test_loss_ret, test_correct / total

    def test1(self):
        # print("test:")
        self.load_networks()
        # print_network(self.symbol)

        total = torch.tensor([0], dtype=torch.float32).cuda()
        test_correct = torch.tensor([0], dtype=torch.float32).cuda()

        log = []
        fp_correct = 0
        fp_error = 0
        sp_correct = 0
        sp_error = 0
        confusionmap = np.zeros([2, 2])
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.val_loader):

                data, target = data.to(self.device), target.to(self.device)
    
                f,output = self.symbol(data[:,:2,:,:])

                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += torch.sum(prediction[1] == target)

                ## threshold mode
                finger_score = exp(output[:][0]) / (exp(output[:][0]) + exp(output[:][1]))
                spoof_score = 1 - finger_score

                # print(prediction.indices[i])
                prediction.indices[0] = (finger_score < 0.5) #0.4
                if prediction.indices[0] != target[0]:
                    if prediction.indices[0] == 0:
                        sp_error = sp_error + 1
                        # log.append("output[0]:%f  output[1]:%f path: %s \n" % (
                        #             finger_score*65536,spoof_score*65536 , self.test_imgs[batch_num][0]))
                    else:
                        fp_error = fp_error + 1
                        # log.append("output[0]:%f  output[1]:%f path: %s \n" % (
                        #             finger_score*65536,spoof_score*65536, self.test_imgs[batch_num][0]))
                else:
                    if prediction.indices[0] == 0:
                        fp_correct = fp_correct+1
                    else:
                        sp_correct = sp_correct+1
                # print(prediction.indices[i])
                confusionmap[target[0]][prediction.indices[0]] += 1

        print('\n****************************************\n')
        confusionmap1 = np.zeros([2, 2])
        confusionmap1[0][0] = fp_correct
        confusionmap1[0][1] = fp_error
        confusionmap1[1][0] = sp_error
        confusionmap1[1][1] = sp_correct

        ## non threshold mode ======end=====
        self.plot_confusion_matrix(confusionmap, True)
        print("---------------------------------------")
        self.plot_confusion_matrix(confusionmap, False)
        print('\n****************************************\n')
        self.plot_confusion_matrix(confusionmap1, False)


        log_print(self.log, "test acc：%1.5f  , total nums:" % (test_correct / total,total))
        #
        # ## save error
        f = open(self.save_dir + "/log.txt", 'a')
        f.writelines(log)
        # f.writelines(error_1sp)
        f.close()
        #
       
        return

    def plot_confusion_matrix(self, cm, normalize=True):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Input
        - cm : 计算出的混淆矩阵的值
        - classes : 混淆矩阵中每一行每一列对应的列
        - normalize : True:显示百分比, False:显示个数
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)

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


def main():
    params = BaseOptions().parse()
    net = Net(params)
    if params.test_accflag:
        net.test0()
    if params.test_savimg_flag:
        net.test()

    print('the test work is over!')


if __name__ == '__main__':
    main()

# %%
