'''Fingerprint ASP training with PyTorch.'''
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from PIL import Image
import torch.utils.data
import torch.backends.cudnn as cudnn
import nni
from torch.autograd  import  Function
import warnings
warnings.filterwarnings("ignore")
from LossF import TPCLoss
from LossF import CenterLoss
from LossF import FocalLoss
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import WeightedRandomSampler
from torch.nn.parallel import DataParallel
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import numpy as np
import argparse
#import matplotlib.pyplot as plt


from torch.autograd import Variable
import config

from mobilenetv3 import MNV30811_LARGE
from mobilenetv3 import MNV30811_SMALL
from mobilenetv3 import MNV30811_SMALL_S
from mobilenetv3 import MNV30811_SMALL_B
from mobilenetv3 import MNV30811_SMALL_MAMIP
from mobilenetv3 import MNV30811_SMALL_MAP
#from MobileNet import MNV30811gg
#from focal_loss import FocalLoss
import datetime 
import re
from random import randint
import random
#import win32api 

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from shutil import copyfile
args = None
def parse_args():
    parser = argparse.ArgumentParser(description="ASP Train Network")
    parser.add_argument('--lr', default=config.lr, type=float, help='start learning rate')
    parser.add_argument('--epoch', default=config.epoch, type=int, help='number of epochs')
    parser.add_argument('--per-batch-size', default=config.batch_size, type=int, help='batch size in each context')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda exist')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    args = parser.parse_args()
    return args


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def log_print(log, content):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    print(time, content)
    log.append(time)
    log.append("  " + content + "\n")
class Dataset_Compare_o(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, is_valid_file=None):
        super(Dataset_ID, self).__init__(root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
    def __getitem__(self, index):
        path, target = self.samples[index]
        path = Path(path)
        sample = self.loader(path)      
        #print(path)
        #if self.transform is not None:
            #sample = self.transform(sample)
        temple_path = path.parent
        #对应编号真手指找temple 作为模板
        #if target == 0:
        #    temple_path = path.parent
        #else:
        #    #id = (str(path.parent.parent.parent.stem))
        #    temple_path = Path(str(path.parent.parent.parent.parent).replace("1neg","0pos"))
        #    child_dir_list = [child for child in temple_path.iterdir() if child.is_dir()]
        #    temple_path = random.choice(child_dir_list)
        #    child_dir_list = [child for child in temple_path.iterdir() if child.is_dir()]
        #    temple_path = random.choice(child_dir_list)
            #print(epoch_now)
        temps = list(temple_path.rglob('*.bmp'))
        try:
            temple = self.loader(random.choice(temps))
        except:
            print(temple_path)
        
        sample = random.choise(self.samples)
        

        #if self.transform is not None:
            #temple = self.transform(temple)
        return sample,target,temple
def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if "bmp" in fname:
                path = os.path.join(root, fname)
                images.append(path)
    return images
def make_root(dir):
    root = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root_tmp, _, fnames in sorted(os.walk(dir)):
        #for fname in fnames:
            #if "bmp" in fname:
                #path = os.path.join(root, fname)
        root.append(root_tmp)
    return root
def get_trans(temp,match):
    return 1,temp
class Dataset_Compare(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        #datas = pd.read_csv(root,header=None)
        self.samples = make_dataset(root)

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        path = self.samples[index]
        temple = Image.open(path).convert('L')
        root_path = os.path.dirname(path)
        reals_path = make_dataset(root_path)
        cum = 0
        while True:
            real_path = random.choise(reals_path)
            real = Image.open(real_path).convert('L')
            flag,real_img = get_trans(temple,real)
            cum += 1
            if cum > 20:
                new_temple_path = random.choise(reals_path)
                temple = Image.open(new_temple_path).convert('L')
                cum = 0
            if flag == True:
                break
        cum = 0
        while True:
            fake_path = random.choise(self.samples)
            fake_root_path = os.path.dirname(fake_path)
            if fake_root_path == root_path:
                continue
            fake = Image.open(fake_path).convert('L')
            flag,fake_img = get_trans(temple,real)
            cum += 1
            if cum > 20:
                new_temple_path = random.choise(reals_path)
                temple = Image.open(new_temple_path).convert('L')
                cum = 0
            if flag == True:
                break
        real_img = transform(real_img)
        fake_img = transform(fake_img)

        return real_img,fake_img
class Dataset_Compare2(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        #datas = pd.read_csv(root,header=None)
        self.samples = make_dataset(root)

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        path = self.samples[index]
        temple = Image.open(path).convert('L')
        root_path = os.path.dirname(path)
        reals_path = make_dataset(root_path)
        cum = 0
        line = NULL
        flag = NULL
        path_txt = path.replace("bmp","txt")

        with open(path_txt,'r') as f:
            try:
                line_list = f.readline()
                #line_list = line.split(" ")
                real_name = random.choise(line_list)
                real_path = root_path + "/" + real_name
                real = Image.open(real_path).convert('L')
                flag,real_img = get_trans(temple,real)
            except:
                line = NULL
        while True and line == NULL:
            real_path = random.choise(reals_path)
            real = Image.open(real_path).convert('L')
            flag,real_img = get_trans(temple,real)
            cum += 1
            if cum > 20:
                new_temple_path = random.choise(reals_path)
                temple = Image.open(new_temple_path).convert('L')
                cum = 0
            if flag == True:
                break
        cum = 0
        while True:
            #fake_root = root_path + "/../../"
            root_path_split = root_path.split("/")
            fake_root = root_path_split[0]
            for i in range(1,len(root_path_split)-2):
                fake_root = fake_root +'/'+root_path_split[i]
            fake_root_list = make_root(fake_root)
            fake_root_path = random.choise(fake_root_list)
            fake_path_list = make_dataset(fake_root_path)
            fake_path = random.choise(fake_path_list)

            #fake_root_path = os.path.dirname(fake_path)
            if fake_root_path == root_path:
                continue
            fake = Image.open(fake_path).convert('L')
            flag,fake_img = get_trans(temple,real)
            cum += 1
            if cum > 20:
                new_temple_path = random.choise(reals_path)
                temple = Image.open(new_temple_path).convert('L')
                cum = 0
            if flag == True:
                break
        real_img = transform(real_img)
        fake_img = transform(fake_img)

        return real_img,fake_img
def Rotation180(img):

    x = random.random()
    if(x>0.5):
        res = transforms.RandomRotation([180,180])
        img=res(img)
    return img
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
    def __init__(self, img_path, transform=None): 
        self.img_path = img_path
        self.imgs=store_dataset(img_path)
        
        self.transform = transform
        self.datalen = len(self.imgs)
  

    def __getitem__(self, index):
        path = self.imgs[index]
        lend = len(self.img_path.split('/'))-1
        ppoint = path.split('/')[lend:]
          
        label = int(ppoint[0])

        #### 2.start
        ori_img = cv2.imread(path)
        b,g,r = cv2.split(ori_img)
        
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
        #### 2.end #################

        cols,rows = sample.size
       
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label
    
    def __len__(self) :
        return self.datalen 

class Net(object):
    def __init__(self, args,args2):
        self.model = None
        self.symbol = None
        self.lr = args2['learning_rate']
        #self.epoch = 0
        self.epochs = config.epoch
        self.batch_size = config.batch_size
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = torch.cuda.is_available()
        self.allcuda = 0
        self.ddp = 0
        #self.local_rank = args.local_rank
        self.local_rank = -1
        self.train_loader = None
        self.val_loader = None
        self.positve_value = None
        self.test_imgs = None
        self.log = []
        self.coe_tpcloss = args2['tpcloss_weight']
        self.hard_negative_threshold = args2['hard_negative_threshold']
        self.alpha=0.5

    def load_data(self):

        train_transform = transforms.Compose([#transforms.CenterCrop(config.input_size),
                                              #transforms.Lambda(lambda img:Rotation180(img)),
                                              #transforms.RandomRotation([-5,5]),
                                              #transforms.RandomCrop([128,32]),
                                              #transforms.Resize([76,40]),
                                              #transforms.Grayscale(),
                                              #transforms.ColorJitter(brightness=(0.9,1.1),contrast=(0.9,1.1)),
                                              #transforms.Lambda(lambda img:junheng(img)),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomVerticalFlip(p=0.5),
                                              transforms.ColorJitter(brightness=(0.8,1.2),contrast=(0.8,1.2)),
                                              #transforms.ColorJitter(brightness=(0.9,1.1)),
                                              transforms.Lambda(lambda img:ColorJitterOne(img)),
                                              #transforms.CenterCrop([132,132]),
                                              #transforms.RandomCrop(config.input_size),
                                              transforms.RandomAffine(degrees = 0,translate=(1/16,1/16)),#仿射变换
                                              transforms.ToTensor(),
                                              #transforms.RandomErasing(p=0.3,scale=(0.02,0.2),ratio=(0.3,3.3),value=0),
                                              #transforms.Normalize()
                                              ])
        test_transform = transforms.Compose([#transforms.CenterCrop(config.input_size),
                                             #transforms.CenterCrop(config.input_size),
                                             #transforms.Resize([76,40]),
                                             #transforms.Grayscale(),
                                             #transforms.Lambda(lambda img:junheng(img)),
                                             #transforms.CenterCrop(config.input_size),
                                             transforms.ToTensor(),
                                             #transforms.Normalize()
                                             ])   #(0.34347,), (0.0479,)   vivo数据集
        #trainDir = win32api.GetShortPathName(config.traindir)
        #valDir = win32api.GetShortPathName(config.valdir)
        train_set = MyDataset(config.traindir, train_transform)#torchvision.datasets.ImageFolder(config.traindir, train_transform)
        classes_idx = train_set.class_to_idx
        appear_times = Variable(torch.zeros(len(classes_idx), 1))
        for label in train_set.targets:
            appear_times[label] += 1
        self.classes_weight = (1./(appear_times / len(train_set))).view( -1)
        weight=list(map(lambda x:self.classes_weight[x],train_set.targets))
        #定义sampler
        print(len(classes_idx))
        print(len(train_set))
        num_add=abs(appear_times[0]-appear_times[1])
        print(appear_times[0])
        print(appear_times[1])
        print(num_add)
        #num_sample=int(len(train_set)+num_add)
        num_sample=int(len(train_set))
        print(num_sample)
        if self.ddp:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', args.local_rank)
            dist.init_process_group(backend='nccl')
            sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            self.batch_size = self.batch_size//4
        else:
            sampler = WeightedRandomSampler(weight, num_sample, replacement=True)
        
        #data_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,shuffle=False, num_workers=0,drop_last=True)
        #self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, sampler=sampler, shuffle=False, collate_fn=detection_collate, num_workers=config.num_worker, pin_memory=True,drop_last=True)
        #self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, sampler=sampler, shuffle=False, num_workers=config.num_worker, pin_memory=True,drop_last=True)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=config.num_worker, pin_memory=True)
        test_set = torchvision.datasets.ImageFolder(config.valdir, test_transform)
        ''' 
        classes_idx = test_set.class_to_idx
        appear_times = Variable(torch.zeros(len(classes_idx), 1))
        for label in train_set.targets:
            appear_times[label] += 1
        self.classes_weight = 1./(appear_times / 256089)
        '''
        self.test_imgs = test_set.imgs
        #self.postive_value = classes_idx['fingerprint']
        self.val_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=config.num_worker, pin_memory=True) #

    def get_symbol(self):
        if self.ddp:
            print("ddp")
        elif self.cuda:
            #torch.cuda.current_device()
            #torch.cuda._initialized = True
            #self.device = torch.device('cuda:2')
            self.device = torch.device('cuda:2')
            #cudnn.benchmark = True
            print("cuda")
        else:
            self.device = torch.device('cpu')
        
        
        #if self.ddp:
            #torch.cuda.set_device(self.local_rank)
            #self.device = torch.device('cuda', args.local_rank)
            #dist.init_process_group(backend='nccl')
        
        if self.allcuda:
            print("allcuda init")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "6, 3, 4, 5"
            device_ids = [0, 1, 2, 3]
            torch.cuda.set_device('cuda:{}'.format(device_ids[0]))
            self.device = torch.device('cuda:0')
            self.symbol = MNV30811_SMALL()
            #self.symbol = MNV30811_SMALL().to(self.device)
            self.symbol = self.symbol.cuda(device=device_ids[0])
            self.symbol = DataParallel(self.symbol, device_ids=device_ids, output_device=device_ids[0])
            
            #self.symbol  = torch.nn.Dataparallel(self.symbol) # 默认使用所有的device_ids 
            #self.symbol  = DataParallel(self.symbol)
            
            
            #self.symbol = self.symbol.cuda()
        else:
            #self.symbol_p = MNV30811_SMALL().to(self.device)
            self.symbol = MNV30811_SMALL().to(self.device)
            #self.lin = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1,
                      #stride=1, padding=0, bias=True).to(self.device)

        if self.ddp:
            self.symbol = DDP(self.symbol, device_ids=[self.local_rank], output_device=self.local_rank)

      

        self.optimizer = optim.Adam(self.symbol.parameters(), lr=self.lr)#, weight_decay=0.00001
        #self.optimizer = optim.Adam([
         #                           {'params':self.symbol.parameters(), 'lr':self.lr,},
          #                          {'params':self.symbol_p.parameters(), 'lr':self.lr,},
           #                         {'params':self.lin.parameters(), 'lr':self.lr,},
            #                        ])
        #self.optimizer = optim.Adam(self.symbol.parameters(), lr=self.lr, weight_decay=0.0005)
        #if self.allcuda:
            #self.optimizer = DataParallel(self.optimizer)
        #self.optimizer = optim.SGD(self.symbol.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        #self.optimizer = optim.Adadelta(self.symbol.parameters(), lr=self.lr)
        #mile = [i for i in range(30,config.epoch, 30)]
        #mile = [4,12,20,40]
        #self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=mile, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss(torch.Tensor([1,1])).to(self.device)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min',factor = 0.5, patience = 5, verbose = True)
        #self.criterion = FocalLoss(2, alpha=self.classes_weight.view(-1),gamma=2).to(self.device)
        #self.criterion = FocalLoss(2,gamma=2).to(self.device)
        self.loss_tpc = TPCLoss().to(self.device)
        self.center_loss = CenterLoss(num_classes=2, feat_dim=80, use_gpu=True)
        #self.optimizer_centloss = optim.SGD(self.center_loss.parameters(), lr=0.5)
    def L1_loss_nbias(self,w):
        loss_l1 = 0 
        for name,param in self.symbol.named_parameters():
            #if 'bias'in name:
            if 'weight'in name:
                loss_l1+=torch.sum(torch.abs(param))
        return loss_l1*w
    def get_patch_data(self,data_o):
        #data_patch = NULL
        #print(data.shape)
        data = data_o.clone()
        data_one = data[:,:,0:33,0:20]
        data_patch = data_one
        data_one = data[:,:,0:33,16:36]
        data_patch = torch.cat((data_patch,data_one),0)
        data_one = data[:,:,30:63,0:20]
        data_patch = torch.cat((data_patch,data_one),0)
        data_one = data[:,:,30:63,16:36]
        data_patch = torch.cat((data_patch,data_one),0)
        data_one = data[:,:,59:92,0:20]
        data_patch = torch.cat((data_patch,data_one),0)
        data_one = data[:,:,59:92,16:36]
        data_patch = torch.cat((data_patch,data_one),0)
        data_one = data[:,:,89:122,0:20]
        data_patch = torch.cat((data_patch,data_one),0)
        data_one = data[:,:,89:122,16:36]
        data_patch = torch.cat((data_patch,data_one),0)
        #print(data_patch.shape)
        return data_patch

    def train(self):
        print("train fingerprint asp:")
        self.symbol.train()
        total,train_loss,train_correct,train_loss_ret = 0,0,0,0 
        loss1=0
        loss2=0	
        loss3=0	
        maxp = nn.AdaptiveMaxPool2d(1)
        #print(len(self.train_loader))
        #st = random.randint(1,len(self.train_loader) - 400)
        for batch_num, (data, target) in enumerate(self.train_loader):
            #if(random.random()<0.95):
                #continue
            data, target = data.to(self.device), target.to(self.device)
            #print(data)
            data = data[:,:2,:,:]
          
            self.optimizer.zero_grad()
          
            features,output = self.symbol(data)
          
            loss2=self.criterion(output, target)
           
            loss = loss2 #+ loss3
            
         
            if loss.detach().cpu().numpy()>self.hard_negative_threshold:
                loss.backward()
                self.optimizer.step()
            else:
                loss.backward()
       
            
            train_loss += loss.detach().cpu().numpy()#loss.cpu().item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
            train_loss_ret = train_loss / (batch_num + 1)
            # print (batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         # % (train_loss_ret, 100. * train_correct / total, train_correct, total))
        print(loss1)
        print(loss2)
        print(loss3)
        return train_loss_ret, train_correct / total

    def resume(self):
        print('Resuming from checkpoint...')
        #assert os.path.isdir('model'), 'Error: no model directory found!'
        checkpoint = torch.load('./model/ckpt_46.pth')
        self.symbol.load_state_dict(checkpoint['net'])
        #best_acc = checkpoint['acc']
        #start_epoch = checkpoint['epoch']

    def test(self):
        print("test:")
        self.symbol.eval()
        #prob = np.empty(shape=[0,2])
        total,test_loss,test_correct,test_loss_ret = 0,0,0,0
        #maxp = nn.AdaptiveMaxPool2d(1)
        #st = random.randint(1,len(self.train_loader) - 400)
        with torch.no_grad():                     
            #for batch_num, (data, target) in enumerate(self.val_loader):
            for batch_num, (data, target) in enumerate(self.val_loader):
                #if(random.random()<0.95):
                    #continue
                #if(batch_num>=400):
                    #break
                data, target = data.to(self.device), target.to(self.device)
            #print(data)
                data = data[:,:2,:,:]
         
                features,output = self.symbol(data)
             
                loss2=self.criterion(output, target)
            
                loss = loss2
            
                test_loss += loss.detach().cpu().numpy()#loss.cpu().item()
                prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
                total += target.size(0)

            # train_correct incremented by one if predicted right
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
                test_loss_ret = test_loss / (batch_num + 1)
                #print (batch_num, len(self.val_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
      
        log_print(self.log, "test loss: %1.5f, test acc：%1.5f" % (test_loss_ret, test_correct / total))
        return test_loss_ret, test_correct / total

    def save(self,epoch):
        model_out_path = "%s/ckpt_%d.pth" % (model_path,epoch)                
        torch.save(self.model, model_out_path)
        '''
        print("Checkpoint saved to {}".format(model_out_path))
        self.log.append("Checkpoint saved to {} \n".format(model_out_path))
        '''
        log_print(self.log, "Checkpoint saved to {}".format(model_out_path))

    def start(self):
        self.load_data()
        self.get_symbol()
        #self.resume()
        #记录acc最高，lost最低的轮数
        accuracy = 0
        accuracyT = 0
        lostvalueT = 1
        scoreT = 0
        bestone = 0
        
        
        train_accuracy = 0
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        test_result = [0., 0.]

        #for epoch in range(self.model['epoch']+1, self.epochs + 1):
        for epoch in range(1, self.epochs + 1):
            if self.ddp:
                self.train_loader.sampler.set_epoch(epoch)
            train_result = self.train()
            #self.scheduler.step(epoch)
            '''
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            print(time, "Epoch[%03d]: %03d/%d    acc=%1.5f    lossvalue=%1.5f" % (epoch,epoch,self.epochs,train_result[1],train_result[0]))
            self.log.append(time)
            self.log.append("  Epoch[%03d]: %03d/%d    acc=%1.5f    lossvalue=%1.5f \tlearning_rate=%1.7f\n" % (epoch,epoch,self.epochs,train_result[1],train_result[0], get_lr(self.optimizer)))
            '''
            log_print(self.log, "\nEpoch[%03d]: %03d/%d    acc=%1.5f    lossvalue=%1.5f \tlearning_rate=%1.7f" % (epoch,epoch,self.epochs,train_result[1],train_result[0], get_lr(self.optimizer)))
            train_loss.append(train_result[0])
            self.scheduler.step(train_result[0])
            train_acc.append(train_result[1])
            if (train_result[1] > train_accuracy and train_result[1] > 0.970) or (train_result[1] > 0.98) or (epoch == self.epochs):
                if self.allcuda:
                    state = {
                        'net': self.symbol.module.cpu().state_dict(),
                        #'net2': self.symbol_p.module.cpu().state_dict(),
                        #'netl': self.lin.module.cpu().state_dict(),
                        'train_acc':train_result[1],
                        'test_acc': test_result[1],
                        'epoch': epoch,
                    }
                else:
                    state = {
                            'net': self.symbol.state_dict(),
                            #'net2': self.symbol_p.state_dict(),
                            #'netl': self.lin.state_dict(),
                            'train_acc':train_result[1],
                            'test_acc': test_result[1],
                            'epoch': epoch,
                    }
                if self.ddp:
                    state = {
                        'net': self.symbol.module.state_dict(),
                        #'net2': self.symbol_p.module.state_dict(),
                        #'netl': self.lin.module.state_dict(),
                        'train_acc':train_result[1],
                        'test_acc': test_result[1],
                        'epoch': epoch,
                    }
                self.model = state
                self.save(epoch)
            train_accuracy = max(train_accuracy, train_result[1])
            nni.report_intermediate_result(train_accuracy)
            #记录acc-lost结果最大的轮数
            #'''
            if train_result[1]-train_result[0]>scoreT:
                accuracyT=train_result[1]
                lostvalueT=train_result[0]
                scoreT=accuracyT-lostvalueT
                bestone=epoch
            #print('\nbest performence:\n[%d]score=%1.5f    accuracy=%1.5f    lossvalue=%1.5f\n' % (bestone, scoreT, accuracyT, lostvalueT))
            #self.log.append('\nbest performence:\n[%d]score=%1.5f    accuracy=%1.5f    lossvalue=%1.5f\n' % (bestone, scoreT, accuracyT, lostvalueT))
            log_print(self.log,'\nbest performence:\n[%d]score=%1.5f    accuracy=%1.5f    lossvalue=%1.5f\n' % (bestone, scoreT, accuracyT, lostvalueT))
            #'''
            #记录test上acc-lost结果最大的轮数
            '''
            if test_result[1]-test_result[0]>scoreT:
                accuracyT=test_result[1]
                lostvalueT=test_result[0]
                scoreT=accuracyT-lostvalueT
                bestone=epoch
            #print('\nbest performence:\n[%d]score=%1.5f    accuracy=%1.5f    lossvalue=%1.5f\n' % (bestone, scoreT, accuracyT, lostvalueT))
            #self.log.append('\nbest performence:\n[%d]score=%1.5f    accuracy=%1.5f    lossvalue=%1.5f\n' % (bestone, scoreT, accuracyT, lostvalueT))
            log_print(self.log,'\nbest performence:\n[%d]score=%1.5f    accuracy=%1.5f    lossvalue=%1.5f\n' % (bestone, scoreT, accuracyT, lostvalueT))
            '''
            test_term = 40
            if train_accuracy > 0.98:
                test_term = 1
            if epoch%test_term == 0:
                test_result = self.test()
                val_loss.append(test_result[0])
                val_acc.append(test_result[1])
                if test_result[1] > accuracy:
                    state = {
                        'net': self.symbol.state_dict(),
                        #'net2': self.symbol_p.state_dict(),
                        #'netl': self.lin.state_dict(),
                        'train_acc':train_result[1],
                        'test_acc': test_result[1],
                        'epoch': epoch,
                    }
                    self.model = state
                    self.save(epoch)
                accuracy = max(accuracy, test_result[1])
                '''
                print('[%d]Accuracy-Highest=%1.5f    lossvalue=%1.5f' % (epoch, accuracy, test_result[0]))
                self.log.append('[%d]Accuracy-Highest=%1.5f    lossvalue=%1.5f\n' % (epoch, accuracy, test_result[0]))
                '''
                log_print(self.log,'[%d]Accuracy-Highest=%1.5f  accuracy=%1.5f  lossvalue=%1.5f' % (epoch, accuracy,test_result[1],test_result[0]))
                #self.scheduler.step(test_result[0])
                print('lr-epoch:', get_lr(self.optimizer), epoch)
                #self.log.append('lr-epoch: %f  %d\n'%(get_lr(self.optimizer), epoch))
                
                #记录acc-lost结果最大的轮数
                '''
                if test_result[1]-test_result[0]>scoreT:
                    accuracyT=test_result[1]
                    lostvalueT=test_result[0]
                    scoreT=accuracyT-lostvalueT
                    bestone=epoch
                print('\nbest performence:\n[%d]score=%1.5f    accuracy=%1.5f    lossvalue=%1.5f\n' % (bestone, scoreT, accuracyT, lostvalueT))
                self.log.append('\nbest performence:\n[%d]score=%1.5f    accuracy=%1.5f    lossvalue=%1.5f\n' % (bestone, scoreT, accuracyT, lostvalueT))
                '''
                if get_lr(self.optimizer) < 1e-5:
                    break
            else :
                val_loss.append(test_result[0])
                val_acc.append(test_result[1])
                
            #record         
            #if epoch%100 == 0:
            #    self.plotacc(train_loss, train_acc, val_loss, val_acc, epoch)
                
            if epoch == self.epochs:
            #    self.plotacc(train_loss, train_acc, val_loss, val_acc, self.epochs)
                '''
                print("Epoch: End of the train, And the Accuracy: %1.5f " % accuracy)
                self.log.append("\n Epoch: End of the train, And the Accuracy: %1.5f \n" % accuracy)
                '''
                log_print(self.log,"Epoch: End of the train, And the Accuracy: %1.5f " % accuracy)
            
            f = open(model_path + "/log.txt", 'a')
            f.writelines(self.log)
            f.close()
            self.log = []
            
            if get_lr(self.optimizer) < 1e-5:
                state = {
                    'net': self.symbol.state_dict(),
                    #'net2': self.symbol_p.state_dict(),
                    #'netl': self.lin.state_dict(),
                    'train_acc':train_result[1],
                    'test_acc': test_result[1],
                    'epoch': epoch,
                }
                self.model = state
                self.save(epoch)
                break
        #nni.report_final_result(accuracy)
        #f = open(model_path + "/acc_%1.5f.txt"%accuracy, 'a')
        #hyper_param="learning_rate:%f\ntpcloss_weight:%f\nhard_negative_threshold:%f\n"%(nni.get_current_parameter('learning_rate'),nni.get_current_parameter('tpcloss_weight'),nni.get_current_parameter('hard_negative_threshold'))
        #f.writelines(hyper_param)
        #f.close()
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

def generate_default_params():
    '''
    Generate default parameters for mnist network.
    '''
    params = {
        'learning_rate': 0.00002,
        'tpcloss_weight':0.005,
        'hard_negative_threshold':0}
    return params

def main():
    global args
    global time
    global model_path
    #params = nni.get_next_parameter()
    #args = parse_args()
    time = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M')
    #experiment_id = nni.get_experiment_id()
    #trial_id = nni.get_trial_id()
    #model_path = "./%s/model_%s%s"%(experiment_id,trial_id,time)
    model_path = "./MODEL_6193/model_%s"%(time)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    copyfile(__file__, Path(model_path)/"train.py")
    copyfile("./config.py", Path(model_path)/"config.py")
    copyfile("./mobilenetv3.py", Path(model_path)/"mobilenetv3.py")
    args2 = generate_default_params()
    #args.update(params)
    net = Net(args,args2)
    net.start()
    print('the train work is over!')

if __name__ == '__main__':
    main()
#CUDA_VISIBLE_DEVICES="0,1,2,3" python3 -m torch.distributed.launch --nproc_per_node 4 train_spoof_mobilenet.