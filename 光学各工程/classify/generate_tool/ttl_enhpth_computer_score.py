import torch
import numpy as np
import torchvision
from torchvision import transforms as transforms
import config
from PIL import Image as Img
from torch.autograd import Variable
from mobilenetv3 import MNV30811_SMALL
import multiprocessing
from models import *

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
import itertools
# import matplotlib.pyplot as plt
import os
from shutil import copyfile
from shutil import move
from math import exp
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

def make_dataset(dir):
    images = []
    #assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if "bmp" in fname:
                path = os.path.join(root, fname)
                images.append(path)
    return images
def get_path_trans(line,envir):
    pathA = line.split("ENROLL:")[-1].split(" verify")[0]
    pathA = pathA.replace('\\','/')
    pathA = envir+'/'+pathA
    
    #pathA = pathA.replace("-auto", "-auto_enhance")
    pathB = line.split("verify:")[-1].split(",trans")[0]
    pathB = pathB.replace('\\','/')
    
    pathB = envir+'/'+pathB
    #pathB = pathB.replace("-auto", "-auto_enhance")
    trans = []
    trans_str_list=line.split("trans:")[-1].split(",")
    for i in range(6):
        #print()
        trans.append(int(trans_str_list[i]))
    trans.append(0)
    trans.append(0)
    trans.append(256)
    Htrans = torch.tensor(trans)
    Htrans = Htrans.reshape(3,3)/256  

    return pathA,pathB,Htrans

def make_dataset_txt(txt_root,flag=0,far_num=1,frr_num=1):
    txt_path=[]
    smples=[]
    labels=[]
    line_txt=[]
    
    #assert os.path.isdir(txt_root), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(txt_root)):
        for fname in fnames:
            if ".txt" in fname:
                path = os.path.join(root, fname)
                if flag==2:   ###全部数据
                  txt_path.append(path)
                elif flag==1:  ## fr数据
                    if 'far' not in fname:
                       txt_path.append(path)
                else:
                    if 'far' in fname: ##fa数据
                        txt_path.append(path)
                # if len(txt_path)>0:
                #     break
                
    for txt in txt_path:
        #if "far" in txt:
            #continue
        with open(txt, "r") as file:
            num= 0
            print(txt)
            envir = txt.split('/')[-2]
            lines = file.readlines()
            lines = lines[1:]
            label = 1
            if "far" in txt:
                #continue
                label = 0
            #else:
                #add_name = "_warp_1_8/"
            for line in lines:
                num = num + 1
                if label == 0 and num !=far_num:#if label == 0 and num !=100 if label == 0 and num !=15 30
                    continue
                if label == 1 and num !=frr_num:#if label == 1 and num !=5: 10
                    continue
                num = 0
                try: 
                    
                #if 1:
                    p_list = line.split("/")
                    #if p_list[1] != p_list[4]:
                        #continue
                    #root = frrfar_path + "/" + path_src + add_name + p_list[1] + "/" + p_list[2]
                    pathA, pathB, trans= get_path_trans(line,envir)
                    s = {"pathA":pathA,"pathB":pathB,"trans":trans}
                    smples.append(s)
                    labels.append(label)
                    line_txt.append(txt + "____" + line)
                except:
                    print("warning",line)
                    continue
    return smples,labels,line_txt

def get_warp_img_o_enhance(enhanceA,enhance_imgB,Htrans):

    level_256_B_M = np.ones_like(enhance_imgB)
    level_256_B_M[level_256_B_M > 0] = 255
    cuth,cutw = enhance_imgB.shape

    w_h = cutw//2
    H_h = cuth//2
    w_h = cutw//2
    H_h = cuth//2
    center_result = torch.sum(Htrans*torch.tensor([w_h,H_h,1]),-1)
    center_result[center_result>0] = center_result[center_result>0]+0.5
    center_result[center_result<0] = center_result[center_result<0]-0.5
    center_n = (center_result + torch.tensor([w_h,H_h,1]))/2
    center_d = torch.tensor([w_h,H_h,1]) - center_n
    center_d[center_d>0] = center_d[center_d>0]+0.5
    center_d[center_d<0] = center_d[center_d<0]-0.5
    center_d_i_x = int(center_d[0])
    center_d_i_y = int(center_d[1])
    Htrans[0,2] = Htrans[0,2] + center_d_i_x
    Htrans[1,2] = Htrans[1,2] + center_d_i_y

    image_warp = inv_warp_image(torch.from_numpy(enhance_imgB), Htrans,size=(cutw,cuth))
    image_warp_B_M = inv_warp_image(torch.from_numpy(level_256_B_M), Htrans,size=(cutw,cuth))
    image_warp_B_M[image_warp_B_M < 255] = 0
    image_warp_B_M[image_warp_B_M >= 255] = 1
    image_warp = image_warp * image_warp_B_M
    image_warp = np.array(image_warp).astype(np.uint8)
    
    enhance_imgA_n = np.zeros((cutw,cuth),dtype=np.uint8)
    if center_d_i_y>0 :
        c_y_t = 0
        c_y_b = cuth - center_d_i_y
        e_y_t = center_d_i_y
        e_y_b = cuth
    else:
        c_y_t = -center_d_i_y
        c_y_b = cuth
        e_y_t = 0
        e_y_b = cuth+center_d_i_y

    if center_d_i_x>0 :
        c_x_t = 0
        c_x_b = cutw - center_d_i_x
        e_x_t = center_d_i_x
        e_x_b = cutw
    else:
        c_x_t = -center_d_i_x
        c_x_b = cutw
        e_x_t = 0
        e_x_b = cutw+center_d_i_x

    enhance_imgA_n[e_y_t:e_y_b,e_x_t:e_x_b] = enhanceA[c_y_t:c_y_b,c_x_t:c_x_b]

    g = image_warp.reshape(image_warp.shape[0],image_warp.shape[1],1)
    r = enhance_imgA_n.reshape(image_warp.shape[0],image_warp.shape[1],1)
    b = np.zeros_like(r)

    img_match = np.concatenate([r,g,b],axis=2)
    return img_match


transorm_enh = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
])

transform_msk= transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize([96, 96]),
            transforms.ToTensor(),
])

from matlab_imresize import imresize
class Dataset_ENH(torch.utils.data.Dataset):
    def __init__(self, files_enhance, transform=None):
        self.imgs = files_enhance
        self.transform = transform
        self.datalen = len(self.imgs)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path).convert('L')
        img = np.array(img, dtype=np.float32)  
        img = imresize(img, method='bilinear', output_shape=[128, 128], mode="vec")
        img = Image.fromarray(img).convert('L')
        if self.transform is not None:
            sample = self.transform(img)
        return sample
    
    def __len__(self) :
        return self.datalen 

class Dataset_Mask(torch.utils.data.Dataset):
    def __init__(self, files_mask, transform=None):
        self.imgs = files_mask
        self.transform = transform
        self.datalen = len(self.imgs)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path).convert('L')     

        if self.transform is not None:
            sample = self.transform(img)

        return sample
    
    def __len__(self) :
        return self.datalen 

def enhance(rootpath,samples,enhance_model,device,mask_model):  #img:debase img

    lens = len(samples)
    files_enhance =[]
    for i in range(lens):
        patha = rootpath + "/" + samples[i]["pathA"]
        pathb = rootpath + "/" + samples[i]["pathB"]
        if patha not in files_enhance:
            files_enhance.append(patha)
        if pathb not in files_enhance:
            files_enhance.append(pathb)

    enh_sets = Dataset_ENH(files_enhance,transorm_enh)
    batchsize = 512
    enh_loader = torch.utils.data.DataLoader(enh_sets, batch_size=batchsize, shuffle=False,
                                            num_workers=8)
    all_datas_enh=[]
    index = 0
    for i, data in enumerate(enh_loader):
        data = data.to(device)
        out = enhance_model(data)
        
        out = out[:,:,2:-2,2:-2]
        out= F.interpolate(out, size=[124, 124], mode="bilinear", align_corners=False)
        out = out.cpu().detach().numpy()
        out = (out + 1) / 2.0 * 255.0
        out = np.maximum(out, 0)
        out = np.minimum(out, 255)


        b,c,h,w = out.shape
        for ind in range(b):  
            outim = out[ind,:,:,:]    
            outim = outim.reshape(124,124)
            out = np.uint8(out)
            s = {'path':files_enhance[index],'enh':outim}
            all_datas_enh.append(s)
            index = index+1
    
    mask_sets = Dataset_Mask(files_enhance,transform_msk)
    msk_loader = torch.utils.data.DataLoader(mask_sets, batch_size=batchsize, shuffle=False,
                                            num_workers=8)
    all_datas_msk=[]
    index = 0
    for i, data0 in enumerate(msk_loader):
        b,c,h,w = data0.shape
        data = data0.to(device)
        out = mask_model(data)
        
        out= F.interpolate(out, size=[124, 124], mode="bilinear", align_corners=False)
        out = out.cpu().detach().numpy()
        b,c,h,w = out.shape
        for ind in range(b):  
            outim = out[ind,:,:,:]
            
            mask = outim.reshape(124,124)
            mask[mask>=0.5]=2
            mask[mask<0.5]=1
            mask[mask==2]=0
            mask = np.uint8(mask) 
            s = {'path':files_enhance[index],'msk':mask}
            all_datas_msk.append(s)
            index = index+1                                        
    
    return all_datas_enh,all_datas_msk


class Dataset_Compare_Txt_enh(torch.utils.data.Dataset):
    def __init__(self, root, txt_path,flag,enhance_model,mask_model,device, transform=None,data_flag='95',far_num=15,frr_num=5):
        self.samples,self.labels,self.line_txt = make_dataset_txt(txt_path,flag,far_num,frr_num)
        self.root = root
        self.transform = transform
        self.targets = self.labels
        self.class_to_idx = [0,1]

        self.enhance_model = enhance_model
        self.mask_model = mask_model
        self.device = device
        
        self.all_datas_enh,self.all_datas_msk =  enhance(self.root,self.samples,self.enhance_model,self.device,mask_model)

        print(len(self.samples),len(self.all_datas_enh))
        

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        
        inf = self.samples[index]
        #print(inf)
        label=self.labels[index]
        line_txt = self.line_txt[index]


        idda = next(x for x in self.all_datas_enh if x["path"]==self.root + "/" + inf["pathA"])
        iddb = next(x for x in self.all_datas_enh if x["path"]==self.root + "/" + inf["pathB"])

        ddam = next(x for x in self.all_datas_msk if x["path"]==self.root + "/" + inf["pathA"])
        ddbm = next(x for x in self.all_datas_msk if x["path"]==self.root + "/" + inf["pathB"])

        img_match = get_warp_img_o_enhance(idda["enh"]*ddam["msk"],iddb["enh"]*ddbm["msk"] , inf["trans"])
        img_match = Img.fromarray(img_match)

        img_match = self.transform(img_match)
        #print("shape",img_match.shape)

        return img_match,label,line_txt

class Dataset_Compare_Txt_enh_honor(torch.utils.data.Dataset):
    def __init__(self, root, txt_path,flag,enhance_model,mask_model,device, transform=None,far_num=15,frr_num=5,datasize=124):
        #datas = pd.read_csv(root,header=None)

        self.samples,self.labels,self.line_txt = make_dataset_txt(txt_path,flag,far_num,frr_num)
        self.root = root
        self.transform = transform
        self.targets = self.labels
        self.class_to_idx = [0,1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        inf = self.samples[index]
        #print(inf)
        label=self.labels[index]
        line_txt = self.line_txt[index]


        idda = next(x for x in self.all_datas_enh if x["path"]==self.root + "/" + inf["pathA"])
        iddb = next(x for x in self.all_datas_enh if x["path"]==self.root + "/" + inf["pathB"])

        ddam = next(x for x in self.all_datas_msk if x["path"]==self.root + "/" + inf["pathA"])
        ddbm = next(x for x in self.all_datas_msk if x["path"]==self.root + "/" + inf["pathB"])

        # img_match = get_warp_img_o_enhance(idda["enh"]*ddam["msk"],iddb["enh"]*ddbm["msk"] , inf["trans"])
        

        flag = random.randint(0,1)
        # if flag==0:
        #     img_match = get_warp_img_o_enhance(idda["enh"]*ddam["msk"],iddb["enh"]*ddbm["msk"] , inf["trans"])
        # else:
        namea = self.root + "/" + inf["pathA"]
        nameb = self.root + "/" + inf["pathB"]
        t1 = namea.split('/')[-1].split('.bmp')[0]
        t2 = nameb.split('/')[-1].split('.bmp')[0]
        t11 = t1+'.jpg'
        t21 = t2+'.jpg'
        namedir1 = namea[:-len(namea.split('/')[-1])]
        namedir2 = nameb[:-len(nameb.split('/')[-1])]
        if not os.path.exists(namedir1+'/'+t11):
            #t11 = t1+'_4.bmp'
            enh1 = idda["enh"]*ddam["msk"]
        else:
            enh1 = cv2.imread(namedir1+'/'+t11,0)

        if not os.path.exists(namedir2+'/'+t21):   
           #t21 = t2+'_4.bmp'
           enh2 = iddb["enh"]*ddbm["msk"]
        else:
           enh2 = cv2.imread(namedir2+'/'+t21,0) 

        img_match = get_warp_img_o_enhance(enh1,enh2 , inf["trans"])

        img_match = Img.fromarray(img_match)
        #print(img_match.size)

        img_match = self.transform(img_match)
        #print("shape",img_match.shape)

        return img_match,label,inf


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


from enhance_model import ResnetGenerator_For_Debase_Zack
def init_enhance_model(enh_flag,device):
    model = ResnetGenerator_For_Debase_Zack(1,2)

    if enh_flag=='ttl15':    
       enpath='./checkpoints/optic_enh/smallttl15_199_net_G.pth'
    elif enh_flag=='st44':
        enpath='./checkpoints/optic_enh/st44_500_net_G.pth'
    
    print(enpath)
    checkpoint = torch.load(enpath, 'cpu')
    model.load_state_dict(checkpoint, strict=True)  #
    enhance_model = model.cuda(device)    #经测试，在cpu上跑model会引起进程阻塞，导致多进程变慢很多
    enhance_model.eval()

    return enhance_model


from mask_model import MNV3_bufen_new
def init_mask_model(device):
    model = MNV3_bufen_new(1,1) 
    enpath = './optic_mask/499_net_G.pth'
   
    # print(enpath)
    checkpoint = torch.load(enpath, 'cpu')
    model.load_state_dict(checkpoint, strict=True)  #
    mask_model = model.cuda(device)    #经测试，在cpu上跑model会引起进程阻塞，导致多进程变慢很多
    mask_model.eval()

    return mask_model

def get_fa_thresholds_readenh(enh_flag,gpu_id):
    time = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M')
    imgsort = 1
    logprint = 1
    # thr = 64800#340
    device = torch.device('cuda:'+str(gpu_id))
    far_inter_num,frr_inter_num = 30,10
    enhance_model = init_enhance_model(enh_flag,device)
    mask_model = init_mask_model(device)

    test_dir =r'/ssd/share/liugq/ttl/honor_124/BNG_bmp-img'
    path_txt=r'/ssd/share/liugq/ttl/honor_124/test'
    
    teacher_flag = False

    confusion_out_path = "./confusion_test_%s_test%s" % (enh_flag,time)
    if not os.path.isdir(confusion_out_path):
        os.mkdir(confusion_out_path)
    # print(test_dir)
    test_transform = transforms.Compose([  # transforms.Resize(config.input_size),
        #transforms.CenterCrop(config.input_size),  #
        #transforms.Grayscale(),
        # transforms.CenterCrop(config.input_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.34347,), (0.0479,))
        # transforms.Normalize()
    ])
    ### 方法一：直接利用pth生成对应的增强图、mask图，生成对应的对位图
    test_set = Dataset_Compare_Txt_enh(test_dir, path_txt,2, enhance_model,mask_model,device,test_transform,data_flag='95',far_num=far_inter_num,frr_num=frr_inter_num)
    ### 方法二: 直接读取增强图、mask图生成对应的对位图
    #test_set = Dataset_Compare_Txt_enh_honor(test_dir, path_txt,flag, enhance_model,mask_model,device,test_transform,far_num=far_inter_num,frr_num=frr_inter_num)
    classnum = 2#len(test_set.classes)
    # print(test_set.imgs)
    appear_times = Variable(torch.zeros(classnum, 1))
    for label in test_set.targets:
        appear_times[label] += 1
    confusionmap = Variable(torch.zeros(classnum, classnum))
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=3200, shuffle=False,
                                            num_workers=8)  # , pin_memory=True
    pth_dir = './checkpoints/optic_classify'  #50

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
        # if int(name[-1].split('_')[3])!=50:
        #     continue
        ttmpname = name[-2]+'_'+name[-1]
        checkpoint = torch.load(pthsss[ipth], map_location=device)

        net = MNV30811_SMALL1(2).to(device)
        # net = MNV30811_LLarg_down(2).to(device)
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
                    # print(data.shape)
                    pat = np.array(list(range(len(target))))

                    data = data[:, :2, : ,:]
                    features, output = net(data)
                    
                    if len(output.shape) == 1:
                        #pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                        out1= output*65536
                    else:
                        out = F.softmax(output,dim=-1)
                        out1 = out[:,1] * 65536
                    #out_class_0 = out1[out1<=mistouch_thr]
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

        icnt=5000
        for thr in class_0_scores[-5000:]:
            

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
            icnt=icnt-1
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
   get_fa_thresholds_readenh(enh_flag='ttl16',gpu_id=0)


