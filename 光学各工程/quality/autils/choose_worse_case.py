# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:10:37 2022

@author: suanfa
"""
import argparse
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"#"2,3,4"#"4,6,7"
from pathlib import Path
#from Models.cls_network import Cls_Network
#from mobilenetv3_brh import MNV3_large2
from mobilenetv3 import MNV30811_SMALL
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
import shutil
import numpy as np
import torchvision
#from grad_cam_test import img_preprocess,comp_class_vec
#from grad_cam import ShowGradCam
def get_parse():
    parser = argparse.ArgumentParser(
        description='absdiff', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m','--modelpath', default="", help='model_path')
    parser.add_argument('-n','--num', default=100, help='choose n nearest sample')
    parser.add_argument('-p', '--path', metavar='s', type=str,
                        default="new_absdiff")
    parser.add_argument('-t', '--template', metavar='s', type=str,
                        default="temple")
    parser.add_argument('-e', '--exp', metavar='s', type=str,
                        default="new_absdiff")
    parser.add_argument('-d', '--del_flag', type=bool, default=False)
    # default="/hdd/file-input/lind/match/classifier_model/absdiff_newmodel_012515/")
    return parser.parse_args()

class Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, is_valid_file=None):
        super(Dataset, self).__init__(root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            #sample = TF.erase(self.transform(sample),i=90,j=90,h=8,w=8,v=self.transform(sample).mean()+0.2)
        if self.target_transform is not None:
            target = self.target_transform(target)
    
        return sample, target, path
    
class Absdiff:
    def OTSU_with_mask(self, img_gray, mask):
        """带mask的大津法  
        mask：0和255
        """
        #start = time.clock()
        mask_size = np.count_nonzero(mask)

        max_g = 0
        suitable_th = 0

        hist = cv2.calcHist([img_gray], [0], mask, [256], [0, 256])
        cum_hist = np.cumsum(hist)
        pixel_sum_hist = [hist[i] * i for i in range(256)]
        cum_pixel_sum_hist = np.cumsum(pixel_sum_hist)

        # 如果可以确定阈值大致范围，可以在这里设置，缩小阈值遍历次数
        th_list = [i for i in range(0, 256) if int(hist[i][0]) != 0]
        for threshold in th_list:
            fore_pix = cum_hist[-1] - cum_hist[threshold]
            back_pix = cum_hist[threshold]
            if 0 == fore_pix:
                break
            if 0 == back_pix:
                continue

            w0 = float(fore_pix) / mask_size
            u0 = float(cum_pixel_sum_hist[-1] -
                       cum_pixel_sum_hist[threshold]) / fore_pix
            w1 = float(back_pix) / mask_size
            u1 = float(cum_pixel_sum_hist[threshold]) / back_pix
            # intra-class variance
            g = w0 * w1 * (u0 - u1) * (u0 - u1)
            if g > max_g:
                max_g = g
                suitable_th = threshold

        #print("thresh: ", suitable_th)
        #print(time.clock() - start)
        return suitable_th

    def absdiff(self, image):
        # 通道分离，注意顺序BGR不是RGB
        (B, G, R) = cv2.split(image)
        # 显示各个分离出的通道
        #cv2.imshow("Red", R)
        #cv2.imshow("Green", G)
        #cv2.imshow("Blue", B)
        mask = self.mask(R, G)
        thresholdR = self.OTSU_with_mask(R, mask)
        _, R = cv2.threshold(R, thresholdR, 255, cv2.THRESH_BINARY)
        thresholdG = self.OTSU_with_mask(G, mask)
        _, G = cv2.threshold(G, thresholdG, 255, cv2.THRESH_BINARY)
        diff = cv2.absdiff(R, G)
        diff = np.where(diff == 0, 128, 255).astype(np.uint8)
        image = cv2.add(diff, np.zeros(
            np.shape(diff), dtype=np.uint8), mask=mask)
        # sss = sss[min_row:max_row,min_col:max_coR·l,:]#去除黑色无用部分
        #  cv2.imshow("absdiff", image)
        return image

    def mask(self, image1, image2):
        mask = np.zeros([image1.shape[0], image1.shape[1]], dtype=np.uint8)
        min_col = -np.ones(image1.shape[0], dtype=np.int64)
        max_col = -np.ones(image1.shape[0], dtype=np.int64)
        min_row = -np.ones(image1.shape[1], dtype=np.int64)
        max_row = -np.ones(image1.shape[1], dtype=np.int64)
        for i in range(image1.shape[0]):
            temp1, temp2 = np.squeeze(image1[i, :]), np.squeeze(image2[i, :])
            cols1, = np.where(temp1 != 0)
            cols2, = np.where(temp2 != 0)
            if len(cols1) and len(cols2):
                min_col[i] = max(min(cols1), min(cols2))
                max_col[i] = min(max(cols1), max(cols2))
        for i in range(image1.shape[1]):
            temp1, temp2 = np.squeeze(image1[:, i]), np.squeeze(image2[:, i])
            rows1, = np.where(temp1 != 0)
            rows2, = np.where(temp2 != 0)
            if len(rows1) and len(rows2):
                min_row[i] = max(min(rows1), min(rows2))
                max_row[i] = min(max(rows1), max(rows2))
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                # if j==0:
                # print("({},{}):min_row:{},max_row{},min_col:{},max_col:{}".format(i,j,min_row[j],max_row[j],min_col[i],max_col[i]))
                if (i in range(min_row[j], max_row[j]+1)) or (j in range(min_col[i], max_col[i]+1)):
                    mask[i, j] = 255
        return mask


# class Cls(object):
    # # FingerPrint Recognition Deep Learning
    # # 提点，描述子，分类，匹配网络
    # def __init__(self, classifier_net_weights, gpu_ids=0):
        # self.device = torch.device('cuda:' + str(gpu_ids))

        # # 加载分类网络
        # self.classifier_net = Cls_Network(in_channle=1, num_class=2)
        # self.classifier_net.load_state_dict(torch.load(classifier_net_weights))
        # self.classifier_net.eval()

        # # 将网络加载到GPU上
        # self.classifier_net = self.classifier_net.to(self.device)

    # def classification(self, image):
        # # 分类
        # data = image
        # data = data.resize((96, 96), resample=Image.BICUBIC)
        # data_tensor = transforms.ToTensor()(data)
        # data = data_tensor.unsqueeze(0)
        # data = data.to(self.device)
        # data = Variable(data)
        # # compute output for patch a
        # #with torch.no_grad():
        # out = self.classifier_net(data)

        # if self.classifier_net.in_channle > 1:
            # probs = F.softmax(out, dim=1)
        # else:
            # probs = torch.sigmoid(out)

        # probs = probs.squeeze(0)
        # pred_prob, pred = torch.max(probs, 0)
        # pred_prob = pred_prob.to(device='cpu').detach().numpy()

        # pred_lable = pred.to(device='cpu').detach().numpy()

        # return pred_prob, pred_lable, out

def img_preprocess(image):
    data = image
    #data = data.resize((96, 96), resample=Image.BICUBIC)
    data_tensor = transforms.ToTensor()(data)
    data = data_tensor.unsqueeze(0)
    data = data.to('cuda:0')
    data = Variable(data)
    #print(data.size())
    return data
if __name__ == '__main__':
    device = 'cuda:0'
    args = get_parse()
    #模型加载
    model_weights = args.modelpath
    G = MNV30811_SMALL().to(device)
    #G.load_state_dict(torch.load(model_weights,map_location='cuda:0')['G'])
    G.load_state_dict(torch.load(model_weights,map_location='cuda:0')['net'])
    G.eval()
    
    #Cls = Cls(model_weights_classifier)
    #Cls.eval()

    experiment_id = args.exp
    time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    savedir = Path("./save/%s_%s" % (Path(args.modelpath).stem, time))
    if not savedir.exists():
        savedir.mkdir(parents=True)
    dir_path = args.path
    print("dir:{}".format(dir_path))
    temple_path = args.template
    log = savedir/'log.txt'
    log.write_text(model_weights+'\n'+dir_path)
    # csv
    #df = pd.read_csv(info_csv)
    #image_path_list = df['hebing'].to_list()
    #worse_case_sample
    t = Path(temple_path)
    temple_list = []
    for f in [f for f in t.rglob('*.bmp') if f.is_file()]:
        if str(f).find('dodo')==-1 and str(Path(f).name).find('副本')==-1:
            temple_list.append(str(Path(f)))
    #print(len(temple_list))
    # dir
    '''
    p = Path(dir_path)
    filename_list_temp = []
    results = []
    # 获取所有需要处理的文件
    for f in [f for f in p.rglob('*.bmp') if f.is_file()]:
        #if 1 or (str(f).find('merge') == -1 and str(f).find('_p') == -1 and str(f).find('_a') == -1):
        #print(str(f))
        filename_list_temp.append(str(Path(f)))
    image_path_list = filename_list_temp
    '''
    #数据加载
    test_transform = transforms.Compose([#transforms.Resize([200,180]),
                                         #transforms.CenterCrop([153,153]),
                                         transforms.Grayscale(),
                                         #transforms.CenterCrop(config.input_size),
                                         transforms.ToTensor(),
                                         #transforms.RandomErasing(p=1, scale=(0.0015,0.0018), ratio=(1,1), value=(0)),
                                         #transforms.Normalize((0.34347,), (0.0479,))
                                         #transforms.Normalize()
                                         ])
    with torch.no_grad():
        dict_del =[]
        with tqdm(total=len(temple_list), desc='获取模板', unit='pic') as pbar:
            for temple_path in temple_list:
                try:
                    temple = cv2.imdecode(np.fromfile(
                        temple_path, dtype=np.uint8), -1)
                    #image_ori = image
                    temple = Image.fromarray(temple).convert('L')
                    #print("temple_path:{}".format(temple_path))
                    pbar.update(1)
                except:
                    print("failed:{}".format(temple_path))
                    pbar.update(1)
                    continue
                pbar.set_description('temple {}'.format(temple_path))
                t_dir = Path(savedir/'dodo'/Path(temple_path).stem)
                if not t_dir.exists():    
                    t_dir.mkdir(parents=True)
                temple_copy = t_dir/'!temple.bmp'
                temple_copy.write_bytes(Path(temple_path).read_bytes())
                temple = img_preprocess(temple)
                feat,output = G(temple)
                feat = feat.reshape(32)
                center = feat
                #print(center.size())
                dict_sample ={}
                #print(len(test_set))
                test_set = Dataset(dir_path, test_transform)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8) #  , pin_memory=True
                with tqdm(total=len(test_set), desc='计算最相似的样本', unit='pic') as pbar: 
                    for _, (data, target, path) in enumerate(test_loader):
                        data, target = data.to(device), target.to(device)
                        feat,output = G(data)
                        feat = feat.reshape(feat.size(0), -1)
                        #print(feat.size())
                        for i in range(len(target)):
                            #print(feat[i].size())
                            inputs = torch.cat((center,feat[i]),dim=0).reshape(2, -1)
                            n=2
                            dist = torch.pow(inputs,2).sum(dim=1,keepdim=True).expand(n,n)
                            dist = dist+dist.t()
                            dist=dist.addmm(inputs,inputs.t(), beta=1, alpha=-2)
                            dist = dist.clamp(min=1e-10).sqrt()
                            if dist[0][1]<2:
                                dict_sample[str(path[i])]=dist[0][1]
                            #print(dist[0][1])
                            pbar.update(1)
    
                dict_sample = sorted(dict_sample.items(), key = lambda kv:(kv[1], kv[0]))
                index = 0
                #with tqdm(total=args.num, desc='复制中', unit='pic') as pbar: 
                for key,value in dict_sample[:args.num]:
                    #print(value.item())
                    if value.item()>0.3:
                        break
                    p = Path(dir_path)
                    p_relative = Path(key).parent.relative_to(p.parent)
                    #print(t_dir)
                    #print(p_relative)
                    #dest = t_dir/p_relative/('{:0>5d}'.format(index)+'_'+str(Path(key).name))
                    dest1 = savedir/p_relative/Path(key).name
                    if not dest1.parent.exists():
                        dest1.parent.mkdir(parents=True)
                    if not dest1.exists():
                        dest1.write_bytes(Path(key).read_bytes())
                    else:
                        dest3 = savedir/'important'/p_relative/Path(key).name
                        if not dest3.parent.exists():
                            dest3.parent.mkdir(parents=True)
                        dest3.write_bytes(Path(key).read_bytes())
                    dest2 = t_dir/p_relative/('{:0>5d}'.format(index)+'_'+'{:.3f}'.format(value.item())+'_'+str(Path(key).name))
                    if not dest2.parent.exists():
                        dest2.parent.mkdir(parents=True)
                    dest2.write_bytes(Path(key).read_bytes())
                    dict_del.append(str(Path(key)))
                    '''
                    try:
                        Path(key).unlink()
                    except:
                        print("img not exists:{}".format(Path(key)))
                    '''
                    index = index + 1
                    #pbar.update(1)
            #'''
            if args.del_flag:    
                with tqdm(total=len(dict_del), desc='删除中', unit='pic') as pbar:
                    for img in dict_del:
                        try:
                            Path(img).unlink()
                        except:
                            pass
                            #print("img has been deleted:{}".format(Path(key)))
                        pbar.update(1)
            #'''