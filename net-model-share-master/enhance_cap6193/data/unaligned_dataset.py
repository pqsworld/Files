import os
import random
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from data.base_datasetgq import BaseDataset
from data.image_folder import make_dataset
import torch.nn.functional as F

def gauss_noise(img):
    mean = 0
    # var = random.uniform(0.12,0.2)
    var = random.uniform(0.005,0.1)
    img = np.array(img)
    img = img/255
    noise = np.random.normal(mean, var, img.shape)
    gaussian_out = img + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)
    gaussian_out = np.uint8(gaussian_out*255)
    # gaussian_out = Image.fromarray(np.uint8(gaussian_out*255)) 
    return gaussian_out

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def random_blur_A(img,tmpf):
    """
        :param img: 输入的图像
        :param pos: 图像截取的位置,类型为元组，包含(x, y)
        :param size: 图像截取的大小
        :return: 返回截取后的图像
    """
    blur_flag = random.randint(1, 10)
    if blur_flag > 6:
        # img = Image.fromarray(img)
        A = img

        w, h = A.size

        wi_random = torch.randint(1, tmpf, size=(1,)).item()
        hi_random = torch.randint(1, tmpf, size=(1,)).item()

        hi = torch.randint(0, h - hi_random + 1, size=(1,)).item()
        wj = torch.randint(0, w - wi_random + 1, size=(1,)).item()

        tmp = A.crop((wj, hi, wj + wi_random, hi + hi_random))
        blur_img = cv2.GaussianBlur(np.uint8(tmp), ksize=(7, 7), sigmaX=random.uniform(0.1, 2))
        blur_img = Image.fromarray(blur_img)
        A.paste(blur_img, (wj, hi, wj + wi_random, hi + hi_random))
        img = A
        # img = Image.fromarray(img).convert("L")
    return img


def createfake_A(img,rangew,rangeh):   
    fake_flag = random.randint(1, 10)
    if fake_flag > 6:#8:   #patch融合
        # img = Image.fromarray(img)
        A = img
        w, h = A.size
        # w, h = img.size
        wi_random = random.randint(15, rangew)
        hi_random = random.randint(35, rangeh)

        hi = random.randint(0, h - hi_random + 1)
        wj = random.randint(0, w - wi_random + 1)
            
        tmp = A.crop((wj, hi, wj + wi_random, hi + hi_random))
        tmpnp = np.array(tmp)
        maskval = random.randint(80,180)
        mask = np.ones_like(tmpnp) * maskval
        ratio = random.randint(20,35)*0.01
        tmpresnp = ratio*tmpnp + mask*(1-ratio)
        tmpres = Image.fromarray(tmpresnp)
        A.paste(tmpres, (wj, hi, wj + wi_random, hi + hi_random))
        img = A
        # img = Image.fromarray(img).convert("L")
    elif 0:#fake_flag > 6: #patch遮盖   315关       
        tmp = 22 #patch size  22
        A = img    
        w, h = A.size

        # wi_random = torch.randint(5, tmp, size=(1,)).item()
        # hi_random = torch.randint(12, tmp, size=(1,)).item()

        wi_random = torch.randint(tmp-10, tmp+10, size=(1,)).item()   #-3 +3
        hi_random = torch.randint(tmp-8, tmp+20, size=(1,)).item()   # -8   +8
        
        hi = torch.randint(0, h - hi_random + 1, size=(1,)).item()
        wj = torch.randint(0, w - wi_random + 1, size=(1,)).item()

        tmpr = A.crop((wj, hi, wj + wi_random, hi + hi_random))
        # blur_img = cv2.GaussianBlur(np.uint8(tmp), ksize=(7, 7), sigmaX=random.uniform(0.1, 2))
        # blur_img = Image.fromarray(blur_img)
        # maskval = random.randint(70,180)
        maskval = random.randint(90,230)  #90 230
        mask = np.ones_like(tmpr) * maskval
        mask = Image.fromarray(mask)
        A.paste(mask, (wj, hi, wj + wi_random, hi + hi_random))
        img = A
        img = Image.fromarray(img).convert("L")
    elif 0:#fake_flag>6:#6:   #6   4 315关
        #读取一个图库，进行融合
        A = img
        w, h = A.size
        # w, h = img.size
        wi_random = random.randint(15, rangew)
        hi_random = random.randint(35, rangeh)

        hi = random.randint(0, h - hi_random + 1)
        wj = random.randint(0, w - wi_random + 1)           
        tmp = A.crop((wj, hi, wj + wi_random, hi + hi_random))
        
        randompath='/home/zhangsn/enhance/datasets/00cap/6193database/alldata/trainuse/01expand/DK7/6193_DK7_rotate_8mul_merge/'
        fid = random.choice(['/L1/','/L2/','/L3/','/R1/','/R2/','/R3/'])
        pid = str(random.randint(0,19)).zfill(4)
        imgid = str(random.randint(0,100)).zfill(4)
        fake_file = randompath+pid+fid+imgid+'_exp.bmp'
        import os
        if not os.path.exists(fake_file):
            img =img
        else:
            fake = Image.open(fake_file).convert('L')
            tmpfake = fake.crop((wj, hi, wj + wi_random, hi + hi_random))
            tmpnp = np.array(tmp)
            tmpfakenp = np.array(tmpfake)
            coe= random.uniform(0.2,0.35)
            tmpresnp = (1-coe)*tmpnp + tmpfakenp*coe
            tmpres = Image.fromarray(tmpresnp)      
            A.paste(tmpres, (wj, hi, wj + wi_random, hi + hi_random))
            img = A
            img = Image.fromarray(img).convert("L")
    part_flag = random.randint(1, 10)
    if 0:#part_flag>8:  #5 315关
        #读取一个图库，进行融合
        A = img       
        randompath='/home/zhangsn/enhance/datasets/00cap/6193database/testdatasets/mask/6193_DK7_merge_test2/'
        fid = random.choice(['/L1/','/L2/','/L3/','/R1/','/R2/','/R3/'])
        pid = str(random.randint(6,9)).zfill(4)
        imgid = str(random.randint(20,199)).zfill(4)
        mask_file = randompath+pid+fid+imgid+'_mask.bmp'
        import os
        if not os.path.exists(mask_file):
            img = img
        else:
            ori_file = mask_file.replace('mask','exp')
            mask = np.array(Image.open(mask_file).convert('L'))
            # mask_tmp = np.ones_like(A)*255
            k=np.ones((9,9),np.uint8)
            mask=cv2.dilate(mask,k,iterations=1)
            mask_tmp = np.array(Image.open(ori_file).convert('L'))
            A = np.where(mask == 0, mask_tmp, A)    
            img = A
            img = Image.fromarray(img).convert("L")       
    left_flag = random.randint(1, 10)
    if 0:#left_flag > 8:  #8
        n = 0
        size = random.randint(1,2)
        aw,ah=img.size
        A = img.crop([size, size, aw-size, ah-size])
        # w, h = A.size
        ranval = random.randint(140,215)
        # print(A.size)
        A = np.pad(A,((size,size),(size,size)),constant_values=ranval)    
        img = A
        img = Image.fromarray(img).convert("L") 
    return img

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        if opt.isTrain:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, 'imfusion3') #opt.phase + 'B2') # create a path '/path/to/data/trainB'
        else:
            self.dir_A = os.path.join(opt.dataroot)  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot)
            # if opt.addmask:
            #     self.dir_C = os.path.join(opt.dataroot)
            # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainB'
            # if opt.addmask:
            #     self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
       
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B 
       
        transform_listA=[
            transforms.Grayscale(1),
            transforms.Resize([self.opt.load_sizeh, self.opt.load_sizew])]
        if opt.isTrain:
            if opt.rblur: 
                transform_listA.append(transforms.Lambda(lambda img: random_blur_A(img,12),))
            if opt.cfake: 
                transform_listA.append(transforms.Lambda(lambda img: createfake_A(img,25,60), ))
        transform_listA.append(transforms.ToTensor())
        transform_listA.append(transforms.Normalize((0.5,), (0.5,)))        
        self.transformA = transforms.Compose(transform_listA)

        self.transformB = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize([self.opt.load_sizeh, self.opt.load_sizew]),  #
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        index_B = index % self.B_size
        B_path = self.B_paths[index_B]
        # print(A_path)
        
            
        A_img = Image.open(A_path).convert('L')
        B_img = Image.open(B_path).convert('L')
            
        if self.opt.isTrain:
            desccrop = 0   #desc_exp_training
            if desccrop == 1:
                from util import imresize
                iw, ih = A_img.size
                # print(iw,ih)
                ch = ih/2
                cw= iw/2
                
                # ranh = random.randint(0,20)
                ranh = 0
                
                # Acrop = A_img.crop([cw-18,ch-40+ranh,cw+18,ch+40+ranh])   #36*80
                Acrop = A_img.crop([cw-14,ch-36+ranh,cw+14,ch+36+ranh])   #裁成小块
                nw,nh = Acrop.size
                
                # print(Acrop.size)
                flag = random.randint(0,3)
                
                #四种L型随机遮盖
                ph = 6
                pw = 11
                # ph = 4
                # pw = 8
                if not random.randint(0,1):
                    tmp = random.randint(0,4) - 2
                    ph = ph + tmp
                    pw = pw + tmp
                    #'constant',constant_values=255
                if flag == 0:
                    Acroptmp = np.array(Acrop.crop([pw,ph,nw,nh]))
                    Acroptt = np.pad(Acroptmp,((ph,0),(pw,0)),'constant',constant_values=255)  #'mean')#
                elif flag ==1:  
                    Acroptmp = np.array(Acrop.crop([0,ph,nw-pw,nh]))
                    Acroptt = np.pad(Acroptmp,((ph,0),(0,pw)),'constant',constant_values=255)
                elif flag==2:
                    Acroptmp = np.array(Acrop.crop([pw,0,nw,nh-ph]))     
                    Acroptt = np.pad(Acroptmp,((0,ph),(pw,0)),'constant',constant_values=255)               
                elif flag == 3:  
                    Acroptmp = np.array(Acrop.crop([0,0,nw-pw,nh-ph]))                 
                    Acroptt = np.pad(Acroptmp,((0,ph),(0,pw)),'constant',constant_values=255)
                        
                Bimg = (np.array(Acrop) - 127.5) / 127.5
                img0 = torch.from_numpy(Bimg).float()
                imgB = torch.unsqueeze(img0, 0)
                B = imgB
                
                Aimg = (Acroptt - 127.5) / 127.5
                img1 = torch.from_numpy(Aimg).float()
                imgA = torch.unsqueeze(img1, 0)
                A = imgA
            elif desccrop == 2:
                from util import imresize
                imgn = np.array(A_img)
                # 去除边界问题像素
                # imgn =imgn[1:118,1:31]
                # 6192(180*36->180*30)
                # print(imgn.shape)
                imgn =imgn[:,3:33]
                # print(imgn.shape)
                # imgn=normalization(imgn) * 255
                imgn = imresize.imresize(imgn, method='bilinear', output_shape=[self.opt.load_sizeh, self.opt.load_sizew], mode="vec")       
                
                #添加gauss noise
                if random.randint(0,1):
                    imgn = gauss_noise(imgn)
                    
                
                tmph = random.randint(0,2)-1
                tmpw = random.randint(0,2)-1
                # tmph = random.randint(0,4)-2
                # tmpw = random.randint(0,8)-4
                # print(imgn.shape)
                
                # Aimg =imgn[(tmph+3):(121-tmph),(tmpw+2):(34-tmpw)]  #134 34
                # Aimg =imgn[(tmph+3):(125-tmph),(tmpw+8):(44-tmpw)]  #134 34
                Aimg =imgn[:,(tmpw+3):(33-tmpw)]  #134 34
                # print(Aimg.shape)
                # Aimg = np.pad(Aimg,((tmph+3,tmph+3),(tmpw+8,tmpw+8)),constant_values=255)
                if 0:#not random.randint(0,5):#random.randint(0,1):
                    ranval = random.randint(100,180)
                    Aimg = np.pad(Aimg,((1,1),(1,1)),constant_values=ranval)  
                    Aimg = np.pad(Aimg,((tmph+2,tmph+2),(tmpw+1,tmpw+1)),constant_values=0)  
                else:
                    # Aimg = np.pad(Aimg,((tmph+3,tmph+3),(tmpw+2,tmpw+2)),constant_values=0)  
                    Aimg = np.pad(Aimg,((0,0),(tmpw+3,tmpw+3)),constant_values=255)
                # print(Aimg.shape)
                Aimg = (Aimg - 127.5) / 127.5
                imgA = torch.from_numpy(Aimg).float()
                imgA = torch.unsqueeze(imgA, 0)
                A = imgA
                
                # print(A_img.size)
                imgB = (imgn - 127.5)/ 127.5
                imgB = torch.from_numpy(imgB).float()
                imgB = torch.unsqueeze(imgB, 0)
                B = imgB
            else:
                A = self.transformA(A_img)
                B = self.transformB(B_img)
                # A,B=self.transformAB(A,B)
                # print(A.shape)
                # print(B.shape)

        else:
            #A = self.transformA(A_img)
            #B = self.transformB(B_img)
            from util import imresize
            _, ih = A_img.size
            imgsrc = np.array(A_img, dtype=np.float32)
            Aimg = imgsrc
            Aimg = (Aimg - 127.5) / 127.5
            img0 = torch.from_numpy(Aimg)
            img0 = img0.float()
            img = torch.unsqueeze(img0, 0)
            A = img
            B = img
            # print(A_path)
        
        # print(A.shape)
        # print(B.shape)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
