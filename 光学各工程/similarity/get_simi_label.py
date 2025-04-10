import os
import shutil
import torch
import math
import pandas as pd
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

from descnet.getdesc import Test
from descnet.tobinary import TOBIN as tobin
from ssim import SSIM as SSIM
from cal_label_light import *

def get_sift_orientation_batch(img, keypoints, patch_size=19, bin_size=10):
    '''
    img:tensor
    '''
    patch_size=19
    w_gauss = torch.tensor([0.0,0.0,0.0,1.0,1.0,2.0,3.0,5.0,6.0,6.0,6.0,5.0,3.0,2.0,1.0,1.0,0.0,0.0,0.0,
    0.0,0.0,1.0,1.0,3.0,5.0,8.0,10.0,12.0,13.0,12.0,10.0,8.0,5.0,3.0,1.0,1.0,0.0,0.0,
    0.0,1.0,2.0,3.0,6.0,11.0,16.0,22.0,26.0,28.0,26.0,22.0,16.0,11.0,6.0,3.0,2.0,1.0,0.0,
    1.0,1.0,3.0,7.0,13.0,22.0,34.0,45.0,55.0,58.0,55.0,45.0,34.0,22.0,13.0,7.0,3.0,1.0,1.0,
    1.0,3.0,6.0,13.0,24.0,42.0,65.0,88.0,106.0,112.0,106.0,88.0,65.0,42.0,24.0,13.0,6.0,3.0,1.0,
    2.0,5.0,11.0,22.0,42.0,73.0,112.0,151.0,182.0,193.0,182.0,151.0,112.0,73.0,42.0,22.0,11.0,5.0,2.0,
    3.0,8.0,16.0,34.0,65.0,112.0,171.0,232.0,279.0,296.0,279.0,232.0,171.0,112.0,65.0,34.0,16.0,8.0,3.0,
    5.0,10.0,22.0,45.0,88.0,151.0,232.0,314.0,378.0,401.0,378.0,314.0,232.0,151.0,88.0,45.0,22.0,10.0,5.0,
    6.0,12.0,26.0,55.0,106.0,182.0,279.0,378.0,456.0,483.0,456.0,378.0,279.0,182.0,106.0,55.0,26.0,12.0,6.0,
    6.0,13.0,28.0,58.0,112.0,193.0,296.0,401.0,483.0,512.0,483.0,401.0,296.0,193.0,112.0,58.0,28.0,13.0,6.0,
    6.0,12.0,26.0,55.0,106.0,182.0,279.0,378.0,456.0,483.0,456.0,378.0,279.0,182.0,106.0,55.0,26.0,12.0,6.0,
    5.0,10.0,22.0,45.0,88.0,151.0,232.0,314.0,378.0,401.0,378.0,314.0,232.0,151.0,88.0,45.0,22.0,10.0,5.0,
    3.0,8.0,16.0,34.0,65.0,112.0,171.0,232.0,279.0,296.0,279.0,232.0,171.0,112.0,65.0,34.0,16.0,8.0,3.0,
    2.0,5.0,11.0,22.0,42.0,73.0,112.0,151.0,182.0,193.0,182.0,151.0,112.0,73.0,42.0,22.0,11.0,5.0,2.0,
    1.0,3.0,6.0,13.0,24.0,42.0,65.0,88.0,106.0,112.0,106.0,88.0,65.0,42.0,24.0,13.0,6.0,3.0,1.0,
    1.0,1.0,3.0,7.0,13.0,22.0,34.0,45.0,55.0,58.0,55.0,45.0,34.0,22.0,13.0,7.0,3.0,1.0,1.0,
    0.0,1.0,2.0,3.0,6.0,11.0,16.0,22.0,26.0,28.0,26.0,22.0,16.0,11.0,6.0,3.0,2.0,1.0,0.0,
    0.0,0.0,1.0,1.0,3.0,5.0,8.0,10.0,12.0,13.0,12.0,10.0,8.0,5.0,3.0,1.0,1.0,0.0,0.0,
    0.0,0.0,0.0,1.0,1.0,2.0,3.0,5.0,6.0,6.0,6.0,5.0,3.0,2.0,1.0,1.0,0.0,0.0,0.0],device=img.device)

    ori_max = 180
    bins = ori_max // bin_size
    batch, c, h, w = img.shape
    offset = patch_size // 2
    device = img.device

    Gx=torch.zeros((batch, c, h+offset*2, w+offset*2), dtype=img.dtype, device=img.device)
    Gy=torch.zeros((batch, c, h+offset*2, w+offset*2), dtype=img.dtype, device=img.device)
    # Gm=torch.zeros((batch, c, h+patch_size, w+patch_size), dtype=img.dtype, device=img.device)
    # Gm[:,:,patch_size:h+patch_size,patch_size:w+patch_size] = 1

    Gx0=torch.zeros_like(img)
    Gx2=torch.zeros_like(img)
    Gy0=torch.zeros_like(img)
    Gy2=torch.zeros_like(img)

    Gx0[:,:,:,1:-1] = img[:,:,:,:-2]*255
    Gx2[:,:,:,1:-1] = img[:,:,:,2:]*255
    Gx[:,:,offset:-offset,offset:-offset] = (Gx0 - Gx2)

    Gy0[:,:,1:-1,:] = img[:,:,:-2,:]*255
    Gy2[:,:,1:-1,:] = img[:,:,2:,:]*255
    Gy[:,:,offset:-offset,offset:-offset] = (Gy2 - Gy0)  #146*146

    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, patch_size), torch.linspace(-1, 1, patch_size)), dim=2)  # 产生两个网格
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.cuda()
    coor_cells = coor_cells.contiguous()
    
    keypoints_num = keypoints.size(1)
    keypoints_correct = torch.round(keypoints.clone())
    keypoints_correct += offset
    
    src_pixel_coords = coor_cells.unsqueeze(0).repeat(batch, keypoints_num,1,1,1)
    src_pixel_coords = src_pixel_coords.float() * (patch_size // 2) + keypoints_correct.unsqueeze(2).unsqueeze(2).repeat(1,1,patch_size,patch_size,1)
    
    src_pixel_coords = src_pixel_coords.view([batch, keypoints_num, -1, 2])
    batch_image_coords_correct = torch.linspace(0, (batch-1)*(h+patch_size-1)*(w+patch_size-1), batch).long().to(device)
    src_pixel_coords_index = (src_pixel_coords[:,:,:,0] + src_pixel_coords[:,:,:,1]*(w+patch_size-1)).long()
    src_pixel_coords_index  = src_pixel_coords_index + batch_image_coords_correct[:,None,None]

    eps = 1e-12
    
    #生成幅值图和角度图
    Grad_Amp = ((torch.sqrt(Gx**2 + Gy**2)) * 256)

    #边界反射
    Grad_Amp[:,:,9] = Grad_Amp[:,:,10]
    Grad_Amp[:,:,-10] = Grad_Amp[:,:,-11]
    Grad_Amp[:,:,:,9] = Grad_Amp[:,:,:,10]
    Grad_Amp[:,:,:,-10] = Grad_Amp[:,:,:,-11]

    degree_value = Gy / (Gx + eps)
    Grad_ori = torch.atan(degree_value)
    Grad_ori = Grad_ori*180 / math.pi #180/(3.1415926)
    a_mask = (Gx >= 0)
    b_mask = (Gy >= 0)
    apbp_mask = a_mask * b_mask
    apbn_mask = a_mask * (~b_mask)
    anbp_mask = (~a_mask) * b_mask
    anbn_mask = (~a_mask) * (~b_mask)
    Grad_ori[apbp_mask] = Grad_ori[apbp_mask]
    Grad_ori[apbn_mask] = Grad_ori[apbn_mask] + 360
    Grad_ori[anbp_mask] = Grad_ori[anbp_mask] + 180
    Grad_ori[anbn_mask] = Grad_ori[anbn_mask] + 180


    #边界反射
    Grad_ori[:,:,9] = Grad_ori[:,:,10]
    Grad_ori[:,:,-10] = Grad_ori[:,:,-11]
    Grad_ori[:,:,:,9] = Grad_ori[:,:,:,10]
    Grad_ori[:,:,:,-10] = Grad_ori[:,:,:,-11]
    
    Grad_ori = Grad_ori % ori_max

    angle = Grad_ori.take(src_pixel_coords_index)

    #高斯加权
    w_gauss /= 512
    Amp = Grad_Amp.take(src_pixel_coords_index)
    Amp = Amp*w_gauss[None,None,:]
    angle_d = ((angle // bin_size)).long() % bins
    angle_d_onehot = F.one_hot(angle_d,num_classes=bins)
    hist = torch.sum(Amp.unsqueeze(-1)*angle_d_onehot,dim=-2) #[0,pi)

    #平滑
    h_t=torch.zeros((batch, keypoints_num, hist.size(-1)+4), dtype=hist.dtype, device=hist.device)
    h_t[:,:,2:-2] = hist
    h_t[:,:,-2:] = hist[:,:,:2]
    h_t[:,:,:2] = hist[:,:,-2:]

    h_p2=h_t[:,:,4:]
    h_n2=h_t[:,:,:-4]
    h_p1=h_t[:,:,3:-1]
    h_n1=h_t[:,:,1:-3]

    Hist = (h_p2 + h_n2 + 4*(h_p1 + h_n1) + 6*hist) / 16
    Hist = Hist.long()
    
    #获取主方向i
    H_p_i = torch.max(Hist,dim=-1).indices
    H_t=torch.zeros((batch, keypoints_num, Hist.size(-1)+2), dtype=Hist.dtype, device=Hist.device)
    H_t[:,:,1:-1] = Hist
    H_t[:,:,-1:] = Hist[:,:,:1]
    H_t[:,:,:1] = Hist[:,:,-1:]

    H_p1=H_t[:,:,2:]
    H_n1=H_t[:,:,:-2]

    H_i_offset = (H_n1 - H_p1) / (2*(H_n1 + H_p1 - 2*Hist) + eps)
    H_p_i_onehot = F.one_hot(H_p_i,num_classes=bins)
    H_p_offset = torch.sum(H_i_offset*H_p_i_onehot,dim=-1)
    H_p = (H_p_i + H_p_offset + 0.5) * bin_size
    H_p = H_p % 180 - 90


    return H_p


def get_ori(overlap_mask,img_t,img_s):
    H, W = 128,128#122, 36
    #产生网格点：
    coor_cells = torch.stack(torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H)), dim=2)  # 产生两个网格
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.cuda()
    coor_cells = coor_cells.contiguous()
    coor_cells = coor_cells.view(H*W,2)

    ori_t = get_sift_orientation_batch(img_t[None,None],coor_cells[None]).squeeze().view(H,W)
    ori_s = get_sift_orientation_batch(img_s[None,None],coor_cells[None]).squeeze().view(H,W)
    ori_diff = abs(ori_s - ori_t)
    ori_diff_180 = 180 - ori_diff
    ori_diff = torch.where(ori_diff < ori_diff_180, ori_diff, ori_diff_180)
    overlap_ori_diff_mean = ori_diff[overlap_mask].mean()
    return overlap_ori_diff_mean

def DescSameSplit(descsA):
    same_mask = torch.tensor([1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1]).unsqueeze(1).repeat(1,8).view(-1).bool()
    
    descs_dim = descsA.size(1)
    factor_dim = 1
    assert descs_dim == 256

    descsA_0, descsA_1 = descsA[:,:128], descsA[:,128:]

    descsA_same    = torch.cat([descsA_0[:,same_mask],descsA_1[:,same_mask]],dim=1)
    descsA_reverse = torch.cat([descsA_0[:,~same_mask],descsA_1[:,~same_mask]],dim=1)

    descsA_new = torch.cat([descsA_same,descsA_reverse],dim=1)

    return descsA_new

def get_ori_hamming(ima_a,img_b,mask,desc_model):
    h,w = ima_a.shape
    pnts_all=[]
    for i in range(h):
        for j in range(w):
    # for i in range(0,h,4):
    #     for j in range(0,w,4):
            if mask[i,j]>10:
                pnts_all.append([j,i])
    

    overlap_mask = (mask[:,:] > 10)
    mean_ori = get_ori(overlap_mask,img_b,ima_a)

    

    # 汉明距离
    # 获得二进制描述子,因为不是扩边图的问题,里面的坐标不改变
    # 获得描述子
    # ima_a = transforms.ToTensor()(ima_a)
    ima_a = ima_a.unsqueeze(0)
    descAs_undetached, patchAs, patchAs_mask, _,angle_A = desc_model.forward_patches_correct_batch_expand(ima_a.unsqueeze(0).cuda(), torch.tensor(pnts_all).float().unsqueeze(0).cuda(), patch_size=16, sample_size=22, correct=True, sift=0, theta=180, trans_theta=0)
    angle_A=((angle_A[:]+90)*math.pi)/180
    # for index in range(Batchsize):
    descA = tobin().Hamming_Hadamard(descAs_undetached[0,:,:])
    # # # 哈德玛同号位异号位
    descA_same = DescSameSplit(descA).float()

    # img_b = transforms.ToTensor()(img_b)
    img_b = img_b.unsqueeze(0)
    descBs_undetached, _, _, _,angle_B = desc_model.forward_patches_correct_batch_expand(img_b.unsqueeze(0).cuda(), torch.tensor(pnts_all).float().unsqueeze(0).cuda(), patch_size=16, sample_size=22, correct=True, sift=0, theta=180, trans_theta=0)
    angle_B=((angle_B[:]+90)*math.pi)/180
    # for index in range(Batchsize):
    descB = tobin().Hamming_Hadamard(descBs_undetached[0,:,:])
    # # # 哈德玛同号位异号位
    descB_same = DescSameSplit(descB).float()
    # # 计算距离
   
    ham = np.sum(descA.cpu().numpy()[:,:]!=descB.cpu().numpy()[:,:])/descA.shape[0]
    # 卡patch的汉明距阈值
    ham_thre = np.sum((descA.cpu().numpy()[:,:]!=descB.cpu().numpy()[:,:]),axis=1)
    ham_thre = np.mean(ham_thre[:]<100)

    # 计算patch的ham满足阈值的比例
    
    # return round(mean_ori,3),round(ham,3)
    return mean_ori.cpu().item(),ham,ham_thre
    # return mean_ori.cpu().item(),ham,ham_1,ham_2,ham_thre







def cal_ssim(imga, imgb,mask,ssim_me):
    mask = mask[:,:] > 10
    mask=torch.tensor(mask).unsqueeze(0).cuda()
    imga = imga.unsqueeze(0)
    imgb = imgb.unsqueeze(0)
    zeros_m=torch.zeros_like(imga)
    imga=torch.where(mask>0,imga,zeros_m)
    imgb=torch.where(mask>0,imgb,zeros_m)
    imga=imga.unsqueeze(0)
    imgb=imgb.unsqueeze(0)
    #print(imga.size())
    im_out=ssim_me(imga*255,imgb*255)
    #print(filename,torch.mean(im_out))
    im_out=im_out*mask
    # print(im_out.size())
    # exit()
    # result=int((torch.mean(im_out)*100)/torch.mean(mask.float()))
    result=(torch.mean(im_out)*100)/torch.mean(mask.float())
    # print(result.cpu().item())
    return result.cpu().item()

def get_byimg_mask_csv(img_path,img_dir, savedir,desc_model,ssim_me):    
       
        img_all = Image.open(img_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img_all))

        img_p = img_tensor[:,:,0]
        img_a = img_tensor[:,:,1]
        mask=np.ones(img_a.shape)
        mask[img_a[:,:]==0]=0
        mask[img_a[:,:]!=0]=255
        img_p = (img_p/255).cuda()
        img_a = (img_a/255).cuda()
          
        mean_ori,mean_ham,ham_thre= get_ori_hamming(img_a,img_p,mask,desc_model)
        # 计算ssim
        ssim = cal_ssim(img_p, img_a,mask,ssim_me)
        img_all.close()
        # return mean_ori, mean_ham, ham1,ham2, ssim
        return mean_ori, mean_ham, ham_thre, ssim



os.environ['CUDA_VICIBLE_DEVICES']='5'
ssim_me=SSIM(win_size=15,win_sigma=4, channel=1)
desc_model = Test(DO_CUDA=True)



if __name__=="__main__":
    # for filename in file_list:
    img_dir=""
    savedir = ""
    all_img = [str(f) for f in Path(img_dir).rglob('*.bmp')]
    all_len = len(all_img)      
    contents = []
    for idx, imgname in enumerate(all_img):
        print(idx,'/',all_len)
        mean_ori, mean_ham, ham_thre, ssim = get_byimg_mask_csv(imgname,img_dir, savedir,desc_model,ssim_me)
        # 合并label
        mean_ori, mean_ham, ham_thre, ssim
        ham=256-mean_ham
        ori = 100-mean_ori
        ssim=ssim
        new_label = int(ori+(ham-100)+ssim*0.8+0.5)

        name1 = imgname.split('/')[-1]
        new_name=str(new_label)+'_ham'+str(int(ham))+'_ssim'+str(int(ssim))+'_'+name1

        save_path = os.path.join(imgname[:-len(name1)], new_name)
        save_path = save_path.replace(img_dir,savedir)
        Path(save_path).parent.mkdir(parents=True,exist_ok=True)
        # os.rename(imgname, os.path.join(imgname[:-len(name1)], new_name))
        shutil.copy(imgname, save_path)
        contents.append([imgname, ori, ham, ham_thre, ssim,new_label,save_path])
        if idx%1000==0:
            df = pd.DataFrame(contents, columns=['img_path', 'ori','ham','ham_thre','ssim','label','new_name'])
            df.to_csv(os.path.join(savedir,'info_1201.csv'))
    df = pd.DataFrame(contents, columns=['img_path', 'ori','ham','ham_thre','ssim','label','new_name'])
    df.to_csv(os.path.join(savedir,'info_1201.csv'))




