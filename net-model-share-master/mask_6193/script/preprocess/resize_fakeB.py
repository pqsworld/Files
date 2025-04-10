import numpy as np
import cv2
#import PIL.Image as im
from PIL import Image
import PIL
from torchvision import transforms as transforms
# 文件遍历
import os
import shutil
path = r'./'
mask_dir = r'./dd2_mask/0000_ext1.bmp'
import torchvision.transforms as transforms
"""
os.walk() ：
root 当前根目录
dirs 前文件夹下的子文件夹list
names 当前文件夹下的文件list

def huatu(Rnum,Gnum):
    for i in range(256):
        if Rnum[i] != 0 or Gnum[i] != 0:
            left=i
            break
    for i in range(255,-1,-1):
        if Rnum[i] != 0 or Gnum[i] != 0:
            right=i
            break
    print(left,right)
    len=range(left,right+1)
    len1=[i-0.15 for i in len]
    len2=[i+0.15 for i in len]
    Rgray=Rnum[left:right+1]
    Ggray=Gnum[left:right+1]
    print(Rgray)
    print(Ggray)
    plt.bar(len1,Rgray,alpha=0.5,width=0.3,color='red')
    plt.bar(len2,Ggray,alpha=0.5,width=0.3,color='blue')
    plt.show()
"""
def addmask2img(image_dir):
    print("Processing Addmask2IMG!")
    for parent, dirnames, filenames in os.walk(image_dir, followlinks=True):
        image_num = len(filenames)
        for filename in filenames:
            file_path = os.path.join(parent, filename)
            if filename.endswith("real_A.png"):
                print('文件完整路径：%s\n' % file_path)
                image_name = filename.split('_')
                image_name = "_".join(image_name[:-2])
                img_path = os.path.join(parent, image_name + "_real_A.png")
                mask_path = os.path.join(parent, image_name + "_fake_B.png")
                bottom = cv2.imread(img_path)
                top = cv2.imread(mask_path)
                top[:, :, 0:2] = 0
                # 权重越大，透明度越低
                overlapping = cv2.addWeighted(bottom, 0.9, top, 0.1, 0)
                # 保存叠加后的图片
                cv2.imwrite(os.path.join(parent, image_name + "_real_AB.png"), overlapping)

def fuse_mask_one(source_path, mask_path,save_dir,save_name): #单图模式
    
    image_ = Image.open(source_path).convert('L')
    try:
        image_mask_temp = Image.open(mask_path).convert('L')
        ima_np = np.array(image_)
        ima_mask_np = np.array(image_mask_temp)
        ima_np = np.hstack((ima_np,ima_mask_np))

        image_c = Image.fromarray(ima_np)
        image_c = image_c.convert('L')
        #target_path = util.mkdir_files(save_dir, source_path, work_dir) #创建文件夹，并返回存储路径
        if(not os.path.exists(save_dir)):
            os.makedirs(save_dir)
        image_c.save(os.path.join(save_dir,save_name))
    except FileNotFoundError: 
        print("Not Found!：%s" % source_path)   
def cal_dice(mask_path, pred_path):
    # calcute dice
    mask_1 = Image.open(mask_path)
    pred_1 = Image.open(pred_path)
    
    # convert image to grey
    mask = mask_1.convert('L')
    pred = pred_1.convert('L')
    w,h = mask_1.size
    if mask.size != pred.size:
        print('error: ', pred_path)
    mask_np = np.array(mask)
    pre_np = np.array(pred)
    d_mask = np.zeros([h, w])
    d_mask[:, :] = mask_np < 128
    d_pred = np.zeros([h, w])
    d_pred[:, :] = pre_np < 128
    if d_mask.sum() == 0:  # 手指图模式
        if d_pred.sum() == 0:
            return 1.1
        elif (d_pred.sum() / (w * h)) < 0.1:
            return 0.7
        else:
            return 0
    else:
        intersect = (d_mask * d_pred).sum()
        total = d_mask.sum() + d_pred.sum()
        dice = (2 * intersect) / total
        return dice
for root,dirs,names in os.walk(path):
    for name in names:
        if "_fake_B.bmp" in name:
            path1=os.path.join(root,name)
            image = Image.open(path1).convert('L')
            res = transforms.Resize([118,32])
            cor = transforms.Pad(2,padding_mode='edge')
            img2 = res(image)
            img2 = cor(img2)
            img2 = np.array(img2)
            
            img2[img2<=128] = 0
            img2[img2>128] = 255
            
            img2 = PIL.Image.fromarray(img2)
            
            name2=name.replace("_fake_B.bmp", "_msk1.bmp")
            root2=root.replace("images", "images_result")
            #name3=name.replace("_msk1.bmp", ".bmp")
            #path1=os.path.join(root,name)
            path2=os.path.join(root2,name2)
            
            #img2.save(os.path.join(root, name2))
            #root2 = root.replace("valid", "valid2")
            #path3=os.path.join(root2,name3)
            #img = fuse_mask_one(path2,path1,root2,name3)
            #path3=root.replace("raw", "bmp")
            if(not os.path.exists(root2)):
                os.makedirs(root2)
            image_c = img2.convert('L')
            image_c.save(path2)
            #shutil.move(path1,path2)
            #print(path1)
            #shutil.copy(path1,path2)
            #os.remove(path1)
            #img = im.open(path1)
            #pix = img.load()
            #width = img.size[0]  # 获得图像的宽度
            #height = img.size[1]  # 获得图像的高度 
            #if width != 32 or height != 128:
            #    print(path1)