import numpy as np
import cv2
#import PIL.Image as im
from PIL import Image

# 文件遍历
import os
import shutil
path = r'./images_result'
path_p = r'./wet_review_p'
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
def fuse_mask2(image_,source_path, target_path):
   
    # image_ = Image.open(source_path).convert('L')
    # image_mask = Image.open(mask_path).convert('L')
    # ima_np = np.array(image_)
    # ima_mask_np = np.array(image_mask)
    # ima_np = np.hstack((ima_np,ima_mask_np))

    # image_c = Image.fromarray(ima_np)
    # image_c = image_c.convert('L')
    # image_c.save(os.path.join(target_path, source_path.split('/')[-1]))

    #image_ = Image.open(source_path).convert('L')
    
    ima_np = np.array(image_)
    ima_mask_np = np.zeros((122,36))
    ima_mask_np[ima_mask_np==0]=255
    ima_np = np.hstack((ima_np,ima_mask_np))

    image_c = Image.fromarray(ima_np)
    image_c = image_c.convert('L')
    image_c.save(os.path.join(target_path, source_path.split('/')[-1]))
for root,dirs,names in os.walk(path):
    for name in names:
        if "_msk1.bmp" in name:
            name2=name.replace("_msk1.bmp", ".bmp")
            print(root)
            root2 = root.replace("images_result", "../../../datasets/6193_2/MASKNet_GT_pin_valid_s")
            print(root2)
            root3 = root.replace("images_result", "images_result_pin")
            path1=os.path.join(root,name)
            path2=os.path.join(root2,name2)
            #root2 = root.replace("test", "test2")
            path3=os.path.join(root3,name2)
            #path4=os.path.join(root2,name4)
            #image_ = Image.open(path1).convert('L')
            mask_c = Image.open(path2).convert('L')
            mask_n = Image.open(path1).convert('L')
            #res = transforms.Resize([122,36])
            #image_ = res(image_)
            if(not os.path.exists(root3)):
                os.makedirs(root3)
            mask_n = np.array(mask_n)
            mask_c = np.array(mask_c)
            #image_ = np.array(image_)
            ima_np = np.hstack((mask_n,mask_c))
            
            
            '''
            na = root.split("/")[:]
            save_data_name = na[2]
            for i_na in range(3, len(na) - 1):
                save_data_name = save_data_name + "_" + na[i_na]
            save_data_name = save_data_name + "_" + name4
            
            path4 = os.path.join(path_p,save_data_name)
            '''
            image_c = Image.fromarray(ima_np)
            image_c = image_c.convert('L')
            image_c.save(path3)
            #img = fuse_mask2(image_,path1,root)
            #path3=root.replace("raw", "bmp")
            #if(not os.path.exists(path2)):
            #    os.makedirs(path2)
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

"""
img = im.open(path)
plt.imshow(img)
plt.show()
im = Image.open('XXX.jpg') 
pix = im.load() 
width = im.size[0] 
height = im.size[1] 
for x in range(width): 
	for y in range(height): 
		r, g, b = pix[x, y]


"""
