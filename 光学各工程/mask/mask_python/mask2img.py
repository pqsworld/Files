import cv2
import os

from PIL import Image
import numpy as np


# img = Image.open(r"E:\new_life_202009\ImageConvertStudy\13.png")
# img_rgb = img.convert("RGB")
# img_array = np.array(img_rgb)
# print(img_array.shape)
# w = img.size[0]
# h = img.size[1]
# print(w)
# print(h)
# img2_array = []
# #pos表示修改像素点的位置，嵌套列表中第一个数值表示行号，第二个数值表示列号
# pos=[[15,18],[18,19],[13,17],[13,7],[13,9],[15,6]]
# for i in range(0, h):
#     for j in range(0, w):
#         temp = [i,j]
#         if temp in pos :
#             img_array[i, j] = [255, 0, 0]
#             #[255, 0, 0]为红色，[255, 255, 255]为白色，[0, 0, 0]为黑色等
#             img2_array.append(img_array[i, j])
#         else:
#             img2_array.append(img_array[i, j])
# img2_array = np.array(img2_array)
# print(img2_array.shape)
# img2_array = img2_array.reshape(28, 28,3)
# img3 = Image.fromarray(img2_array)
# img3.show()
# img3.save(r"E:\new_life_202009\ImageConvertStudy\ss1.png")

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


if __name__ == "__main__":
    work_dir = "/hdd/file-input/yey/work/mask/results/fbufen_96_bottleneck7_d4_g4_pix2pix/zhangxiaochi_200/images"
    addmask2img(work_dir)
