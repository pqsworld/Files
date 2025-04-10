import os
import numpy as np
from PIL import Image
import pandas as pd


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
            return 1
        elif (d_pred.sum() / (w * h)) < 0.1:
            return 0.7
        else:
            return 0
    else:
        intersect = (d_mask * d_pred).sum()
        total = d_mask.sum() + d_pred.sum()
        dice = (2 * intersect) / total
        return dice

def cal_area(mask_path, pred_path):
    # calcute dice
    mask_1 = Image.open(mask_path)
    pred_1 = Image.open(pred_path)

    # convert image to grey
    mask = mask_1.convert('L')
    pred = pred_1.convert('L')
    if mask.size != pred.size:
        print('error: ', pred_path)
    mask_np = np.array(mask)
    pre_np = np.array(pred)
    d_mask = np.zeros([96, 96])
    d_mask[:, :] = mask_np >= 100
    d_pred = np.zeros([96, 96])
    d_pred[:, :] = pre_np >= 50

    return (1 - d_pred.sum() / (96 * 96))


def DiceJudge(image_path, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    contents = []
    content = []
    for parent, dirnames, filenames in os.walk(image_path, followlinks=True):
        for filename in filenames:
            file_path = os.path.join(parent, filename)
            if filename.endswith("fake_B.png"):
                print('文件完整路径：%s\n' % file_path)
                process_image = filename.split('_')

                image_name = "_".join(process_image[:-2])
                real_image = os.path.join(parent, image_name + "_real_B.png")
                fake_image = os.path.join(parent, image_name + "_fake_B.png")
                dice_value = cal_dice(real_image, fake_image)
                contents.append([image_name, dice_value])
                content.append(dice_value)

                AB_image = os.path.join(parent, image_name + "_real_AB.png")
                image_AB_PIL = Image.open(AB_image)

                image = os.path.join(parent, image_name + "_real_A.png")
                image_PIL = Image.open(image).convert('L')

                image_real_PIL = Image.open(real_image).convert('L')
                image_fake_PIL = Image.open(fake_image).convert('L')

                if dice_value >= 0.8:
                    prename = "High_"
                elif dice_value < 0.8 and dice_value >= 0.6:
                    prename = "medium_"
                else:
                    prename = "poor_"
                save_path1 = os.path.join(save_path,prename)
                if not os.path.isdir(save_path1):
                    os.mkdir(save_path1)
                image_newname = os.path.join(save_path1, prename + image_name + "_" + str(dice_value) + ".png")
                image_AB_newname = os.path.join(save_path1, prename + image_name + ".png")
                image_real_newname = os.path.join(save_path1, prename + image_name + "_real_B.png")
                image_fake_newname = os.path.join(save_path1, prename + image_name + "_fake_B.png")
                if dice_value <= 1:
                    image_PIL.save(image_newname)
                    image_AB_PIL.save(image_AB_newname)
                    image_real_PIL.save(image_real_newname)
                    image_fake_PIL.save(image_fake_newname)
    print("Dice Test:")
    mean_value = np.mean(content)
    print("mean_value: ", mean_value)  #junzhi
    max_value = np.max(content)
    print('max_value: ', max_value)
    min_value = np.min(content)
    print('min_value: ', min_value)
    std_value = np.std(content)
    print('std_value: ', std_value)   #fangcha
    contents.append(['mean_value: ', mean_value])
    contents.append(['max_value: ', max_value])
    contents.append(['min_value: ', min_value])
    contents.append(['std_value: ', std_value])
    # df = pd.DataFrame(contents, columns=['imagename', 'iou'])
    df = pd.DataFrame(contents, columns=['imagename', 'dice'])
    save_path = os.path.join(save_path + 'test_dice_170.csv')
    df.to_csv(save_path, index=False)


def AreaJudge(image_path, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    contents = []
    content = []
    for parent, dirnames, filenames in os.walk(image_path, followlinks=True):
        for filename in filenames:
            file_path = os.path.join(parent, filename)
            if filename.endswith("fake_B.png"):
                print('文件完整路径：%s\n' % file_path)
                process_image = filename.split('_')

                image_name = "_".join(process_image[:-2])
                real_image = os.path.join(parent, image_name + "_real_B.png")
                fake_image = os.path.join(parent, image_name + "_fake_B.png")
                dice_value = cal_area(real_image, fake_image)
                contents.append([image_name, dice_value])
                content.append(dice_value)

                AB_image = os.path.join(parent, image_name + "_real_AB.png")
                image_AB_PIL = Image.open(AB_image)

                image = os.path.join(parent, image_name + "_real_A.png")
                image_PIL = Image.open(image).convert('L')

                image_real_PIL = Image.open(real_image).convert('L')
                image_fake_PIL = Image.open(fake_image).convert('L')

                if dice_value >= 0.98:
                    prename = "High_"
                elif dice_value < 0.98 and dice_value >= 0.9:
                    prename = "medium_"
                else:
                    prename = "poor_"

                image_newname = os.path.join(save_path, prename + image_name + "_" + str(dice_value) + "_.png")
                image_AB_newname = os.path.join(save_path, prename + image_name + "_.png")
                image_real_newname = os.path.join(save_path, prename + image_name + "_real_B.png")
                image_fake_newname = os.path.join(save_path, prename + image_name + "_fake_B.png")
                if dice_value <= 1:
                    image_PIL.save(image_newname)
                    image_AB_PIL.save(image_AB_newname)
                    image_real_PIL.save(image_real_newname)
                    image_fake_PIL.save(image_fake_newname)
    print("Area Test:")
    mean_value = np.mean(content)   #均值
    print("mean_value: ", mean_value)
    max_value = np.max(content)
    print('max_value: ', max_value)
    min_value = np.min(content)
    print('min_value: ', min_value)
    std_value = np.std(content)   #标准差
    print('std_value: ', std_value)
    contents.append(['mean_value: ', mean_value])
    contents.append(['max_value: ', max_value])
    contents.append(['min_value: ', min_value])
    contents.append(['std_value: ', std_value])
    # df = pd.DataFrame(contents, columns=['imagename', 'iou'])
    df = pd.DataFrame(contents, columns=['imagename', 'dice'])
    save_path = os.path.join(save_path + 'test_area_170.csv')
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    image_path = r'./results/6193mask_85/valid2_283/images'
    save_path = r'./results/6193mask_85/valid2_283/warning'
    DiceJudge(image_path, save_path)
