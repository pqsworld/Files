from PIL import Image
import os
import os.path
import random


IMG_EXTENSIONS = [
    '.bmp', '.BMP', '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM','raw','RAW',
]

TXT_EXTENSIONS = [
    '.txt', '.TXT',
]

NO_USE_EXTENSIONS = [
    '.DAT', '.dat','cal','bin','.txt','.Txt'
]
NO_USE_EXTENSIONS1 = [
    '/test/','/bin/'
]



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_txt_file(filename):
    return any(filename.endswith(extension) for extension in TXT_EXTENSIONS)

def is_nouse_file(filename):
    return any(filename.endswith(extension) for extension in NO_USE_EXTENSIONS)


def store_dataset(dir):
    imagesA = []
    all_pathA = []

    for root, _, fnames in sorted(os.walk(dir)):#
        for fname in fnames:
            if is_image_file(fname):
                pathA = os.path.join(root, fname)
                tmp = os.stat(pathA)
                # print(tmp.st_size)
                # if tmp.st_size<27702:
                #     # print(pathA)
                #     continue
                # if tmp.st_size>24454:
                #     # print(pathA)
                #     continue
                if tmp.st_size<24454:
                    # print(pathA)
                    continue
                # if tmp.st_size<27702:
                #     # print(pathA)
                #     continue
                    # print(tmp)
                #     os.remove(pathA)
                # imgA = Image.open(pathA).convert('L')
                # imagesA.append(imgA)

                # src = path
                # dst = path[:-6] + '.bmp'
                # os.rename(src,dst) # 用于更改图像名
                #print(path)
                if os.path.exists(pathA):
                   all_pathA.append(pathA)

    return all_pathA


def store_datasetAB(dira,dirb):
    all_pathA = []

    for root, _, fnames in sorted(os.walk(dira)):#
        for fname in fnames:
            if is_image_file(fname):
                pathA = os.path.join(root, fname)
                tmp = os.stat(pathA)
                # print(tmp.st_size)
                # if tmp.st_size<44950:
                #     # print(pathA)
                #     continue
                # if tmp.st_size<27702:
                #     # print(pathA)
                #     continue
                if tmp.st_size<24454:
                    # print(pathA)
                    continue
                all_pathA.append(pathA)
    for root, _, fnames in sorted(os.walk(dirb)):#
        for fname in fnames:
            if is_image_file(fname):
                pathA = os.path.join(root, fname)
                tmp = os.stat(pathA)
                # print(tmp.st_size)
                # if tmp.st_size<44950:
                #     # print(pathA)
                #     continue
                # if tmp.st_size<27702:
                #     # print(pathA)
                #     continue
                if tmp.st_size<24454:
                    # print(pathA)
                    continue
                all_pathA.append(pathA)

    return all_pathA

def store_datasetABC(dira,dirb,dirc):
    all_pathA = []

    for root, _, fnames in sorted(os.walk(dira)):#
        for fname in fnames:
            if is_image_file(fname):
                pathA = os.path.join(root, fname)
                tmp = os.stat(pathA)
                # print(tmp.st_size)
                # if tmp.st_size<44950:
                #     # print(pathA)
                #     continue
                # if tmp.st_size<27702:
                #     # print(pathA)
                #     continue
                if tmp.st_size<24454:
                    # print(pathA)
                    continue
                all_pathA.append(pathA)
    for root, _, fnames in sorted(os.walk(dirb)):#
        for fname in fnames:
            if is_image_file(fname):
                pathA = os.path.join(root, fname)
                tmp = os.stat(pathA)
                # print(tmp.st_size)
                # if tmp.st_size<44950:
                #     # print(pathA)
                #     continue
                # if tmp.st_size<27702:
                #     # print(pathA)
                #     continue
                if tmp.st_size<24454:
                    # print(pathA)
                    continue
                all_pathA.append(pathA)


    for root, _, fnames in sorted(os.walk(dirc)):#
        for fname in fnames:
            if is_image_file(fname):
                pathA = os.path.join(root, fname)
                tmp = os.stat(pathA)
                # print(tmp.st_size)
                # if tmp.st_size<44950:
                #     # print(pathA)
                #     continue
                # if tmp.st_size<27702:
                #     # print(pathA)
                #     continue
                if tmp.st_size<24454:
                    # print(pathA)
                    continue
                all_pathA.append(pathA)

    return all_pathA