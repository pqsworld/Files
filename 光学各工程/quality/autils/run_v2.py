import os
import glob
import shutil
import random

from tqdm import tqdm
from pathlib import Path

from tool import get_file, delete_file



if __name__ == "__main__":

    # ========== 重命名==========
    # path = r'F:\qint\data\MS_3D\6157A\20240312_3D_blacksilicone'
    # # save_path = os.path.join(os.path.dirname(path), os.path.basename(path) + '_new_p')
    
    # fils = get_file(path)

    # for f in tqdm(fils):
    #     if str(f).find('Ori') != -1 or str(f).find('.bmp') == -1:
    #         continue
        
    #     assert os.path.exists(f)
    #     # f_ori = str(f).replace('Base', 'Ori')
    #     # assert os.path.exists(f) and os.path.exists(f_ori), "Ori ?!!!"

    #     name = Path(f).name
    #     name_new = str(int(Path(f).stem) + 40).rjust(4, '0') + '.bmp'

    #     f_base_new = Path(f).parent / name_new
    #     # f_ori_new = Path(f_ori).parent / name_new

    #     os.rename(f, str(f_base_new))
    #     # os.rename(f_ori, str(f_ori_new))



    # ========== 改变文件目录结构 ==========
    # path = r'F:\qint\data\oppo\HH_match\BOE_WXN'
    # save_path = os.path.join(os.path.dirname(path), os.path.basename(path) + '_new_p')
    
    # fils = get_file(path)

    # for f in tqdm(fils):
    #     if ('2D' in f or '2.5D' in f) and ('!base' not in f):
    #         rel_path = os.path.relpath(f, path)
    #         rel_splited = rel_path.split("\\")

    #         nm_ = '2D' if '2D' in f else '2.5D'
    #         rel_splited.pop(-3)
    #         rel_splited.pop(-2)
    #         rel_splited.insert(0, rel_splited[-2])
    #         nm_ = nm_ + '_' + rel_splited[-3]
    #         rel_splited.insert(1, nm_)
    #         rel_splited.pop(-2)

    #         rel_path = Path(rel_splited[0], rel_splited[1], '_'.join(rel_splited[2:]))
    #         save_file = Path(save_path, rel_path)

    #         if not os.path.isdir(os.path.dirname(save_file)):
    #             os.makedirs(os.path.dirname(save_file))
            
    #         shutil.copy(f, save_file)

    #     else:
    #         rel_path = os.path.relpath(f, path)
    #         rel_splited = rel_path.split("\\")
    #         rel_splited.insert(0, rel_splited[-3])
    #         rel_splited.insert(1, rel_splited[-2])
    #         rel_splited.pop(-3)
    #         rel_splited.pop(-2)

    #         rel_path = Path(rel_splited[0], rel_splited[1], '_'.join(rel_splited[2:]))
    #         save_file = Path(save_path, rel_path)

    #         if not os.path.isdir(os.path.dirname(save_file)):
    #             os.makedirs(os.path.dirname(save_file))
            
    #         shutil.copy(f, save_file)



    # pass

    # ========== 改手指顺序 ==========
    def find_with_keylist(name: str, keylist: list=[]):
        '''
        查找name中是否包含keylist中的任一关键字
        返回keylist中的关键字位置,没有返回-1
        '''
        index = -1
        if len(keylist) == 0:
            return index

        for idx, key in enumerate(keylist):
            index = str(name).find(str(key))
            if index != -1:
                return idx
        
        return index

    finger = list(['L0', 'L1', 'L2', 'L3', 'R0', 'R1', 'R2', 'R3'])
                    #R3    R2    R1    R0    L3    L2    L1    L0

    path = r'F:\qint\data\oppo\cmr\Camry_frr_sh_随机手指顺序'

    for root, dirs, files in tqdm(os.walk(path)):
        
        if '!base' in dirs:   # 目录下有!base文件夹
            for d in dirs:
                basePath = os.path.join(root, d)
                idx = find_with_keylist(d, keylist=finger)    # 
                if idx != -1:
                    # start rename
                    name_new = str(finger[len(finger) - 1 - idx]) + '_s'    # _s后缀表示swaped
                    dst_Path = Path(os.path.dirname(basePath), name_new)

                    try:
                        os.rename(basePath, dst_Path)
                    except:
                        print("rename err!!! : {} {}".format(basePath, name_new))
                    
                    pass
        



