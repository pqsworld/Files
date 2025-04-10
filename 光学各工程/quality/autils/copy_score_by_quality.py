import os
import random
import shutil


def copy_images_every_100(source_dir, dest_dir):
    # 遍历源目录
    idx = 0
    for root, _, files in os.walk(source_dir):
        # 如果当前目录下没有文件，继续下一个目录
        if not files:
            continue

        # 过滤出所有的 .bmp 文件
        bmp_files = [f for f in files if f.endswith('.bmp')]

        # 如果当前目录有符合条件的图片文件
        if bmp_files:
            # 获取相对路径
            rel_path = root[len(source_dir):].lstrip(os.sep)

            # 每100张选择一张
            for i in range(0, len(bmp_files), 30):
                # 从每100张中随机选择一张
                selected_file = random.choice(bmp_files[i:i+30])
                print("idx: "+ str(idx) + 'score: '+ selected_file.split('_')[0])
                # rename = '00'+ str(idx)+'.bmp'
                # 创建目标目录，保留原路径结构
                dest_path = os.path.join(dest_dir, rel_path)
                os.makedirs(dest_path, exist_ok=True)

                # 复制文件到目标目录
                src_file = os.path.join(root, selected_file)
                dest_file = os.path.join(dest_path, selected_file)
                shutil.copy2(src_file, dest_file)
                idx+=1


if __name__ == "__main__":

    source_directory = '/hdd/share/quality/optic_quality/datasets/lif_nvwanew/'
    destination_directory = '/hdd/share/quality/optic_quality/datasets/lif_nvwanew_2k1'
    copy_images_every_100(source_directory, destination_directory)
