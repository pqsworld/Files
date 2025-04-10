import os

ccl= [ 0.8,0.9,0.95,0.98]
gb= [ 0.2,0.3,0.755,0.85]
gn= [ 0.65,0.75,0.833,0.9]
in1= [ 0.7,0.8,0.85,0.9]
jpeg= [ 0.56,0.7,0.85,0.9]
msh= [ 0.9,0.92,0.94,0.96]
nepn= [ 0.9,0.8,0.76,0.73]


def rename_bmp_files_with_prefix(directory, prefix='q'):
    # 遍历目录及其子目录
    for root, _, files in os.walk(directory):
        for file_name in files:


            if file_name.endswith('.bmp'):
                # 构建旧文件路径
                old_file_path = os.path.join(root, file_name)

                # 构建新文件名和路径
                try:
                    # num_ori = file_name.split('_')[0].split('q')[1]
                    num_ori = file_name.split('_')[0]
                except IndexError:
                    continue

                file_name_new = ''
                if '_ccl_' in file_name:
                    level = int(file_name.split('_')[-3])
                    file_name_new = str(int(int(num_ori)*ccl[4-level])) + '_' + '_'
                if '_gb_' in file_name:
                    level = int(file_name.split('_')[-3])
                    file_name_new = str(int(int(num_ori)*gb[4-level])) + '_'
                if '_gn_' in file_name:
                    level = int(file_name.split('_')[-3])
                    file_name_new = str(int(int(num_ori)*gn[4-level])) + '_'
                if '_in_' in file_name:
                    level = int(file_name.split('_')[-3])
                    file_name_new = str(int(int(num_ori)*in1[4-level])) + '_'
                if '_jpeg_' in file_name:
                    level = int(file_name.split('_')[-3])
                    file_name_new = str(int(int(num_ori)*jpeg[4-level])) + '_'
                if '_msh_' in file_name:
                    level = int(file_name.split('_')[-3])
                    file_name_new = str(int(int(num_ori)*msh[4-level])) + '_'
                if '_nepn_' in file_name:
                    level = int(file_name.split('_')[-3])
                    file_name_new = str(int(int(num_ori)*nepn[4-level])) + '_'


                # num_new = int(int(num_ori)*100/35)
                # if num_new>100:
                #     num_new=100
                new_file_name = file_name_new+file_name
                # new_file_name = file_name.replace('q','')
                new_file_path = os.path.join(root, new_file_name)

                # 重命名文件
                os.rename(old_file_path, new_file_path)
                # print(f"Renamed: {old_file_path} -> {new_file_path}")


if __name__ == "__main__":
    directory = '/hdd/share/quality/optic_quality/RankIQA-master/RankIQA-master/data/rank_tid2013'
    rename_bmp_files_with_prefix(directory)
