import os


def rename_bmp_files_with_prefix(directory):
    numbers=[]
    l_score_1=[0,0,0,0,0,0,0,0,0,0,0]
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
                    numbers.append(int(num_ori))
                    # print(num_ori)
                    l_score_1[int(int(num_ori)/10)]+=1
                except IndexError:
                    continue
    print(l_score_1)
    return numbers
def save_num2txt(nums,outfile):
    """
    docstring
    """
    with open(outfile,'w') as f:
        for num in nums:
            f.write(f"{num}\n")


if __name__ == "__main__":
    directory = '/hdd/share/quality/optic_quality/RankIQA-master/RankIQA-master/data/train4/train'
    path_txt = 'score_train4.txt'
    nums= rename_bmp_files_with_prefix(directory)
    save_num2txt(nums,path_txt)

