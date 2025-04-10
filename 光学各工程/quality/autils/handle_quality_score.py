
import os
import shutil
import argparse
from pathlib import Path
import xlsxwriter
import datetime


def create_sheet(book):
    #创建 一个Workbook 实列


    #添加一个工作簿
    sheet=book.add_worksheet(u'gsl7015a_default')
    sheet.set_column(0,20,8)
    format_title = {    'font_size': 14,  # 字体大小
    'bold': True,  # 是否粗体    #
    #'bg_color': '#101010',  # 表格背景颜色
    #'fg_color': '#00FF00',
    #'font_color': '#0000FF',  # 字体颜色
    'align': 'center',  # 水平居中对齐
    'valign': 'vcenter',  # 垂直居中对齐    #
    'num_format': '#,##0',#'yyyy-mm-dd H:M:S',# 设置日期格式    # 后面参数是线条宽度
    'border': 1, # 边框宽度
    'top': 1,  # 上边框
    'left': 1,  # 左边框
    'right': 1,  # 右边框
    'bottom': 1  # 底边框
    }
    format_text = {    'font_size': 14,  # 字体大小
    'bold': False,  # 是否粗体    #
    #'bg_color': '#101010',  # 表格背景颜色
    #'fg_color': '#00FF00',
    #'font_color': '#0000FF',  # 字体颜色
    'align': 'center',  # 水平居中对齐
    'valign': 'vcenter',  # 垂直居中对齐    #
    'num_format': '#,##0',#'yyyy-mm-dd H:M:S',# 设置日期格式    # 后面参数是线条宽度
    'border': 1, # 边框宽度
    'top': 1,  # 上边框
    'left': 1,  # 左边框
    'right': 1,  # 右边框
    'bottom': 1  # 底边框
    }
    style_title = book.add_format(format_title)   # 设置样式format是一个字典  # # write_row 写入
    style = book.add_format(format_text)   # 设置样式format是一个字典  # # write_row 写入
    sheet.write(0,0,"idx",style_title)#写入序号值
    sheet.write(0,1,"spoof",style_title)#写入序号值
    sheet.write(0,2,"cover",style_title)#写入序号值
    sheet.write(0,3,"mean",style_title)#写入序号值
    sheet.write(0,4,"DC",style_title)#写入序号值
    sheet.write(0,5,"0xDC",style_title)#写入序号值
    sheet.write(0,6,"name",style_title)#写入序号值

    return sheet


def parse_args():
    parser = argparse.ArgumentParser(description="ASP Train Network")
    parser.add_argument('-p','--path', default=r"/home/panq/dataset/spoof/6193/data/data-8-0816-真人误卡采图11人-quality")
    args = parser.parse_args()
    return args

# if __name__ == '__main__':
#     args = parse_args()
#     path = args.path
#     count = 0
#     path = Path(path)

#     for p in path.rglob("*.bmp"):
#         if p.name.find("quality")==-1:
#             p.unlink()
#             continue
#         str1 = p.stem.split("_quality_")
#         dst = p.parent/(str1[-1]+"_"+str1[0]+".bmp")
#         if count%1000==0:
#             print(count*1000)
#         count+=1
#         # print(dst)
#         os.rename(p,dst)
if __name__ == '__main__':

    book=xlsxwriter.Workbook(u'score_0fp.xlsx')
    sheet = create_sheet(book)

    book1=xlsxwriter.Workbook(u'score_1sp.xlsx')
    sheet1 = create_sheet(book1)
    args = parse_args()
    path = args.path
    count = 0
    path = Path(path)

    c0_0=0
    c0_1=0
    c0_2=0
    c0_3=0
    c0_4=0
    c0_5=0
    c0_6=0
    c0_7=0
    c0_8=0
    c0_9=0
    c1_0=0
    c1_1=0
    c1_2=0
    c1_3=0
    c1_4=0
    c1_5=0
    c1_6=0
    c1_7=0
    c1_8=0
    c1_9=0
    cnt_s=0
    cnt_1=0
    cnt=0

    row=1

    time=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    path_all = "/home/panq/dataset/spoof/6193/Quality/"+time
    path_1=os.path.join(path_all,'1')
    path_0=os.path.join(path_all,'0')
    if not os.path.isdir(path_1):
            os.makedirs(path_1)
    if not os.path.isdir(path_0):
            os.makedirs(path_0)

    # copyfile(path[i], out_path + "/%d_%s.bmp"%(finger_score*65536,pathout))


    for p in path.rglob("*.bmp"):

        if 'quality' not in p.stem:
            print(p.parts[8:11])
            break
        score = int(p.stem.split("quality_")[-1].split('.')[0])
        # path_bmp=p.__str__
        # path_new=path_bmp.replace('/', '_')
        cnt+=1

            # sheet.write_number(row,2,int(score))#写入序号值

            # sheet1.write_number(row-113318,2,int(score))#写入序号值
        row+=1
    print(c0_0)
    print(c0_1)
    print(c0_2)
    print(c0_3)
    print(c0_4)
    print(c0_5)
    print(c0_6)
    print(c0_7)
    print(c0_8)
    print(c0_9)

    print(c1_0)
    print(c1_1)
    print(c1_2)
    print(c1_3)
    print(c1_4)
    print(c1_5)
    print(c1_6)
    print(c1_7)
    print(c1_8)
    print(c1_9)

    print('all '+str(cnt))
    print('1spoof '+str(cnt_1))
    print('score '+str(cnt_s))
    # book.close()
    # book1.close()
                # f.write('-'.join(p.parts)+'\n')
    # f.close()
        # if int(score) <= 27:
        #     print(p)
        #     os.remove(p)