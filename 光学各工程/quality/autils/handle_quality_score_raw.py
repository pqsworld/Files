
import os
import shutil
import argparse
from pathlib import Path
import xlsxwriter


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
    parser.add_argument('-p','--path', default=r"/home/panq/dataset/spoof/6193/data/test-10-qua03050100")
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

    book=xlsxwriter.Workbook(u'score_0fp.xls')
    sheet = create_sheet(book)

    book1=xlsxwriter.Workbook(u'score_1sp.xls')
    sheet1 = create_sheet(book1)
    args = parse_args()
    path = args.path
    count = 0
    path = Path(path)
    row=1

    for p in path.rglob("*.bmp"):

        if 'quality' not in p.stem:
            print(p.parts[8:11])
            break
        score = int(p.stem.split("_")[-1])
        # if score<=31:
        # if score <= 30 or score >= 50:
        if score > 30 and score < 50:
            p.unlink()
            # print(score)

        # if p.parts[7]=='0fp':
        #     sheet.write_number(row,2,int(score))#写入序号值
        # else:
        #     sheet1.write_number(row,2,int(score))#写入序号值
        # row+=1

    book.close()
    book1.close()
                # f.write('-'.join(p.parts)+'\n')
    # f.close()
        # if int(score) <= 27:
        #     print(p)
        #     os.remove(p)