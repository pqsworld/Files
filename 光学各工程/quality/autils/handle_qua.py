import os
# import xlwt
import numpy as np
import time
import sys
import re
import xlsxwriter

path="/home/panq/tool/aosp/vendor"
# path="/home/panq/dataset/spoof/6250/0705sh/0705sh/1hdg/1hdg.log"

if __name__ == '__main__':
    ret =os.path.exists(path)
    if ret == False:
        print("not exist")



    #创建 一个Workbook 实列
    book=xlsxwriter.Workbook(u'quality.xlsx')

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


    row = 1




    fp = open(path,"w")
    for line in fp.readlines():
        line = line.strip()

        value_c = re.findall(r"(?<=-C)\d+",line)[0]
        value_m = re.findall(r"(?<=-M)\d+",line)[0]
        # value_dc = re.findall(r"(?<=-DC)\d+",line)
        value_0xdc =  re.findall(r"0x[0-9a-fA-F]+",line)[0]
        value_dc = int(value_0xdc, 16)
        value_spoof = re.findall(r"(\d+)-DC",line)[0]

        sheet.write(row,0,row,style)#写入序号值
        sheet.write(row,1,int(value_spoof, 10),style)#写入序号值
        sheet.write(row,2,int(value_c, 10),style)#写入序号值
        sheet.write(row,3,int(value_m, 10),style)#写入序号值
        sheet.write(row,4,value_dc,style)#写入序号值
        sheet.write(row,5,value_0xdc,style)#写入序号值
        sheet.write(row,6,line,style)#写入序号值
        row += 1

    book.close()
    # print(mem_dict)
    #exit(sys)