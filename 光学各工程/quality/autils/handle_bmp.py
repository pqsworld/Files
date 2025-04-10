import os
# import xlwt
import numpy as np
import time
import sys
import re
import xlsxwriter
path="/home/panq/dataset/spoof/6250/0718ID/0fp/1spoof.txt"
# path="/home/panq/dataset/spoof/6250/0705sh/0705sh/1hdg/1hdg.log"

if __name__ == '__main__':
    ret =os.path.exists(path)
    if ret == False:
        print("not exist")



    #创建 一个Workbook 实列
    book=xlsxwriter.Workbook(u'77sh_1.xlsx')

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


    list_v = [1,2,4,6]
    list_count_v=[0,0,0,0,0]

    fp = open(path,"r")
    for line in fp.readlines():
        line = line.strip()
        if "enroll" in line:
            continue
        value_s = re.findall(r"(?<=-Spoof_)\d+",line)[0]
        # print(value_s)
        flag_listv=0
        for i in range(len(list_v)):
            if int(value_s) == list_v[i]:
                # print(value_s+'='+list_v[i])
                list_count_v[i]+=1
                flag_listv=1
        if not flag_listv:
            list_count_v[4]+=1



        row += 1

    c_1 = list_count_v[0]+list_count_v[1]+list_count_v[2]+list_count_v[3]
    c_0 = list_count_v[4]
    print("list_1: "+str(list_count_v[0]))
    print("list_2: "+str(list_count_v[1]))
    print("list_4: "+str(list_count_v[2]))
    print("list_6: "+str(list_count_v[3]))
    print("list_0fp: "+str(list_count_v[4]))


    print("acc_1spoof: "+str(c_1*100.0/(c_0+c_1))+"%")
    # print("acc_0fp: "+str(list_count_v[4]))/
    book.close()
    # print(mem_dict)
    #exit(sys)