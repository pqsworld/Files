from tkinter import *

import tkinter
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
import os
import shutil
from natsort import ns, natsorted

img_path = []
img_path_c = []
img_total = 0
img_now = 0
newscore = 0
# 注册窗口
win = Tk()
win.title('图像打分')
win.geometry('880x600')
win.resizable(0, 0)


def on_scroll_value(event):
    if event.delta > 0:
        newscore = newscore + 1
        # uname_new.config(text=score, font=font_style)

    else:
        newscore = newscore - 1
        # uname_new.config(text=score, font=font_style)

    return


# 图像路径
Label(text='img_path:').place(x=530, y=150)
path_img = Entry(win)
path_img.place(x=600, y=150)
font_style = ("Arial", 16, "italic")

# 原分数文本框
Label(text='ori_score_g:').place(x=530, y=170)
uname = Label(win)
uname.place(x=600, y=170)

Label(text='ori_score_sl:').place(x=530, y=200)
uname1 = Label(win)
uname1.place(x=600, y=200)

Label(text='ori_score_sh:').place(x=530, y=230)
uname2 = Label(win)
uname2.place(x=600, y=230)

#修改分数文本框
# Label(text='new_score:').place(x=760, y=250)
# uname_new = Label(win)
# uname_new.place(x=780, y=250)
# uname_new.config(text=newscore, font=font_style)
stContent = scrolledtext(win, font=font_style)
stContent.insert('0.0', newscore)
stContent.place(x=760, y=250, width=60, height=20)
stContent.bind('<Control-MouseWheel>', on_scroll_value)
amend_score = Entry(win, width=7)
# amend_score.insert('0', score)
amend_score.place(x=690, y=200)

img_jump = Entry(win, width=7)
img_jump.place(x=690, y=300)

lableShowImage = Label(win)

lableShowImage.pack()

#lableShowImage.config(image=img)
lableShowImage.place(x=100, y=100)


def get_img_score(path):
    #print(path)
    #s = path.split("\\")[-1].split("_")[0].split("q")[-1]
    s = path.split("\\")[-1].split("_")[-3].split("g")[-1]
    s1 = path.split("\\")[-1].split("_")[-2].split("sl")[-1]
    s2 = path.split("\\")[-1].split("_")[-1].split("sh")[-1].split(".")[0]
    print(s, s1, s2)
    #print(s)
    try:
        s_int = int(s)
        s1_int = int(s1)
        s2_int = int(s2)
    except ValueError:
        s_int = None
        s1_int = None
        s2_int = None
    return s_int, s1_int, s2_int


def get_img_path():
    global img_total
    global img_path
    global img_path_c
    img_path = []
    path_im = path_img.get()
    #get_img_path(path_im):
    for root, dirs, names in os.walk(path_im):
        names = natsorted(names, alg=ns.PATH)
        for name in names:
            if ".bmp" in name:
                path1 = os.path.join(root, name)
                img_path.append(path1)
    img_path_c = img_path.copy()
    img_total = len(img_path)
    if img_total == 0:
        return -1
    upload_pic()
    img_total_text.config(text=img_total)


    #img_total_text.text=img_total
def open_image(path1):
    img = Image.open(path1)
    w, h = img.size
    b = 400 / h
    if w > h:
        b = 400 / w
    w_new = (int)(w * b + 0.5)
    h_new = (int)(h * b + 0.5)
    img = img.resize((w_new, h_new))
    img = ImageTk.PhotoImage(img)
    return img


#显示图像
def show_img(img_n):
    path1 = img_path[img_n]
    # global score,amend_score
    score, score1, score2 = get_img_score(path1)
    amend_score.insert('0', score)
    newscore = score
    img_now_text.config(text=img_n + 1)
    path_now_text.config(text=path1)
    uname.config(text=score, fg="red", font=font_style)
    uname1.config(text=score1, font=font_style)
    uname2.config(text=score2, font=font_style)
    img = open_image(path1)
    lableShowImage.config(image=img)
    #lableShowImage.place(x=100, y=100)
    lableShowImage.image = img
    return score


# 获取图像
def upload_pic():
    global img_now
    global amend_score
    #path = r"C:\Users\user\Desktop\6193\6193\0001\0010-0100-auth_mraw-700101-080625059.bmp"
    img_now = 0
    show_img(img_now)


# 获取图像
def upload_pic_up():
    #global img
    global img_now
    global amend_score
    amend_score.delete(0, "end")
    #path = r"C:\Users\user\Desktop\6193\6193\0001\0010-0100-auth_mraw-700101-080625059.bmp"
    img_now = img_now - 1
    if img_total == 0:
        return -1
    if img_now < 0:
        img_now = img_now + 1
        return 0
    show_img(img_now)


def upload_pic_next():
    #global img
    global img_now
    global amend_score
    amend_score.delete(0, "end")
    #path = r"C:\Users\user\Desktop\6193\6193\0001\0010-0100-auth_mraw-700101-080625059.bmp"
    img_now = img_now + 1
    if img_total == 0:
        return -1
    if img_now > img_total - 1:
        img_now = img_now - 1
        return 0
    show_img(img_now)
    #path1 = img_path[img_now]
    #score = get_img_score(path1)
    #img_now_text.config(text=img_now+1)
    #path_now_text.config(text=path1)
    #uname.config(text=score)
    #img = open_image(path1)

    #img = Image.open(path1)
    #w,h = img.size
    #img = img.resize((w*3,h*3 ))
    #img = ImageTk.PhotoImage(img)
    #canvas.create_image(300,50,image = img)
    #image_Label = Label(w, image=photo)
    #image_Label.grid(row=0, column=1)

    #lableShowImage = Label(win)

    #lableShowImage.pack()

    #canvas.pack


    #canvas.create_image( 0, 0, anchor=NW, image=img)
    #lableShowImage.config(image=img)
    #lableShowImage.place(x=100, y=100)
    #lableShowImage.image = img
def img_jump_f():
    global img_now, score
    num = img_jump.get()
    try:
        num = int(num)
        num = num - 1
        if (num >= 0 and num < img_total):
            img_now = num
            score = show_img(img_now)
        else:
            return -1

    except ValueError:
        return -1


def amend_quality_score():
    s = amend_score.get()
    try:
        s_int = int(s)
        path1 = img_path_c[img_now]
        path2 = img_path[img_now]
        img_name = path1.split("\\")
        #img_ori_score = img_name[-1].split("_")[0].split("q")[-1]
        #img_ori_score = int(img_ori_score)
        #if s_int == img_ori_score:
        #return 0
        img_name_last = str(s_int) + "_" + img_name[-1]
        img_name[-1] = img_name_last
        img_name_new = img_name[0]
        for i in range(len(img_name) - 1):
            img_name_new = img_name_new + "\\" + img_name[i + 1]
        #print(img_name_new)
        shutil.move(path2, img_name_new)
        img_path[img_now] = img_name_new
        uname.config(text=s_int)
        path_now_text.config(text=img_path[img_now])
    except ValueError:
        s_int = None


#upload_pic()
#path = r"C:\Users\user\Desktop\6193\6193\0001\0010-0100-auth_mraw-700101-080625059.bmp"
#default_pic = PhotoImage(file=path)

# 注册
#def register():
#user_name = uname.get()
#path_im = path_img.get()
#get_img_path(path_im):

#password = pwd.get()
#head_path = 'head.png'
#with open('data.txt', 'a', encoding='utf-8') as f:
#f.write('{},{},{}\n'.format(user_name,password,head_path))
Button(text='打开图像', command=get_img_path).place(x=760,
                                                y=150,
                                                width=60,
                                                height=20)
bt1 = Button(text='上一张', command=upload_pic_up)
bt1.place(x=530, y=300, width=60, height=20)
bt2 = Button(text='下一张', command=upload_pic_next)
bt2.place(x=610, y=300, width=60, height=20)

Button(text='跳转图像', command=img_jump_f).place(x=760,
                                              y=300,
                                              width=60,
                                              height=20)

#Button(text='修改分数', command=amend_quality_score).place(x=760, y=200, width=60, height = 20)
Button(text='修改分数', command=amend_quality_score).place(x=760,
                                                       y=200,
                                                       width=60,
                                                       height=20)

Label(text='img_total:').place(x=530, y=250)
img_total_text = Label(win)
img_total_text.place(x=600, y=250)
Label(text='img_now:').place(x=660, y=250)
img_now_text = Label(win)
img_now_text.place(x=720, y=250)
Label(text='path_now:').place(x=530, y=350)
path_now_text = Label(win)
path_now_text.place(x=600, y=350)
#path_now_text.config(justify='right')
#path_now_text.config(height=4)
path_now_text.config(width=30)
path_now_text.config(wraplength=200)


def show_key(event):
    # 查看触发事件的按钮
    #print(event)
    num_gl = (int)(event.delta)
    if (num_gl > 0 or event.keysym == "Up" or event.keysym == "Left"
            or event.keysym == "A" or event.keysym == "W"):
        upload_pic_up()
    elif (num_gl < 0 or event.keysym == "Down" or event.keysym == "Right"
          or event.keysym == "S" or event.keysym == "D"):
        upload_pic_next()
    if (event.keysym == "Return"):
        amend_quality_score()
    #s=event.keysym
    # 将其显示在按钮控件上
    #lb.config(text=s)


# win.bind_all('<MouseWheel>', on_scroll_value)
win.bind_all('<KeyPress>', show_key)
#lb.bind('<Key>',show_key) #使用bind方法绑定（实例绑定）事件
#lb.focus_set() # 设置获取焦点， Label 控件获取焦点后才能接收键盘事件

#Button(command=upload_pic)
#head_button = Button(command=upload_pic)
#head_button.place(x=150, y=10, width=80, height=80)

win.mainloop()
