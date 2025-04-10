

"""需要将里面的.ini,.txt等杂文件删除掉"""
import os
from natsort import natsorted
pathv="/home/lif/im/AMBER/amber-BOE/BOE"
num=0
listfinger=["L0","L1","L2","R0","R1","R2"]
for namev in natsorted(os.listdir(pathv)):
    path=os.path.join(pathv,namev)
    num=0
    for namei in natsorted(os.listdir(path)):
        os.rename(os.path.join(path,namei),os.path.join(path,"{:0>4d}".format(num)))
        #num+=1
        patht=os.path.join(path,"{:0>4d}".format(num))
        num+=1
        numt=0
        for namej in natsorted(os.listdir(patht)):
            os.rename(os.path.join(patht,namej),os.path.join(patht,listfinger[numt]))
            numt+=1


for root,_,file in os.walk(pathv):
    file.sort()
    num=0
    #for fi in file:
    for fi in natsorted(file):
        if ".bmp" in fi:
            pathold=os.path.join(root,fi)
         
            pathnew=os.path.join(root,"{:0>4d}.bmp".format(num))
            os.rename(pathold,pathnew)
            num+=1