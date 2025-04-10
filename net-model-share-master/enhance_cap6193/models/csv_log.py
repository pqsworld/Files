import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import seaborn as sns

def reslog(dirPath):
    dPathList = glob.glob(dirPath+"/*")  
    # 将所有log文件中数据合并
    dfList = []    
    for dfilePath in dPathList:
        if os.path.isdir(dfilePath):
            dfList.append(dfilePath)
    for path in dfList:
        # scoresum = NULL
        scoresum=np.zeros(1)
        gridscore=np.zeros(1)
        fileNames = os.listdir(path)
        for fileName in fileNames:
            if fileName.endswith('csv'):
                if fileName.find('Frr')!=-1:
                    #print(fileName)
                    csvpath = path+"/"+fileName
            
                    data = pd.read_csv(csvpath,sep=',',usecols=['nScoreSum','nGridScore'])
                    # nScoreSum nGridScore nOverlap nInNumO        
                    data = np.array(data)
                    # print(data.shape)
                    tmpss = data[:,0]  #0:scoresum   1:gridscore
                    scoresum=np.hstack((scoresum, tmpss))
                    tmpgs = data[:,1]  #0:scoresum   1:gridscore
                    gridscore=np.hstack((gridscore, tmpgs))
                    # print(scoresum)
        scoresum = np.delete(scoresum, 0)
        gridscore = np.delete(gridscore, 0)
        # print(scoresum)
    return scoresum,gridscore   
    # my = np.hstack((scoresum,gridscore))
if __name__ == '__main__':     
    dirPath1 = '/home/zhangsn/enhance/评价相关/cap/log/alllog/ct'
    dirPath2 = '/home/zhangsn/enhance/评价相关/cap/log/alllog/211'
    dirPath3 = '/home/zhangsn/enhance/评价相关/cap/log/alllog/211norm'
    
    scoresum1,gridscore1 = reslog(dirPath1)
    scoresum2,gridscore2 = reslog(dirPath2)
    scoresum3,gridscore3 = reslog(dirPath3)
    print(np.min(scoresum1))
    print(np.min(scoresum2))
    print(np.min(scoresum3))
    plt.figure()
    plt.title('ScoresumFR')
    sns.kdeplot(scoresum1,shade=True,label='ct',color='r')
    sns.kdeplot(scoresum2,shade=True,label='net',color='g')
    sns.kdeplot(scoresum3,shade=True,label='netnorm',color='y')
    plt.legend()
    plt.xlim((250, 500))
    plt.savefig('scoresumfrr0.png')

    plt.figure()
    plt.title('GridscoreFR')
    sns.kdeplot(gridscore1,shade=True,label='ct',color='r')
    sns.kdeplot(gridscore2,shade=True,label='net',color='g')
    sns.kdeplot(gridscore3,shade=True,label='netnorm',color='y')
    plt.legend()
    plt.savefig('gridscorefrr0.png')
    
    # plt.figure()
    # plt.title('Scoresum')
    # plt.hist(scoresum1,bins=100, rwidth=0.8, range=(300,500), label='ct', align='left')
    # plt.hist(scoresum2,bins=100, rwidth=0.8, range=(300,500), label='net', align='left',alpha=0.5)
    # plt.legend()
    # plt.savefig('scoresum.png')

    # plt.figure()
    # plt.title('Gridscore')
    # plt.hist(gridscore1,bins=50, rwidth=0.8, range=(100,250), label='ct',align='left')
    # plt.hist(gridscore2,bins=50, rwidth=0.8, range=(100,250), label='net', align='left',alpha=0.5)  #
    # plt.legend()
    # plt.savefig('gridscore.png')