# -*- coding: utf-8 -*-
import numpy as np
from os import listdir
from sklearn.svm import SVC

def im2vector(filname):#对数据进行扁平化处理
    returnvect=np.zeros((32,32))
    file=open(filname)
    linestr=file.readlines()
    index=0
    for i in linestr:
        words=i.strip()
        words=list(words)
        returnvect[index,:]=words[0:32]
        index+=1
    returnvect=returnvect.flatten()    
    return returnvect

def handwritingclasstest():
    hwlabels=[]
    trainingflilelist=listdir('trainingdigits')
    m=len(trainingflilelist)
    trainningmat=np.zeros((m,1024))
    for i in range(m):
        filenamestr=trainingflilelist[i]
        classnumber=int(filenamestr.split('_')[0])
        hwlabels.append(classnumber)
        trainningmat[i,:]=im2vector('trainingDigits/'+filenamestr)
    clf=SVC(C=200)
    clf.fit(trainningmat,hwlabels)
    testfilelist=listdir('testDigits')
    errorcount=0
    mtest=len(testfilelist)
    for i in range(mtest):
        filenamestr=testfilelist[i]
        classnumber=int(filenamestr.split('_')[0])
        vectorundertest=im2vector('testDigits/'+filenamestr)
        classifiterresult=clf.predict(vectorundertest.reshape(1,-1))
        print("分类返回结果为%d\t真实结果为%d" % (classifiterresult, classnumber))
        if classifiterresult!=classnumber:
            errorcount+=1
    print('共预测错误%d个数\n错误率为%f%%'%(errorcount,errorcount/mtest*100))

if __name__=="__main__":
    
#    filename='D:/程序/python/dl/trainingDigits/0_0.txt'
#    im2vector(filename)
    handwritingclasstest()