# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np 
import random
import time

class optstruct:
    def __init__(self,datamatin,classlabels,c,toler):
        self.x = datamatin
        self.labelmat = classlabels
        self.c = c
        self.tol = toler
        self.b = 0 
        self.m = np.shape(datamatin)[0]
        self.alphas = np.mat(np.zeros((self.m , 1)))
        self.ecache = np.mat(np.zeros((self.m,2)))

def loaddataset(filename):
    datamat = []
    labelmat =[]
    fr = open(filename)
    for line in fr.readlines():
        linearr = line.strip().split('\t')
        datamat.append([float(linearr[0]),float(linearr[1])])
        labelmat.append(float(linearr[2]))
    return datamat,labelmat

def calcek(os,k):
    fxk = float(np.multiply(os.alphas,os.labelmat).T * (os.x * os.x[k,:].T))+os.b
    ek = fxk - float(os.labelmat[k])
    return ek

def selectjrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

def selectj(i,os,ei):
    maxk = -1 
    maxdeltae=0
    ej=0
    os.ecache[i] = [1,ei]
    validecachelist = np.nonzero(os.ecache[:,0].A)[0]
    if (len(validecachelist))>1:
        for k in validecachelist:
            if k == i:
                continue
            ek = calcek(os,k)
            deltae = abs(ei - ek)
            if (deltae > maxdeltae):
                maxk = k 
                maxdeltae = deltae
                ej = ek
        return maxk , ej
    else:
        j=selectjrand(i,os.m)
        ej = calcek(os,j)
    return j,ej

def updataek(os,k):
    ek = calcek(os,k)
    os.ecache[k] = [1,ek]
    
def clipalpha(aj,h,l):
    if aj>h:
        aj=h
    if l > aj:
        aj=l
    return aj

def innerl(i,os):
    ei = calcek(os,i)
    if ((os.labelmat[i]*ei < -os.tol) and (os.alphas[i] < os.c)) or ((os.labelmat[i]*ei > -os.tol) and (os.alphas[i] > 0)):
        j,ej = selectj(i,os,ei)
        alphaiold = os.alphas[i].copy()
        alphajold = os.alphas[j].copy()
        if (os.labelmat[i]) != os.labelmat[j]:
            l = max(0,os.alphas[j] + os.alphas[i])
            h = min(os.c ,os.c+os.alphas[j] - os.alphas[i])
        else:
            l = max(0,os.alphas[j] + os.alphas[i] - os.c)
            h = min(os.c,os.alphas[j]+os.alphas[i])
        if l == h:
            print('l==h')
            return 0
        eta = 2.0 * os.x[i,:] * os.x[j,:].T - os.x[i,:] * os.x[i,:].T - os.x[j,:] * os.x[j,:].T
        if eta >= 0:
            print("eta>=0")
            return 0
        #步骤4：更新alpha_j
        os.alphas[j] -= os.labelmat[j] * (ei - ej)/eta
        #步骤5：修剪alpha_j
        os.alphas[j] = clipalpha(os.alphas[j],h,l)
        #更新Ej至误差缓存
        updataek(os, j)
        if (abs(os.alphas[j] - alphajold) < 0.00001):
            print("alpha_j变化太小")
            return 0
        os.alphas[i] += os.labelmat[j] * os.labelmat[i] * (alphaiold - os.alphas[j])
        updataek(os,i)
        b1 = os.b - ei - os.labelmat[i] * (os.alphas[i] - alphaiold) * os.x[i,:] * os.x[i,:].T- os.labelmat[j]*(os.alphas[j]-alphajold)*os.x[i,:]*os.x[j,:].T
        b2 = os.b - ej - os.labelmat[i] * (os.alphas[i] - alphaiold) * os.x[i,:] * os.x[j,:].T- os.labelmat[j]*(os.alphas[j]-alphajold)*os.x[j,:]*os.x[j,:].T
        if (0 < os.alphas[i]) and (os.c > os.alphas[i]):
            os.b =b1
        elif (0 < os.alphas[j]) and (os.c > os.alphas[j]):
            os.b = b2
        else:
            os.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

def smop(datamatin,classlabels,c,toler,maxiter):
    os = optstruct(np.mat(datamatin),np.mat(classlabels).transpose(),c,toler)
    iter = 0
    entireset = True
    alphapairschanged = 0
    while (iter < maxiter) and ((alphapairschanged > 0) or (entireset)):
        alphapairschanged= 0
        if entireset:
            for i in range(os.m):
                alphapairschanged += innerl(i,os)
                print('all sample:the %d iteration,sampel:%d,aplpha optimize num:%d'%(iter,i,alphapairschanged))
            iter +=1
        else:
            nonboundis = np.nonzero((os.alphas.A >0) * (os.alphas.A < c))[0]
            for i in nonboundis:
                alphapairschanged +=innerl(i,os)
                print('nonboundis:the %d iteration,sampel:%d,aplpha optimize num:%d'%(iter,i,alphapairschanged))        
            iter +=1
        if entireset:
            entireset = False
        elif (alphapairschanged == 0):
            entireset = True
        print('iteration num : %d'%iter)
    return os.b,os.alphas

def showclassifer(datamat,labelmat,w,b):
    data_plus=[]
    data_minus = []
    for i in range(len(datamat)):
        if labelmat[i]>0:
            data_plus.append(datamat[i])
        else:
            data_minus.append(datamat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0],np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0],np.transpose(data_minus_np)[1])
    x1 = max (datamat)[0]
    x2 = min(datamat)[0]
    a1,a2 = w
    b = float(b)
    a1= float(a1[0])
    a2 = float(a2[0])
    y1,y2 = (-b-a1*x1)/a2,(-b-a1*x2)/a2
    plt.plot([x1,x2],[y1,y2])
    for i ,alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x,y= datamat[i]
            plt.scatter([x],[y],s=150,c='none',alpha=0.7,linewidths=1.5,edgecolors='red')
    plt.ylim(-20,20)
    plt.show()

def get_w(alphas,dataarr,classlabels):
     x = np.mat(dataarr)
     labelmat = np.mat(classlabels).transpose()
     m,n = np.shape(x)
     w = np.zeros((n,1))
     for i in range(m):
         w +=np.multiply(alphas[i] * labelmat[i],x[i,:].T)
     return w
   
if __name__=='__main__':
    datamat,labelmat = loaddataset('testSet.txt')
    start = time.time()
    b,alphas = smop(datamat,labelmat,0.6,0.001,40)
    w = get_w(alphas,datamat,labelmat)
    showclassifer(datamat,labelmat,w,b)
    end = time.time()
    print('times is:%d'%(end-start) )