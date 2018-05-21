import matplotlib.pyplot as plt
import numpy as np 
import random

def loaddataset(filename):
    datamat = []
    labelmat =[]
    fr = open(filename)
    for line in fr.readlines():
        linearr = line.strip().split('\t')
        datamat.append([float(linearr[0]),float(linearr[1])])
        labelmat.append(float(linearr[2]))
    return datamat,labelmat

    
def selectjrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

def clipalpha(aj,h,l):
    if aj>h:
        aj=h
    if l > aj:
        aj=l
    return aj

def smosimple(datamatin,classlabels,c,toler,maxiter):
    datamatrix = np.mat(datamatin)
    labelmat = np.mat(classlabels).transpose()
    b=0
    m,n=np.shape(datamatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter_num = 0
    while (iter_num < maxiter):
        alphapairschanged = 0
        for i in range(m):
            fxi = float(np.multiply(alphas,labelmat).T * (datamatrix * datamatrix[i,:].T))+b
            ei=fxi - float(labelmat[i])
            if ((labelmat[i]*ei < -toler) and (alphas[i] <c)) or ((labelmat[i] * ei >toler) and (alphas[i] > 0)):
                j = selectjrand(i,m)
                fxj= float(np.multiply(alphas,labelmat).T * (datamatrix * datamatrix[j,:].T))+b
                ej=fxj - float(labelmat[j])
            
                alphaiold = alphas[i].copy()
                alphajold = alphas[j].copy()
                if (labelmat[i] != labelmat[j]):
                    l= max(0,alphas[j]-alphas[i])
                    h= min(c,c+alphas[j] - alphas[i])
                else :
                    l = max(0,alphas[j] + alphas[i] -c)
                    h = min(c, alphas[j] + alphas[i])
                if l == h :
                    print('l==h')
                    continue
                eta = 2.0 * datamatrix[i,:]*datamatrix[j,:].T - datamatrix[i,:] * datamatrix[i,:].T - datamatrix[j,:] * datamatrix[j,:].T
                if eta >=0:
                    print('eta>=0')
                    continue
                alphas[j] -=labelmat[j] * (ei - ej)/eta
                alphas[j] = clipalpha(alphas[j],h,l)
                if (abs(alphas[j] - alphajold) < 0.00001):
                    print('alpha_ja too small')
                    continue
                alphas[i] +=labelmat[j] * labelmat[i] * (alphajold - alphas[j])
                b1 = b - ei -labelmat[i]*(alphas[i]-alphaiold)*datamatrix[i,:]*datamatrix[i,:].T - labelmat[j]*(alphas[j]-alphajold)*datamatrix[i,:]*datamatrix[j,:].T
                b2 = b - ei -labelmat[i]*(alphas[i]-alphaiold)*datamatrix[i,:]*datamatrix[j,:].T - labelmat[j]*(alphas[j]-alphajold)*datamatrix[j,:]*datamatrix[j,:].T
                if (0 < alphas[i]) and (c > alphas[i]):
                    b=b1
                elif (0 < alphas[j]) and (c > alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                
                alphapairschanged +=1
    
                print('the %d iteration,sampel:%d,aplpha optimize num:%d'%(iter_num,i,alphapairschanged))
        if (alphapairschanged == 0):
            iter_num +=1
        else:
            iter_num = 0
        print('iteration num : %d:'% iter_num)
    return b,alphas

def showclassifer(datamat,labelmat,w,b):
    data_plus=[]
    data_minus = []
    for i in range(len(datamat)):
        if labelmat[i]>0:
            data_plus.append(datamat[i])
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
    plt.show()
            
def get_w(datamat,labelmat,alphas):
    alphas,datamat,labelmat=np.array(alphas),np.array(datamat),np.array(labelmat)
    w = np.dot((np.tile(labelmat.reshape(1,-1).T,(1,2))*datamat).T,alphas)
    return w.tolist()
if __name__=='__main__':
    datamat,labelmat = loaddataset('testSet.txt')
    b,alphas = smosimple(datamat,labelmat,0.6,0.001,40)
    w = get_w(datamat,labelmat,alphas)
    showclassifer(datamat,labelmat,w,b)
