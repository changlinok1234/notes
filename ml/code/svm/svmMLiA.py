
#coding=utf-8
#-----------------svm的迷你版程序，只能进行二分类-------------------
#-----------------在当前程序所在文件夹下运行python后，要import svmMLiA，之后写入各种命令才会生效----------
''' @author: Peter  '''



from numpy import *
from time import sleep

def loadDataSet(fileName):
    dataMat=[]; labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')   #.strip()删除空白符
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat


#--------------------------------------------------------SMO算法部分-----------------------------------------------------



def selectJrand(i,m):  #i是alpha下标，m是所有alpha的数目
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L): #用于调整大于H或小于L的alpha值
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

#简化版smo算法,
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):  #数据集，标签集，类别标签，常数C，容错率，退出前最大的循环次数
    
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()  #.transpose()转置
    b=0; m,n=shape(dataMatrix)
    alphas = mat(zeros((m,1)))  #构建alpha列矩阵，由于有约束条件\求和{alpha*label}=0，改变一个alpha可能会导致该约束条件失效，因此总是同时改变两个alpha
    iter=0
    while (iter < maxIter):
        alphaPairsChanged=0
        for i in range (m):
            fXi=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b  #模型公式，就是笔记里的（6-12）
            Ei=fXi-float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)): #如果alpha可以被优化：随机选择另外一个数据向量
                    j=selectJrand(i,m)  #随机选择第二个alpha
                    fXj=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                    Ej=fXj-float(labelMat[j])
                    alphaIold=alphas[i].copy()
                    alphaJold=alphas[j].copy()
                    if (labelMat[i] != labelMat[j]):      #保证
                        L=max(0,alphas[j]-alphas[i])       #alpha
                        H=min(C,C+alphas[j]-alphas[i])     #在
                    else:                                  #0和C
                        L=max(0,alphas[j]+alphas[i]-C)     #之
                        H=min(C,alphas[j]+alphas[i])       #间
                    if L==H:print "L=H"; continue
                    eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                    if eta >=0: print "eta>=0"; continue
                    alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                    alphas[j]=clipAlpha(alphas[j],H,L)
                    if (abs(alphas[j]-alphaJold)<0.00001): print "j not moving enougn"; continue
                    alphas[i] +=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])   #对i进行修改
                    b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T   #设
                    b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T     #置
                    if (0 < alphas[i]) and (C > alphas[i]): b=b1        #常
                    elif(0 < alphas[j]) and (C > alphas[j]): b=b2     #数
                    else: b=(b1+b2)/2.0         #项b
                    alphaPairsChanged += 1    #用于记录alpha是否已经优化
                    print "iter: %d i: %d, pairs changed %d" %(iter,i,alphaPairsChanged)
        if (alphaPairsChanged ==0 ): iter += 1
        else: iter=0
        print "iteration number: %d" % iter
    return b,alphas

#完整版Platt SMO的支持函数
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):   #kTup是包含核函数信息的元祖
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zeros((self.m,2)))    #误差缓存？ 第一列给出的是eCache是否有效的标志位，第二列给出的是实际E值
        self.K=mat(zeros((self.m,self.m)))
        for i in range (self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:], kTup)

def calcEk(oS, k):  #计算误差E
    fXk=float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)
    Ek=fXk-float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):    #用于选择第二个alpha或者说内循环的alpha，目标是选择合适的第二个alpha值以保证在每次优化中采用最大步长
    maxK= -1; maxDeltaE=0; Ej=0
    oS.eCache[i]=[1,Ei]
    validEcacheList=nonzero(oS.eCache[:,0].A)[0]  #构建了一个非零表，nonzero（）在这里返回的是非零E值所对应的alpha值(似乎是列表值)
    if (len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i: continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if (deltaE > maxDeltaE):   #选择具有最
                maxK=k; maxDeltaE=deltaE; Ej=Ek     #大步长的j
        return maxK, Ej
    else:
        j=selectJrand(i, oS.m)
        Ej=calcEk(oS,j)
    return j,Ej

def updateEk(oS,k):      #计算误差值并存入缓存当中
    Ek=calcEk(oS, k)
    oS.eCache[k]=[1,Ek]


#完整Platt SMO算法中的优化例程
def innerL(i, oS):
    Ei=calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej=selectJ(i,oS,Ei)   #选择第二个alpha
        alphaIold=oS.alphas[i].copy(); alphaJold=oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L=max(0, oS.alphas[j]-oS.alphas[i])
            H=min(oS.C, oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0, oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C, oS.alphas[j]+oS.alphas[i])
        if L==H: print "L==H"; return 0
        eta=2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
        if eta >=0: print "eta>0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j)   #更新误差缓存
        if (abs(oS.alphas[j]-alphaJold) < 0.00001):
            print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS, i)
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b=b1
        elif (0 <oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b=b2
        else: oS.b=(b1+b2)/2.0
        return 1
    else: return 0

#完整版Platt SMO的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS=optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter=0
    entireSet=True; alphaPairsChanged=0
    while (iter < maxIter) and ((alphaPairsChanged >0 ) or (entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):   #遍历所有的alpha值
                alphaPairsChanged += innerL(i, oS)
            print "fullSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs=nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]  #****.A返回自身数据的2维数组的一个视图（没有做任何的拷贝）
            for i in nonBoundIs:   #遍历所有的非边界alpha值，也就是不在边界0或c上的值
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i: %d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet =False
        elif (alphaPairsChanged == 0): entireSet = True
        print "iteration number %d" % iter
    return oS.b,oS.alphas

#在python下输入：
#>>>import svmMLiA
#>>>dataArr,labelArr=svmMLiA.loadDataSet('testSet.txt')
#>>>b,alphas=svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)
#就可以得到b和alpha的值了

#计算w
def calcWs(alphas,dataArr,classLabels):
    X=mat(dataArr); labelMat=mat(classLabels).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)  #多个乘积
    return w

#>>>ws=svmMLiA.calcWs(alphas,dataArr,labelArr)  求w
#以上，就可以训练出分类超平面了。
#测试时，在python下输入
#>>>dataMat=mat(dataArr)
#>>>dataMat[i（i为某个具体行数）]*mat(ws)+b   得到一个结果值，若该值大于0，属于1类，小于0属于-1类


#--------------------------------------------------------SMO算法部分结束-----------------------------------------------------

#--------------------------------------------------------核函数部分-----------------------------------------------------

#核转换函数

def kernelTrans(X, A, kTup):
    m,n=shape(X)
    K=mat(zeros((m,1)))
    if kTup[0]=='lin': K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
    K=exp(K/(-1*kTup[1]**2))      #sigma就用在这里了
    else: raise NameError('Houston We Have a Problrm -- That Kernel is not recognized')
    return K

#利用核函数进行分类的径向基测试函数
def testRbf(k1=1.3):
    dataArr, labelArr=loadDataSet('testSetRBF.txt')
    b,alphas=smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat=mat(dataArr); labelMat=mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=dataMat[svInd]   ##get matrix of only support vectors
    labelSV=labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n=shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]): errorCount +=1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr, labelArr=loadDataSet('testSetRBF2.txt')
    errorCount=0
    dataMat=mat(dataArr); labelMat=mat(labelArr).transpose()
    m,n=shape(dataMat)
    for i in range(m):
        kernelEval=kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict=kernelEval.T * multiply(labelSV, alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount)/m)

#在python下输入
#>>>reload(svmMLiA)
#<module 'svmMLiA' from  'svmMLiA.pyc'>
#>>>svmMLiA.testRbf()

#---------------------------------------------手写识别问题（这里的问题不是针对图像的，只是举了个例子）---------------------------------


def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList=listdir(dirName)  #获得指定目录中的内容
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)  #这里模拟了两分类问题，碰到数字9，输出类别标签-1，否则输出1
        else: hwLabels.append(1)
        trainingMat[i,:]=img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):     #这里也可以自己输入核函数的类型和sigma
    dataArr, labelArr=loadImages('trainingDigits')
    b, alphas=smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat=mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV=labelMat[svInd]
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n=shape(datMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs, datMat[i,:], kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]): errorCount +=1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr, labelArr=loadImages('testDigits')
    errorCount=0
    datMat=mat(dataArr); labelMat=mat(labelArr).transpose()
    m,n=shape(datMat)
    for i in range(m):
        kernelEval=kernelTrans(sVs, datMat[i,:], kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]): errorCount +=1
    print "the test error is: %f" % (float(errorCount)/m)

#>>>svmMLiA.testDigits(('rbf', 20))




