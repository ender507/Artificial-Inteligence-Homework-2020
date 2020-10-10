import math

# 读取训练集
def readTrainSet():
    trainSet = []
    with open('train.csv','r') as ts:
        for eachLine in ts:
            s = eachLine.split(',')
            # 去掉最后的回车
            s[40] = s[40][0:-1]
            # 进行数据类型转换，从字符串转换成数字
            for i in range(40):
                s[i] = float(s[i])
            s[40] = int(s[40])
            # 添加一维常数项,在下标为40的位置，41的位置为标签
            s.append(s[40])
            s[40] = 1
            # 加入记录训练集的列表
            trainSet.append(s)
    return trainSet

# 进行训练
def train(trainSet, learningRate):
    w = [ 0 for i in range(41)]
    c=0
    while c != 300:
        c += 1
        L = [0 for i in range(41)]  # 损失函数的梯度
        for eachSample in trainSet:
            pix = 0                             # pi(x)
            for i in range(41):
                pix += ( - w[i] * eachSample[i] )
            pix = 1 / (1 + math.exp(pix) )
            for i in range(41):
                L[i] += (eachSample[41] - pix) * eachSample[i]

        for i in range(41):
            w[i] = w[i] + learningRate * L[i] / 7000
    return w

# 验证结果
def valid(w):
    all = 0
    right = 0
    with open('validation.csv','r') as validSet:
        for eachLine in validSet:
            s = eachLine.split(',')
            # 去掉最后的回车
            s[40] = s[40][0:-1]
            # 进行数据类型转换，从字符串转换成数字
            num = 0
            for i in range(40):
                num = num + float(s[i]) * w[i]
            num += w[40]
            if (num>0 and s[40]=='1') or (num<0 and s[40]=='0'):
                right += 1
            all += 1
    print(right, all, right/all)


trainSet = readTrainSet()
learningRate = 0.01
w = train(trainSet, learningRate)
valid(w)
