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
            # 将0的标签改为-1，方便训练过程
            if s[40] == 0:
                s[40] = -1
            # 加入记录训练集的列表
            trainSet.append(s)
    return trainSet

def PLA2(trainSet, learningRate, w, b, failSample):
    # 遍历每个样本
    c = 0
    for eachSample in trainSet:
        c += 1
        # 计算当前样本的预测值
        num = 0.0
        for i in range(40):
            num = num + w[i] * eachSample[i]
        num += b
        # 如果预测值和真实值不符则更新w和b并重新检验
        if num==0 or (num>0 and eachSample[40]==-1) or (num<0 and eachSample[40]==1):
            if c not in failSample.keys():
                failSample[c] = 1
            elif failSample[c] < 3:
                failSample[c] += 1
            else:
                continue

            for i in range(40):
                w[i] = w[i] + eachSample[40] * eachSample[i] * learningRate
            b += (learningRate * eachSample[40])
            return False, w, b
    return True, w, b

def PLA(trainSet, learningRate, w, b):
    for eachSample in trainSet:
        num = 0
        for i in range(40):
            num = num + w[i] * eachSample[i]
        num += b
        if num == 0 or (num > 0 and eachSample[40] == -1) or (num < 0 and eachSample[40] == 1):
            for j in range(40):
                w[j] = w[j] + eachSample[40] * eachSample[j] * learningRate
            b += (learningRate * eachSample[40])
            return w, b
    return w, b

# 进行训练
def train(trainSet, learningRate):
    w = [ 0 for i in range(40)]
    b = 0
    times = 0
    while times <= 10000:
        times += 1
        w, b = PLA(trainSet, learningRate, w, b)
    return w, b

# 进行训练
def train2(trainSet, learningRate):
    w = [ 0 for i in range(40)]
    b = 0
    flag = False
    failSample = {}
    while flag == False:
        flag, w, b = PLA2(trainSet, learningRate, w, b, failSample)
    return w, b


# 验证结果
def valid(w, b):
    all = 0
    right = 0
    with open('validation.csv','r') as validSet:
        for eachLine in validSet:
            s = eachLine.split(',')
            # 去掉最后的回车
            s[40] = s[40][0:-1]
            num = 0
            for i in range(40):
                num = num + float(s[i]) * w[i]
            num += b
            if (num>0 and s[40]=='1') or (num<0 and s[40]=='0'):
                right += 1
            all += 1
        print(right, all, right/all)


trainSet = readTrainSet()
learningRate = 1
w, b = train(trainSet, learningRate)
valid(w, b)
w, b = train2(trainSet, learningRate)
valid(w, b)