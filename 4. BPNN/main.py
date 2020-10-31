import math
import random
# 对每个样本，下标0为编号，下标1~15为特征值，16为标签

def train(trainSet, learningRate, miniBatchSize, times, hideNodeCount):
    w1 = [[random.randint(-10,10)/100 for i in range(15)] for j in range(hideNodeCount)]
    w2 = [random.randint(-10,10)/100 for i in range(hideNodeCount)]
    w1Gar = [[0 for i in range(15)] for j in range(hideNodeCount)]
    w2Gar = [0 for i in range(hideNodeCount)]
    while times != 0:
        times -= 1
        # print(times)
        for eachSample in trainSet:
            if eachSample[0] == 1:      # 每次重新学习整个训练集时，将梯度归零
                w1Gar = [[0 for i in range(15)] for j in range(hideNodeCount)]
                w2Gar = [0 for i in range(hideNodeCount)]
            hiddenNodeVal = [0 for i in range(hideNodeCount)]
            predictVal = 0
            # 正向传播
            for i in range(hideNodeCount):
                # 从输入层到隐藏层
                for j in range(15):
                    hiddenNodeVal[i] = hiddenNodeVal[i] + w1[i][j] * eachSample[j+1]
                hiddenNodeVal[i] = math.tanh(hiddenNodeVal[i])  # 激活函数为tanh函数
                # 从隐藏层到输出层
                predictVal = predictVal + hiddenNodeVal[i] * w2[i]
            # 反向传播
            # 计算梯度
            for i in range(hideNodeCount):
                for j in range(15):
                    w1Gar[i][j] = w1Gar[i][j] + ((predictVal - eachSample[16]) * w2[i] * eachSample[j+1] * (1 - hiddenNodeVal[i] * hiddenNodeVal[i]))/miniBatchSize
                w2Gar[i] = w2Gar[i] + ((predictVal - eachSample[16]) * hiddenNodeVal[i]) / miniBatchSize
            # 达到minibatch大小后，进行梯度更新
            if eachSample[0] % miniBatchSize == 0:
                for i in range(hideNodeCount):
                    for j in range(15):
                        w1[i][j] = w1[i][j] - learningRate * w1Gar[i][j]
                    w2[i] = w2[i] - learningRate * w2Gar[i]
                w1Gar = [[0 for i in range(15)] for j in range(hideNodeCount)]
                w2Gar = [0 for i in range(hideNodeCount)]
    return w1, w2


def date2md(date):
    s = date.split('/')             # 拆分成年月日
    return [int(s[1]),int(s[2])]    # 返回月和日的列表

def readTrainSet():
    trainSet = []
    with open('train.csv','r')as train:
        for eachSample in train:
            s = eachSample.split(',')
            if s[0] == 'instant':
                continue
            # 进行数据类型的转换
            s[-1] = s[-1][:-1]
            s[0] = int(s[0])
            s[14] = int(s[14])
            for i in range(2, 10):
                s[i] = int(s[i])
            for i in range(10,14):
                s[i] = float(s[i])
            # 对日期作特殊变化
            s = [s[0]]+date2md(s[1])+s[2:-1]+[1]+s[-1:]
            # 进行数据归一化
            s_min = min(s[1:16])
            s_max = max(s[1:16])
            for i in range(1,16):
                s[i] = (s[i]-s_min)/(s_max-s_min)
            trainSet.append(s)
    return trainSet

def valid(w1, w2, hideNodeCount):
    loss = 0
    with open('train.csv','r')as validSet:
        for eachSample in validSet:
            s = eachSample.split(',')
            if s[0] == 'instant':
                continue
            s[-1] = s[-1][:-1]
            s[0] = int(s[0])
            for i in range(2, 10):
                s[i] = int(s[i])
            for i in range(10,14):
                s[i] = float(s[i])
            sample = [s[0]]+date2md(s[1])+s[2:-1]+[1]+s[-1:]
            s_min = min(sample[1:16])
            s_max = max(sample[1:16])
            for i in range(1, 16):
                sample[i] = (sample[i] - s_min) / (s_max - s_min)
            predictVal = 0.0
            hiddenLayer = [0.0 for i in range(hideNodeCount)]
            for i in range(hideNodeCount):
                for j in range(15):
                    hiddenLayer[i] = hiddenLayer[i] + w1[i][j] * sample[j+1]
                hiddenLayer[i] = math.tanh(hiddenLayer[i])
            for i in range(hideNodeCount):
                predictVal = predictVal + hiddenLayer[i] * w2[i]
            # print(int(predictVal), sample[16])
            loss = loss + ((predictVal - int(sample[16]))**2)/2/1000
        print(loss)

trainSet = readTrainSet()
hideNodeCount = 50
miniBatchSize = 100
learningRate = 0.01
times = 500
w1, w2 = train(trainSet, learningRate, miniBatchSize, times, hideNodeCount)
valid(w1, w2, hideNodeCount)
