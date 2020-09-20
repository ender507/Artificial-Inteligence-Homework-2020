from tf_idf import *
import math
K = 6
p = 12

# 使得所有的概率之和为1
def standard(predictEmt):
    total = sum(predictEmt)
    for i in range(0,6):
        predictEmt[i] /= total

# 预测六种概率
def predict(simSentence, emt):
    predictEmt = [0 for i in range(0,6)]
    for i in range(0,6):
        for eachSentence in simSentence.keys():
            predictEmt[i] = predictEmt[i] + emt[eachSentence][i]/simSentence[eachSentence]
    return predictEmt

# 计算并输出相关系数
def cor(x, y):
    avgX = sum(x)/len(x)
    avgY = sum(y)/len(y)
    tmp1 = 0.0
    tmp2 = 0.0
    tmp3 = 0.0
    for i in range(0,len(x)):
        tmp1 = tmp1 + (x[i]-avgX)*(y[i]-avgY)
        tmp2 = tmp2 + (x[i]-avgX)*(x[i]-avgX)
        tmp3 = tmp3 + (y[i]-avgY)*(y[i]-avgY)
    tmp1 = abs(tmp1)
    # print(tmp1/math.sqrt(tmp2*tmp3))
    return tmp1/math.sqrt(tmp2*tmp3)

def findSimSentence(sentenceCount, tf_idf, idf, sentence ,simSentence, k):
    simSentence.clear()
    for i in range(0, sentenceCount):
        diff = 0.0
        for eachWord in tf_idf.keys():
            if tf_idf[eachWord][i] != 0 or (eachWord in sentence):
                diff = diff + abs(tf_idf[eachWord][i] - sentence.count(eachWord)*idf[eachWord]) ** p
                # diff = max(diff, abs((tf_idf[eachWord][i] - sentence.count(eachWord) * idf[eachWord])))
        # diff = pow(diff, 1.0 / p)             # 开根号不会影响距离结果
        if diff == 0:
            diff += 0.000000001
        # 若找到的文档数量比K小，直接加入simSentence
        if len(simSentence) != k:
            simSentence[i] = diff
        # 否则进行判断，去掉diff最大的文档并加入当前文档
        elif diff < max(simSentence.values()):
            for eachSentence in simSentence:
                if simSentence[eachSentence] == max(simSentence.values()):
                    del simSentence[eachSentence]
                    break
            simSentence[i] = diff


def valid(tf_idf, idf, emt, sentenceCount):
    #整个验证集预测的6个概率
    predictAnger = []
    predictDisgust = []
    predictFear = []
    predictJoy = []
    predictSad = []
    predictSurprise = []
    #整个验证集实际的6个概率
    trueAnger = []
    trueDisgust = []
    trueFear = []
    trueJoy = []
    trueSad = []
    trueSurprise = []
    num = 1
    with open("validation_set.csv","r")as validSet:
        for eachLine in validSet:
            s = eachLine.split(',')
            sentence = s[0].split(' ')
            s[-1] = s[-1][0:-1]
            emotion = s[1:7]
            if emotion[0] == 'anger':               # 排除测试集的第一行
                continue
            for i in range(0, 6):
                emotion[i] = float(emotion[i])      # 将概率大小由字符串转为浮点数

            trueAnger.append(emotion[0])
            trueDisgust.append(emotion[1])
            trueFear.append(emotion[2])
            trueJoy.append(emotion[3])
            trueSad.append((emotion[4]))
            trueSurprise.append(emotion[5])

            simSentence = {}
            findSimSentence(sentenceCount, tf_idf, idf, sentence ,simSentence, K)
            predictEmt = predict(simSentence, emt)
            standard(predictEmt)

            predictAnger.append(predictEmt[0])
            predictDisgust.append(predictEmt[1])
            predictFear.append(predictEmt[2])
            predictJoy.append((predictEmt[3]))
            predictSad.append(predictEmt[4])
            predictSurprise.append(predictEmt[5])
            # print(num)
            num += 1
            # print(predictEmt)
            # print(emotion)
    print((cor(predictAnger,trueAnger)+cor(predictDisgust,trueDisgust)+cor(predictFear,trueFear)+cor(predictJoy,trueJoy)+cor(predictSad,trueSad)+cor(predictSurprise,trueSurprise))/6)

def test(tf_idf, idf, emt, sentenceCount):
    with open("18340057_ HuTingxi_KNN_regression.csv","w")as outputFile:
        outputFile.write("textid,anger,disgust,fear,joy,sad,surprise\n")
        with open("test_set.csv", "r")as testSet:
            for eachLine in testSet:
                s = eachLine.split(',')
                num = s[0]
                sentence = s[1].split(' ')
                if num == 'textid':  # 排除测试集的第一行
                    continue

                simSentence = {}
                findSimSentence(sentenceCount, tf_idf, idf, sentence ,simSentence, K)
                predictEmt = predict(simSentence, emt)
                standard(predictEmt)

                outputFile.write(num)
                outputFile.write(',')
                for i in range(0,6):
                    outputFile.write(str(predictEmt[i]))
                    outputFile.write(',')
                outputFile.write('\n')

tf_idf, idf, emt, sentenceCount = summonTF_IDF()
# valid(tf_idf, idf, emt, sentenceCount)
test(tf_idf, idf, emt, sentenceCount)