from tf_idf import *

K = 8                    # KNN算法的变量K
p = 2                     # Lp距离的变量p


def test(tf_idf, idf, emt,sentenceCount):
    simSentence = {}  # 相似的句子。simSentence[i] = j ,i为句子编号，j为差异diff
    with open("18340057_ HuTingxi_KNN_classification.csv", "w") as outputFile:
        with open("test_set.csv", "r") as testSet:
            for eachLine in testSet:
                s = eachLine.split(',')
                num = s[0]
                sentence = s[1].split(' ')      # 文档
                if num == 'textid':              # 去掉第一行
                    continue

        # with open("classification_simple_test.csv")as testSet:
        #     for eachLine in testSet:
        #         sentence = eachLine.split()
        #         if sentence[0] == "Words":
        #             continue

                findSimSentence(sentenceCount, tf_idf, idf, sentence ,simSentence, K)
                emotionCount = emtCount(simSentence, emt)

                mostEmt = findMostEmt(emotionCount, sentence, simSentence, sentenceCount, tf_idf, idf, K)

                # print(mostEmt)
                outputFile.write(s[0])
                outputFile.write(',')
                outputFile.write(mostEmt)
                outputFile.write('\n')


def findSimSentence(sentenceCount, tf_idf, idf, sentence ,simSentence, k):
    simSentence.clear()
    for i in range(0, sentenceCount):
        diff = 0.0
        for eachWord in tf_idf.keys():
            if tf_idf[eachWord][i] != 0 or (eachWord in sentence):
                diff = diff + abs(tf_idf[eachWord][i] - sentence.count(eachWord)*idf[eachWord]) ** p
        # diff = pow(diff, 1.0 / p)             # 开根号不会影响距离结果

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

def emtCount(simSentence, emt):
    emotionCount = {}
    for eachSentence in simSentence:
        if emt[eachSentence] not in emotionCount:
            emotionCount[emt[eachSentence]] = 1
        else:
            emotionCount[emt[eachSentence]] += 1
    return emotionCount

def findMostEmt(emotionCount, sentence, simSentence, sentenceCount, tf_idf, idf, k):
    mostEmt = None
    for eachEmt in emotionCount.keys():
        # 若当前遍历到的情绪标签是最多的，记录下来
        if emotionCount[eachEmt] == max(emotionCount.values()):
            if mostEmt == None:
                mostEmt = eachEmt
        # 否则，有多个情绪标签的数量相同，则减少k值并继续计算
            else:
                findSimSentence(sentenceCount, tf_idf, idf, sentence, simSentence, k-1)
                emotionCount = emtCount(simSentence, emt)
        # 递归调用自己，找到数量最多的情绪
                mostEmt = findMostEmt(emotionCount, sentence, simSentence, sentenceCount, tf_idf, idf, k-1)
                break
    return mostEmt


def valid(tf_idf, idf, emt,sentenceCount):
    all = 0
    right = 0
    simSentence = {}  # 相似的句子。simSentence[i] = j ,i为句子编号，j为差异diff
    with open("validation_set.csv", "r") as testSet:
        for eachLine in testSet:
            s = eachLine.split(',')
            ans = s[1][0:-1]                # 标签的实际结果
            sentence = s[0].split(' ')      # 文档
            if ans == 'label':              # 去掉第一行
                continue

            findSimSentence(sentenceCount, tf_idf, idf, sentence ,simSentence, K)
            emotionCount = emtCount(simSentence, emt)
            mostEmt = findMostEmt(emotionCount, sentence, simSentence, sentenceCount, tf_idf, idf, K)

            all += 1
            if mostEmt == ans:
                right += 1
            print(all, right / all)




tf_idf, idf, emt, sentenceCount = summonTF_IDF()
# valid(tf_idf, idf, emt,sentenceCount)
test(tf_idf, idf, emt,sentenceCount)
