import math

def makeEMT(emt, num, emotion):
    for i in range(0,6):
        emotion[i] = float(emotion[i])
    emt[num] = emotion

# 计算tf矩阵，每次调用处理一行的数据
def makeTF(sentence, tf, sentenceCount, num):
    words = sentence.split(' ')
    # 把句子拆分成单词
    for eachWord in words:
        # 若字典中不存在该单词，加入字典
        if eachWord not in tf.keys():
            tf[eachWord] = [0.0 for x in range(0, sentenceCount)]
        tf[eachWord][num] = tf[eachWord][num] + 1.0 / len(words)

# 由tf构造idf字典
def makeIDF(tf, idf, sentenceCount):
    for eachKey in tf.keys():
        if eachKey not in idf.keys():
            idf[eachKey] = 1.0    # 依据公式log(|D|/|Di|+1)要加上1，所以初始化为1
        # 统计每个单词出现的次数
        for i in range(0,sentenceCount):
            if tf[eachKey][i] != 0:
                idf[eachKey] += 1.0
    # 到此处，变量idf记录的值是|Di|+1.0
    # 代入公式
    for eachKey in idf.keys():
        idf[eachKey] = math.log(sentenceCount/idf[eachKey],10)

def summonTF_IDF():
    # tf['word'][i]表示单词'word'在第i个句子出现的次数/第i个句子的单词数
    # idf['word']表示单词'word'在idf中的值
    tf = {}
    idf = {}
    emt = {}
    # 打开文件
    with open("train_set.csv", "r") as trainSet:
        sentenceCount = len(trainSet.readlines())  # 统计文件行数
        trainSet.seek(0, 0)  # 文件指针复位

        num = 0
        for eachLine in trainSet:
            s = eachLine.split(',')
            sentence = s[0]
            s[-1]=s[-1][0:-1]       #去掉最后的回车符
            emotion = s[1:7]
            if emotion[0] == "anger":
                continue  # 不计入第一行

            # 用每一行的数据构造tf
            makeTF(sentence, tf, sentenceCount, num)
            makeEMT(emt, num, emotion)

            num += 1

    # 对tf按照键值进行排序
    tfList = sorted(tf.items(), key=lambda item: item[0])
    tf.clear()
    for i in range(len(tfList)):
        tf[tfList[i][0]] = tfList[i][1]

    # 使用已经建好的tf建立idf
    makeIDF(tf, idf, sentenceCount)
    # 用idf的数据更新tf内的值，从而生成tf_idf矩阵
    for eachKey in idf.keys():
        for i in range(0, sentenceCount):
            tf[eachKey][i] *= idf[eachKey]  # 直接用原来的变量tf存储tf_idf的数据

    return tf, idf, emt, sentenceCount-1
