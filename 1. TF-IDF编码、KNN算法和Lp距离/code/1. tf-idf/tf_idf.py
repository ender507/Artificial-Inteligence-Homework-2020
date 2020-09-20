import math
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
    # 打开文件
    with open("semeval.txt", "r") as text:
        sentenceCount = len(text.readlines())   # 统计文件行数
        text.seek(0, 0)                         # 文件指针复位

        num = 0     # 记录当前是第几行（从0开始计数）
        for eachLine in text:
            # 拆分每一行并将第三个部分的句子赋值给sentence
            sentence = eachLine.split('\t')[2][0:-1]    # 取[0:-1]是为了把句末的回车符去掉
            # 用每一行的数据构造tf
            makeTF(sentence, tf, sentenceCount, num)
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

    return tf, idf, sentenceCount


if __name__ ==  '__main__':
    # 生成tf_idf矩阵和emt矩阵
    # tf_idf['word'][i]为单词'word'在第i个句子中的tf_idf值
    # sentenceCount为文本中句子数
    tf_idf, idf, sentenceCount = summonTF_IDF()
    # 输出tf_idf矩阵为txt文件
    with open("18340057_HuTingxi_TFIDF.txt", "w") as tf_idfFile:
        for i in range(0, sentenceCount):
            for eachKey in tf_idf.keys():
                tf_idfFile.write(str(tf_idf[eachKey][i]))
                tf_idfFile.write(' ')
            tf_idfFile.write('\n')
