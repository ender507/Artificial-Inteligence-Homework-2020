import math
from ID3 import calcHDA
from ID3 import validID3

# 计算特定特征的信息熵
def calcHAD(data, HAD, i):
    # 条件熵初始化为0
    HAD[i] = 0.0
    # 找出选定特征全部的种类和个数，如tmp['low']=10表示该特征下特征值为'low'的有10个
    tmp = {}
    total = 0
    for eachSample in data:
        if eachSample[i] not in tmp:
            tmp[eachSample[i]] = 0
        tmp[eachSample[i]] += 1
        total += 1
    for eachKey in tmp.keys():
        HAD[i] += - tmp[eachKey]/total * math.log2(tmp[eachKey]/total)


def trainC4_5(attr, C4_5tree, node, data):
    # 计算经验熵
    # 标签只有0和1，分别统计0和1的个数后计算经验熵
    count0 = 0
    count1 =0
    for eachSample in data:
        if eachSample[6] == '0':
            count0 += 1
        else:
            count1 += 1
    # 当前数据只包含一种标签，则生成叶子节点
    if count0 == 0:
        C4_5tree[node] = 1
        return
    elif count1 == 0:
        C4_5tree[node] = 0
        return
    else:
        HD = - count0/(count0+count1)*math.log2(count0/(count0+count1)) - count1/(count0+count1)*math.log2(count1/(count0+count1))

    # 计算条件熵
    # 条件熵初始化为HD，即初始化信息增益为0。索引0~5分别表示六个特征下的条件熵
    HDA = [HD for i in range(6)]
    flag = 1
    for i in range(6):
        # 只有没有被当做决策点的特征才会被计算实际的条件熵
        if attr[i] == 1:
            calcHDA(data, HDA, i)
            flag = 0
    # 若flag没有被改成0，说明所有属性都已经被用来生成过节点，此处直接生成叶子节点
    if flag == 1:
        if count0 > count1:
            C4_5tree[node] = 0
        else:
            C4_5tree[node] = 1
        return

    # 计算信息增益比
    gDA = [(HD - HDA[i])for i in range(6)]
    HAD = [0 for i in range(6)]
    for i in range(6):
        if attr[i] == 1:
            calcHAD(data, HAD, i)
    gRDA = [0 for i in range(6)]
    for i in range(6):
        if attr[i] == 1:
            gRDA[i] = gDA[i] / HAD[i]
    gRDA_maxNum = 0
    gRDA_maxId = -1
    for i in range(6):
        if gRDA[i] > gRDA_maxNum:
            gRDA_maxId = i           # 信息增益最大的属性
            gRDA_maxNum = gDA[i]     # 信息增益最大的值

    # 开始生成新的节点
    key = []
    for eachSample in data:
        if eachSample[gRDA_maxId] not in key:
            key.append(eachSample[gRDA_maxId])
    newNode = node*2+2      # 使用树的兄弟-儿子表示法，每个节点的左指针指向下一个兄弟，右指针指向第一个儿子
    newData = []            # 按照选择的属性选取每个子节点的数据集子集
    newAttr = attr[:]       # 将选择的属性记录为已经选择过
    newAttr[gRDA_maxId] = 0
    C4_5tree[node] = [gRDA_maxId,key,count1>count0]
    for eachKey in key:
        newData.clear()
        # 选出新节点的数据集子集
        for eachSample in data:
            if eachSample[gRDA_maxId] == eachKey:
                newData.append(eachSample)
        # 生成子节点
        trainC4_5(newAttr, C4_5tree, newNode, newData)
        newNode = newNode*2+1   # 记录下一个兄弟节点



def validC4_5(C4_5tree):
    validID3(C4_5tree)
