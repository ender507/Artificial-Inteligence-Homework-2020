# 依据第i个特征是否为i将数据集data划分为两个子集
def subD(data, value, i):
    subData1 = []
    subData2 = []
    for eachSample in data:
        if eachSample[i] == value:
            subData1.append(eachSample)
        else:
            subData2.append(eachSample)
    return subData1, subData2

def Gini(data):
    count0 = 0
    count1 = 0
    for eachSample in data:
        if int(eachSample[6]) == 0:
            count0 += 1
        else:
            count1 += 1
    return 1 - (count0/len(data))**2 - (count1/len(data))**2

def trainCART(CARTtree, node, data):
    # 先判断终止条件，即当前数据集的标签是否完全一致
    summonLeave = 1
    for eachSample in data:
        if eachSample[6] != data[0][6]:
            summonLeave = 0
            break
    if summonLeave == 1:
        CARTtree[node] = int(data[0][6])
        return

    data1 = None                    # 记录基尼指数最小时划分出的两个数据子集
    data2 = None
    attr = -1                       # 记录基尼指数最小时选择的特征和特征值
    attrVal = -1
    minGiniDA = 1                   # 基尼指数，初始化为基尼指数的理论最大值
    for i in range(6):              # 遍历所有的特征
        a = {}                      # a用来存储当前数据集下当前特征所有的可能的取值
        for eachSample in data:
            if eachSample[i] not in a.keys():
                a[eachSample[i]] = 0
            a[eachSample[i]] += 1
        if len(a)==1:                # 若当前特征只有一种取值则不纳入考虑范围
            continue
        # 计算当前特征下不同特征取值对应的基尼指数
        
        for eachValue in a.keys():
            GiniDA = 0
            subData1, subData2 = subD(data, eachValue, i)
            GiniDA += a[eachValue]/len(data) * Gini(subData1) + (1-a[eachValue]/len(data)) * Gini(subData2)
            if GiniDA < minGiniDA:
                minGiniDA = GiniDA
                data1 = subData1[:]
                data2 = subData2[:]
                attr = i
                attrVal = eachValue

    if minGiniDA == 1:
        count0 = 0
        count1 = 0
        for eachSample in data:
            if int(eachSample[6]) == 0:
                count0 += 1
            else:
                count1 += 1
        if count0 > count1:
            CARTtree[node] = 0
        else:
            CARTtree[node] = 1
    else:
        CARTtree[node] = [attr,attrVal]
    trainCART(CARTtree, node*2+1, data1)
    trainCART(CARTtree, node*2+2, data2)

def validCART(CARTtree):
    all = 0
    right = 0
    with open("car_valid.csv","r") as validSet:
        for eachLine in validSet:
            s = eachLine.split(',')
            if s[0] == 'buying':
                continue
            s[-1] = s[-1][0:-1]                     # 去掉最后一个单词的结尾处的回车

            node = 0
            while CARTtree[node] != 0 and CARTtree[node] != 1:
                if CARTtree[node] is None:
                    break
                if s[CARTtree[node][0]] == CARTtree[node][1]:
                    node = node * 2 + 1
                else:
                    node = node * 2 + 2

            all += 1
            if int(s[6]) == CARTtree[node]:
                right += 1
    print(all,end='\t\t')
    print(right,end='\t\t')
    print(right/all,end='\t\t\n')
