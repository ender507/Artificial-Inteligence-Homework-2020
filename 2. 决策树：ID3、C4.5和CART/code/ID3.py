import math

# 计算特定特征下的条件熵
def calcHDA(data, HDA, i):
    # 条件熵初始化为0
    HDA[i] = 0.0
    # 找出选定特征全部的种类
    key = []
    for eachSample in data:
        if eachSample[i] not in key:
            key.append(eachSample[i])
    # 该特征下的每一种情况分别计算条件熵并加和
    for eachKey in key:
        count0 = 0
        count1 = 0
        for eachSample in data:
            if eachSample[i] == eachKey:
                if eachSample[6] == '0':
                    count0 += 1
                else:
                    count1 += 1
        # 单独讨论count0或count1==0的情形，防止出现log2(0)的计算错误
        if count0 == 0:
            HDA[i] += (count0+count1)/len(data)*(-count1/(count0+count1)*math.log2(count1/(count0+count1)))
        elif count1 == 0:
            HDA[i] += (count0+count1)/len(data)*(-count0/(count0+count1)*math.log2(count0/(count0+count1)))
        else :
            HDA[i] += (count0+count1)/len(data)*(-count0/(count0+count1)*math.log2(count0/(count0+count1))-count1/(count0+count1)*math.log2(count1/(count0+count1)))

def trainID3(attr, ID3tree, node, data):
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
        ID3tree[node] = 1
        return
    elif count1 == 0:
        ID3tree[node] = 0
        return
    else:
        HD = - count0/(count0+count1)*math.log2(count0/(count0+count1)) - count1/(count0+count1)*math.log2(count1/(count0+count1))

    # 计算条件熵
    # 条件熵初始化为HD，即初始化信息增益为0。索引0~5分别表示六个特征下的条件熵
    HDA = [HD for i in range(6)]
    for i in range(6):
        # 只有没有被当做决策点的特征才会被计算实际的条件熵
        if attr[i] == 1:
            calcHDA(data, HDA, i)

    # 计算信息增益
    gDA = [(HD - HDA[i])for i in range(6)]
    gDA_maxNum = 0
    gDA_maxId = -1
    for i in range(6):
        if gDA[i] > gDA_maxNum:
            gDA_maxId = i           # 信息增益最大的属性
            gDA_maxNum = gDA[i]     # 信息增益最大的值
    # 若最大信息增益为0，说明所有属性都已经被用来生成过节点，此处直接生成叶子节点
    if gDA_maxNum == 0:
        if count0 > count1:
            ID3tree[node] = 0
        else:
            ID3tree[node] = 1
        return

    # 生成新的节点
    key = []
    for eachSample in data:
        if eachSample[gDA_maxId] not in key:
            key.append(eachSample[gDA_maxId])
    newNode = node*2+2      # 使用树的兄弟-儿子表示法，每个节点的左指针指向下一个兄弟，右指针指向第一个儿子
    newData = []            # 按照选择的属性选取每个子节点的数据集子集
    newAttr = attr[:]       # 将选择的属性记录为已经选择过
    newAttr[gDA_maxId] = 0
    # 当前分支节点的表示：使用列表表示，第一个值为特征索引，
    # 第二个值为列表，记录特征所有可能的取值，顺序依次对应第1、2、3...个子节点
    # 第三个值为当前节点下样本结果为1和0的个数大小比较，用于回溯查找结果
    ID3tree[node] = [gDA_maxId,key,count1>count0]
    for eachKey in key:
        newData.clear()
        # 选出新节点的数据集子集
        for eachSample in data:
            if eachSample[gDA_maxId] == eachKey:
                newData.append(eachSample)
        # 生成子节点
        trainID3(newAttr, ID3tree, newNode, newData)
        newNode = newNode*2+1   # 记录下一个兄弟节点


def validID3(ID3tree):
    all = 0
    right = 0
    with open("car_valid.csv","r") as validSet:
        for eachLine in validSet:
            s = eachLine.split(',')
            if s[0] == 'buying':
                continue
            s[-1] = s[-1][0:-1]                     # 去掉最后一个单词的结尾处的回车

            node = 0			# 从根节点开始
            newNode = 0		    # 记录下一个要跳转的节点
            while type(ID3tree[node]) != type(1):       # 当当前节点的值为1或0，即预测结果时退出循环
                # 如果当前节点的属性对应的属性值都匹配不上验证/预测样本，则回溯
                if s[ID3tree[node][0]] not in ID3tree[node][1]:
                    tmp = ID3tree[int((node-1)/2)]
                    if type(tmp) == type(1):
                        ID3tree[node] = tmp
                        break
                    if ID3tree[int((node-1)/2)][2] == True:
                        ID3tree[node] = 1
                    else:
                        ID3tree[node] = 0
                    break
                # 否则进入判断进入哪个子节点
                newNode = node * 2 + 2          # 初始化为第一个子节点
                for i in range(len(ID3tree[node][1])):
                    # 若匹配则选中当前子节点
                    if s[ID3tree[node][0]] == str(ID3tree[node][1][i]):
                        break
                    # 否则考虑下一个子节点
                    else:
                        newNode = newNode * 2 +1
                # 更新node节点的值
                node = newNode

            all += 1
            if int(s[6]) == ID3tree[node]:
                right += 1
    print(all, end='\t\t')
    print(right, end='\t\t')
    print(right / all, end='\t\t\n')