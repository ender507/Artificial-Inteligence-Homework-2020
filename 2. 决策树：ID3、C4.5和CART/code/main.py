from ID3 import *
from C4_5 import *
from CART import *

def trainSetRepro():
    data = []
    with open("car_train.csv") as trainSet:
        for eachLine in trainSet:
            s = eachLine.split(',')
            if s[0] == 'buying':                    # 去掉第一行
                continue
            s[-1] = s[-1][0:-1]                     # 去掉最后一个单词的结尾处的回车

            data.append([s[i] for i in range(7)])
    return data

# data的每一行为一个样本，索引从0到6依次为六个特征和最终结果
data = trainSetRepro()
# attr索引0~5分别表示六个特征，为1表示当前特征未被决策树作为子节点划分，否则为0
attr = [1, 1, 1, 1, 1, 1]
print('\t\t样本数\t正确数\t准确率')
# 用数组构造ID3树
ID3tree = [None for i in range(10000000)]
# print('ID3:',end='\t')
# trainID3(attr, ID3tree, 0, data)
# validID3(ID3tree)
# 同理可以构造C4.5树
C4_5tree = [None for i in range(10000000)]
# print('C4.5:',end='\t')
# trainC4_5(attr, C4_5tree, 0, data)
# validC4_5(C4_5tree)
# 构造CART树
CARTtree = [None for i in range(10000000)]
print('CART:',end='\t')
trainCART(CARTtree, 0, data)
validCART(CARTtree)
