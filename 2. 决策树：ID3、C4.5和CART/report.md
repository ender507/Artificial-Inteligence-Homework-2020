# 人工智能实验二 实验报告

18340057  18级计算机科学二班

## 一、 ID3决策树

### 1. 算法原理

#### 1.1 决策树的通用算法

决策树的生成算法分为以下几个步骤：

1. 初始化：创建根节点，拥有全部的数据集和特征
2. 选择特征：遍历当前节点的数据集和特征，依据某种原则选择一个特征
3. 划分数据：依据所选特征的不同取值，将当前数据集划分为若干个子集
4. 创建节点：为每个子数据集创建一个子节点，并删去刚刚选中的特征
5. 递归建树：对每个子节点，回到第二步进行递归调用，直到达到边界条件，则回溯
6. 完成建树：叶子节点采用多数投票的方式判定自身的类别

其中，若当前节点的数据集为$D$，特征集为$A$，则边界条件的判断方式如下（满足其一即可）：

- 若$D$中的样本属于同一类别$C$，则将当前的节点标记为$C$类叶节点

- $A$为空集，或$D$中所有样本在$A$中所有特征上取值相同，则无法划分。当前节点标记为叶节点，类别为$D$中出现最多的类
- $D$为空集，则将当前节点标记为叶节点，类别为父节点中出现最多的类

#### 1.2 ID3决策树的信息增益

ID3决策树指定了上述决策树生成算法第二、三步中，选取特征和取值的原则。

ID3决策树是采用信息增益来决定通过哪个特征作为决策点的。信息增益越大，说明该特征对得到结果的帮助越大，则优先选用信息增益最大的特征作为决策点。信息增益的算法如下：

假设训练数据集为$D$，$|D|$表示样本容量，样本有$K$个类，记为$C_k, k=1,2,...,K$，其中$|C_k|$表示该类的样本个数。依据特征A的n个不同取值$\{a_1,a_2,...,a_n\}$，将D划分为n个子集$\{D_1,...,D_n\}$，记子集$D_i$中属于类$C_k$的样本集合为$D_{ik}$。

1. 计算数据集$D$的经验熵：

$$
\displaystyle H(D)=-\sum_{k=1}^K\frac{|C_k|}{|D|}\log_2\frac{|C_k|}{|D|}
$$

2. 计算特征$A$对数据集$D$的经验条件熵：

$$
\displaystyle H(D|A)=\sum^n_{i=1}\frac{|D_i|}{|D|}H(D_i)=-\sum^n_{i=1}\frac{|D_i|}{|D|}\sum^K_{k=1}\frac{|D_{ik}|}{|D_i|}\log_2\frac{|D_{ik}|}{|D_i|}
$$

3. 计算信息增益：

$$
g(D,A)=H(D)-H(D|A)
$$

对所有的特征，都计算出相应的信息增益。比较各个特征的信息增益，最终选择使得信息增益最大的特征作为决策点。每次优先选择信息量最多的属性，即使得熵变得最小的属性，以构造使得熵下降最快的决策树，到叶子节点时熵为0或接近0。

-----

### 2. 伪代码

#### 2.1 训练

考虑到数据样本的标签只有0和1，首先要统计样本中标签为0的个数和标签为1的个数，从而能够计算经验熵。

##### 计算经验熵（含生成叶节点判断）

```pseudocode
Input:ID3tree, node, data
/* 输入：ID3决策树、当前节点、当前数据集 */
Output:HD
/* 输出：经验熵 */
def calcHD(ID3tree, node, data){
	count0 = 0			/* 统计标签为0的样本数 */
	count1 = 0			/* 统计标签为1的样本数 */
	/* 遍历每个样本,统计不同标签对应的样本数 */
	for eachSample in data 		 
		if eachSample的标签为0
			then count0++
		else count1++
	end
	
	/* 若当前数据集的所有样本的标签都相同则直接生成叶子节点 */
	HD = 0
	if count0 == 0
		then ID3tree[node] = 1
	else if count1 = 0
		then ID3tree[node] = 0
	/* 否则计算经验熵 */
	HD = - count0/(count0+count1)*log2(count0/(count0+count1)) - count1/(count0+count1)*log2(count1/(count0+count1))
	return HD
}
```

##### 计算经验条件熵

接下来要计算经验条件熵，需要对所有的特征进行遍历，计算每个特征的经验条件熵。计算过程如下：

```pseudocode
Input: data, A
/* 输入：数据集，特征 */
Output: HDA
/* 输出：数据集在特征A下的条件熵 */
def calcHDA(data, A){
	HDA = 0
	/* 对A中的每一个类分别求条件熵分量 */
	for eachClass in A
    	count0 = 0
        count1 = 0
        /* 计算特征A在特征值为eachClass下数据集的经验熵 */
        for eachSample in data:
            if eachSample的特征A等于eachClass
            	/* 统计标签种类和个数 */
            	then if eachSample的标签为0:
                     	then count0++
                     else: count1++
                /*  计算条件熵分量 */
        	HDA[i] += (count0+count1)/len(data)*(-count0/(count0+count1)*log2(count0/(count0+count1))-count1/(count0+count1)*log2(count1/(count0+count1)))
        end
	end
	return HDA
}
```

##### 计算信息增益并选定作为决策点的特征

每个特征下数据集的信息增益`gDA`即为上面求得的经验熵和经验条件熵的差值`HD - HDA`。找到使得`gDA`最大的特征即可。

##### 划分新的数据集，并递归调用

对于选定的特征`A`，以`A`的不同取值为依据划分数据集。

```pseudocode
newData = []
for eachClass in A
	for eachSample in data
		if eachSample的特征A的取值为eachClass
			then 将eachSample加入newData
	end
	/* 递归调用生成节点的函数 */
	summonNode(ID3tree, newNode, newData)
end
```

### 3. 代码展示

#### 3.1 数据的预处理

本次实验的每个样本有6个特征和1个标签，首先将训练样本存储在列表`data`中，`data`里的每一个节点包含7个数，索引从0到6，分别记录这六个特征和一个标签。

```python
def trainSetRepro():
    data = []
    with open("car_train.csv") as trainSet:
        for eachLine in trainSet:
            s = eachLine.split(',')
            if s[0] == 'buying':                    # 去掉第一行的样本说明
                continue
            s[-1] = s[-1][0:-1]                     # 去掉最后一个单词的结尾处的回车

            data.append([s[i] for i in range(7)])	# 将6个特征值和1个标签值组成的列表分别加入data
    return data
```

#### 3.2 多叉树的表示和特征使用判断

多叉树可以用树的兄弟-儿子表示法用二叉树表示：对于任意一个节点，有两个指针：左指针为下一个兄弟节点，右节点为第一个儿子节点。在我的代码中，我使用数组来表示该二叉树。对于`ID3tree`的节点`node`，左指针的节点索引为`2*node+1`，右指针的节点索引为`2*node+2`。若不存在该节点，则节点值为`None`。

ID3树每一次分叉需要选择一个特征作为判断条件，而某个特征在作为决策点后不会再次作为决策点。我使用了一个数组`attr`来表示。`attr`有6个1，分别表示6个特征是否能够用来选作决策点。若可以则为1。当某个特征被选作决策点后，将对应的值改为0。

二者初始化如下：

```python
attr = [1, 1, 1, 1, 1, 1]
ID3tree = [None for i in range(10000000)]
```

#### 3.3 训练过程

训练时调用`trainID3(attr, ID3tree, node, data)`函数。其中`node`参数在第一次调用时为0，表示`ID3tree`的根节点。

首先计算数据集的经验熵`HD`。如果数据集的标签只有一种时，直接生成叶子节点。叶子节点的值即为预测的标签值。

```python
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
    # 否则进行正常的经验熵运算
    else:
        HD = - count0/(count0+count1)*math.log2(count0/(count0+count1)) - count1/(count0+count1)*math.log2(count1/(count0+count1))

```

接着计算经验条件熵。需要注意的是，不是所有的特征都可以被选作决策点：之前已经被用来做决策点的特征之后不能再选择作为决策点。这时就需要上述的`attr`列表进行判断。

```python
    # 计算条件熵
    # 条件熵初始化为HD，即初始化信息增益为0。索引0~5分别表示六个特征下的条件熵
    HDA = [HD for i in range(6)]
    for i in range(6):
        # 只有没有被当做决策点的特征才会被计算实际的条件熵
        if attr[i] == 1:
            calcHDA(data, HDA, i)
```

计算选定特征的条件熵`HDA`的函数如下：

```python
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
            HDA[i] += (count0+count1)/len(data)*(-count0/(count0+count1)*math.log2(count0/(count0+count1))\
                                                 -count1/(count0+count1)*math.log2(count1/(count0+count1)))
```

得到了数据集的经验`HD`和各个特征下数据集的经验条件熵`HDA[0]~HDA[5]`后，就可以计算各个特征下的信息增益了。需要注意的是，之前讨论过的不可作为决策点的特征，其经验熵初始化为`HD`，也就是说计算得到的信息增益为0，从而不会被选作特征。还有一种特殊情况是，所有特征的信息增益都为0，也就是说，所有的特征都被选作决策点过，这时需要生成叶子节点，叶子节点的值采用多数投票原则，选取数据集中标签数量最多的那个作为叶子节点的值。

```python
# 计算信息增益
    gDA = [(HD - HDA[i])for i in range(6)]
    gDA_maxNum = 0
    gDA_maxId = -1
    for i in range(6):
        if gDA[i] > gDA_maxNum:
            gDA_maxId = i           # 使得信息增益最大的属性索引
            gDA_maxNum = gDA[i]     # 信息增益最大值
    # 若最大信息增益为0，说明所有属性都已经被用来生成过节点，此处直接生成叶子节点
    if gDA_maxNum == 0:
        if count0 > count1:
            ID3tree[node] = 0
        else:
            ID3tree[node] = 1
        return
```

上述过程都有几次生成叶子节点的讨论。如果还会往下执行，说明该节点不是叶子节点，需要生成子节点。生成子节点的方式如同之前的兄弟-儿子表示法所述。若当前节点的索引为`node`，则第一个子节点为`2*node+2`，之后是`2*(2*node+2)+1`、`2*(2*(2*node+2)+1)+1`...依次类推。注意第一个子节点是`+2`而不是`+1`。通过选取的特征划分数据集，依据上述节点算法找到新的节点位置，进行递归调用即可。当前的节点作为分支节点，还要储存特征和取值信息，从而在测试时可以知道，到达该分支节点时如何选取下一个子节点。

```python
# 开始生成新的节点
    key = []
    # 记录所选特征下的特征值所有可能的取值
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
        # 生成子节点（递归调用）
        trainID3(newAttr, ID3tree, newNode, newData)
        newNode = newNode*2+1   # 记录下一个兄弟节点
```

#### 3.4 验证/测试过程

验证和测试代码基本相同，以验证为例。函数原型和数据初始化如下：

```python
def validID3(ID3tree):
    all = 0			# 验证样本总数
    right = 0		# 预测正确的验证样本数
    with open("car_valid.csv","r") as validSet:
        for eachLine in validSet:
            s = eachLine.split(',')
            if s[0] == 'buying':					# 不考虑第一行的样本数据说明
                continue
            s[-1] = s[-1][0:-1]                     # 去掉最后一个单词的结尾处的回车
```

接下来是预测部分，即验证/测试的核心代码：

```python
            node = 0			# 从根节点开始
            newNode = 0			# 记录下一个要跳转的节点
            while type(ID3tree[node]) != type(1):	# 当当前节点的值为1或0，即预测结果时退出循环
                # 如果当前节点的属性对应的属性值都匹配不上验证/预测样本，则回溯
                if s[ID3tree[node][0]] not in ID3tree[node][1]:
                    tmp = ID3tree[int((node-1)/2)]
                    # 若回溯的节点为预测结果（0或1），则作为当前样本的结果
                    if type(tmp) == type(1):
                        ID3tree[node] = tmp
                        break
                    # 若回溯的节点为分支节点，则采用多数投票方法作为当前样本的结果
                    if ID3tree[int((node-1)/2)][2] == True:
                        ID3tree[node] = 1
                    else:
                        ID3tree[node] = 0
                    break
                # 否则进入判断进入哪个子节点
                newNode = node * 2 + 2	# 初始化为第一个子节点
                for i in range(len(ID3tree[node][1])):
                    # 若匹配则选中当前子节点
                    if s[ID3tree[node][0]] == ID3tree[node][1][i]:
                        break
                    # 否则考虑下一个子节点
                    else:
                        newNode = newNode * 2 +1
                # 更新node节点的值
                node = newNode
```

最后对预测结果进行统计即可。

```python
            all += 1
            if int(s[6]) == ID3tree[node]:
                right += 1
    print(all,end='\t\t')
    print(right,end='\t\t')
    print(right/all,end='\t\t\n')
```

------

### 4. 实验结果及分析

在本次实验中，只提供了一份数据集，需要自己划分训练集和验证集。一共有1728个样本，我尝试由大到小设置验证集所占的比例，看看实际的验证结果准确率。

首先选取全部样本中的最后432个样本做测试集，也就是验证集中样本数约为样本数的25%。代码运行结果如下：

![1](pic\\1.png)

接着选取最后346个样本作为验证集。也就是说，验证集中样本数约为样本数的20%。代码运行结果如下：

![2](pic\\2.png)

然后选取最后259个样本作为验证集。也就是说，验证集中样本数约为样本数的15%。代码运行结果如下：

![3](pic\\3.png)

然后选取最后173个样本作为验证集。也就是说，验证集中样本数约为样本数的10%。代码运行结果如下：

![4](pic\\4.png)

最后选取最后86个样本作为验证集。也就是说，验证集中样本数约为样本数的5%。代码运行结果如下：

![5](pic\\5.png)

样本验证集划分的越多，训练集就越少。训练集过少容易导致欠拟合，容易导致准确率的降低；而验证集过少容易导致验证集的泛化性不够，即验证集难以包含样本可能出现的全集的多数情况，容易导致过拟合。适中的训练集和测试集的划分才能保证模型训练的效果。

之后的决策树，都使用验证集占验证集和训练集之和的10%作为验证标准。



## 二、 C4.5决策树

### 1. 算法原理

C4.5决策树是对ID3决策树的改进。ID3决策树以息增益作为划分训练数据集的特征，存在偏向于选择取值较多的特征的问题，即该属性划分出的子类很多的话，信息增益就会更大。使用信息增益比可以对这一问题进行校正。所以C4.5决策树使用信息增益比作为选取属性作为决策点的判断依据。首先能够通过ID3决策树上述提到的算法计算出数据集的信息增益，得到信息增益后再计算信息增益比。

特征$A$对数据集$D$的信息增益比定义为信息增益与数据集$D$关于特征$A$的值的熵的比，即：
$$
g_R(D,A)=\frac{g(D,A)}{H_A(D)}
$$
即$n$为特征$A$的每种取值的个数，则：
$$
H_A(D)=-\sum^n_{i=1}\frac{|D_i|}{|D|}log_2\frac{|D_i|}{|D|}
$$
通过信息增益除以$H_A(D)$，考虑到了特征的多种取值对信息增益的影响。每次选择信息增益比最大的属性作为决策点生成分支节点即可。

-------

### 2. 伪代码

C4.5的很多部分与ID3决策树相同，而ID3上文已经讨论过，下面只讨论C4.5特有的部分。

通过与ID3相同的步骤，可以得出数据集的信息增益。下面要计算数据集的信息增益比，先要算出数据集$data$关于特征$A$的信息熵：

```pseudocode
Input: data, A
/* 输入数据集，特征 */
Output: HAD
/* 输出数据集关于特征A的信息熵 */
def calcHAD(data, A){
    /* 条件熵初始化为0 */
    HAD = 0.0
    /* 找出选定特征全部的种类和个数，如tmp['low']=10表示该特征下特征值为'low'的有10个 */
    tmp = {}
    total = 0
    for eachSample in data
        if tmp中不包含该eachSample的特征取值
            then tmp[eachSample的特征值] = 0
        tmp[eachSample的特征值] += 1
        total += 1
    end
    /* 计算HAD中每个特征值对应的分量并进行加和 */
    for eachKey in tmp:
        HAD += - tmp[eachKey]/total * log2(tmp[eachKey]/total)
}
```

计算信息增益比，只要将二者相除即可：

```pseudocode
gRDA = gDA / HAD
```

得到各个属性的$g_R(D,A)$后，选择使得信息增益比最大的特征，并进行子数据集的划分和子节点的生成，与ID3基本相同，不在赘述。

最后生成的树的构造方式和ID3完全相同，可以直接用和ID3完全相同的方式进行样本的验证和测试。

------

### 3. 代码展示

训练过程中计算信息增益的部分和ID3基本相同。计算信息增益比的具体代码如下：

首先计算数据集`data`关于特征`data[i]`的信息熵：

```python
# 对于每个特征计算各自的HAD，分别记为HAD[0]~HAD[5]
HAD = [0 for i in range(6)]
    for i in range(6):
        if attr[i] == 1:
            calcHAD(data, HAD, i)
```

具体计算某个特征下的信息熵如下：

```python
# 计算特定特征的信息熵
def calcHAD(data, HAD, i):
    # 条件熵初始化为0
    HAD[i] = 0.0
    # 找出选定特征全部的种类和个数，如tmp['low']=10表示该特征下特征值为'low'的有10个
    tmp = {}
    total = 0
    # 用tmp统计数据集中特征data[i]所有的取值和个数
    for eachSample in data:
        if eachSample[i] not in tmp:
            tmp[eachSample[i]] = 0
        tmp[eachSample[i]] += 1
        total += 1
    # 将各个特征下的信息熵进行加和
    for eachKey in tmp.keys():
        HAD[i] += - tmp[eachKey]/total * math.log2(tmp[eachKey]/total)
```

接下来的找信息增益比最大的属性，进行数据集的划分和子节点的生成，过程和ID3基本相同。

测试/验证过程中，因为树的构成方法相同，代码可以通用：

```python
def validC4_5(C4_5tree):
    validID3(C4_5tree)
```

-----

### 4. 实验结果及分析

对于划分出的10%的验证集，运行结果如下：

![6](pic\\6.png)

本次实验所给的数据中，各个特征的可能的取值个数都为3个或4个，上下限相差不大，C4.5的特性难以体现，所以ID3和C4.5决策树的训练结果相差不大。





## 三、 CART决策树

#### 1. 算法原理

CART决策树为二叉树，使用基尼指数作为选取特征作为决策点的依据。CART对特征进行二分类，对于有多个不同特征值的特征$A$，结果只会将结果分成“特征值为$a$”和"特征值不为$a$"两种。

对每个特征进行遍历，在每个特征下又对每个特征值进行遍历，依据特征值的不同将数据集$D$划分为两个子集：
$$
D_1=\{(x,y)\in D|A(x)=a\}
$$

$$
D_2=D-D_1
$$

在特征$A$是否为$a$的条件下划分的两个子集得到的数据集的基尼指数为：
$$
Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)
$$
其中，每个子集的基尼指数为：
$$
Gini(D)=1-\sum^K_{k=1}(\frac{|C_k|}{|D|})^2
$$
其中$C_k$表示标签的不同类别。

通过计算得到基尼指数后，选择基尼指数最小的特征$A$和对应的划分自己的特征值$a$作为决策点。决策树的生成算法和ID3部分的讨论。

----

#### 2. 伪代码

决策树的叶节点和分支节点的生成的代码都已经讨论过，此处不再赘述。下面只介绍基尼质数计算的代码。

首先对所有特征进行遍历，在某个特征下的特征值也进行变量，将数据集划分为两个子集：

```pseudocode
Input: data, A, a
/* 输入：数据集、特征A、特征值a */
Output: subData1, subData2
/* 输出：两个子数据集 */
 
def subD(data, A, a)
    subData1 = []
    subData2 = []
    for eachSample in data:
        if eachSample的特征A取值为a:
            then 将eachSample加入subData1
        else:
            将eachSample加入subData2
    end
    return subData1, subData2
```

划分子集的基尼指数使用以下函数计算:

```pseudocode
Input: data
/* 输入：数据集（即之前得到的数据子集） */
Output: gini
/* 输出：数据集的基尼指数 */
def Gini(data):
	count0 = 0
    count1 = 0
    for eachSample in data:
        if eachSample的标签为0
            then count0 += 1
        else:
            count1 += 1
    end
    return 1 - (count0/len(data))^2 - (count1/len(data))^2
```

考虑到数据集的标签，即最终结果的取值只有0和1，使用`count0`和`count1`分别记录标签为0或1的训练样本数，从而计算出基尼指数求和符号下的两个分量。

#### 3. 代码展示

调用训练函数，函数原型为:

```python
def trainCART(CARTtree, node, data)
```

CART树和ID3与C4.5的一个区别在于，CART树作为决策点的特征能够在之后再次被用作决策点。因此生成叶子节点只有2种情况：当前数据集的标签完全相同或当前数据集的特征值完全相同。

第一个条件直接在递归调用的开头作为截止条件：

```python
    # 先判断终止条件，即当前数据集的标签是否完全一致
    summonLeave = 1
    for eachSample in data:
        if eachSample[6] != data[0][6]:
            summonLeave = 0
            break
    if summonLeave == 1:
        CARTtree[node] = int(data[0][6])
        return
```

如果当前数据集的标签完全一致，则用该标签作为结果生成叶子节点。

接下来初始化相关变量，以及遍历各个特征，记录特征下的特征取值：

```python
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
```

接下来，依据不同的特征$A$和特征值$a$划分数据集，计算基尼指数并记下基尼指数最小时的基尼指数`minGiniDA`、特征`attr`、特征值`attrVal`、子数据集`data1`，`data2`。

子数据集生成的函数如下：

```python
def subD(data, value, i):
    subData1 = []
    subData2 = []
    for eachSample in data:
        if eachSample[i] == value:
            subData1.append(eachSample)
        else:
            subData2.append(eachSample)
    return subData1, subData2
```

计算单个数据集的基尼指数的函数如下：

```python
def Gini(data):
    count0 = 0
    count1 = 0
    for eachSample in data:
        if int(eachSample[6]) == 0:
            count0 += 1
        else:
            count1 += 1
    return 1 - (count0/len(data))**2 - (count1/len(data))**2
```

和之前伪代码的思路相同。

如果计算后得到的基尼指数为1，说明当前数据集下，所有的样本的特征值完全相同，则统计标签0和1的个数，选取最大的那个作为结果生成叶子节点。否则生成分支节点。分支节点的内容有两项：特征和划分子集的特征值。特征值相同的样本进入左子节点，索引变为`2*node+1`，否则进入右子节点，索引变为`2*node+2`。

CART树为二叉树，故直接采取左右节点存储两个儿子节点的方式存储，而不用兄弟-儿子表示法。

```python
# 若所有样本的特征值相同，采用多数投票生成叶子节点
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
# 否则生成分支节点
    else:
        CARTtree[node] = [attr,attrVal]
```

最后递归调用，生成子节点：

```python
    trainCART(CARTtree, node*2+1, data1)
    trainCART(CARTtree, node*2+2, data2)
```

在验证/测试代码中，基本过程和ID3与C4.5树相同，需要注意的是，满足节点的特征值的样本进入左节点，否则进入右节点。

```python
            node = 0
            while CARTtree[node] != 0 and CARTtree[node] != 1:
                if CARTtree[node] is None:# 理论上不会出现进入None节点的情况，因为每种特征每种取值都讨论到了
                    break
                if s[CARTtree[node][0]] == CARTtree[node][1]:
                    node = node * 2 + 1
                else:
                    node = node * 2 + 2
```

------------

### 4. 实验结果及分析

代码运行结果如下：

![7](pic\\7.png)

CART树采用二元划分，二叉树不易产生数据碎片，精确度往往也会高于多叉树。 在验证集占比为10%的情况下，可以看出，其精确度的确略高于ID3和C4.5生成的决策树。

---------

## 四、 思考题

1. 决策树有哪些避免过拟合的方法？ 

   可以通过剪枝的方法避免决策树的过度拟合：

   - 预剪枝：在生成子节点时，如果此处的决策点在验证集上的准确率不提高，则不生成子节点，而是直接将当前节点设置为叶子节点
   - 后剪枝：后序遍历完整的决策树，对于每个非叶节点，考虑将其变为叶子节点，如果在验证集上的准确率不会降低，则设置为叶子节点

2. C4.5相比于ID3的优点是什么，C4.5又可能有什么缺点？ 

   ID3只考虑信息增益作为选择特征作为决策点的判例，在某个特征可选的特征值很多时，往往会使得信息增益更高。换句话说，信息增益偏向取值较多的特征，从而使得决策树产生大量分支。C4.5考虑到了特定特征下的数据集信息熵，从而解决了ID3的这一缺陷。但是C4.5需要对数据进行多次的扫描和计算，效率较低，只适合小规模数据集。并且，ID3和C4.5都只能处理离散数据而不能处理连续性数据。

3. 如何用决策树来进行特征选择（判断特征的重要性）？

   采用不同的测试属性及其先后顺序将会生成不同的决策树。选择某些特征可能可以很快地确定样本标签，而选择某些特征可能使得判断相当艰难，需要更多的特征信息。所以，选取的样本应该要足够有效，有足够明显的确定样本的标签。ID3采用的信息增益、C4.5采用的信息增益比，考虑的都是特征能够确定标签的程度，而CART采用的基尼指数可以理解为特征下的标签混乱程度。使得熵减最大的特征，或者说使得我们能够更加确信样本标签的特征，往往是更好的特征。