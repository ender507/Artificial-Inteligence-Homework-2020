# 人工智能实验一 实验报告

18340057	18级计算机科学二班

## 一、 TF-IDF矩阵表示

### 1. 算法原理

计算TF-IDF矩阵，需要先分别计算出TF矩阵和IDF向量。

TF矩阵为词频归一化后的概率表示，公式为:
$$
tf_{i,d} = \frac{n_{i,d}}{\sum_vn_{v,d}}
$$
其中，`d`为文档编号，`i`为文档中的某个单词。分子中$n_{i,d}$表示文档`d`中单词`i`出现的次数，分母对文档`d`中的单词进行求和，即该文档中的单词总数。简单来说，$tf_{i,d}$表示文档`d`中单词`i`出现的次数除以该文档中的单词总数。TF矩阵表示了某个文档中的特定单词的权重。

IDF向量为逆向文档频率，公式为：
$$
idf_i = \log\frac{|C|}{|C_i|}
$$
其中，`i`为某个单词，$C_i$为单词`i`在多少篇文档中出现了，而`C`为文档总数。IDF向量衡量了每个单词在所有文档中出现的频率，能度量该词语的普遍重要性。

为了同时考虑到单词在某篇文档和全部文档中的重要性，将TF和IDF合并，产生了TF-IDF矩阵。其公式如下：
$$
tf\_idf_{i,j} = \frac{n_{i,j}}{\sum_{k=1}^Vn_{i,k}} * \lg\frac{|D|}{|D_i|+1}
$$
IDF的部分中，分母要加一是为了防止出现分母等于零的情况。简单来说，就是将IDF的值作为权重乘到了TF中。IDF中单词`i`对应的值，乘到TF中单词`i`表示的每一项中。

依据上述公式和原理计算TF、IDF，最后就能算出TF-IDF。

-------

### 2. 伪代码

首先需要计算TF。考虑到每个文档的内容是一句话，单独占一行，程序可以每次读取一行，再对TF进行维护。

```pseudocode
tf = {}			
/*	 
	tf为字典，键值为单词，包含全部的单词。数值为列表，位置i表示文档i的tf值
	例如：tf['hello'][12]，表示单词hello在12号文档对应的tf值
*/

/* 每次只读取一个文档，即一行 */
for eachLine in txt
	/* 遍历该句的所有单词 */
    for eachWord in eachLine
		if tf的键值中不包括eachWord
        	then 将eachWord加入tf的键值，对应的数值列表全0，列表大小为文档数        
        tf[eachWord][当前文档编号] += 1/当前文档长度
    end
end
```

接着计算IDF。IDF的建立需要汇总全部文档的信息，通过一行行读入难以建立。可以由包含了全部文档信息的TF矩阵来建立。

```pseudocode
idf = {}
/* idf为字典，键值为单词，数值为该单词对应的idf值 */
for eachWord in tf内的全部单词
	if idf的键值不包括eachWord
		then 在idf的键值中加入eachWord，对应的数值为1		/* 对应公式中分母中的+1 */
    for tf中的全部文档
    	if 该文档中出现了eachWord
    		then idf[eachWord]+=1					/* 对应公式中的分母 */
   	end
end
for eachWord in idf内的全部单词
	idf[eachWord] = log10(文档总数/idf[eachWord])	/* 对应公式的完整结果 */
end
```

计算了TF和IDF后，就可以算出要求的TF-IDF。

```pseudocode
tf_idf = {}
/* tf_idf的索引方式和tf相同，为tf_idf[word][document] */
for eachKey in idf内的全部单词
	for 每一篇文档
		tf_idf[eachKey][当前文档] = tf[eachKey][当前文档] * idf[eachKey]
	end
end
```

-------

### 3. 代码展示 

#### 3.1 TF矩阵的构造

首先建立`idf`字典。

```python
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
```

打开文件`semeval.txt`后，首先统计文件行数并存入变量`sentenceCount`，紧接着的`text.seek(0,0)`是为了复位指针，使得之后的文件遍历能正常进行。`num`为文档编号。

接着依次遍历文档的每一行。每一行有三个部分：编号（因为文档中的编号不连续，会影响结果，所以我使用了变量`num`来表示编号）、心情、文档，三个部分用`\t`分开，所以使用`split`拆解，用`[2]`取第三个部分，即文档。最后的`[0:-1]`去掉文档的最后一个字符，即回车符，避免影响分析。

接着调用函数`makeTF`，用当前遍历到的文档来更新`tf`字典。最后`num+=1`表示下一个文档。

`makeTF`函数的实现如下：

```python
def makeTF(sentence, tf, sentenceCount, num):
    words = sentence.split(' ')
    # 把句子拆分成单词
    for eachWord in words:
        # 若字典中不存在该单词，加入字典
        if eachWord not in tf.keys():
            tf[eachWord] = [0.0 for x in range(0, sentenceCount)]
        tf[eachWord][num] = tf[eachWord][num] + 1.0 / len(words)
```

将句子拆分成单词，依次遍历各个单词，若有新单词则加入到`tf`中。

对于出现的单词，已经确定了单词和文档编号，于是更新`tf`对应位置的值，加上`1/len(words)`。如果单词多次出现，会多次加上该值，从而保证了结果的正确性。

在构造`idf`前，还需要将`tf`的键值按照字母顺序排序。代码如下：

```python
# 对tf按照键值进行排序
tfList = sorted(tf.items(), key=lambda item: item[0])
tf.clear()
for i in range(len(tfList)):
    tf[tfList[i][0]] = tfList[i][1]
```

通过`sorted`函数传入`tf`的键值对，依据`item[0]`，即键值进行排序。返回的结果存入`tfList`中，为列表。列表中为按顺序排好的元组，元组的第一个值为原先`tf`的键值，第二个值为键值对应的`tf`原先的数值。将`tf`清空后，按键值对的对应关系依次赋值回`tf`即可。

当生成了正确的`tf`后，就能使用`tf`来构造`idf`。

#### 3.2 IDF向量的构造

```python
def makeIDF(tf, idf, sentenceCount):
    for eachKey in tf.keys():
        if eachKey not in idf.keys():
            idf[eachKey] = 1.0    # 依据公式log(|D|/|Di|+1)分母要加上1，所以初始化为1
        # 统计每个单词出现的次数
        for i in range(0,sentenceCount):
            if tf[eachKey][i] != 0:
                idf[eachKey] += 1.0
    # 到此处，变量idf记录的值是|Di|+1.0
    # 代入公式
    for eachKey in idf.keys():
        idf[eachKey] = math.log(sentenceCount/idf[eachKey],10)
```

依次遍历`tf`的各个单词，并加入`idf`的单词表中。每个单词对应的值初始化为1，因为计算公式中$\lg\frac{|D|}{|Di|+1}$的分母为1。之后对特定的单词，看其是否在`tf`的各篇文档中出现过。出现过则再加一。这里计算了公式中的|Di|+1部分。又因为文档数|D|值就是`sentenceCount` 的变量值，代入公式算出最终的`idf`。

#### 3.3 TF-IDF矩阵的计算和文件输出

有了`idf`和`tf`后就能计算出`tf_idf`

```python
# 用idf的数据更新tf内的值，从而生成tf_idf矩阵
for eachKey in idf.keys():
    for i in range(0, sentenceCount):
        tf[eachKey][i] *= idf[eachKey]  # 直接用原来的变量tf存储tf_idf的数据
```

为了编程简便和节省空间，我在代码里直接用变量`tf`记录`tf_idf`的值。函数返回后传递的参数再重新命名为`tf_idf`。

最后输出结果，用到以下函数：

```python
def writeTF_IDF(tf_idf, inputEnd):
    with open("18340057_HuTingxi_TFIDF.txt","w") as tf_idfFile:
        for i in range(0,inputEnd):
            for eachKey in tf_idf.keys():
                tf_idfFile.write(str(tf_idf[eachKey][i]))
                tf_idfFile.write(' ')
            tf_idfFile.write('\n')
```

两层循环先遍历文档再遍历单词。也就是说，输出结果的每一行表示一个文档中各个单词的`tf_idf`值。各个值用空格隔开。

--------

### 4. 实验结果及分析

为了便于审查实验结果，我简单的将上述的`writeTF_IDF`函数中的输出文本后缀名`txt`改为`csv`，将间隔符号空格`' '`改为逗号`','`，输出结果的部分内容如下：

<img src="pic\\1.png" alt="1" style="zoom:50%;" />

这只是整个`tf_idf`矩阵的左上角的一部分。按照字典序排列单词，第一列表示单词`a`。可以看到，第6行（对应文档编号为5）和第19行（对应文档编号为18）的值不为0。查阅`semeval.txt`文件发现这两句分别为`pm havana deal a good experi`和`we re a pretti kind bulli`，两句都出现了单词`a`且两句总的单词数相同，都为6。这证明了矩阵位置的正确性。借助统计工具可以查出，1246文档中有74篇文档含有单词`a` 。$\frac{1}{6}*\lg\frac{1246}{1+74}=0.20340946...$和结果数值0.20341一致，这证明了矩阵数值的正确性。

完整的文件附在作业压缩包中。（按照实验要求，我上交的文件是以空格分隔的`txt`文件）

## 二、 KNN分类任务

### 1. 算法原理

首先对训练集进行学习。在本题中，训练集的每个数据由两部分组成：文档和标签。文档是一句话，含有多个单词，标签是一个描述情绪的单词，是需要用文档预测的结果。使用KNN进行分类的基本过程是：将文档进行编码，统计文档中的单词。在测试时比较测试样本和训练样本的距离，选取若干个距离最近的训练样本，由这几个训练样本的标签来判断测试样本的标签。具体算法如下：

首先使用TF-IDF矩阵形式编码文档，具体做法参见上文。

同样，读入测试集后，每个测试集样本也用TF-IDF编码表述为向量形式。需要注意的是测试集在计算TF-IDF矩阵时使用的IDF向量应该是通过训练集的出来的IDF，从而保证每个单词在整个文档中的重要程度的一致。将测试样本和训练集样本一一比较，并且选择距离最小的几个。具体选择几个记为变量K，K的取值不同会对模型产生影响，这会在之后的实验结果分析部分讨论。描述训练集和测试集样本距离的方式是Lp距离：
$$
\displaystyle Lp(x_i,x_j)=(\sum^n_{l=1}|x_i^{(l)}-x_j^{(l)}|^p)^{\frac{1}{p}}
$$
该公式表示向量$x_i$和$x_j$之间的Lp距离，公式内的`l`表示向量的维度。将两个向量各个维度对应的数值相减后求q次方，再把各个维度的结果加和，最后再开p次方根，就得到了Lp距离的结果。这里的`p`也是一个可以调整大小的变量，取值在之后的实验结果分析进行讨论。

在找出了K个和测试样本最近的训练样本后，由训练样本的标签决定测试样本的标签。这里采用多数投票的方法，即选取K个训练样本中出现次数最多的那个标签作为结果。

-----

### 2. 伪代码

首先要将训练集进行TF-IDF编码。具体做法和上文一样。

在对文档进行编码的同时，也要将文档对应的标签给记录下来：

```pseudocode
emt = {}
/* emt即emotion，记录情绪标签。emt为字典，如emt[5]='sad'表示5号文档的标签为sad */
for eachEmt in 全部文档:
	emt[当前文档编号] = eachEmt
end
```

这样一来，就能记录各个文档的TF-IDF编码以及对应的标签`emt`了。

在测试时，将测试样本进行TF-IDF编码可以不需要读入整个训练集再计算。因为使用的IDF向量已经由训练集得到，每次读取一行，即一个测试样本，直接对其进行距离运算即可。

```pseudocode
 for i in 全部的训练样本			/* 和全部的训练样本进行比较 */
    diff = 0.0					/* diff表示样本距离 */
   	for eachWord in tf_idf:		/* 检索每个单词，如果在当前训练样本或测试样本中出现则更新diff */
    	if eachWord出现在测试样本或当前训练样本中	
    		then diff += 因eachWord增加的Lp距离项
    end
    diff = diff ^ (1/p)
end
```

这里我假设所有在测试样本中出现的单词都在字典中。如果不是，那些不在字典中的单词没有经过学习，不能作为判断标签的依据，依次不纳入`diff`的统计。

得到各个训练样本和当前测试样本之间的距离后，要判断哪些训练样本需要留下，当做判断测试样本标签的依据。实现如下：

```pseudocode
simSentence = {}
/*
simSentence表示和测试样本最相近的K个训练样本，记录方式为simSentence[i]=diff
即第i句和当前测试样本的距离为diff
*/
for i in 全部的训练样本
	if simSentence大小不到k
		then 将i加入simSentence
/* 否则进行判断，去掉diff最大的文档并加入当前文档 */
		else if 如果i的距离小于simSentence中的距离最大值:
			then 删去simSentence距离最大的样本，将i加入simSentence 
end
```

这样就能得到和当前测试样本最相近的K个训练样本。只要统计这些样本的标签，将数量最多的标签作为分类标签即可。

```pseudocode
for sentence in SimSentence
	sentence的标签票数+1
end
return 票数最多的标签
```

----

###  3. 代码展示 

#### 3.1 训练样本的处理

首先生成TF-IDF矩阵。代码和第一题几乎一致，在报告中不再赘述。需要注意的是，除了`tf_idf`矩阵的计算和样本大小`sentenceCount`的统计外，还需要返回各个样本的情绪标签的统计`emt`和TF-IDF矩阵计算的中间量`idf`矩阵。

```python
tf_idf, idf, emt, sentenceCount = summonTF_IDF()
```

`emt`的计算只需要在每次读取一个训练样本时调用以下函数即可：

```python
def makeEMT(emt, num, emotion):
    emt[num] = emotion
```

其中，`num`为训练样本编号，`emotion`为该训练样本的标签。

#### 3.2 测试样本的预测

之后开始进行测试。这里我以验证集为例展示代码。函数原型和初始化的变量如下：

```python
def valid(tf_idf, idf, emt,sentenceCount):
    all = 0
    right = 0
    simSentence = {}  # 相似的句子。simSentence[i] = j ,i为句子编号，j为差异diff
```

传入的参数名和之前的声明一致。`all`和`right`分别用来统计训练集（这里是验证集）的总个数和预测正确的个数，用于计算最后的正确率。`simSentence`和之前讨论的一样，用来记录K个和训练样本最相近的训练样本。

对验证集的预处理如下：

```python
    with open("validation_set.csv", "r") as testSet:
        for eachLine in testSet:
            s = eachLine.split(',')
            ans = s[1][0:-1]                # 标签的实际结果
            sentence = s[0].split(' ')      # 文档
            if ans == 'label':              # 去掉第一行的说明
                continue  
```

通过逗号`,`将每一行分开，`ans`为验证集的实际标签，`sentence`为被拆分好的文档，是包含当前文档中所有单词的列表。

接着调用以下三个函数，`findSimSentence`用于找到最相近的K个训练样本，`emotionCount`用来统计这些训练样本的标签，而`findMostEmt`用于在统计好的标签中找到出现次数最多的标签。这三个函数在之后会有详细说明。

```python
 			findSimSentence(sentenceCount, tf_idf, idf, sentence ,simSentence, K)
            emotionCount = emtCount(simSentence, emt)
            mostEmt = findMostEmt(emotionCount, sentence, simSentence, sentenceCount, tf_idf, idf, K)
```

最后进行预测标签和实际标签的比较，并**输出当前的正确率**（之后的参数调优会使用到）。

```python
            all += 1
    		if mostEmt == ans:
                right += 1
            print(all, right / all)		# 输出当前已经验证到第几个样本，以及当前的正确率
```

#### 3.3 测试时调用的三个函数的具体说明

首先是`findSimSentence`函数，用来找到K个与当前测试样本（即参数列表中的`sentence`）最相近的`k`个训练样本。

 ```python
def findSimSentence(sentenceCount, tf_idf, idf, sentence ,simSentence, k):
 ```

要找出最相近的训练样本，必须先要计算每个训练样本和测试样本之间的差距，即Lp距离。

```python
    simSentence.clear()
    for i in range(0, sentenceCount):			# 遍历每个训练样本，计算距离
        diff = 0.0								# diff用来计算Lp距离
        for eachWord in tf_idf.keys():			#查找词典中每一个单词
            if tf_idf[eachWord][i] != 0 or (eachWord in sentence):	# 该单词在任一样本出现，则更新diff
                diff += abs(tf_idf[eachWord][i]-sentence.count(eachWord)*idf[eachWord]/len(sentence)) ** p
        # diff = pow(diff, 1.0 / p)             # 开根号与否不会影响距离大小比较
```

每次取一个训练样本和测试样本比较。比较时遍历所有单词，只要该单词出现在训练样本中或测试样本中，则需要更新距离。此处加上的距离为Lp距离根号下的一项，即$(x_i^{(l)}-x_j^{(l)})^p$对某个特定的`l`的部分。因为`p`为常数，最后的开根号不影响距离的单调性，所以可以不开，减少一定的计算量。

得到了当前训练样本和测试样本的距离后，要判断该训练样本是否能作为判断测试样本标签的依据，即距离是不是足够小。

```python
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
```

如果还没找满K个训练样本，直接j加入`simSentence`即可。而如果已经找了K个训练样本，则比较这些样本中距离最大的和当前训练样本的距离，取更小的那个加入`simSentence`中。

找到所有K个训练样本后，对这些样本的标签进行统计：

```python
def emtCount(simSentence, emt):
    emotionCount = {}
    for eachSentence in simSentence:
        if emt[eachSentence] not in emotionCount:
            emotionCount[emt[eachSentence]] = 1
        else:
            emotionCount[emt[eachSentence]] += 1
    return emotionCount
```

这个函数的逻辑很简单：`emotionCount`记录各个情绪的个数，如:`emotionCount['joy']=2`。遍历K个样本的标签，若该标签不在`emotionCount`中，则加入并初始化为1次，若在则次数加一。最后返回`emotionCount`

统计了所有的情绪标签后，通过下面的函数找出出现次数最多的标签：

```python
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
```

本来此处的逻辑很简单，在`emotionCount`中找到数值最大的那一项，返回其键值即可。但是有可能会出现这种情况：在所有的标签中，出现次数最多的标签不止一个。换句话说，就是K个样本的投票出现了票数相同的情况。这时我采用了改变k值的方法，重新调用上述统计相似训练样本的函数，并重新找出现次数最多的标签。

将测试结果输出为文件不涉及相关算法，基本语法和第一题展示的输出文件的代码相同，在报告中不再赘述。

----

### 4. 创新点

在本题中，测试样本最终标签的确定是用最相近的K个训练样本投票得出的，那就很有可能会出现**票数相同**的情况。针对这一点，我加入了部分细节上的改进。具体代码见上面的`findMostEmt`函数。

试想，当两个标签（也可以是多个），比如`joy`和`sad`的票数相同且二者都为票数最多的标签时，基本就能确定最后结果的标签就是这几者中的一个。需要最终选定一个就要判断孰优孰劣。鉴于二者票数相同，能够比较的就只有这几个标签对应的样本和测试样本的距离。比如标签为`joy`的训练样本和测试样本之间的距离总的来说小于标签为`sad`样本，此时选择`joy`作为最后的预测结果即可。

这种想法简单的说就是“关注距离更近的样本、忽视距离更远的样本”。通过减小K值，重新计算最近的几个样本即可。若减小K值还是有票数相同的情况，则继续减小`K`值，直到有唯一且票数最多的标签为止。也就是说，整个测试过程中有两个`K`值：一个是全局的`K`值（大写表述），是依据KNN算法一开始预设好的`K`值；另一个是局部的`k`值（小写表示），初始化为全局的`K`值，但如果针对当前的测试样本出现了票数相同的情况，则减一。还是相同则继续减一。当`k=1`时必然不会出现票数相同的情况，故保证了该方法不会得不出结果。

具体代码的实现体现在`mostEmt`的变量值上。该变量初始化为`None`。如果找到票数最多的标签则进行赋值：

```python
    mostEmt = None
    for eachEmt in emotionCount.keys():
        # 若当前遍历到的情绪标签是最多的，记录下来
        if emotionCount[eachEmt] == max(emotionCount.values()):
			if mostEmt == None:
                mostEmt = eachEmt
```

如果找到票数最多的标签时`mostEmt`不为`None`，说明出现了多个票数相同且最多的标签。此时减小`k`值并重新调用统计`simSentence`的函数即可。重新找最多票数的标签是一个递归过程，直到找到为止。

```python
        # 否则，有多个情绪标签的数量相同，则减少k值并继续计算
            else:
                findSimSentence(sentenceCount, tf_idf, idf, sentence, simSentence, k-1)
                emotionCount = emtCount(simSentence, emt)
        # 递归调用自己，找到数量最多的情绪
                mostEmt = findMostEmt(emotionCount, sentence, simSentence, sentenceCount, tf_idf, idf, k-1)
                break
```

-------

### 5. 实验结果及分析

本题含有两个参数：一个是KNN算法中的相近样本数`K`，一个是Lp距离计算中的参数`p`。以验证集为测试对象，对二者调优的结果如下：

首先是对`K`值的调优。其中准确率为预测正确的样本个数/验证样本总个数，对`K`调优时使用的`p=2`。

| K值    | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      | 9      | 10     |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 准确率 | 0.4148 | 0.4148 | 0.3923 | 0.4148 | 0.4084 | 0.4116 | 0.4341 | 0.4437 | 0.4373 | 0.4277 |

| K值    | 11     | 12     | 13     | 14     | 15     | 16     | 17     | 18     | 19     | 20     |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 准确率 | 0.4148 | 0.4051 | 0.3987 | 0.3891 | 0.3959 | 0.3826 | 0.3826 | 0.3794 | 0.3859 | 0.3667 |

<img src="pic\\2.png" alt="2" style="zoom:50%;" />

我的模型在`K`的取值不同时，各个准确度的差距不大，但也有一定的趋势：在`K`取值1~8时准确率总体上升，之后下降。准确率最大时`K=8`。在`K`值较小时，决定预测结果的样本太少，参考价值不大，产生欠拟合的问题。而在`K`值较大时，用于预测的样本太多，容易受到距离较远的样本的影响，产生过拟合。适中的`K`值才能产生较好的结果。

现在取`K=8`，对`p`进行调优：

| p值    | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      | 9      | 10     | ∞      |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 准确率 | 0.4135 | 0.4437 | 0.4199 | 0.4115 | 0.4135 | 0.4116 | 0.4103 | 0.4084 | 0.4199 | 0.4212 | 0.3526 |

<img src="pic\\3.png" alt="3" style="zoom:50%;" />

其中，求`p=∞`时的结果需要将之前代码计算`diff`的步骤改为：

```python
diff = max(diff,abs((tf_idf[eachWord][i] - sentence.count(eachWord)*idf[eachWord])))
```

可以看出，`p=2`时准确率要高一些。

对测试集进行测试的部分输出结果如下：

<img src="pic\\4.png" alt="4" style="zoom: 67%;" />

各种标签都能正常输出，标签分布没有明显的问题。全部的结果见`result`文件夹下的文件。

## 三、 KNN回归任务

### 1. 算法原理

在本题中，一共有六项数值需要进行回归预测，anger, disgust, fear, joy, sad, surprise的数值作为样本的结果。同样，首先将训练样本进行TF-IDF编码后进行预测。并依据上一题中详细解释过的KNN算法和Lp算法得到K个与测试样本距离最近的训练样本。但是与分类任务不同的是，回归任务不能通过投票预测结果，而是需要另外的算法得到回归结果：

假设通过KNN算法得出了K个与测试样本最相近的训练样本，分别记为$train_1,train_2,...,trian_K$，用$d(train_1,test)$表示$train_1$和测试样本之间的Lp距离，$prob$表示某个回归值，则有：
$$
\displaystyle prob(test)=\sum_{k=1}^K\frac{prob(train_k)}{d(train_k,test)}
$$


即测试样本的某项回归值等于`K`个训练样本的回归值除以该训练样本与测试样本之间的Lp距离之商的和。

这样得出来的结果之和可能不为1，而依据题目要求，六种概率之和必须为1，因此可以对六种数据做以下处理：
$$
\displaystyle porb_i=\frac{prob_i}{\sum^6_{j=1}prob_j}
$$
也就是每个概率都除以六个概率之和，本质上是对六种概率进行相同的线性变化，使得六者的和相加为1。

与分类问题不同的是，回归问题的预测结果不可能和实际结果完全准确，为了使得结果更加精确，引入下面的指标相关系数作为判断预测结果和实际结果差距的依据。
$$
COR(X,Y)=\frac{cov(X,Y)}{\sigma_X\sigma_Y}=\frac{\sum^n_{i=1}(X_i-\overline X)(Y_i-\overline Y)}{\sqrt{\sum^n_{i=1}(X_i-\overline X)^2\sum^n_{i=1}(Y_i-\overline Y)^2}}
$$
本题中的结果有六个概率值。先分别计算六个维度上的真实概率值和预测概率值的相关系数，然后对六个维度取平均，计算得到最终相关系数作为判断依据。相关系数的绝对值越大，预测值和真实值的线性相关程度就越好。

-----

### 2. 伪代码

首先创建TF-IDF矩阵，并且通过KNN算法和Lp距离找出和测试样本最近的`K`个训练样本。和之前的一样，不再重复说明。

找到`K`个训练样本后，根据这些样本的标签对测试样本的回归值进行预测：

```pseudocode
for 每种要预测的概率 i 
	i = 0
    for eachSentence in KNN最近邻
    	i = i + eachSentence的i值 / eachSentence的Lp距离
    end
end
```

得到了每个要预测的回归值后，还需要将这些值归一化，即使得这些概率的和为1。

```pseudocode
每种概率值 = 每种概率值 / 所有概率值之和
```

只要将每种概率值除以所有概率之和，这时所有概率相加为：原来的所有概率之和/原来的所有概率之和=1。

------

### 3. 代码展示

#### 3.1 测试流程

生成TF-IDF的矩阵的代码和上述内容相同，不再重复展示。

之后开始测试，以验证集为例。声明下面的函数并进行相关数据的初始化：

```python
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
```

接下来打开验证集文件。

```python
    with open("validation_set.csv","r")as validSet:
        for eachLine in validSet:
            s = eachLine.split(',')				# 依据逗号拆分每个验证样本
            sentence = s[0].split(' ')			# 验证样本的文档
            s[-1] = s[-1][0:-1]					# 将最后的回车去掉
            emotion = s[1:7]					# 文档对应的6种概率
            if emotion[0] == 'anger':			# 排除测试集的第一行
                continue
            for i in range(0, 6):
                emotion[i] = float(emotion[i])	# 将概率大小由字符串转为浮点数
```

得到了文档对应的实际的六种概率则可以加入上述列表中：

```python
            trueAnger.append(emotion[0])
            trueDisgust.append(emotion[1])
            trueFear.append(emotion[2])
            trueJoy.append(emotion[3])
            trueSad.append((emotion[4]))
            trueSurprise.append(emotion[5])
```

接着是对`K`个最近邻的查找，通过`findSimSentence`函数完成。得到后调用`predict`函数得到六种预测的概率，再调用`standard`函数将六个概率标准化，即使得六种概率之和为1。这几个函数的详细代码会在之后讨论。其中`findSimSentence`和第二题完全相同，不再展示代码。需要注意的是，为了防止之后出现计算概率时除以0的情况，距离`diff`需要设置最小值。

```python
            simSentence = {}
            findSimSentence(sentenceCount, tf_idf, idf, sentence ,simSentence, K)	#查找最近邻
            predictEmt = predict(simSentence, emt)		# 依据最近邻预测概率
            standard(predictEmt)						# 将概率标准化
```

得到了预测的六种概率之后，也分别加入对应的列表中。

```python
            predictAnger.append(predictEmt[0])
            predictDisgust.append(predictEmt[1])
            predictFear.append(predictEmt[2])
            predictJoy.append((predictEmt[3]))
            predictSad.append(predictEmt[4])
            predictSurprise.append(predictEmt[5])
```

有了所有验证样本的实际结果和预测结果组成的六个向量（列表）后，调用`cor`函数计算六个相关系数，并求平均值输出。

```python
print((cor(predictAnger,trueAnger)+cor(predictDisgust,trueDisgust)+cor(predictFear,trueFear)+cor(predictJoy,trueJoy)+cor(predictSad,trueSad)+cor(predictSurprise,trueSurprise))/6)
```

#### 3.2 概率值的计算

计算出K个最近邻后，调用以下函数计算六种概率的预测值：

```python
def predict(simSentence, emt):
    predictEmt = [0 for i in range(0,6)]		# 大小为6的向量，分别对应六种情绪的预测概率值
    for i in range(0,6):
        for eachSentence in simSentence.keys():
            # 概率预测值 = 所有最近邻的概率/Lp距离之和
            predictEmt[i] = predictEmt[i] + emt[eachSentence][i]/simSentence[eachSentence]
    return predictEmt
```

得到六种概率的预测值后，调用下面的函数将其标准化：

```python
def standard(predictEmt):
    total = sum(predictEmt)
    for i in range(0,6):
        predictEmt[i] /= total
```

求得所有概率值之和，并将所有概率除以该值即可。

#### 3.3 相关系数的计算

```python
def cor(x, y):
    avgX = sum(x)/len(x)			# x的平均值
    avgY = sum(y)/len(y)			# y的平均值
    tmp1 = 0.0						# tmp1用于计算相关系数的分母
    tmp2 = 0.0						# tmp2用于计算x的标准差
    tmp3 = 0.0						# tmp3用于计算y的标准差
    for i in range(0,len(x)):
        # 依据公式计算变量值
        tmp1 = tmp1 + (x[i]-avgX)*(y[i]-avgY)
        tmp2 = tmp2 + (x[i]-avgX)*(x[i]-avgX)
        tmp3 = tmp3 + (y[i]-avgY)*(y[i]-avgY)
    tmp1 = abs(tmp1)
    return tmp1/math.sqrt(tmp2*tmp3)
```

-----

### 4. 实验结果及分析

同样对`K`和`p`进行调整，具体结果如下：

当`p=2`时，对K调优：

| K值      | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      | 9      | 10     |
| -------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 相关系数 | 0.3070 | 0.3445 | 0.3544 | 0.3562 | 0.3540 | 0.3709 | 0.3621 | 0.3535 | 0.3472 | 0.3516 |

| K值      | 11     | 12     | 13     | 14     | 15     | 16     | 17     | 18     | 19     | 20     |
| -------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 相关系数 | 0.3402 | 0.3316 | 0.3214 | 0.3228 | 0.3156 | 0.3030 | 0.3004 | 0.2919 | 0.2950 | 0.2996 |

<img src="pic\\5.png" alt="5" style="zoom:50%;" />

可以看出，在`K`取6的时候相关系数达到最大值。

当`K=6`时，对`p`进行调优：

| p值      | 1      | 2      | 4      | 6      | 8      | 10     | 12     | 14     | 16     | ∞      |
| -------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 相关系数 | 0.3100 | 0.3709 | 0.3739 | 0.3827 | 0.3866 | 0.3883 | 0.3912 | 0.3907 | 0.3856 | 0.1916 |

<img src="pic\\6.png" alt="6" style="zoom:50%;" />

`p`取12的时候相关系数最大。

输出的结果的部分如下：

<img src="pic\\7.png" alt="7" style="zoom: 67%;" />

可以看到，数据都在0到1之间且6个概率之和为1。

------

### 5. 思考题

1. 为什么计算回归值时使用距离的倒数？

   因为距离越大，则当前训练样本与测试样本的相似性越小，需要相应地减小当前训练样本对预测结果的影响。取倒数的话，距离越大，倒数值越小，从而计算时对结果的影响越小。

2. 如果要求得到的每一种标签的概率的和等于1，应该怎么处理？

   将每个标签值除以所有标签值的总和。