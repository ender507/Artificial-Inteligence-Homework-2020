# 人工智能实验四 实验报告

| 姓名     |                        |
| -------- | ---------------------- |
| **学号** | **18340057**           |
| **班级** | **18级计算机科学二班** |

## 反向传播神经网络BPNN 

### 1. 算法原理

<img src="pic\\1.png" style="zoom: 50%;" />

​		一般的反向传播神经网络如上图所示，由一个输入层、若干个隐藏层和一个输出层组成。每层有若干个节点，称为神经元。除了输入层外，每层的节点连接着上一层的每个神经元。每个神经元由两部分组成：一个线性变化和一个激活部分。

#### 1.1 前向传播

##### 1.1.1 加权计算（线性变化）

每个神经元的线性变化部分类似于之前实验中的感知机算法。每个神经元连接上一层的全部节点，上一层的每个节点作为该神经元的输入数据。该节点保存着一个权重向量$\bold w=[w_1,w_2,...,w_n,w_0]$，其中$n$表示上一层的节点数，$w_0$代表偏置，对应的输入值恒为1。对于上一层的输入$\bold x = [x_1,x_2,...,x_n,x_0]$，其中$x_0=1$，加权计算的值为两个向量的点乘，即：
$$
h=\sum_{i=1}^nw_ix_i+w_0=\sum_{i=0}^nw_ix_i
$$

##### 1.1.2 激活函数（非线性变化）

考虑到数据集的分布不一定是线性的，而加权求和的计算结果必然是线性结果，因此需要在神经网络中加入非线性变化的部分，从而更好地拟合数据分布。激活函数就是神经网络中的非线性部分。激活函数是一个非线性函数，将之前加权计算得到的结果作为输入，通过激活函数将得到的结果进行输出，从而产生非线性变化。常用的激活函数有：

- sigmoid函数：$f(x)=\frac{1}{1+e^{-x}}$
- softmax函数：$f(x)=\frac{\exp(W_jx)}{\sum_{i=1}^K\exp(W_jx)}$
- ReLU函数：$f(x)=\max(0,x)$
- tanh函数：$f(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$
- Leaky ReLU函数：$f(x)=ax,x<0;f(x)=x,x>0$

选择激活函数需要依据具体的需求，如二元分类常用sigmoid函数，多分类常用softmax函数等。

激活函数的结果作为该节点的输出值，同时作为下一个节点的一个输入值。

#### 1.2 反向传播

依据前向传播的算法，最终能得到样本的一个预测值。预测值和实际值之间的差距需要用一个损失函数来表示。如本次实验使用的欧氏距离的1/2：
$$
E=\frac{1}{2}(\hat y-y)^2
$$
其中，$\hat y$为预测值，$y$为真实值。整个训练集的损失函数值为每个样本的损伤函数值取均值。

为了使得整个神经网络模型更加接近数据分布，需要对模型进行调参，使得损失函数减小。这里我们使用梯度下降算法，即对每个参数$w_i$，更新的公式为：
$$
w_{i(new)}=w_{i(old)}-\eta\frac{\partial E}{\partial w_{i(old)}}
$$
其中，$\eta$为学习率，是自己设置的参数。通过对损失函数用参数进行求导得到损失梯度，应用上述公式，从而使得$w_i$向损失函数的极小值点移动，从而减小损失函数值。

在对损失函数求导的过程中，可以用到求导的链式法则：
$$
\frac{\partial}{\partial z}p(q(z))=\frac{\partial p}{\partial q}\frac{\partial q}{\partial z}
$$
也就是对函数的求导可以化为对函数的子函数求导与对子函数的自变量求导的乘积。链式法则可以继续递归地延伸下去。如果遇上一个自变量在多个并列的子函数出现的情况，将二者分开求导并将结果加和即可。反向传播进行的顺序是从输出层依次往前，直到输入层进行调参，使用链式法则时求导层层递进，公式前面的对子函数求导的部分相同，在较大的神经网络中可以重复利用数据进行计算。

以输出层到最后一层隐藏层的传播为例，最后一层隐藏层的一个参数$w_i$的更新方法为：
$$
\frac{\partial E}{\partial w_i}=\frac{\partial}{\partial w_i}\frac{1}{2}(y-\hat y)^2\\=-(y-\hat y)\frac{\partial \hat y}{\partial w_i}
\\=-(y-\hat y)f'(h)\frac{\partial}{\partial w_i}\sum_iw_ix_i
\\=-(y-\hat y)f'(h)x_i
$$
其中$f(h)$为激活函数。记：
$$
\delta=(y-\hat y)f'(h)
w_i=w_i+\eta\delta x_i
$$
则能从第h+1层的误差算出第h层的误差：
$$
\delta^h_j=\sum_j W_j\delta_k^{h+1}f'(h_j)
$$
每一层参数的更新值为：
$$
w_{ij(new)}=w_{ij(old)}-\Delta w_{ij}=w_{ij(old)}-\eta\delta^h_jx_i
$$
经过多次的正向传播计算预测值和反向传播更新参数后，模型逐渐能够更好地拟合数据分布。

#### 1.3 优化

在本次实验中，我采用了mini-batch梯度下降的方法对训练集进行学习。传统的反向传播神经网络在整个数据集上计算损失函数并使用该函数值进行参数更新，即批梯度下降。而mini-batch下降算法中，将数据集分为多个子部分，每个部分为一个mini-batch。在每个mini-batch上计算出损失函数后，利用该损失函数值进行参数更新。这样在处理大量数据时，有利于参数更快速地更新同时节约内存和计算资源，在内存较小、不能同时训练整个数据集的机器也可以训练模型。

### 2. 流程图和伪代码

<img src="pic\\2.png" alt="2" style="zoom: 67%;" />

再进行样本的训练前，通常需要将数据归一化（标准化），利用标准化后得数据进行数据分析。不同评价指标往往具有不同的量纲和量纲单位，这样的情况会影响到数据分析的结果。为了消除指标之间的量纲影响，需要进行数据标准化处理，以解决数据指标之间的可比性。原始数据经过数据标准化处理后，各指标处于同一数量级，适合进行综合对比评价。

我采用的标准化方法为，将每个样本的特征值减去当前样本的最小特征值，再除以特征值的极差，即：
$$
x_i=\frac{x_i-x_{min}}{x_{max}-x_{min}}
$$
之后开始正向传播。考虑到本次实验的具体实现中只有一层隐藏层，就以该模型编写代码。

```pseudocode
input: x[n], n, hiddenNodeNum, w1[hiddenNodeNum][n], w2[hiddenNodeNum]
/* 输入：特征向量, 特征向量大小, 隐藏层节点数,输出层到隐藏层的权重向量, 隐藏层到输出层的权重向量 */
output: predictValue, hiddenLayer[hiddenNodenum] 				
/* 输出：预测结果、隐藏层结果 */

for i=0 to hiddenNodeNum-1 do	/* 遍历所有隐藏层节点 */
	/* 从输出层到隐藏层的传播 */
	for j=0 to n-1 do			/* 遍历所有特征值 */
		hiddenLayer[i] = hiddenLayer[i] + w1[i][j]*x[j]		/* 加权求和 */
	end
	hiddenLayer[i] = tanh(hiddenLayer[i])					/* 激活函数使用tanh */
	/* 从隐藏层到输出层的传播 */
	predictValue = predictValue + hiddenLayer[i] * w2[i]
end
```

接下来是对权重参数更新值的计算。直接代入上述的公式即可：

```pseudocode
input: predictValue, trueValue, w1[hiddenNodeNum][n], w2[hiddenNodeNum], x[n], n, miniBatchSize, hiddenLayer[hiddenNodenum]
/* 输入：预测值、真实值、权重向量w1和w2、特征向量, 特征向量大小、minibatch大小、隐藏层结果 */
output: w1Gar[hiddenNodeNum][n], w2Gar[hiddenNodeNum]]
/* 输出：w1和w2各个参数的梯度*/

for i=0 to hideNodeCount-1
	for j=0 to n:
		w1Gar[i][j] += ((predictValue - trueValue) * w2[i] * x[j] *
        	(1 - hiddenNodeVal[i]^2)/miniBatchSize
		w2Gar[i] += ((predictValue - trueValue) * hiddenNodeVal[i]) / miniBatchSize
	end
end
```

需要注意的是，tanh的倒数为：$tanh'(x)=1-tanh^2(x)$。上述部分只是计算了当前样本对梯度的影响，并没有应用反向传播。因为采用mini-batch方法进行更新，当样本数达到mini-batch规定数目时才进行参数的更新：

```pseudocode
input: miniBatchSize, hiddenNodeNum, w1[hiddenNodeNum][n], w2[hiddenNodeNum], w1Gar[hiddenNodeNum][n], w2Gar[hiddenNodeNum]], learningRate
/* 输入：minibatch大小、隐藏层节点数、权值向量w1和w2、w1和w2的梯度 */
output: w1[hiddenNodeNum][n], w2[hiddenNodeNum]
/* 输出：更新后的w1和w2 */
if 样本数达到mini-batch个数 then
	for i=0 to hiddenNodeNum-1
		for j=0 to n-1:
			w1[i][j] = w1[i][j] - learningRate * w1Gar[i][j]
		end
		w2[i] = w2[i] - learningRate * w2Gar[i]
	end
endif
```

### 3. 代码展示

#### 3.1 数据预处理

针对每个样本进行数据预处理的代码如下：

```python
        for eachSample in train:			# 遍历每个训练样本
            s = eachSample.split(',')		
            if s[0] == 'instant':			# 除去第一行的说明
                continue
            s[-1] = s[-1][:-1]				# 除去最后的回车符
            # 进行数据类型的转换
            s[0] = int(s[0])
            s[14] = int(s[14])
            for i in range(2, 10):
                s[i] = int(s[i])
            for i in range(10,14):
                s[i] = float(s[i])
            # 对日期作特殊变化
            s = [s[0]]+date2md(s[1])+s[2:-1]+[1]+s[-1:]
            # 进行数据归一化
            s_min = min(s[1:16])
            s_max = max(s[1:16])
            for i in range(1,16):
                s[i] = (s[i]-s_min)/(s_max-s_min)
            # 将结果加入训练集列表
            trainSet.append(s)
```

除了基本的处理外，有两点需要注意：一个是在读入数据时就要进行数据的归一化：每个训练样本的特征值重新赋值为该特征值和该样本中最小特征值的差，除以最大特征值减去最小特征值的商。实现数据归一化有利于一致分析各个特征值，上文已经提及过。

另外，特征值中有一个特殊项：日期，表示方式为字符串`'yyyy/mm/dd'`。经过对整个数据集的分析，我发现所有数据的年份都相同，对预测结果没有影响，因此可以丢弃。而月份和日期可以作为两个特征值加入样本，处理方式如下：

```python
def date2md(date):
    s = date.split('/')					# 拆分成年月日
    return [int(s[1]),int(s[2])]		# 返回月和日的列表
```

至此，对于每一个训练样本`trainSet[i][j]`，下标`j`为0时表示其编号，下标1~15为特征值，16为标签，即真实值

#### 3.2 正向传播

需要说明的是，正向传播和反向传播都被我封装在函数`train`中。在报告中拆开来介绍是为了条理的清晰性。函数原型如下：

```python
def train(trainSet, learningRate, miniBatchSize, times, hideNodeCount)
```

参数列表的参数分别为训练集、学习率、mini-batch大小、训练次数、隐藏层节点个数，返回值为训练好的权重`w1`he `w2`。

首先对`w1`和`w2`进行随机初始化，取值从-0.10到0.10：

```python
    w1 = [[random.randint(-10,10)/100 for i in range(15)] for j in range(hideNodeCount)]
    w2 = [random.randint(-10,10)/100 for i in range(hideNodeCount)]
```

接下来进行`times`次训练。每次训练的正向传播过程如下：

```python
        for eachSample in trainSet:
        	# 初始化中间层和输出层的值为0
            hiddenNodeVal = [0 for i in range(hideNodeCount)]
            predictVal = 0
            # 正向传播
            for i in range(hideNodeCount):
                # 从输入层到隐藏层
                for j in range(15):
                    hiddenNodeVal[i] = hiddenNodeVal[i] + w1[i][j] * eachSample[j+1]
                hiddenNodeVal[i] = math.tanh(hiddenNodeVal[i])		# 激活函数为tanh函数
                # 从隐藏层到输出层
                predictVal = predictVal + hiddenNodeVal[i] * w2[i]
```

各个节点与权重向量相乘得到下一层的节点值，再用激活函数处理。因为结果是从0到600+的数字的回归任务结果，输出层不进行激活，直接以输出层的值为预测结果。

#### 3.3 反向传播

反向传播需要计算`w1`和`w2`中各个参数的梯度，维度大小分别和两个向量相同（实际上`w1`为高维张量）：

```python
    w1Gar = [[0 for i in range(15)] for j in range(hideNodeCount)]
    w2Gar = [0 for i in range(hideNodeCount)]
```

需要注意的是，使用mini-batch算法训练时，一份份地从训练集中取出小部分训练子集，最后可能会剩余很少部分训练样本。可以丢弃这些极少部分样本，或是拿出来单独训练。考虑到小部分样本单独训练可能对模型产生误导，我选择将它们进行了丢弃。这样一来，在重复训练时需要清除这小部分产生的梯度，以免之后代入进行计算：

```python
        for eachSample in trainSet:
            if eachSample[0] == 1:		# 每次重新学习整个训练集时，将梯度归零
                w1Gar = [[0 for i in range(15)] for j in range(hideNodeCount)]
                w2Gar = [0 for i in range(hideNodeCount)]
```

在每次计算完预测值后，计算当前样本带来的梯度变化：

```python
# 计算梯度
for i in range(hideNodeCount):
    for j in range(15):
        w1Gar[i][j] = w1Gar[i][j] + ((predictVal - eachSample[16]) * w2[i] * 
        	eachSample[j+1] * (1 - hiddenNodeVal[i] * hiddenNodeVal[i]))/miniBatchSize
        w2Gar[i] = w2Gar[i] + ((predictVal - eachSample[16]) * hiddenNodeVal[i]) / miniBatchSize
```

直接代入梯度公式即可。因为是mini-batch算法，需要对每个mini-batch中的样本带来的梯度取平均，最后除以`miniBatchSize`即可。

计算梯度后，不要马上更新，而是等到样本数达到`miniBatchSize`后在进行权值的更新：

```python
            # 达到minibatch大小后，进行梯度更新
            if eachSample[0] % miniBatchSize == 0:
                for i in range(hideNodeCount):
                    for j in range(15):
                        w1[i][j] = w1[i][j] - learningRate * w1Gar[i][j]
                    w2[i] = w2[i] - learningRate * w2Gar[i]
```

更新后要对`w1Gar`和`w2Gar`清零，再接着计算下一个mini-batch。

### 4. 采用的优化建议

1. mini-batch的使用，在上文有详细的描述
2. tanh激活函数的使用。其中对tanh函数的求导过程如下：

$$
f(x)=\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}\\
f'(x)=\frac{(e^x+e^{-x})(e^x+e^{-x})-(e^x-e^{-x})(e^x-e^{-x})}{(e^x+e^{-x})^2}\\
=\frac{(e^x+e^{-x})^2-(e^x-e^{-x})^2}{(e^x+e^{-x})^2}\\
=1-(\frac{e^x-e^{-x}}{e^x+e^{-x}})^2\\
=1-f^2(x)
$$

即：
$$
\frac{\partial \tanh(x)}{\partial x}=1-\tanh^2(x)
$$


### 5. 实验结果及分析

在8619个训练样本中，我选取了最后1000个样本作为验证集，前面7619个作为训练集，验证集约占全部样本的11.6%。

在隐藏层节点数为50、学习率为0.001、学习次数为500的前提下，对mini-batch大小进行调参的结果如下：

| `miniBatchSize` | 100     | 500     | 1000    | 2000    | 7000    |
| --------------- | ------- | ------- | ------- | ------- | ------- |
| Loss            | 3763.96 | 7560.55 | 7957.59 | 7638.78 | 6171.71 |

<img src="pic\\4.png" style="zoom:50%;" />

就理论上而言，mini-batch算法的使用主要是为了不用读入全部的训练集，节省计算和内存资源的下策。如果使用全部的训练集进行训练，每次参数更新时必然是在整个训练集梯度下降最快的方向。在训练次数较少的情况下，本身的准确度应当比大批量小。然而在我的实验中出现了小批量反而有着更小的损失函数的情况。考虑到得到的损失函数值都很大，很有可能是模型并未收敛。为了能让模型收敛，理应提高学习率或学习次数。然而在我的训练过程中，500次学习次数造成的时间开销已经非常大了，而学习率之所以设置为0.001这么小是因为在mini-batch大小较大而不是很大的情况下，学习100多次的时候会出现梯度爆炸现象。为了统一参数进行比较我才设置了这么小的学习率。就结果来看，尽管理论上在训练整个训练集是最有效地降低损失函数的方法，但在学习率较小、学习次数有限时，小批量的学习更加有效。之后的数据分析都以整个训练集进行训练。

当隐藏节点数为50，学习率为0.01时，损失函数随着学习次数的变化：

| 学习次数   | 1        | 100     | 200     | 300     | 400     | 500     |
| ---------- | -------- | ------- | ------- | ------- | ------- | ------- |
| 验证集Loss | 13013.47 | 5706.23 | 5963.77 | 5994.87 | 6042.48 | 6089.14 |
| 训练集Loss | 20151.02 | 6007.08 | 5564.82 | 5377.99 | 5258.62 | 5155.07 |

<img src="pic\\3.png" style="zoom:50%;" />

在上述条件下，可以看到，训练集的损失函数值一直在下降，而验证集的损失函数下降到一定程度后反而增长了。可能是出现了过拟合的现象，且样本中存在噪声。训练集上的Loss一直在下降，但明显放缓了，参数的变化已经不大。这是因为模型已经一定程度地拟合了训练集。需要更加贴近需要大量增加训练次数。

在学习率为0.01、学习次数为500时，对隐藏层节点数的测试如下：

| 隐藏层节点数 | 1       | 10      | 50      |
| ------------ | ------- | ------- | ------- |
| 验证集Loss   | 6221.91 | 5726.84 | 6095.22 |
| 训练集Loss   | 9323.01 | 5314.73 | 5121.49 |

理论上，节点数越多，学到的特征也越多，训练集上的损失函数的值也越小。结果基本也符合该规律。

### 6.  思考题

- 尝试说明下其他激活函数的优缺点

  - sigmoid：因为sigmoid函数能将单个输入映射到[0,1]的区间内，所以很容易当做二分类的概率值解释。函数有界、单调、容易求导。但是在输入值较大或较小的情况下，输出值非常相近，经过多层处理很容易出现梯度消失的情况
  - softmax：softmax的输出区间也在[0,1]，但是输入为多个数据，因此很容易当做多分类的概率解释。softmax函数也很容易求导。但设计到多个值的幂运算，在实际深度神经网络的训练和预测过程中可能极大影响效率
  - tanh：tanh和sigmoid类似，[-1,1]的输出范围也适合二分类，同时函数有界、单调、易求导，但也和sigmoid一样存在梯度消失的问题
  - ReLU：函数简单，求导和运算都简单，同时还能防止梯度消失。但因为x<0时激活函数结果都为0，可能导致梯度计算时一直为0，使得权重更新不再变化

- 有什么方法可以实现传递过程中不激活所有节点?

  ​	加权求和的过程在数值计算的角度上说，其实就是对输入值进行一个线性变化；而激活函数是一个非线性函数，从而在整个模型中引入了非线性的变化。如果要不激活节点，则使用线性变化的激活函数即可。所有形如$y=\bold w^T\bold x+b$形式的函数都可以，也可以直接去掉激活函数（相当于激活函数为$y=[1, 1, ..., 1]\cdot\bold x$）

-  梯度消失和梯度爆炸是什么？可以怎么解决？

  - 梯度消失主要是因为使用Sigmoid、tanh等类似的函数作为激活函数，导数值都较小。在计算时如果输入值较大或较小，导数值往往很小，或是很多层的累加使得多个导数值的乘积变得很小，使得梯度几乎降为0。要解决梯度消失，需要选取较好的学习率和激活函数
  - 梯度爆炸主要是权重值较大时产生的，经过多层神经网络的累加，导致梯度中导数的乘积变得很大，使得梯度非常大，参数变化幅度也非常大。需要选取合适的学习率，也可以通过梯度截断（设置梯度阈值，超过阈值的梯度修改为阈值）等方法解决