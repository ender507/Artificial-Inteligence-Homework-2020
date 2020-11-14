import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
def getWordVec():
    wordVec = {}
    with open('glove.6B.100d.txt','r',encoding='utf—8') as glove:
        for eachLine in glove.readlines():
            eachLine = list(eachLine.split())
            for i in range(1, len(eachLine)):
                eachLine[i] = float(eachLine[i])
            wordVec[eachLine[0]] = eachLine[1:]
    return wordVec

def getTrainSet():
    trainSet = list()
    with open('Laptops_Test_Data_phaseB.xml','r',encoding='utf—8') as train:
        for eachLine in train:
            if eachLine[:28] == "            <aspectTerm term":
                term = re.search('term=".*?"', eachLine).span()
                str1 = eachLine[term[0]+6:term[1]-1]
                if str1[-1] == ',' or str1[-1] == '.':
                    str1 = str1[:-1]
                trainSet[-1].append(str1.lower())
            elif eachLine[:14] == '        <text>':
                trainSet.append([eachLine[14:-9].lower()])
    return trainSet

def getTestSet():
    testSet = list()
    with open('laptops-trial.xml','r',encoding='utf—8') as train:
        for eachLine in train:
            if eachLine[:28] == "            <aspectTerm term":
                term = re.search('term=".*?"', eachLine).span()
                str1 = eachLine[term[0]+6:term[1]-1]
                if str1[-1] == ',' or str1[-1] == '.':
                    str1 = str1[:-1]
                testSet[-1].append(str1.lower())
            elif eachLine[:14] == '        <text>':
                testSet.append([eachLine[14:-9].lower()])
    return testSet

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)#RNN的思想，combined=本次输入+上次的输出
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def train(sample):
    hidden = rnn.initHidden()
    size = len(sample)
    for eachWord in sample[0].split():
        if eachWord[-1] == ',' or eachWord[-1] == '.':
            eachWord = eachWord[:-1]
        rnn.zero_grad()
        if eachWord in wordVec.keys():
            output, hidden = rnn(torch.Tensor(wordVec[eachWord]).resize(1, 100), hidden)
        else:
            output, hidden = rnn(rnn.initHidden(), hidden)
        ans = 0
        if eachWord in sample[1:]:
            ans = 1
        else:
            for i in range(1, size):
                if eachWord in sample[i].split():
                    if eachWord == sample[i].split()[-1]:
                        ans = 3
                    else:
                        ans =2

        loss = criterion(output, torch.LongTensor([ans]))
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item()

def test(sample):
    rnn.zero_grad()
    size = len(sample)
    print('answer:')
    for i in range(1, size):
        print(sample[i],end=', ')
    print('\noutput:')
    hidden = rnn.initHidden()
    for eachWord in sample[0].split():
        if eachWord[-1] == ',' or eachWord[-1] == '.':
            eachWord = eachWord[:-1]
        if eachWord in wordVec.keys():
            output, hidden = rnn(torch.Tensor(wordVec[eachWord]).resize(1, 100), hidden)
        else:
            output, hidden = rnn(hidden, hidden)
        output = torch.argmax(output.data, 1)
        if output[0] == 1 or output[0] == 3:
            print(eachWord, end=', ')
        elif output[0] == 2:
            print(eachWord, end=' ')
    print()

wordVec = getWordVec()
trainSet = getTrainSet()
testSet = getTestSet()
input_size = 100
hidden_size  = 100
output_size = 4
learning_rate = 0.005
rnn = RNN(input_size, hidden_size, output_size)
criterion = nn.NLLLoss()
for i in range(1, 20):
    loss = 0
    for eachSample in trainSet:
        output,loss2 = train(eachSample)
        loss += loss2
    print(loss)
for eachSample in testSet:
    test(eachSample)