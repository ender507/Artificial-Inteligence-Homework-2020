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

import Datapreprocess as d
#Transforms for train data
transformsTrainData = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
#Transforms for test data 
transformsTestData = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
#Load Data
trainSet = d.Datapreprocess(
    root = 'cifar-10-python',
    train = True,
    transform = transformsTrainData
)

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = 100, shuffle = True, num_workers = 0)

testSet = d.Datapreprocess(
    root = 'cifar-10-python',
    train = False,
    transform = transformsTestData
)

testLoader = torch.utils.data.DataLoader(testSet, batch_size = 100, shuffle = False, num_workers = 0)

#classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#define CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #1st segment
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        #2nd segment
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        #3rd segment
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        #4th segment
        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        #5th segment
        self.conv11 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv12 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv13 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()

        #FC segment
        self.fc14 = nn.Linear(256 * 4 * 4, 1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)
        return x
    #train
    def train_sgd(self, device):
        optimizer = optim.Adam(self.parameters(), lr = 0.0001)
        #save parameters
        path = 'weights.zip'
        initEpoch = 0

        if os.path.exists(path) is not True:
            loss = nn.CrossEntropyLoss()
        else:
            checkPoint = torch.load(path)
            self.load_state_dict(checkPoint['model_state_dict'])
            optimizer.load_state_dict(checkPoint['optimizer_state_dict'])
            initEpoch = checkPoint['epoch']
            loss = checkPoint['loss']
        #run 100 epochs
        for epoch in range(initEpoch, 100):
            startTime = time.time()
            runningLoss = 0.0
            #calculate accuracy of each epoch 
            total = 0
            correct = 0
            for i, data in enumerate(trainLoader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                #forward、backward、optimise
                outputs = self(inputs)
                l = loss(outputs, labels)
                l.backward()
                optimizer.step()

                runningLoss += l.item()
                #print every 500 iteration
                #500 iteration = 1 epoch
                if i % 500 == 499:
                    print('[%d, %5d] loss : %.4f' % (epoch, i, runningLoss / 500))
                    runningLoss = 0.0
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    print('Accuracy of the network on the %d train images: %.3f %%' % (total, 100.0 * correct / total))
                    total = 0
                    correct = 0
                    #save weights
                    torch.save({
                        'epoch':epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':loss
                    }, path)
            print('epoch %d cost %3f sec' % (epoch, time.time() - startTime))
        print("Finished Training")
    #test
    def test(self, device):
        correctClass = np.zeros((10, 10))
        total = 0
        correct = 0
        with torch.no_grad():
            for data in testLoader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                #record each predicted case
                for i in range(100):
                    oriLabel = labels[i]
                    predLabel = predicted[i]
                    correctClass[oriLabel, predLabel] += 1
                    if oriLabel == predLabel:
                        correct += 1
        df = pd.DataFrame(correctClass, index = classes, columns = classes)

        print('Predicted Result\n', df)
        print('Accuracy of the network on the 10000 test images: %.3f %%' % (
                100.0 * correct / total))
        #calculate accuracy of each kind label photo
        for i in range(10):
            print("Accuracy of %5s : %.2f %%" % (classes[i], 100.0 * correctClass[i,i] / np.sum(correctClass[i])))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net = net.to(device)
net.train_sgd(device)
net.test(device)


# In[ ]:




