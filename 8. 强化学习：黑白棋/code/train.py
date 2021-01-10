import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

score_map = [
    [100, -35, 10, 5, 5, 10, -35, 100], 
    [-35, -35, 2, 2, 2, 2, -35, -35], 
    [10, 2, 5, 1, 1, 5, 2, 10], 
    [5, 2, 1, 2, 2, 1, 2, 5],
    [5, 2, 1, 2, 2, 1, 2, 5],
    [10, 2, 5, 1, 1, 5, 2, 10], 
    [-35, -35, 2, 2, 2, 2, -35, -35],
    [100, -35, 10, 5, 5, 10, -35, 100]
]


class QL(nn.Module):
    def __init__(self):
        super(QL, self).__init__()
        self.fc0 = nn.Linear(65, 128)
        self.relu0 = nn.ReLU()
        self.fc1 = nn.Linear(128, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(256, 64)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = self.relu0(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x

    def train(self, inputs, labels):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss = nn.MSELoss()
        outputs = self(inputs)
        l = loss(outputs, torch.FloatTensor([labels]))
        l.backward()
        optimizer.step()

def nextPos(r, c, i, dr, dc, chesses, color):
    if r+dr[i]<8 and r+dr[i]>=0 and c+dc[i]<8 and c+dc[i] >= 0:
        r += dr[i]
        c += dc[i]
        if chesses[r][c] == -color:
            return nextPos(r, c, i, dr, dc, chesses, color)
        if chesses[r][c] == color:
            return True
    return False

def isValidPos(x , y, color, chesses):
    if chesses[x][y] != 0:
        return False
    dr = [0,   1, 1, 1, 0, -1, -1, -1]
    dc = [-1, -1, 0, 1, 1,  1,  0, -1]
    ans = False
    for i in range(8): # 表示8个方向
        if x+dr[i]<8 and x+dr[i]>=0 and y+dc[i]<8 and y+dc[i]>=0:
            r = x + dr[i]
            c = y + dc[i]
            if chesses[r][c] == -color:
                ans = ans or nextPos(r, c, i, dr, dc, chesses, color)
        if ans:
            return ans
    return ans

def getNextStepPos(chesses, color):
    pos = []
    for i in range(8):
        for j in range(8):
            if isValidPos(i ,j, color, chesses):
                pos.append((i,j))
    return pos

def evaluate(chesses, color):
    score = 0
    for i in range(8):
        for j in range(8):
            if chesses[i][j] == color:
                score += score_map[i][j]
    return score

def putChess(chesses, p, color):
    dr = [0, 1, 1, 1, 0, -1, -1, -1]
    dc = [-1, -1, 0, 1, 1, 1, 0, -1]
    x = p[0]
    y = p[1]
    chesses[x][y] = color
    for i in range(8):  # 表示8个方向
        if x + dr[i] < 8 and x + dr[i] >= 0 and y + dc[i] < 8 and y + dc[i] >= 0:
            r = x + dr[i]
            c = y + dc[i]
            # 判断该方向上是否会棋子翻面
            if chesses[r][c] == -color:
                if nextPos(r, c, i, dr, dc, chesses, color):
                    while chesses[r][c] == -color:
                        chesses[r][c] = color
                        r += dr[i]
                        c += dc[i]

def play(color, pos, chesses, ai, _ai, flag = True):
    size = len(pos)
    # 10%的几率在可以下的位置随机选择一个
    if random.randint(0, 99) < 20 and flag:
        p = pos[random.randint(0, size-1)]
    # 否则选择神经网络打分最大的位置
    else:
        max_res = -10000
        p = None
        for i in range(size):
            pos64 = pos[i][0] * 8 + pos[i][1]
            res = ai(torch.cat((chesses.reshape(64),torch.FloatTensor([pos64])),0).reshape(65))
            if res > max_res:
                max_res = res
                p = pos[i]
    # 计算该位置的真实值
    score1 = evaluate(chesses, color)
    putChess(chesses, p, color)
    score2 = evaluate(chesses, color)
    next_pos = getNextStepPos(chesses, -color)
    next_pos_max_res = 0
    for eachPos in next_pos:
        pos64 = eachPos[0] * 8 + eachPos[1]
        next_res = _ai(torch.cat((chesses.reshape(64),torch.FloatTensor([pos64])),0).reshape(65))
        if next_pos_max_res < next_res:
            next_pos_max_res = next_res
    pos64 = p[0] * 8 + p[1]
    ai.train(torch.cat((chesses.reshape(64),torch.FloatTensor([pos64])),0).reshape(65), score2 - score1 - 0.2 * next_pos_max_res)

if __name__ == "__main__":
    load = True
    if not load:    # 创建一个先手的ai，一个后手的ai模型
        first_ai = QL()
        last_ai = QL()
    else:       # 读入已经训练好的模型
        first_ai = torch.load('first_ai.pkl')
        last_ai = torch.load('last_ai.pkl')
    times = 3000   # 训练局数
    time = times
    # 执行比赛
    while times != 0:
        print(time-times)
        times -= 1
        # 创建棋盘，对应棋盘的64个位置。0：空，-1：黑，1：白
        chesses = torch.FloatTensor(np.zeros((8, 8)))
        chesses[3][3] = 1
        chesses[3][4] = -1
        chesses[4][3] = -1
        chesses[4][4] = 1
        while 1:
            flag = True
            # 查找黑方下一步可以下的位置
            pos = getNextStepPos(chesses, -1)
            if pos != []:
                flag = False
                play(-1, pos, chesses, first_ai, last_ai)
            # 查找拜访下一步可以下的位置
            pos = getNextStepPos(chesses, 1)
            if pos != []:
                flag = False
                play(1, pos, chesses, last_ai, first_ai)
            if flag:   # 双方都无处可下，直接结束游戏
                break
    # 保存模型
    torch.save(first_ai, 'first_ai25000.pkl')
    torch.save(last_ai, 'last_ai25000.pkl')





