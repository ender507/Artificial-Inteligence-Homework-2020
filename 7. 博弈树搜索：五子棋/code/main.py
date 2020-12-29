import pygame

def draw(pos = None):
    # 绘制棋盘底色
    screen.fill((150, 80, 0))
    # 绘制网格
    for x in range(0, cross_size * cross_num, cross_size):
        pygame.draw.line(screen, (255, 255, 255), (x + space, 0 + space),
                         (x + space, cross_size * (cross_num - 1) + space), 1)
    for y in range(0, cross_size * cross_num, cross_size):
        pygame.draw.line(screen, (255, 255, 255), (0 + space, y + space),
                         (cross_size * (cross_num - 1) + space, y + space), 1)
    # 绘制棋子
    for x in range(cross_num):
        for y in range(cross_num):
            color = chesses[(x,y)]
            if color != -1:
                xi = space + x * cross_size
                yi = space + y * cross_size
                pygame.draw.circle(screen, (color, color, color), (xi, yi),15,0)
    # 绘制胜利的五连
    if pos is not None:
        pygame.draw.line(screen, (255, 0, 0), (space + cross_size * pos[0][0], space + cross_size * pos[0][1]), \
                         (space + cross_size * pos[1][0], space + cross_size * pos[1][1]), 5)
    pygame.display.update()

def getScore(color ,chess):
    # 优先直接取得胜利
    if chess == (color, color, color, color, color, color) or chess == (color, color, color, color, color, 255 - color) or \
        chess == (255-color, color, color, color, color, color) or chess == (-1, color, color, color, color, color) or \
        chess == (color, color, color, color, color, -1):
        return 10000
    # 其次防止对方下一步制胜
    if chess == (255-color, 255-color, 255-color, 255-color, color, 255-color) or chess ==(255-color, color, 255-color, 255-color, 255-color, 255-color) or\
        chess == (255-color, 255-color, color, 255-color, 255-color, 255-color) or chess == (255-color, 255-color, 255-color, color, 255-color, 255-color)or\
        chess == (color, 255-color, 255-color, 255-color, 255-color, color) or \
        chess == (255-color, 255-color, color, 255-color, 255-color, -1) or chess == (-1, 255-color, 255-color, color, 255-color, 255-color) or\
        chess == (255-color, color, 255-color, 255-color, 255-color, -1) or chess == (-1, 255-color, 255-color, 255-color, color ,255-color) or \
        chess == (255 - color, color, 255 - color, 255 - color, 255 - color, color) or chess == (color, 255 - color, 255 - color, 255 - color, color, 255 - color) or\
        chess == (-1, 255-color, color, 255-color, 255-color, 255-color) or chess == (255-color, 255-color, 255-color, color, 255-color, -1) :
        return 8000
    # 其次制造必胜棋
    if chess == (-1, color, color, color, color, -1):
        return 8000
    # 再其次破坏对方的必胜棋
    if chess == (-1, color, color-255, color-255, color-255, -1) or chess == (-1, color-255, color-255, color-255, color, -1)\
        or chess == (-1, color-255, color-255, color, color-255, -1) or chess == (-1, color-255, color, color-255, color-255, -1):
        return 4000
    if chess == (-1, color-255, color-255, -1, color-255, color) or chess == (color, color-255, -1, color-255, color-255, -1)or\
        chess == (-1, color-255, -1, color-255, color-255, color) or chess == (color, color-255, color-255, -1,color-255, -1):
        return 2000
    # 最次造棋
    if chess == (-1, color, color, color, -1, -1) or chess == (-1, -1, color, color, color, -1) or\
        chess == (-1, color, color, -1, color ,-1) or chess == (-1, color, -1, color, color, -1):
        return 1000
    if chess == (-1, color, color, -1, -1, -1) or chess == (-1, -1, -1, color, color, -1) or\
        chess == (-1, -1, color, color, -1, -1)or\
        chess == (-1, color, -1, color, -1, -1) or chess == (-1, -1, color, -1, color, -1):
        return 20
    if chess == (-1, 255 - color, color, -1, -1, -1):
        if color == ai_color:
            return 10
        else:
            return 10
    return 0

def skipCross(x,y):
    if x == 0 or y == 0 or x == cross_num-1 or y == cross_num-1:
        return False
    if chesses[(x-1,y-1)] == chesses[(x-1,y)] == chesses[(x-1,y+1)] == \
        chesses[(x, y-1)] ==chesses[(x,y)] == chesses[(x,y+1)] == \
            chesses[(x+1, y-1)] ==chesses[(x+1,y)] == chesses[(x+1,y+1)] == -1:
        return True
    return False

def evaluate(chess, color):
    score = 0
    # 横着的部分
    for x in range(cross_num):
        for y in range(cross_num-5):
            score += getScore(color, (chess[(x,y)], chess[(x,y+1)], \
                      chess[(x,y+2)], chess[(x,y+3)], chess[(x,y+4)], chess[(x,y+5)]))
    # 竖着的部分
    for x in range(cross_num-5):
        for y in range(cross_num):
            score += getScore(color, (chess[(x,y)], chess[(x+1,y)], \
                      chess[(x+2,y)], chess[(x+3,y)], chess[(x+4,y)], chess[(x+5,y)]))
    # 左上到右下
    for x in range(cross_num-5):
        for y in range(cross_num-5):
            score += getScore(color, (chess[(x,y)], chess[(x+1,y+1)], \
                      chess[(x+2,y+2)], chess[(x+3,y+3)], chess[(x+4,y+4)], chess[(x+5,y+5)]))
    # 右上到左下
    for x in range(cross_num-5):
        for y in range(cross_num-5):
            score += getScore(color, (chess[(x+5,y)], chess[(x+4,y+1)], \
                      chess[(x+3,y+2)], chess[(x+2,y+3)], chess[(x+1,y+4)], chess[(x,y+5)]))
    return score


def nextStep(chess, dep, color, last_alpha, last_beta):
    chesses_tmp = chess.copy()
    # 探索深度到底，进行打分
    if dep == 0:
        score = evaluate(chesses_tmp, ai_color) - evaluate(chesses_tmp, player_color)
        return score, score, (-1,-1)
    # 依据节点类型，对alpha和beta初始化
    if color == ai_color:
        alpha = float('-inf')
        beta = last_beta
    else:
        alpha = last_alpha
        beta = float('inf')
    # 遍历下一步可以下的地方，生成子节点
    for x in range(cross_num):
        for y in range(cross_num):
            if skipCross(x,y):
                continue
            if chesses_tmp[(x,y)] == -1:
                chesses_tmp[(x,y)] = color
                next_alpha, next_beta,_ = nextStep(chesses_tmp, dep-1, 255-color, alpha, beta)
                chesses_tmp[(x,y)] = -1
                if color == ai_color and alpha < next_beta:
                    alpha = next_beta
                    pos = (x,y)
                if color == player_color and beta > next_alpha:
                    beta = next_alpha
                    pos = (x,y)
                if beta < alpha:
                    return alpha, beta, pos
    return alpha, beta, pos


def AIturn():
    score,_,pos = nextStep(chesses, depth, ai_color, float('-inf'), float('inf'))
    print('AI得分：',score)
    chesses[pos] = ai_color
    trace.append(pos)

def checkWinner():
    winner = -1
    for x in range(cross_num):
        for y in range(cross_num-4):
            if chesses[(x,y)] == chesses[(x,y+1)] == chesses[(x,y+2)] ==\
                    chesses[(x,y+3)] == chesses[(x,y+4)] != -1:
                winner =  chesses[(x,y)]
                pos = [(x,y), (x,y+4)]
    for x in range(cross_num-4):
        for y in range(cross_num):
            if chesses[(x,y)] == chesses[(x+1,y)] == chesses[(x+2,y)] == \
                    chesses[(x+3,y)] == chesses[(x+4,y)] != -1:
                winner = chesses[(x,y)]
                pos = [(x, y), (x+4, y)]
    # 左上到右下
    for x in range(cross_num-4):
        for y in range(cross_num-4):
            if chesses[(x,y)] == chesses[(x+1,y+1)] == chesses[(x+2,y+2)] ==\
                    chesses[(x+3,y+3)] == chesses[(x+4,y+4)] != -1:
                winner = chesses[(x,y)]
                pos = [(x, y), (x+4, y+4)]
    # 右上到左下
    for x in range(cross_num-4):
        for y in range(cross_num-4):
            if chesses[(x+4,y)] == chesses[(x+3,y+1)] == chesses[(x+2,y+2)] ==\
                    chesses[(x+1,y+3)] == chesses[(x,y+4)] != -1:
                winner = chesses[(x+4,y)]
                pos = [(x+4, y), (x, y+4)]

    if winner != -1:
        if winner == ai_color:
            print('AI wins')
        else:
            print('player wins')
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    pygame.quit()
                    exit()
            draw(pos)


# 游戏参数
trace = []      # 记录下棋的位置
chesses = {}    # 记录所有的落子
cross_num = 11     # 交叉点的个数
depth = 2#int(input())
for x in range(cross_num):
    for y in range(cross_num):
        chesses[(x,y)] = -1

# 窗口参数
pygame.init()
space = 60          # 四周留下的边距
cross_size = 40     # 交叉点的间隔
grid_size = cross_size * (cross_num - 1) + space * 2     # 棋盘的大小
screencaption = pygame.display.set_caption('18340057-五子棋')  # 窗口标题
screen = pygame.display.set_mode((grid_size, grid_size))  # 设置窗口长宽
screen.fill((0,0,0))
pygame.draw.rect(screen, (255,255,255),((grid_size/2, 0), (grid_size/2, grid_size)), 0)
pygame.display.update()
choose_color = True
while choose_color:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.MOUSEBUTTONUP:      # 松开鼠标
            x, y = pygame.mouse.get_pos()           # 获取鼠标位置
            choose_color = False
            if x < grid_size / 2:
                ai_color = 255
                player_color = 0
            else:
                ai_color = 0
                player_color = 255

# 初始化棋局
chesses[(4,5)] = 255
chesses[(5,5)] = 0
chesses[(6,5)] = 0
chesses[(5,6)] = 255
if ai_color == 0:
    AIturn()
    # chesses[(int(cross_num / 2), int(cross_num / 2))] = 0


draw()
# 游戏过程
while True:
    for event in pygame.event.get():
        # 退出游戏
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        # 获取鼠标点击位置
        if event.type == pygame.MOUSEBUTTONUP:      # 松开鼠标
            x, y = pygame.mouse.get_pos()           # 获取鼠标位置
            x = round((x - space) / cross_size)     # 获取到x方向上取整的序号
            y = round((y - space) / cross_size)     # 获取到y方向上取整的序号
            if x >= 0 and x < cross_num and y >= 0 and y < cross_num and \
                    chesses[(x,y)] == -1:
                chesses[(x,y)] = player_color  # 将落子加入棋子列表
                trace.append((x,y))
                draw()
                checkWinner()
                AIturn()
                draw()
                checkWinner()

        if event.type == pygame.KEYDOWN:
            key = pygame.key.get_pressed()
            if key[pygame.K_SPACE]:
                draw()
                continue
            chesses[trace[-1]] = -1
            chesses[trace[-2]] = -1
            trace = trace[:-2]
            draw()
