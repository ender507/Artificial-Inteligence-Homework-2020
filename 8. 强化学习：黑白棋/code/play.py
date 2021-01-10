from train import *
import time
import pygame

def draw(chesses):
    space = 60          # 四周留下的边距
    cross_size = 40     # 交叉点的间隔
    cross_num = 9
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
    black = 0
    white = 0
    for x in range(cross_num-1):
        for y in range(cross_num-1):
            color = chesses[(x,y)]
            if color != 0:
                xi = space + (x + 0.5) * cross_size
                yi = space + (y + 0.5) * cross_size
                if color == 1:
                    white += 1
                    pygame.draw.circle(screen, (255, 255, 255), (xi, yi),15,0)
                else:
                    black += 1
                    pygame.draw.circle(screen, (0, 0, 0), (xi, yi), 15, 0)
    print('-------------')
    print('black:',black)
    print('white:', white)
    pygame.display.update()

def aiPlay(ai, ai_color, chesses, _ai):
    pos = getNextStepPos(chesses, ai_color)
    if pos != []:
        play(ai_color, pos, chesses, ai, _ai, False)

def endGame(chesses, ai_color):
    black = 0
    white = 0
    for i in range(8):
        for j in range(8):
            if chesses[i][j] == 1:
                white += 1
            elif chesses[i][j] == -1:
                black += 1
    print('-------------')
    print('black:',black)
    print('white', white)
    if black == white:
        print('draw')
    if black > white:
        if ai_color == 1:
            print('you win!')
        else:
            print('ai_wins!')
    else:
        if ai_color == 1:
            print('ai wins!')
        else:
            print('you win!')
    exit(0)


if __name__ == '__main__':
    # 初始化图形界面
    pygame.init()
    space = 60          # 四周留下的边距
    cross_size = 40     # 交叉点的间隔
    cross_num = 9
    grid_size = cross_size * (cross_num - 1) + space * 2     # 棋盘的大小
    screencaption = pygame.display.set_caption('18340057-黑白棋')  # 窗口标题
    screen = pygame.display.set_mode((grid_size, grid_size))  # 设置窗口长宽
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 255, 255), ((grid_size / 2, 0), (grid_size / 2, grid_size)), 0)
    pygame.display.update()
    # 选择颜色
    ai_color = player_color = 0
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
                    ai_color = 1
                    player_color = -1
                else:
                    ai_color = -1
                    player_color = 1
    # 依据颜色选择ai
    if ai_color == 1:
        ai = torch.load('last_ai.pkl')
        _ai = torch.load('first_ai.pkl')
    else:
        ai = torch.load('first_ai.pkl')
        _ai = torch.load('last_ai.pkl')
    # 初始化棋局
    chesses = torch.FloatTensor(np.zeros((8, 8)))
    chesses[3][3] = 1
    chesses[3][4] = -1
    chesses[4][3] = -1
    chesses[4][4] = 1

    # ai 先手
    if ai_color == -1:
        aiPlay(ai, ai_color, chesses, _ai)
    draw(chesses)

    while 1:
        for event in pygame.event.get():
            # 退出游戏
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            # 获取鼠标点击位置
            if event.type == pygame.MOUSEBUTTONUP:  # 松开鼠标
                x, y = pygame.mouse.get_pos()  # 获取鼠标位置
                x = round((x - space - 0.5 * cross_size) / cross_size)  # 获取到x方向上取整的序号
                y = round((y - space - 0.5 * cross_size) / cross_size)  # 获取到y方向上取整的序号

                # 如果玩家下的位置合法则可以落子
                if x >= 0 and x < cross_num-1 and y >= 0 and y < cross_num-1 and \
                        chesses[(x, y)] == 0 and isValidPos(x , y, player_color, chesses):
                    chesses[(x, y)] = player_color  # 将落子加入棋子列表
                    putChess(chesses, (x,y), player_color)
                    draw(chesses)
                    # 稍微等待一下，
                    time.sleep(0.5)
                    aiPlay(ai, ai_color, chesses, _ai)
                    draw(chesses)
                    # 人下不了则ai一直下
                    while getNextStepPos(chesses, player_color) == []:
                        time.sleep(0.5)
                        aiPlay(ai, ai_color, chesses, _ai)
                        # 都下不了则结束游戏
                        if getNextStepPos(chesses, ai_color) == [] and getNextStepPos(chesses, player_color) == []:
                            endGame(chesses, ai_color)
                        draw(chesses)

