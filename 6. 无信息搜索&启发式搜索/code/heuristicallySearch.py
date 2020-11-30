# 判断当前节点是否可以拓展并加入开放列表
def validNode(open_list, open_parent_node, g, x, y, x_expand, y_expand, min_idx):
    if maze[x_expand][y_expand] == '0' or maze[x_expand][y_expand] == 'E':
        if [x_expand, y_expand] in open_list:
            idx = open_list.index([x_expand, y_expand])
            if g[idx] < g[min_idx] + 1:
                g[idx] = g[min_idx] + 1
                open_parent_node[idx] = [x,y]
        else:
            open_list.append([x_expand,y_expand])
            open_parent_node.append([x,y])
            g.append(g[min_idx]+1)


# 拓展开放列表中的节点，启发函数为Lp距离
def expandNode(open_list, close_list, open_parent_node, close_parent_node, g):
    min_cost = 100000   # 当前最小开销
    min_idx = 0         # 开销最小的节点的下标
    p = 1               # Lp距离的p

    # 找到可扩展的开销最小的节点
    for i in range(len(open_list)):
        h = (abs(open_list[i][0] - end_node[0])**p + abs(open_list[i][1] - end_node[1])**p ) ** (1/p)
        # h = max(abs(open_list[i][0] - end_node[0]), abs(open_list[i][1] - end_node[1]))
        # h = 0
        h *= 4
        f = h + g[i]
        if f < min_cost:
            min_cost = f
            min_idx = i

    # 将该节点的可扩展节点加入开放列表
    x = open_list[min_idx][0]
    y = open_list[min_idx][1]
    maze[x][y] = '1'
    validNode(open_list, open_parent_node, g, x, y, x+1, y, min_idx)
    validNode(open_list, open_parent_node, g, x, y, x-1, y, min_idx)
    validNode(open_list, open_parent_node, g, x, y, x, y+1, min_idx)
    validNode(open_list, open_parent_node, g, x, y, x, y-1, min_idx)
    print(len(open_list))
    # 将该节点从开放列表中删除并加入关闭列表
    close_list.append([x,y])
    close_parent_node.append(open_parent_node[min_idx])
    expanded.add((x,y))
    del open_list[min_idx]
    del open_parent_node[min_idx]
    del g[min_idx]

# 读入迷宫数据
maze = []
with open('MazeData.txt', 'r') as f:
    for eachLine in f:
        line = []
        for eachPos in eachLine:
            if eachPos == '\n':
                break
            line.append(eachPos)
        maze.append(line)

# 找到起点和终点坐标，并加入开启列表
row = len(maze)
col = len(maze[0])
open_list = []              # 开启列表
open_parent_node = [None]   # 开启列表中每个节点对应的父节点
g = [0]                     # 开启列表中每个节点对应的 g(x) 值
close_list = []             # 关闭列表
close_parent_node = []      # 关闭列表中每个节点对应的父节点
end_node = []
expanded = set()
for i in range(row):
    for j in range(col):
        if maze[i][j] == 'S':
            open_list.append([i,j])
        if maze[i][j] == 'E':
            end_node = [i,j]

# 开始搜索
find_ans = False
while 1:
    # 若开启列表为空，则迷宫无解
    if open_list == []:
        find_ans = False
        break
    # 对节点进行拓展
    expandNode(open_list, close_list, open_parent_node, close_parent_node, g)
    # 若开放节点中存在终点节点则找到解
    if end_node in open_list:
        find_ans = True
        break

if find_ans == False:
    print('no answer for this maze')
else:
    idx = open_list.index(end_node)
    last_node = open_parent_node[idx]
    path = [end_node]
    while last_node is not None:
        path.append(last_node)
        last_node = close_parent_node[close_list.index(last_node)]
    path = path[::-1]
    print(len(expanded) + 1)
    print(len(path))
    print(path)
    # maze = []
    # with open('MazeData.txt', 'r') as f:
    #     for eachLine in f:
    #         maze.append(eachLine)
    # for i in range(row):
    #     for j in range(col):
    #         if [i,j] in path:
    #         if (i,j) in expanded:
    #             maze[i] = maze[i][:j] + 'X' + maze[i][j+1:]
    # with open('ans.txt', 'w') as f:
    #     for eachLine in maze:
    #         f.write(eachLine)
