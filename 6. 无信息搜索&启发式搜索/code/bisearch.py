def expandNode(node_set, path_set):
    new_node_set = []
    new_path_set = []
    while node_set != []:
        x = node_set[0][0]
        y = node_set[0][1]
        if maze[x+1][y] == '0':
            new_node_set.append([x+1,y])
            new_path_set.append(path_set[0]+[[x+1,y]])
            expanded.add((x+1,y))
        if maze[x-1][y] == '0':
            new_node_set.append([x-1,y])
            new_path_set.append(path_set[0]+[[x-1,y]])
            expanded.add((x-1,y))
        if maze[x][y+1] == '0':
            new_node_set.append([x,y+1])
            new_path_set.append(path_set[0]+[[x,y+1]])
            expanded.add((x,y+1))
        if maze[x][y-1] == '0':
            new_node_set.append([x,y-1])
            new_path_set.append(path_set[0]+[[x,y-1]])
            expanded.add((x,y-1))
        maze[x][y] = '1'
        expanded.add((x,y))
        del node_set[0]
        del path_set[0]
    return new_node_set[:], new_path_set[:]

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

# 找到起点和终点坐标，并加入可扩展节点
row = len(maze)
col = len(maze[0])
node_from_start = []
path_from_start = []
node_from_end = []
path_from_end = []
expanded = set()
for i in range(row):
    for j in range(col):
        if maze[i][j] == 'S':
            node_from_start.append([i,j])
            path_from_start.append([[i, j]])
            expanded.add((i,j))
        if maze[i][j] == 'E':
            node_from_end.append([i, j])
            path_from_end.append([[i, j]])
            expanded.add((i, j))

# 开始搜索，宽度优先，每次起点和终点所有节点拓展一步
find_ans = False
while 1:
    # 无拓展节点，即无解。退出循环
    if node_from_start == [] or node_from_end == []:
        find_ans = False
        break

    # 拓展从起点出发的节点
    node_from_start, path_from_start = expandNode(node_from_start, path_from_start)
    # 如果从起点和终点出发的节点重复，则找到解，退出循环
    for eachNode in node_from_start:
        if eachNode in  node_from_end:
            find_ans = True
            break
    if find_ans:
        break
    # 拓展从终点出发的节点
    node_from_end, path_from_end = expandNode(node_from_end, path_from_end)
    # 如果从起点和终点出发的节点重复，则找到解，退出循环
    for eachNode in node_from_start:
        if eachNode in  node_from_end:
            find_ans = True
            break
    if find_ans:
        break

if find_ans == False:
    print('no answer for this maze')
else:
    for eachNode in node_from_start:
        if eachNode in  node_from_end:
            cross_node = eachNode
            break
    start_idx = node_from_start.index(cross_node)
    end_idx = node_from_end.index(cross_node)
    ans = path_from_start[start_idx][:-1] + path_from_end[end_idx][::-1]
    print(len(expanded))
    print(len(ans))
    print(ans)

    # 将解的路径输出成txt文件
    # maze = []
    # with open('MazeData.txt', 'r') as f:
    #    for eachLine in f:
    #        maze.append(eachLine)
    # for i in range(row):
    #    for j in range(col):
            # if [i,j] in ansL
    #         if (i,j) in expanded:
    #             maze[i] = maze[i][:j] + 'X' + maze[i][j+1:]
    # with open('ans.txt', 'w') as f:
    #     for eachLine in maze:
    #         f.write(eachLine)
