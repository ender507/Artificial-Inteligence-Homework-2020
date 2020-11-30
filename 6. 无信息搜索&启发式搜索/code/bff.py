def expand(x,y):
    if maze[x][y] == '0' or maze[x][y] == 'E':
        node_list.append([x,y])
        path.append(path[0]+[[x,y]])
        expanded.add((x,y))
        maze[x][y] = '1'

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
node_list = []
path = []
end_node = []
expanded = set()
for i in range(row):
    for j in range(col):
        if maze[i][j] == 'S':
            node_list.append([i,j])
            path.append([[i,j]])
            expanded.add((i,j))
        if maze[i][j] == 'E':
            end_node = [i,j]
            
while 1:
    x = node_list[0][0]
    y = node_list[0][1]
    expand(x+1,y)
    if end_node in node_list:
        break
    expand(x-1,y)
    if end_node in node_list:
        break
    expand(x,y+1)
    if end_node in node_list:
        break
    expand(x,y-1)
    if end_node in node_list:
        break
    del node_list[0]
    del path[0]

print(len(expanded))
print(len(path[-1]))        
print(path[-1])
