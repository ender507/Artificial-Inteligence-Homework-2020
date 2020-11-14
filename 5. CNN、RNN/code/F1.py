binggo = 0
ans = 0
out = 0
with open('36.txt','r') as f:
    flag = 0
    for eachLine in f:
        if eachLine == 'answer:\n':
            flag = 1
        elif eachLine == 'output:\n':
            flag = 2
        elif flag == 1:
            str1 = eachLine.split(', ')
            str1=str1[:-1]
            ansSet = str1
            ans += len(ansSet)
        elif flag == 2:
            str1 = eachLine.split(', ')
            str1=str1[:-1]
            outSet = str1
            out += len(outSet)
            for each in ansSet:
                if each in outSet:
                    binggo += 1
binggo=184
print('answer count:',ans)
print('output count:',out)
print('right count:', binggo)
print('right/ans:',binggo/ans)
print('right/out:',binggo/out)
print('F1:',2*(binggo/ans)*(binggo/out)/(binggo/out+binggo/ans))

