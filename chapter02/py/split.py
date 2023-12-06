import sys
num = int(sys.argv[1])
d_n = 2780//num
rem = 2780 % num
file = sys.argv[2]
result = []
save = []
sum_len = 0

print(f'{num}分割します。各ファイル(リスト)の行数は{d_n}か{d_n+1}になります。')

for i, l in enumerate(open(file), 1):
    if i % d_n == 0:
        save.append(l.rstrip())
        result.append(save)
        save = []
    elif i - (2780 - rem) > 0:
        for n, li in enumerate(result[i - (2780 - rem):], i - (2780 - rem)):
            result[n - 1].append(li.pop(0))
        result[-1].append(l)
    else:
        save.append(l.rstrip())
        
for r in result:
    r = len(r)
    print(r)
    sum_len += r
print(f'合計は{sum_len}')