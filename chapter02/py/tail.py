import sys
num = int(sys.argv[1])
file = sys.argv[2]
tail = []
with open(file, 'r') as f:
    f_reversed = reversed(f.readlines())
    for _, l in zip(range(num), f_reversed):
        tail.insert(0, l.rstrip())
for t in tail:
    print(t)