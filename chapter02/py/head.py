import sys
num = int(sys.argv[1])
file = sys.argv[2]
with open(file, 'r') as f:
    for _, l in zip(range(num), f):
        print(l.rstrip())