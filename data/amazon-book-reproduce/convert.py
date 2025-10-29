import numpy as np

if __name__ == '__main__':
    res = []
    with open('train.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            nums = line.strip().split()
            nums = [int(i) for i in nums]
            res.append(nums)
    with open('train_new.txt', 'w') as f:
        for line in res:
            line_str = ' '.join([str(i) for i in line])
            f.write(line_str + '\n')
    res = []
    with open('test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            nums = line.strip().split()
            nums = [int(i) for i in nums]
            res.append(nums)
    with open('test_new.txt', 'w') as f:
        for line in res:
            line_str = ' '.join([str(i) for i in line])
            f.write(line_str + '\n')
            