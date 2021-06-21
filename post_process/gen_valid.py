
root = "../extra_data/labels/val.txt"


thresh = 0.7

pos = thresh + 0.01
neg = thresh - 0.01

data = []
with open(root, 'r') as f:
    for line in f.readlines():
        name, label = line.strip().split(' ')
        if label == '0': #pos
            score = pos
        else:
            score = neg
        data.append(name + ' {:.4f}'.format(score))

with open('valid_thresh_{}.txt'.format(thresh), "w") as f:
    f.write("\n".join(s for s in data))
    