
def get_list_data(txt_file):
    res = []
    with open(txt_file,'r') as f:
        lines = f.readlines()
        for line in lines:
            name = line.strip().split(' ')[0]
            res.append(name)
    return res

def calc_data(list1,list2):
    count = 0
    for item in list1:
        if item not in list2:
            print(item)
            count+=1
    print(count)

train_1 = get_list_data('./train_label.txt')
train_2 = get_list_data('./train_v2.txt')
valid_1 = get_list_data('./valid_label.txt')
valid_2 = get_list_data('./valid_v2.txt')
calc_data(train_1,train_2)
calc_data(valid_1,valid_2)
print(len(train_1),len(train_2))
print(len(valid_1),len(valid_2))
