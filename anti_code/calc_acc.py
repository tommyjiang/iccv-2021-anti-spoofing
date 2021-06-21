# 计算模型预测结果与“真值”的一个acc

pred_res = '../results/ccl_resnext101_se_light_model_best_submit.txt'
label_gt = './val.txt'

pred_dic = {}
label_dic = {}
with open(pred_res, 'r') as f:
    lines = f.readlines()
    for line in lines:
        name,pred, *_ = line.strip().split(' ')
        if pred > str(0.5):
            pred = 0
        else:
            pred = 1
        # print(name,pred)
        # print(pred)
        pred_dic[name] = str(pred)

with open(label_gt, 'r') as f:
    lines = f.readlines()
    for line in lines:
        name,pred,*_ = line.strip().split(' ')
        # print(name,pred)
        if int(pred)>1:
            pred = 1
        label_dic[name] = str(pred)
        
res_txt = open('./light.txt', 'w')
count = 0
for i in range(len(pred_dic)):
    name = '%04d.png'%(i+1)
    if int(pred_dic[name])!=int(label_dic[name]):
        res_txt.write(name+' '+pred_dic[name]+' '+label_dic[name]+'\n')
        count+=1

print(count / len(pred_dic) , count)
    


