import os
from os import PRIO_PGRP
import numpy as np
from sklearn import metrics
import shutil

def eval_res(pred, label, isShow=False, eval_type="ERR_strict",eval_val='self', thresh=None):

    name2idx = {}
    idx = 0
    data = [] 
    with open(pred, "r") as f:
        for line in f.readlines():
            name, score = line.strip().split(" ")
            # name = os.path.join('train',name)
            name2idx[name] = idx
            data.append([float(score), -1])
            idx += 1
    with open(label, "r") as f:
        for line in f.readlines():
            name, label = line.strip().split(" ") #0 3 -1 [0:real] [1:mask1] [3:mask3] [-1:unknown]
            if name in name2idx:
                if eval_val=='self':
                    data[name2idx[name]][1] = 1-int(label)
                else:
                    data[name2idx[name]][1] = int(label)
            else:
                print("{} not in the predict txt".format(name))
    data = np.array(data)
    if thresh is not None:
        thres = thresh
        FN = np.sum((data[:, 0] < thres) & ((data[:, 1] == 0)))
        FP = np.sum((data[:, 0] >= thres) & ((data[:, 1] == 1) | (data[:, 1] == 3)))
        BPCER = 1. * FN / (np.sum(data[:, 1] == 0))
        APCER = 1. * FP / (np.sum((data[:, 1] == 1) | (data[:, 1] == 3)))
        ACER = (BPCER + APCER) / 2.
        if isShow:
            print("Thres:{:.3f} APCER:{:.4f} BPCER:{:.4f} ACER:{:.4f}".format(thres, APCER, BPCER, ACER))
    
        return thres, APCER, BPCER, ACER

    if eval_type == "min_acer":
        scores = []
        for i in range(100):
            thres = i / 100
            FN = np.sum((data[:, 0] < thres) & ((data[:, 1] == 0))) + np.sum(data[:, 1]==-1)
            FP = np.sum((data[:, 0] >= thres) & ((data[:, 1] == 1) | (data[:, 1] == 3))) + np.sum(data[:, 1]==-1)
            BPCER = 1. * FN / (np.sum(data[:, 1] == 0))
            APCER = 1. * FP / (np.sum(data[:, 1] != 0))
            ACER = (BPCER + APCER) / 2.
            scores.append([thres, APCER, BPCER, ACER])
        scores = np.array(scores)
        idx = np.argmin(scores, axis=0)[3]
        if isShow:
            print("Thres:{:.3f} APCER:{:.4f} BPCER:{:.4f} ACER:{:.4f}".format(scores[idx, 0], scores[idx, 1], scores[idx, 2], scores[idx, 3]))
    
        return scores[idx, 0], scores[idx, 1], scores[idx, 2], scores[idx, 3]
    
    elif eval_type == "ERR":
        scores = []
        for i in range(100):
            thres = i / 100
            FN = np.sum((data[:, 0] < thres) & (data[:, 1] == 0))
            FP = np.sum((data[:, 0] >= thres) & (data[:, 1] != 0))
            BPCER = 1. * FN / np.sum(data[:, 1] == 0) #FNR / FRR
            APCER = 1. * FP / np.sum(data[:, 1] != 0) #FPR / FAR
            ACER = (BPCER + APCER) / 2.
            scores.append([thres, APCER, BPCER, ACER])
        scores = np.array(scores)
        delta = abs(scores[:, 1] - scores[:, 2])
        idx = np.argmin(delta, axis=0)

        if isShow:
            print("Thres:{:.3f} APCER:{:.4f} BPCER:{:.4f} ACER:{:.4f}".format(scores[idx, 0], scores[idx, 1], scores[idx, 2], scores[idx, 3]))
    
        return scores[idx, 0], scores[idx, 1], scores[idx, 2], scores[idx, 3]
    
    elif eval_type == "ERR_strict":
        scores = []
        for i in range(100):
            thres = i / 100
            FN = np.sum((data[:, 0] < thres) & ((data[:, 1] == 0))) + np.sum(data[:, 1]==-1)
            FP = np.sum((data[:, 0] >= thres) & ((data[:, 1] == 1) | (data[:, 1] == 3))) + np.sum(data[:, 1]==-1)
            BPCER = 1. * FN / (np.sum(data[:, 1] == 0))
            APCER = 1. * FP / (np.sum(data[:, 1] != 0))
            
            ACER = (BPCER + APCER) / 2.
            scores.append([thres, APCER, BPCER, ACER])
        scores = np.array(scores)
        delta = abs(scores[:, 1] - scores[:, 2])
        idx = np.argmin(delta, axis=0)

        if isShow:
            print("Thres:{:.3f} APCER:{:.4f} BPCER:{:.4f} ACER:{:.4f}".format(scores[idx, 0], scores[idx, 1]*100, scores[idx, 2]*100, scores[idx, 3]*100))
    
        return scores[idx, 0], scores[idx, 1], scores[idx, 2], scores[idx, 3]
    
    return 0, 0, 0, 0

def calc_acc(pred_res,label_gt,eval_val="self"):
    root_pth = '../raw_data/phase1'
    target_data = '../wrong_pic'
    pred_dic = {}
    label_dic = {}
    with open(pred_res, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name,pred, *_ = line.strip().split(' ')
            # name = os.path.join('train',name)
            # print(name)
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
            if eval_val=='self':
                pred = 1-int(pred)
            else:
                if int(pred)>1:
                    pred = 1
            label_dic[name] = str(pred)
    # if eval_val=='offical':
    res_txt = open('./se_101.txt', 'w')
    count = 0
    # print(pred_dic,label_dic)
    for name in pred_dic.keys():
        # print(name)
        # name = '%04d.png'%(i+1)
        # print(name)
        if int(pred_dic[name])!=int(label_dic[name]):
            # if eval_val=='self':
            #     shutil.copyfile(os.path.join(root_pth,name),os.path.join(target_data,"_".join(name.split('/'))))
            # else:
            res_txt.write(name+' '+pred_dic[name]+' '+label_dic[name]+'\n')
            count+=1

    print(count / len(pred_dic) , count)

if __name__ == "__main__":
    pass
