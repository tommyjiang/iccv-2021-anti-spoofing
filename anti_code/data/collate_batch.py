# encoding: utf-8
import torch


def train_collate_fn(batch):
    imgs, _ , label, maskid, _ = zip(*batch)
    label = torch.tensor(label, dtype=torch.int64)
    return torch.stack(imgs, dim=0), label, maskid


def val_collate_fn(batch):
    imgs,_, label, maskid,img_pth = zip(*batch)
    label = torch.tensor(label, dtype=torch.int64)
    return torch.stack(imgs, dim=0), label, maskid, img_pth

def ccl_collate_fn(batch):
    img0, img1, label_pair, label0, label1, maskid0, maskid1 = zip(*batch)
    label_pair = torch.tensor(label_pair, dtype=torch.int64)
    label0 = torch.tensor(label0, dtype=torch.int64)
    label1 = torch.tensor(label1, dtype=torch.int64)
    mask0 = torch.tensor(maskid0, dtype=torch.int64)
    mask1 = torch.tensor(maskid1, dtype=torch.int64)
    return torch.stack(img0, dim=0),torch.stack(img1,dim=0),label_pair,label0,label1,mask0,mask1

def ult_collate_fn(batch):
    w_img, s_img, labelid = zip(*batch)
    labelid = torch.tensor(labelid, dtype=torch.int64)

    return torch.stack(w_img,dim=0),torch.stack(s_img,dim=0),labelid

def TTA_collate_fn(batch) :
    img_lsts, _, label, maskid,img_pth = zip(*batch)
    label = torch.tensor(label, dtype=torch.int64)
    return torch.stack([torch.stack(img_lst, dim=0) for img_lst in img_lsts], dim=0), label, maskid, img_pth

def p_collate_fn(batch):
    imgs,_, label, label0, maskid,img_pth = zip(*batch)
    label = torch.tensor(label, dtype=torch.int64)
    label0 = torch.tensor(label0, dtype=torch.int64)

    return torch.stack(imgs, dim=0), label, label0, maskid, img_pth