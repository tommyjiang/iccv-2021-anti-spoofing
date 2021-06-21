# encoding: utf-8

import os.path as osp
from PIL import Image
from torch import argmax
from torch.utils import data
from torch.utils.data import Dataset
import random
import cv2
import numpy as np
from albumentations.pytorch.functional import img_to_tensor
import math
from ..augmentation.augmentation_class import StrongAugmentation,WeakAugmentation
import random

def add_light_numpy(img,p=0.1,strength = 250):
    if random.uniform(0,1) >= p:
        return img
    rows,cols = img.shape[:2]
    center_x = random.randint(0,rows)
    center_y = random.randint(0,cols)
    min_radius = int(math.sqrt(math.pow(center_x,2)+math.pow(center_y,2)))
    max_radius = int(math.sqrt(math.pow(rows,2)+math.pow(cols,2)))
    radius = random.randint(min_radius,max_radius)
    dst = np.zeros((rows,cols,3), dtype = 'uint8')
    x_src = np.ones((rows,cols))
    y_src = np.ones((rows,cols))
    for i in range(rows):
        x_src[i,:] = i
    for j in range(cols):
        y_src[:,j] = j
    x_center = np.ones((rows,cols))*center_x
    y_center = np.ones((rows,cols))*center_y
    ## 计算离光源点的距离
    dist_ = (x_src-x_center)**2+(y_src-y_center)**2
    radius_ma = np.ones((rows,cols))*radius
    ## 添加亮度的基本逻辑
    thresh = np.ones((rows,cols))*radius*radius
    val = (1.0-np.sqrt(dist_) / radius_ma)*strength
    ## 构建一个mask，屏蔽掉不满足条件的位置
    mask = dist_- thresh
    mask[mask>0] = 0
    mask[mask<0] = 1
    res = mask*val
    add_val = np.repeat(res[:,:,np.newaxis],3,axis=2)
    dst = img+add_val
    # 对值进行截断
    dst[dst<0] = 0
    dst[dst>255] = 255
    dst = np.uint8(dst)
    return dst
def enlarge(landmarks, scale=1.3):
    # if len(landmarks)==0:
    #     return landmarks
    assert(len(landmarks) == 4)
    x1, y1, x2, y2 = landmarks

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    enlarge_x1 = x_center - scale / 2 * w
    enlarge_x2 = x_center + scale / 2 * w
    enlarge_y1 = y_center - scale / 2 * h
    enlarge_y2 = y_center + scale / 2 * h
    enlarge_landmarks = [enlarge_x1, enlarge_y1, enlarge_x2, enlarge_y2]

    enlarge_landmarks = list(map(int, enlarge_landmarks))

    return enlarge_landmarks

def CropFace(im, box, exp_ratio=1.3):
    """
    params:
    im: np.ndarray (h, w, c) [BGR]
    box: [x1, y1, x2, y2]
    """
    if len(box)==0:
        return im
    img_h, img_w = im.shape[:2]
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    assert (bw > 0) and (bh > 0), "invalid box"
    
    cx = box[0] + bw / 2.
    cy = box[1] + bh / 2.
    # print(box)
    if bh > bw:
        box[0] = max(cx - bh/2., 0) 
        box[2] = min(cx + bh/2., img_w)
    else:
        box[1] = max(cy - bw/2., 0)
        box[3] = min(cy + bw/2., img_h)
    
    bw = box[2] - box[0]
    bh = box[3] - box[1]

    box_e = [max(box[0] - bw * (exp_ratio - 1) / 2., 0),
                max(box[1] - bh * (exp_ratio - 1) / 2., 0),
                min(box[2] + bw * (exp_ratio - 1) / 2., img_w),
                min(box[3] + bh * (exp_ratio - 1) / 2., img_h),]
    
    box_e = [int(num) for num in box_e]
    # print(box_e)
    new_im = im[box_e[1]:box_e[3], box_e[0]:box_e[2]]
    # print(new_im.shape)
    return new_im

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            # img = Image.open(img_path).convert('RGB')
            img = cv2.imread(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def CropFace_jiang(im, box, exp_ratio=1.3):

    height, width, _ = im.shape
    landmarks = box
    # enlarge 人脸 bbox
    if len(landmarks) != 0:
        landmarks = enlarge(landmarks, exp_ratio)
    else:
        landmarks = [0, 0, width - 1, height - 1]

    # 根据 enlarge 的 bbox 将原图做仿射变换
    pts1 = np.float32([[0, 0], [0, 256 - 1], [256 - 1, 256 - 1]])
    pts2 = np.float32([[landmarks[0], landmarks[1]], [landmarks[0], landmarks[3]], [landmarks[2], landmarks[3]]])
    M = cv2.getAffineTransform(pts2, pts1)
    n_im = cv2.warpAffine(im, M, (256, 256))
    
    return n_im

class ImageDataset(Dataset):
    """Image Dataset"""

    def __init__(self, dataset,transform=None,pid_index=None):
        self.dataset = dataset
        self.transform = transform
        self.pid_index = pid_index
        self.normalize = {"mean":[0.5, 0.5, 0.5],
                        "std":[0.5, 0.5, 0.5]}
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        ## 修改这里生成样本对
        img_path, pid, labelid, maskid, points_list = self.dataset[index]
        img = read_image(img_path)
        # print(points_list)
        img = CropFace(img,points_list)
        # print(img.shape,img_path,points_list)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            data = self.transform(image=img)
            img = data['image']
            img = img_to_tensor(img,self.normalize)
        return img, pid, labelid, maskid, img_path


class CCLDataset(Dataset):
    def __init__(self, dataset, transform=None, pid_index=None):
        self.dataset = dataset
        self.transform = transform
        self.pid_index = pid_index
        self.normalize = {"mean":[0.5, 0.5, 0.5],
                        "std":[0.5, 0.5, 0.5]}


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        img_path, pid, labelid, maskid, points_list = self.dataset[index]
        should_postive = random.randint(0,1)
        # should_postive = 1
        ## 这里其实是有可能陷入循环影响训练的时间的,或许有更好的实现方式
        if should_postive:
            # print(should_postive)
            while True:
                img_path_1, pid_1, labelid_1, maskid_1, points_list1 = self.pid_index[pid][random.randint(0,len(self.pid_index[pid])-1)]
                if labelid_1 == labelid:
                    break
                else:
                    continue
        else:
            while True:
                img_path_1, pid_1, labelid_1, maskid_1, points_list1 = self.pid_index[pid][random.randint(0,len(self.pid_index[pid])-1)]
                if labelid_1 != labelid:
                    break
                else:
                    continue
        img1 = read_image(img_path_1)
        img1 = add_light_numpy(img1)
        img1 = CropFace(img1,points_list1)
        # img1 = add_light_numpy(img1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        data1 = self.transform(image=img1)
        img1 = data1["image"]
        img1 = img_to_tensor(img1,self.normalize)



        img = read_image(img_path)
        # img = add_light_numpy(img)
        img = CropFace(img,points_list)
        img = add_light_numpy(img)
        # print(points_list)
        # print(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = self.transform(img)
        data = self.transform(image=img)
        img = data["image"]
        img = img_to_tensor(img,self.normalize)
        return img, img1, should_postive,labelid,labelid_1,maskid,maskid_1

class UnlabelsImageDataset(Dataset):
    """Image Dataset"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.weak_transform = WeakAugmentation(224,[0.5,0.5,0.5],[0.5,0.5,0.5],True,True,True,False)
        self.strong_transform = StrongAugmentation(224,[0.5,0.5,0.5],[0.5,0.5,0.5], True, True, "fixmatch", False,0.5)
        # self.transform = transform
        # self.pid_index = pid_index
        # self.normalize = {"mean":[0.5, 0.5, 0.5],
        #                 "std":[0.5, 0.5, 0.5]}
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        ## 修改这里生成样本对
        img_path, pid, labelid, maskid, points_list = self.dataset[index]
        img = read_image(img_path)
        # print(points_list)
        img = CropFace(img,points_list)
        # print(img.shape,img_path,points_list)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        w_aug_image = self.weak_transform(img)
        s_aug_image = self.strong_transform(img)
        # if self.transform is not None:
        #     data = self.transform(image=img)
        #     img = data['image']
        #     img = img_to_tensor(img,self.normalize)
        return w_aug_image,s_aug_image,labelid 

class PesudoDataset(ImageDataset):
    def __init__(self, dataset, det_label_dir="../post_process/det_labels.txt", transform=None,  data_scores = "../post_process/619/temp_res_test_post.txt"):

        super().__init__(dataset, transform=transform, pid_index=None)

        self.thresh = 0.9
        self.det_labels_dict = {}

        with open(det_label_dir, "r") as f:
            for line in f.readlines():
                name, det_label = line.strip().split(' ')
                idx = int(name.split('.')[0])
                self.det_labels_dict[idx] = int(det_label) # 0 1 2
        
        self.base_dataset = dataset
        self.dataset = []
        self.update_labels(data_scores, self.thresh)

    def update_labels(self, data_scores, thresh):
        print("label update")
        # self.bucket = dict()  # 记录每个 bucket 对应的 image
        pesudo_scores_dict = {}
        with open(data_scores, "r") as f:
            for line in f.readlines():
                name, score = line.strip().split(' ')
                idx = int(name.split('.')[0])
                pesudo_scores_dict[idx] = float(score) # 0 1 2
        
        dataset = []
        for imgpth,t1,label,t2,points_list in self.base_dataset:
            idx = int(imgpth.split('/')[-1].split('.')[0])
            det_label = self.det_labels_dict[idx]
            score = pesudo_scores_dict[idx]
            if det_label == 2:
                if score > thresh:
                    label = 0
                elif score < (1-thresh):
                    label = 1
                else:
                    continue

                dataset.append([imgpth,t1,label,t2,points_list])
        self.dataset = dataset


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        ## 修改这里生成样本对
        img_path, pid, labelid, maskid, points_list = self.dataset[index]

        img = read_image(img_path)
        # print(points_list)
        img = CropFace(img,points_list)
        # print(img.shape,img_path,points_list)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            data = self.transform(image=img)
            img = data['image']
            img = img_to_tensor(img,self.normalize)
        return img, pid, labelid,  maskid, img_path


class MPesudoDataset(ImageDataset):
    def __init__(self, dataset, det_label_dir="../post_process/det_labels.txt", transform=None,  data_scores = "./35_multi_res.txt"):

        super().__init__(dataset, transform=transform, pid_index=None)

        self.thresh = 0.9
        self.det_labels_dict = {}

        with open(det_label_dir, "r") as f:
            for line in f.readlines():
                name, det_label = line.strip().split(' ')
                idx = int(name.split('.')[0])
                self.det_labels_dict[idx] = int(det_label) # 0 1 2
        
        self.base_dataset = dataset
        self.dataset = []
        self.update_labels(data_scores, self.thresh)

    def update_labels(self, data_scores, thresh):
        print("label update")
        # self.bucket = dict()  # 记录每个 bucket 对应的 image
        pesudo_scores_dict = {}
        with open(data_scores, "r") as f:
            for line in f.readlines():
                name, s0, s1, s2 = line.strip().split(' ')
                idx = int(name.split('.')[0])
                pesudo_scores_dict[idx] = [float(s0), float(s1), float(s2)] # 0 1 2
        
        dataset = []
        for imgpth,t1,label,t2,points_list in self.base_dataset:
            idx = int(imgpth.split('/')[-1].split('.')[0])
            det_label = self.det_labels_dict[idx]
            scorelst = pesudo_scores_dict[idx]
            if det_label == 2:
                if max(scorelst) > thresh:
                    label = np.argmax(np.array(scorelst))
                else:
                    continue
                dataset.append([imgpth,t1,label,t2,points_list])
        self.dataset = dataset


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        ## 修改这里生成样本对
        img_path, pid, labelid, maskid, points_list = self.dataset[index]

        img = read_image(img_path)
        # print(points_list)
        img = CropFace(img,points_list)
        # print(img.shape,img_path,points_list)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            data = self.transform(image=img)
            img = data['image']
            img = img_to_tensor(img,self.normalize)
        if labelid == 0:
            label = 0
        else:
            label = 1
        return img, pid, label, labelid, maskid, img_path

class TTADataset(Dataset):
    """Image Dataset"""

    def __init__(self, dataset,transform=None, TTA_lst=None, pid_index=None):
        self.dataset = dataset
        self.transform = transform
        self.TTA_lst = TTA_lst
        self.pid_index = pid_index
        self.normalize = {"mean":[0.5, 0.5, 0.5],
                        "std":[0.5, 0.5, 0.5]}
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        ## 修改这里生成样本对
        img_path, pid, labelid, maskid, points_list = self.dataset[index]
        img = read_image(img_path)
        # print(points_list)
        img = CropFace(img,points_list)
        # print(img.shape,img_path,points_list)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs = []
        if self.transform is not None:
            data = self.transform(image=img)
            imgs.append(img_to_tensor(data['image'],self.normalize))
            for TTA in self.TTA_lst:
                imgs.append(img_to_tensor( (TTA(image=data['image']))['image'], self.normalize))
        return imgs, pid, labelid,  maskid, img_path
