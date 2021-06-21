# encoding: utf-8
import glob
import re
import os
import os.path as osp

from .bases import BaseImageDataset


class Anti_mask(BaseImageDataset):
    """
    Anti mask:HIFI mask data
    """
    dataset_dir = 'phase1'
    # normal_dir = ''

    def __init__(self, root='./raw_data/phase1', verbose=True, **kwargs):
        super(Anti_mask, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        # self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train,pid_index = self._process_dir(self.train_dir)
        val = self._process_val(self.val_dir)
        # gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Anti Mask loaded")
            self.print_dataset_statistics(train, val)

        self.train = train
        self.val = val
        self.pid_index = pid_index
        # self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_labels,self.num_train_masks = self.get_imagedata_info(self.train)
        self.num_val_pids, self.num_val_imgs, self.num_val_labels,self.num_val_masks = self.get_imagedata_info(self.val)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))

    def _process_dir(self, dir_path,raw_data_pth='../raw_data/phase1/train'):

        img_folder = os.listdir(dir_path)
        pid_container = set()
        dataset = []
        pid_index = {}
        no_use_list = []
        with open('./no_use.txt','r') as f:
            lines = f.readlines()
            for line in lines:
                no_use_list.append(line.strip())
        for folder in img_folder:
            # cur_mask = 1
            # print(folder)
            if folder in no_use_list:
                continue
            cur_label = 0
            cur_mask = int(folder.split('_')[2])
            cur_pid = int(folder.split('_')[1])
            if cur_mask==3:
                cur_mask-=1
            if cur_pid not in pid_index:
                pid_index[cur_pid] = []
            # if cur_mask not in pid_index[cur_pid]:
            #     pid_index[cur_pid][cur_mask]=[]
            # cur_mask = int(folder.split('_')[])
            pid_container.add(cur_pid)
            assert 0<= cur_mask <= 3
            if cur_mask>0:
                cur_label=1
            # if cur_label not in pid_index:
            #     pid_index[cur_label] = []
            cur_img_root_pth = osp.join(dir_path,folder)
            raw_img_root_pth = osp.join(raw_data_pth,folder)
            cur_folder_list = os.listdir(cur_img_root_pth)
            # points_list = []
            for img_name in cur_folder_list:
                points_list = []
                label_txt_pth = os.path.join(cur_img_root_pth,img_name)
                with open(label_txt_pth, 'r') as f:
                    lines = f.readlines()
                    try:
                        if len(lines)>=2:
                            points_list = list(map(int,lines[0].split(',')[:4]))
                        else:
                            points_list = []
                    except:
                        pass
                img_name = img_name.split('.')[0]+'.png'
                img_pth = osp.join(raw_img_root_pth,img_name)
                ## 按照pid
                pid_index[cur_pid].append((img_pth,cur_pid,cur_label,cur_mask,points_list))
                ## 按照实际的label取样
                # pid_index[cur_label].append((img_pth,cur_pid,cur_label,cur_mask,points_list))
                ## 重复采样透明面具
                if cur_mask==1:
                    for i in range(6):
                        dataset.append((img_pth,cur_pid,cur_label,cur_mask,points_list))
                else:
                    dataset.append((img_pth,cur_pid,cur_label,cur_mask,points_list))
        return dataset, pid_index
    def _process_val(self, dir_path,raw_data_pth='../raw_data/phase1/val',val_label = './val.txt'):

        img_folder = os.listdir(dir_path)
        pid_container = set()
        dataset = []
        # pid_index = {}
        label_dic = {}
        with open(val_label,'r') as f:
            lines = f.readlines()
            for line in lines:
                name,gt = line.strip().split(' ')
                label_dic[name] = gt
        # count = 0
        for folder in img_folder:
            # cur_mask = 1
            # print(folder)
            cur_label = 0
            cur_mask = 0
            cur_pid = 0
            # cur_mask = int(folder.split('_')[])
            pid_container.add(cur_pid)
            # assert 0<= cur_mask <= 3
            # if cur_mask>0:
            #     cur_label=1
            cur_img_root_pth = osp.join(dir_path,folder)
            raw_img_root_pth = osp.join(raw_data_pth,folder)
            cur_folder_list = os.listdir(cur_img_root_pth)
            # points_list = []
            for img_name in cur_folder_list:
                points_list = []
                label_txt_pth = os.path.join(cur_img_root_pth,img_name)
                with open(label_txt_pth, 'r') as f:
                    lines = f.readlines()
                    try:
                        if len(lines)>=2:
                            points_list = list(map(int,lines[0].split(',')[:4]))
                        else:
                            points_list = []
                    except:
                        pass
                img_name = img_name.split('.')[0]+'.png'
                img_pth = osp.join(raw_img_root_pth,img_name)
                cur_mask = int(label_dic[img_name])
                if cur_mask>0:
                    cur_label=1
                dataset.append((img_pth,cur_pid,cur_label,cur_mask,points_list))
        return dataset
    