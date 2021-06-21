# encoding: utf-8
# from anti-spoofing-challenge.anti_code.data import datasets
import os.path as osp

from .bases import BaseImageDataset
import os

class Self_anti(BaseImageDataset):
    """
    Anti mask:HIFI mask data
    """
    dataset_dir = 'phase1'
    # normal_dir = ''

    def __init__(self, root='../raw_data/phase1', train_label="./labels/train_label.txt",valid_label="./labels/valid_label.txt",verbose=True, **kwargs):
        super(Self_anti, self).__init__()
        self.root = root
        self.train_dir = train_label
        self.val_dir = valid_label
        # self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train,pid_index = self._process_dir(self.train_dir)
        val,_ = self._process_dir(self.val_dir)
        # val,_ = self._process_offical_dir()
        # gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Self anti loaded")
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
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.isfile(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.isfile(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))

    def _process_dir(self, file_txt,train=True,bbox_root = '../extra_data/pts/phase1'):
        pid_index = {}
        dataset = []
        with open(file_txt,'r') as f:
            lines = f.readlines()
            for line in lines:
                # print(line)
                pth, label = line.strip().split(' ')
                # print(label)
                imgpth = osp.join(self.root,pth)
                _,pid,maskid,_,_,_ = pth.split('/')[1].split('_')
                pid = int(pid)
                maskid = int(maskid)
                if maskid==3:
                    maskid-=1
                if train:
                    label = 1-int(label)
                else:
                    if int(label)>0:
                        label = 1
                    else:
                        label = int(label)
                if pid not in pid_index:
                    pid_index[pid] = []
                ## 读入bbox内容
                points_list = []
                bbox_pth = pth.replace('.png','.txt')
                with open(osp.join(bbox_root,bbox_pth),'r') as ff:
                    l = ff.readlines()
                    try:
                        points_list = list(map(int,l[0].split(',')[:4]))
                    except:
                        pass
                pid_index[pid].append((imgpth,pid,label,maskid,points_list))
                dataset.append((imgpth,pid,label,maskid,points_list))
        # print(dataset[0])
        return dataset,pid_index
    def _process_offical_dir(self, folder_pth="../raw_data/phase1/val",train=True,bbox_root = '../extra_data/pts/phase1/val'):
        folder_list = os.listdir(folder_pth)
        dataset = []
        pid_index = {}
        for folder in folder_list:
            cur_pth = osp.join(folder_pth,folder)
            for name in os.listdir(cur_pth):
                points_list = []
                imgpth = osp.join(cur_pth,name)
                cur_bbox_pth = osp.join(bbox_root,folder,name.replace('.png','.txt'))
                with open(cur_bbox_pth,'r') as f:
                    lines = f.readlines()
                    try:
                        points_list = list(map(int,lines[0].split(',')[:4]))
                    except:
                        pass
                dataset.append((imgpth,0,0,0,points_list)) 
        return dataset,pid_index       

