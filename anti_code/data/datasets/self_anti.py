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
 
    def __init__(self, root='../raw_data/phase1', train_label="../raw_data/phase1/train_label.txt",
                valid_label="../extra_data/labels/phase2_val_training.txt", test_label="../extra_data/labels/test_label.txt", verbose=True, **kwargs):

        super(Self_anti, self).__init__()
        self.root = root
        self.train_dir = train_label
        self.val_dir = valid_label
        self.test_dir = test_label

        self._check_before_run()

        train,pid_index = self._process_dir(self.train_dir)
        val, _ = self._process_dir_val(self.val_dir)
        test, _ = self._process_dir_val(self.test_dir)
        
        if verbose:
            print("=> Self anti loaded")
            self.print_dataset_statistics(train, val, test)

        self.train = train
        self.val = val
        self.test = test
        self.pid_index = pid_index

        self.num_train_pids, self.num_train_imgs, self.num_train_labels,self.num_train_masks = self.get_imagedata_info(self.train)
        self.num_val_pids, self.num_val_imgs, self.num_val_labels,self.num_val_masks = self.get_imagedata_info(self.val)
        self.num_test_pids, self.num_test_imgs, self.num_test_labels,self.num_test_masks = self.get_imagedata_info(self.test)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.isfile(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.isfile(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.isfile(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))


    def _process_dir(self, file_txt,train=True,bbox_root = '../extra_data/pts_v2/phase1'):
        pid_index = {}
        dataset = []
        with open(file_txt,'r') as f:
            lines = f.readlines()
            for line in lines:
                pth, label = line.strip().split(' ')
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
                    ff_lines = ff.readlines()
                    try:
                        if len(ff_lines)>=2:
                            points_list = list(map(int,ff_lines[0].split(',')[:4]))
                        else:
                            points_list = []
                    except:
                        pass
                pid_index[pid].append((imgpth,pid,label,maskid,points_list))
                dataset.append((imgpth,pid,label,maskid,points_list))

        return dataset,pid_index

    def _process_dir_val(self, file_txt,bbox_root = '../extra_data/pts_v2/phase1'):
        dataset = []
        with open(file_txt,'r') as f:
            for line in f.readlines():
                pth, label = line.strip().split(' ')
                imgpth = osp.join(self.root,pth)
                label = int(label)
                if label > 0:
                    label = 1
                elif label == 0:
                    label = 0
                else:
                    label = -1  #ignore

                ## 读入bbox内容
                points_list = []
                bbox_pth = pth.replace('.png','.txt')
                with open(osp.join(bbox_root,bbox_pth),'r') as ff:
                    ff_lines = ff.readlines()
                    try:
                        if len(ff_lines)>=2:
                            points_list = list(map(int,ff_lines[0].split(',')[:4]))
                        else:
                            points_list = []
                    except:
                        pass
                dataset.append((imgpth,0,label,0,points_list))

        return dataset,None
        

