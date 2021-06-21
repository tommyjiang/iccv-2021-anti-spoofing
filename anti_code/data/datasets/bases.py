# encoding: utf-8
import numpy as np


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, labels, masks = [], [], []
        for _, pid, labelid, maskid,_ in data:
            pids += [pid]
            labels += [labelid]
            masks += [maskid]
        pids = set(pids)
        masks = set(masks)
        labels = set(labels)
        num_pids = len(pids)
        num_masks = len(masks)
        num_imgs = len(data)
        num_labels = len(labels)
        return num_pids, num_imgs, num_labels, num_masks

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, val, test):
        num_train_pids, num_train_imgs, num_train_labels,num_train_masks = self.get_imagedata_info(train)
        num_val_pids, num_val_imgs, num_val_labels,num_val_masks = self.get_imagedata_info(val)
        num_test_pids, num_test_imgs, num_test_labels, num_test_masks = self.get_imagedata_info(test)

        # num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # masks")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_masks))
        print("  test     | {:5d} | {:8d} | {:9d}".format(num_val_pids, num_val_imgs, num_val_masks))
        print("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_masks))
        # print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")