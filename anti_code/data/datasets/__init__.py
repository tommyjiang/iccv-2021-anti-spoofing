# encoding: utf-8
from .dataset_loader import ImageDataset,CCLDataset, UnlabelsImageDataset, PesudoDataset, TTADataset, MPesudoDataset
from .anti_mask import Anti_mask
from .self_anti import Self_anti
from .gmm_mask import Gmm_mask
from .test_mask import Test_mask

__factory = {
    'anti_mask':Anti_mask,
    'self_anti':Self_anti,
    'gmm_mask': Gmm_mask,
    'test_mask':Test_mask
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)