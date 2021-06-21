# encoding: utf-8
# from anti-spoofing-challenge.anti_code.data import transforms
import torchvision.transforms as T

from .transforms import RandomErasing
import albumentations as A


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.TARGET_TRAIN),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform


def build_albu_transforms(cfg, is_train=True):
    if is_train:
        transforms = A.Compose([
            A.Resize(width = 291, height=291),
            A.RandomSunFlare(flare_roi=[0, 0, 0.3, 0.8],num_flare_circles_lower=0,num_flare_circles_upper=1,src_radius=200,p=0.1),
            A.RandomCrop(width=224,height=224),
            A.JpegCompression(quality_lower=80,quality_upper=100),
            A.RandomBrightnessContrast(p=0.5),
            A.FancyPCA(p=0.1),
            A.GaussNoise(p=0.3),
            A.HueSaturationValue(hue_shift_limit=5,sat_shift_limit=5,val_shift_limit=20),
            A.HorizontalFlip(p=0.5),
            A.Cutout(p=0.5)
        ])
    else:
        transforms = A.Compose([
            A.Resize(width = 291, height=291),
            A.CenterCrop(height = 224, width=224),
        ])
    return transforms

def build_albu_transforms_TTA(cfg):

    transforms_lst = [A.Compose([A.HorizontalFlip(p=1)])]

    return transforms_lst