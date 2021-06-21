# encoding: utf-8
# from face_attr_dali.dataset.byted_kv_dali import batch_size
from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn, ccl_collate_fn, ult_collate_fn, TTA_collate_fn, p_collate_fn
from .datasets import init_dataset, ImageDataset, CCLDataset, UnlabelsImageDataset, PesudoDataset, TTADataset, MPesudoDataset
# from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms, build_albu_transforms, build_albu_transforms_TTA


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_labels
    train_set = ImageDataset(dataset.train,train_transforms)
    train_loader = DataLoader(train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,shuffle=True, num_workers=num_workers,collate_fn=train_collate_fn)
    # if cfg.DATALOADER.SAMPLER == 'softmax':
    #     train_loader = DataLoader(
    #         train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
    #         collate_fn=train_collate_fn
    #     )
    # else:
    #     train_loader = DataLoader(
    #         train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
    #         sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
    #         # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
    #         num_workers=num_workers, collate_fn=train_collate_fn
    #     )

    val_set = ImageDataset(dataset.val, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    test_set = ImageDataset(dataset.val, val_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader, test_loader, num_classes

def make_ccl_loader(cfg):
    if cfg.DATALOADER.TRANSFORMS=='torch':
        train_transforms = build_transforms(cfg, is_train=True)
        val_transforms = build_transforms(cfg, is_train=False)
    else:
        train_transforms = build_albu_transforms(cfg, is_train=True)
        val_transforms = build_albu_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_labels
    # num_classes = dataset.num_train_masks
    train_set = CCLDataset(dataset.train,train_transforms,dataset.pid_index)
    train_loader = DataLoader(train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,shuffle=True, num_workers=num_workers,collate_fn=ccl_collate_fn, pin_memory=True,drop_last=True)
    # if cfg.DATALOADER.SAMPLER == 'softmax':
    #     train_loader = DataLoader(
    #         train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
    #         collate_fn=train_collate_fn
    #     )
    # else:
    #     train_loader = DataLoader(
    #         train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
    #         sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
    #         # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
    #         num_workers=num_workers, collate_fn=train_collate_fn
    #     )

    val_set = ImageDataset(dataset.val, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    test_set = ImageDataset(dataset.test, val_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader, test_loader, num_classes

def make_ult_loader(cfg):
    
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    ult_dataset = UnlabelsImageDataset(dataset.test)
    ult_loader = DataLoader(ult_dataset,batch_size=cfg.SOLVER.IMS_PER_BATCH,shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS,collate_fn=ult_collate_fn, pin_memory=True,drop_last=True)
    return ult_loader


def make_pesudo_loader(cfg):
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    # val_transforms = build_albu_transforms(cfg, is_train=False)
    train_transforms = build_albu_transforms(cfg, is_train=True)
    pesudo_dataset = PesudoDataset(dataset.test, det_label_dir="../post_process/det_labels.txt", transform=train_transforms, data_scores = "")
    # pesudo_dataset = PesudoDataset(dataset.test, det_label_dir="../post_process/det_labels.txt", transform=val_transforms, data_scores = "../post_process/test_ori.txt")
    pesudo_loader = DataLoader(pesudo_dataset,batch_size=cfg.SOLVER.IMS_PER_BATCH,shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS,collate_fn=val_collate_fn, pin_memory=True,drop_last=True)
    return pesudo_loader

def make_mpesudo_loader(cfg):
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    # val_transforms = build_albu_transforms(cfg, is_train=False)
    train_transforms = build_albu_transforms(cfg, is_train=True)
    pesudo_dataset = MPesudoDataset(dataset.test, det_label_dir="../post_process/det_labels.txt", transform=train_transforms, data_scores = "./35_multi_res.txt")
    pesudo_loader = DataLoader(pesudo_dataset,batch_size=cfg.SOLVER.IMS_PER_BATCH,shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS,collate_fn=p_collate_fn, pin_memory=True,drop_last=True)
    return pesudo_loader

def make_TTA_loader(cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    val_transforms = build_albu_transforms(cfg, is_train=False)
    transforms_lst = build_albu_transforms_TTA(cfg)

    val_set = ImageDataset(dataset.val, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    test_set = TTADataset(dataset.test, val_transforms, TTA_lst=transforms_lst)
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=TTA_collate_fn
    )

    return val_loader, test_loader, 2