# encoding: utf-8
# from anti-spoofing-challenge.anti_code.modeling.backbones.ssl_ccl import SSL_CCL
from .baseline import Baseline
from .backbones import EfficientNetAntiSpoof
from .backbones import CCL
from .backbones import SSL_CCL


def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    if "efficient" in cfg.MODEL.NAME:
        model = EfficientNetAntiSpoof(arch=cfg.MODEL.ARCH)
    elif "ccl" in cfg.MODEL.NAME:
        model0 = CCL(num_class=num_classes,encoder_name=cfg.MODEL.ENCODER)
        model1 = CCL(num_class=num_classes,encoder_name=cfg.MODEL.ENCODER)
        return model0,model1
    elif "ssl" in cfg.MODEL.NAME:
        model0 = SSL_CCL(num_class=num_classes,encoder_name = cfg.MODEL.ENCODER)
        model1 = SSL_CCL(num_class=num_classes,encoder_name = cfg.MODEL.ENCODER)
        return model0,model1
    else:
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    return model
