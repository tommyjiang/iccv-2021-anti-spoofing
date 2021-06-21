from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import argparse
import os
import sys
import torch

from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_ccl_loader
# from engine.trainer import do_train, do_train_with_center
from modeling import build_model
from layers import make_ccl_loss
from solver import make_optimizer, WarmupMultiStepLR

from utils.logger import setup_logger
from utils.meter import AverageMeter
from utils.accuracy import get_accuracy
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from time import strftime, localtime
from test import test_val,test_self_val
CLF=GaussianMixture(n_components=5, covariance_type='full')#GaussianMixture(n_components=2, covariance_type='full')
SCALER=StandardScaler()
from utils.gmm import Gmm_train
import numpy as np
from os import mkdir

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def train(cfg,logger):
    # prepare dataset
    train_loader, val_loader,  num_classes = make_ccl_loader(cfg)
    # print(num_classes)
    # prepare model
    model, _ = build_model(cfg, 2)
    print(model)
    model.load_param(cfg.TEST.WEIGHT)
    model.cuda()
    model.eval()
    res_txt = open('./gmm_res.txt', 'w')
    res_train = open('./gmm_train.txt', 'w')
    real_reatures = []
    with torch.no_grad():
        for i, (img0,img1,pair_label,label0,label1,maskid0,maskid1) in enumerate(train_loader):
            print("%i / %i"%(i,len(train_loader)))
            img0 = img0.cuda()
            score,feat = model(img0,img0)
            real_reatures.append(feat.cpu().numpy())
        real_reatures = np.concatenate(real_reatures,axis=0)
        print(real_reatures.shape)
        gmm = Gmm_train(clf=CLF,scaler=SCALER)
        gmm.train_projector(real_reatures,"./gmm_projector_0000")

        # predic = gmm.load_projector("./gmm_projector_0000")
        for j,(img,_,_,pth) in enumerate(val_loader):
            print('%i / %i'%(j,len(val_loader)))
            img = img.cuda()
            _,feat_val = model(img,img)
            for feat_,p in zip(feat_val,pth):
                feat_ = torch.unsqueeze(feat_,0)
                score_val = str("{:.5f}".format(gmm.project(feat_.cpu().numpy())[0]))
                pic_name = p.split('/')[-1]
                res_txt.write(pic_name+' '+str(score_val)+'\n')   
        res_txt.close()
        for i, (img0,img1,pair_label,label0,label1,maskid0,maskid1) in enumerate(train_loader):
            print("%i / %i"%(i,len(train_loader)))
            img0 = img0.cuda()
            score,feat = model(img0,img0)
            for feat__ in feat:
                feat__ = torch.unsqueeze(feat__,0)
                score_val = str("{:.5f}".format(gmm.project(feat__.cpu().numpy())[0]))
                res_train.write(str(score_val)+'\n')
        res_train.close()



def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True
    train(cfg,logger)

if __name__=="__main__":
    main()




    # if cfg.MODEL.IF_WITH_CENTER == 'no':
    

    # arguments = {}

    # do_train(
    #     cfg,
    #     model0,
    #     model1,
    #     train_loader,
    #     val_loader,
    #     optimizer,
    #     scheduler,      # modify for using self trained model
    #     loss_func,
    #     loss_constra,
    #     logger    # add for using self trained model
    # )