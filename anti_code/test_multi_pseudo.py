# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch
from tqdm import tqdm
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_ccl_loader, make_pesudo_loader, make_TTA_loader
# from engine.trainer import do_train, do_train_with_center
from modeling import build_model
from layers import make_ccl_loss
from solver import make_optimizer, WarmupMultiStepLR

from utils.logger import setup_logger
from utils.meter import AverageMeter
from utils.accuracy import get_accuracy
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from eval_acer import eval_res

from time import strftime, localtime
from test import test_val,test_self_val
from layers.arch_face import OCCL


# 滑动平均的更新策略
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # alpha = 1-(1-alpha)*
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def test(model, test_loader, thresh=0.5, save_pth='./temp_res_test.txt', label_path="./"):
    model.eval()
    # model_name = cfg.TEST.WEIGHT.split('/')[-2]+'_'+cfg.TEST.WEIGHT.split('/')[-1].split('.')[0]
    res_txt = open(save_pth, 'w')
    # res_dic = {}
    with torch.no_grad():
        for i,(imgs,_,_,img_pth) in tqdm(enumerate(test_loader)):
            imgs = imgs.cuda()
            # b, t, c, h, w = imgs.size()
            # imgs = imgs.view(-1, c, h, w)
            score, score_mlti = model(imgs, None, isPair=False)
            
            pred_logit = torch.nn.Softmax(dim=1)(score_mlti)
            # pred_logit = pred_logit.view(b, t, -1)
            # pred_logit = torch.mean(pred_logit, dim=1)
            for pth, logit in zip(img_pth, pred_logit.cpu().numpy()):
                logit = str("{:.5f} {:.5f} {:.5f}".format(logit[0], logit[1], logit[2]))
                pic_name = pth.split('/')[-1]
                res_txt.write(pic_name+' '+logit+'\n')
    res_txt.close()
    # thresh,apcer,bpcer,acer = eval_res(os.path.join(save_pth, "temp_res_test.txt"), label_path, eval_val="heihei", thresh=thresh)
    # return thresh,apcer,bpcer,acer



def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
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
    pesudo_dir = os.path.join(cfg.OUTPUT_DIR, "pesudo_scores")

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if pesudo_dir and not os.path.exists(pesudo_dir):
        os.makedirs(pesudo_dir)


    cudnn.benchmark = True

    train_loader, val_loader, test_loader, num_classes = make_ccl_loader(cfg)
    model, _ = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    

    pesudo_loader = make_pesudo_loader(cfg)
    epoch_idx = 0
    pesudo_pth = os.path.join("35_multi_res.txt")
    test(model,test_loader, save_pth=pesudo_pth, thresh=0.5, label_path="")
    pesudo_loader.dataset.update_labels(data_scores=pesudo_pth, thresh=0.8)


if __name__ == '__main__':
    main()