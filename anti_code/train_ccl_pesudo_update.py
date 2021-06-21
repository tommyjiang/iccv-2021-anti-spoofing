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
from data import make_ccl_loader, make_pesudo_loader
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

import shutil

# def save_checkpoint(state, filename='checkpoint.pth.tar', bestname='model_best.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, bestname)


# 滑动平均的更新策略
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # alpha = 1-(1-alpha)*
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def draw_figure(epoch_list, train_loss_list, train_accu_list, test_loss_list, test_accu_list, save_dir, figure_name):
    plt.clf()
    plt.plot(epoch_list, train_loss_list, color = "yellow", marker='o', linewidth = 2, label = "train_loss")
    plt.plot(epoch_list, train_accu_list, color = "blue", marker='o', linewidth = 2, label = "train_accu") 
    plt.plot(epoch_list, test_loss_list, color = "red", marker='*', linewidth = 3, label = "test_loss")
    plt.plot(epoch_list, test_accu_list, color = "green", marker='o', linewidth = 2, label = "test_accu") 
    
    min_ce = min(test_loss_list)
    min_ce_index = test_loss_list.index(min_ce)
    plt.text(epoch_list[min_ce_index], min_ce + 0.001, '%.4f' % min_ce, ha='center', va= 'bottom', fontsize=9)  
    plt.text(epoch_list[min_ce_index], train_loss_list[min_ce_index] + 0.001, '%.4f' % train_loss_list[min_ce_index], ha='center', va= 'bottom', fontsize=9)  
    plt.text(epoch_list[min_ce_index], test_accu_list[min_ce_index] + 0.001, '%.4f' % test_accu_list[min_ce_index], ha='center', va= 'bottom', fontsize=9)  
    plt.text(epoch_list[min_ce_index], train_accu_list[min_ce_index] + 0.001, '%.4f' % train_accu_list[min_ce_index], ha='center', va= 'bottom', fontsize=9)  
    plt.vlines([epoch_list[min_ce_index]], 0, max(test_loss_list), color='black', linewidth=1.0, linestyle='--')

    max_train_accu = max(train_accu_list)
    max_train_accu_index = train_accu_list.index(max_train_accu)
    plt.text(epoch_list[max_train_accu_index], max_train_accu + 0.001, '%.4f' % max_train_accu, ha='center', va= 'bottom', fontsize=9, color = "red")

    max_test_accu = max(test_accu_list)
    max_test_accu_index = test_accu_list.index(max_test_accu)
    plt.text(epoch_list[max_test_accu_index], max_test_accu + 0.001, '%.4f' % max_test_accu, ha='center', va= 'bottom', fontsize=9, color = "red")

    plt.legend()
    plt.title(figure_name + strftime("%Y-%m-%d-%H-%M-%S", localtime()))
    plt.savefig(save_dir, dpi=300)


def val(model, val_loader, save_pth='./temp_res_val.txt', label_path="./"):
    model.eval()
    res_txt = open(save_pth, 'w')

    # res_dic = {}
    with torch.no_grad():
        for i,(img,_,_,img_pth) in tqdm(enumerate(val_loader)):
            if torch.cuda.is_available():
                img = img.cuda()
                model = model.cuda()
            if 'ccl' in cfg.MODEL.NAME:
                score = model(img,img)
            else:
                score = model(img)
            pred_logit = torch.nn.Softmax(dim=1)(score)
            for pth, logit in zip(img_pth, pred_logit.cpu().numpy()):
                logit = str("{:.5f}".format(logit[0]))
                pic_name = pth.split('/')[-1]
                res_txt.write(pic_name+' '+logit+'\n')
    res_txt.close()
    thresh,apcer,bpcer,acer = eval_res(save_pth, label_path, eval_val="heihei")
    return thresh,apcer,bpcer,acer

def test(model, test_loader, thresh=0.5, save_pth='./temp_res_test.txt', label_path="./"):
    model.eval()
    res_txt = open(save_pth, 'w')

    # res_dic = {}
    with torch.no_grad():
        for i,(img,_,_,img_pth) in tqdm(enumerate(test_loader)):
            if torch.cuda.is_available():
                img = img.cuda()
                model = model.cuda()
            if 'ccl' in cfg.MODEL.NAME:
                score = model(img,img)
            else:
                score = model(img)
            pred_logit = torch.nn.Softmax(dim=1)(score)
            for pth, logit in zip(img_pth, pred_logit.cpu().numpy()):
                logit = str("{:.5f}".format(logit[0]))
                pic_name = pth.split('/')[-1]
                res_txt.write(pic_name+' '+logit+'\n')
    res_txt.close()
    

def do_train(cfg, model0,model1, train_loader, val_loader, test_loader, pesudo_loader, optimizer, scheduler, loss_func,loss_constra,logger):
    # losses = AverageMeter()
    # # loss_maes = AverageMeter()
    # accu_datas = AverageMeter()
    # val_datas = AverageMeter()

    test_dir = cfg.OUTPUT_DIR + "/test_ckpt"
    pesudo_dir = cfg.OUTPUT_DIR + "/pesudo_ckpt"
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

    if not os.path.isdir(pesudo_dir):
        os.makedirs(pesudo_dir)

    loss_cls = torch.nn.CrossEntropyLoss()

    best_acer = 100
    occl_loss = OCCL(margin=3.0, feat_dim=2048)
    loss_mse = torch.nn.MSELoss()
    if torch.cuda.device_count()>1:
        model0 = torch.nn.DataParallel(model0)
        model0 = model0.cuda()
        model1 = torch.nn.DataParallel(model1)
        model1 = model1.cuda()
    # epoch_list, train_loss_list, train_accu_list, test_loss_list, test_accu_list = [], [], [], [], []
    # model.train()
    for epoch in range(0,cfg.SOLVER.MAX_EPOCHS):
        # epoch_list.append(epoch+1)
        losses = AverageMeter()
    # loss_maes = AverageMeter()
        accu_datas = AverageMeter()
        val_datas = AverageMeter()
        test_loss_datas = AverageMeter()

        model0.train()
        model1.train()
        for i, ((img0,img1,pair_label,label0,label1,maskid0,maskid1),
        (img,label,_,_))  in enumerate(zip(train_loader, pesudo_loader)):
            # print(label,maskid)
            # print(img.shape)
            target_data = pair_label.long()
            label0 = label0.long()
            label1 = label1.long()
            maskid0 = maskid0.long()
            maskid1 = maskid1.long()
            if torch.cuda.is_available():
                img0 = img0.cuda()
                img1 = img1.cuda()
                label0 = label0.cuda()
                label1 = label1.cuda()
                maskid0 = maskid0.cuda()
                maskid1 = maskid1.cuda()
                # target_data = label.long()
                target_data = target_data.cuda()
                img = img.cuda()
                label = label.long().cuda()

            score0,score_mask0,feat0,score1,score_mask1,feat1,feat_c1, feat_c2 = model0(img0,img1)
            score0_ema,_,feat0_ema,score1_ema,_,feat1_ema,_,_ = model1(img0,img1,exp=True)


            scorep,score_maskp,featp,scorep_,score_maskp_,featp_,feat_c1_p, feat_c2_p = model0(img,img)
            loss_pesudo = loss_cls(scorep, label)

            ## 添加mse监督
            # loss_mse = torch.nn.MSELoss()
            loss_ema_m = loss_mse(feat0,feat0_ema)+loss_mse(feat1,feat1_ema)
            # loss_occl = occl_loss(feat_c1,label0)+occl_loss(feat_c2,label1)
            # 添加三分类分支监督
            loss_data = 0.5 * loss_pesudo + loss_ema_m + loss_func(score_mask0,maskid0,score_mask1,maskid1)+loss_func(score0,label0,score1,label1)+0.7*(loss_constra(feat0,feat1_ema, target_data)+loss_constra(feat0_ema,feat1, target_data)) / 2
            # loss_mse = torch.nn.MSELoss()
            acc = get_accuracy(score0_ema,label0,topk=(1,))
            optimizer.zero_grad()
            loss = loss_data
            losses.update(loss.item(),img0.size(0))
            accu_datas.update(acc[0].item(),1)
            loss.backward()
            optimizer.step()
            update_ema_variables(model0,model1,0.996,epoch*len(train_loader)+i)
            # print accu_gender
            if i % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'LossSum {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Accu {accu.val:.4f}% ({accu.avg:.4f}%)\t'
                            'learning_rate {learning_rate:.7f}'.format(
                    epoch, i, len(train_loader),loss=losses,
                    accu=accu_datas, learning_rate = scheduler.get_last_lr()[0]))
        scheduler.step()

        if ((epoch+1) % cfg.SOLVER.PESUDO_UPDATE_PERIOD == 0) and ((epoch+1) >= cfg.SOLVER.PESUDO_SKIP):
            pesudo_pth = os.path.join(pesudo_dir, "pesudo_scores_{}.txt".format(epoch))
            test(model1,test_loader, save_pth=pesudo_pth, thresh=0.5, label_path="")
            pesudo_loader.dataset.update_labels(data_scores=pesudo_pth, thresh=0.9)


        if (epoch+1) % cfg.SOLVER.EVAL_PERIOD==0:
            thresh, apcer,bpcer,acer = val(model1,val_loader, save_pth=os.path.join(test_dir, "temp_res_val_{}.txt".format(epoch)), label_path="../extra_data/labels/val.txt")
            logger.info("VALID APCER:{:.4f} BPCER:{:.4f} ACER:{:.4f}".format( apcer*100, bpcer*100, acer*100))
            if ((epoch+1) % cfg.SOLVER.PESUDO_UPDATE_PERIOD == 0) and ((epoch+1) >= cfg.SOLVER.PESUDO_SKIP):
                shutil.copy(pesudo_pth, os.path.join(test_dir, "temp_res_test_{}.txt".format(epoch)))
            else:
                test(model1,test_loader, save_pth=os.path.join(test_dir, "temp_res_test_{}.txt".format(epoch)), thresh=0.5, label_path=None)

        
        if (epoch+1) % cfg.SOLVER.CHECKPOINT_PERIOD==0:
            torch.save(model1,os.path.join(cfg.OUTPUT_DIR,'ema_checkpoint_%i.pth'%epoch))
            torch.save(model0,os.path.join(cfg.OUTPUT_DIR,'checkpoint_%i.pth'%epoch))



def train(cfg,logger):
    # prepare dataset
    train_loader, val_loader, test_loader, num_classes = make_ccl_loader(cfg)
    pesudo_loader = make_pesudo_loader(cfg)
    # print(num_classes)
    # prepare model
    model0, model1 = build_model(cfg, num_classes)
    print(model0)

    # if cfg.MODEL.IF_WITH_CENTER == 'no':
    print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
    # 只对model0进行权重更新
    optimizer = make_optimizer(cfg, model0)
    # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
    #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    loss_func,loss_constra = make_ccl_loss(cfg, num_classes)     # modified by gu
    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
        print('Start epoch:', start_epoch)
        path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
        print('Path to the checkpoint of optimizer:', path_to_optimizer)
        model0.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
        optimizer.load_state_dict(torch.load(path_to_optimizer))
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                        cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        start_epoch = 0
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                        cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    else:
        start_epoch = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                        # cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        # print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

    # arguments = {}

    do_train(
        cfg,
        model0,
        model1,
        train_loader,
        val_loader,
        test_loader,
        pesudo_loader,
        optimizer,
        scheduler,      # modify for using self trained model
        loss_func,
        loss_constra,
        logger    # add for using self trained model
    )

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
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("anti_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True
    train(cfg,logger)


if __name__ == '__main__':
    main()