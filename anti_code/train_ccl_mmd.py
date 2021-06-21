# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

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
import math
from layers.regulaizer import Distribution_Loss
import torch.nn.functional as F


# def save_checkpoint(state, filename='checkpoint.pth.tar', bestname='model_best.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, bestname)

## add mmd supervise
LABELED_FEAT_TABLES=None
UNLABELED_FEAT_TABLES=None
def get_mask(logits,threshold, num_class=10):
    ent = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    threshold = threshold * math.log(num_class)
    mask = ent.le(threshold).float()
    return mask
def update_feat_table(cur_feat_l,cur_feat_u,feat_table_size_l=-1,feat_table_size_u=-1,mask_l=None, mask_u=None):
    global LABELED_FEAT_TABLES,UNLABELED_FEAT_TABLES
    if mask_l is not None:
        mask_l = mask_l.nonzero().flatten()
        mask_u = mask_u.nonzero().flatten()
        cur_feat_l=cur_feat_l[mask_l]
        cur_feat_u=cur_feat_u[mask_u]
    if feat_table_size_l>0:
        if LABELED_FEAT_TABLES is None:
            LABELED_FEAT_TABLES = cur_feat_l
            UNLABELED_FEAT_TABLES = cur_feat_u
        else:
            LABELED_FEAT_TABLES = torch.cat([LABELED_FEAT_TABLES,cur_feat_l])
            UNLABELED_FEAT_TABLES = torch.cat([UNLABELED_FEAT_TABLES,cur_feat_u])
            if len(LABELED_FEAT_TABLES) > feat_table_size_l:
                LABELED_FEAT_TABLES = LABELED_FEAT_TABLES[-feat_table_size_l:]
            if len(UNLABELED_FEAT_TABLES) > feat_table_size_u:
                UNLABELED_FEAT_TABLES = UNLABELED_FEAT_TABLES[-feat_table_size_u:]
        feat_l = LABELED_FEAT_TABLES
        feat_u = UNLABELED_FEAT_TABLES
        LABELED_FEAT_TABLES=LABELED_FEAT_TABLES.detach()
        UNLABELED_FEAT_TABLES=UNLABELED_FEAT_TABLES.detach()
    else:
        feat_l = cur_feat_l
        feat_u = cur_feat_u
    
    return feat_l, feat_u

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

def do_train(cfg, model0,model1, train_loader, val_loader, optimizer, scheduler, loss_func,loss_constra,logger):
    # losses = AverageMeter()
    # # loss_maes = AverageMeter()
    # accu_datas = AverageMeter()
    # val_datas = AverageMeter()
    best_acer = 100
    if torch.cuda.device_count()>1:
        model0 = torch.nn.DataParallel(model0)
        model0 = model0.cuda()
        model1 = torch.nn.DataParallel(model1)
        model1 = model1.cuda()
    # epoch_list, train_loss_list, train_accu_list, test_loss_list, test_accu_list = [], [], [], [], []
    # model.train()
    loss_mse = torch.nn.MSELoss()
    loss_mmd = Distribution_Loss(loss='mmd').cuda()
    for epoch in range(0,cfg.SOLVER.MAX_EPOCHS):
        # epoch_list.append(epoch+1)
        losses = AverageMeter()
    # loss_maes = AverageMeter()
        accu_datas = AverageMeter()
        val_datas = AverageMeter()
        test_loss_datas = AverageMeter()
        model0.train()
        model1.train()
        for i, (img0,img1,pair_label,label0,label1,maskid0,maskid1) in enumerate(train_loader):
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
            score0,score_mask0,feat0,score1,score_mask1,feat1 = model0(img0,img1)
            score0_ema,_,feat0_ema,score1_ema,_,feat1_ema = model1(img0,img1,exp=True)
            ## 添加mmd loss监督
            L_super = F.cross_entropy(score0,label0)
            L_mmd = torch.zeros_like(L_super)
            mmd_mask_l = get_mask(score0,0.7,num_class=2)
            mmd_mask_u = get_mask(score1, 0.7, num_class=2)
            if mmd_mask_l.sum()>0 or mmd_mask_u.sum()>0:
                cur_feat_l, cur_feat_u = update_feat_table(feat0, feat1, feat_table_size_l=128, feat_table_size_u=128, mask_l=mmd_mask_l,mask_u=mmd_mask_u)
                if epoch > 5 and len(cur_feat_l)>20:
                # print("calc mmd")
                    L_mmd = loss_mmd(cur_feat_l,cur_feat_u)
            ## 添加mse监督
            # loss_mse = torch.nn.MSELoss()
            loss_ema_m = loss_mse(feat0,feat0_ema)+loss_mse(feat1,feat1_ema)
            # 添加三分类分支监督
            loss_data =50*L_mmd+0.5*loss_ema_m + loss_func(score_mask0,maskid0,score_mask1,maskid1)+loss_func(score0,label0,score1,label1)+0.7*(loss_constra(feat0,feat1_ema, target_data)+loss_constra(feat0_ema,feat1, target_data)) / 2
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
        # train_loss_list.append(losses.avg)
        # train_accu_list.append(accu_datas.avg)
        scheduler.step()
        if (epoch+1) % cfg.SOLVER.EVAL_PERIOD==0:
            _,apcer,bpcer,acer = test_val(model1,val_loader)
            # _,apcer,bpcer,acer = test_self_val(model1,val_loader)
            logger.info(" APCER:{:.4f} BPCER:{:.4f} ACER:{:.4f}".format( apcer*100, bpcer*100, acer*100))
            # model1.eval()
            # with torch.no_grad():
            #     for j,(img,label,maskid,_) in enumerate(val_loader):
            #         target_data = label.long()
            #         if torch.cuda.is_available():
            #             img = img.cuda()
            #             # target_data = label.long()
            #             target_data = target_data.cuda()
            #         score = model1(img,img)
            #         test_loss_data = loss_func(score,target_data,score,target_data)
            #         val_acc = get_accuracy(score,target_data,topk=(1,))
            #         val_datas.update(val_acc[0].item(),1)
            #         test_loss_datas.update(test_loss_data.item(),img.size(0))
            #         if (j+1) % 400 == 0:
            #             logger.info('Test: [{0}/{1}/{2}]\t'
            #                 'Accu {accu.val:.4f}% ({accu.avg:.4f}%)\t'.format(
            #         j+1, len(val_loader) // cfg.TEST.IMS_PER_BATCH, len(val_loader),
            #         accu=val_datas))
            if acer < best_acer:
                best_acer = acer
                torch.save(model1,os.path.join(cfg.OUTPUT_DIR,'model_best.pth'))
        # test_accu_list.append(val_datas.avg)
        # test_loss_list.append(test_loss_datas.avg)
        # figure_name = cfg.MODEL.NAME
        # save_pth = '../results/%s'%figure_name
        # draw_figure(epoch_list,train_loss_list,train_accu_list,test_loss_list,test_accu_list,save_pth,figure_name)
        if (epoch+1) % cfg.SOLVER.CHECKPOINT_PERIOD==0:
            torch.save(model1,os.path.join(cfg.OUTPUT_DIR,'checkpoint_%i.pth'%epoch))
    logger.info("best acer is {:.4f}".format(best_acer))

def train(cfg,logger):
    # prepare dataset
    train_loader, val_loader,  num_classes = make_ccl_loader(cfg)
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