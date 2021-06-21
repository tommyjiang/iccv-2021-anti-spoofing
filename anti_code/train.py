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
from data import make_data_loader
# from engine.trainer import do_train, do_train_with_center
from modeling import build_model
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR

from utils.logger import setup_logger
from utils.meter import AverageMeter
from utils.accuracy import get_accuracy
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from time import strftime, localtime
from test import test_val


# def save_checkpoint(state, filename='checkpoint.pth.tar', bestname='model_best.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, bestname)

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

def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_func,logger):
    # losses = AverageMeter()
    # # loss_maes = AverageMeter()
    # accu_datas = AverageMeter()
    # val_datas = AverageMeter()
    best_acer = 0
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
    epoch_list, train_loss_list, train_accu_list, test_loss_list, test_accu_list = [], [], [], [], []
    # model.train()
    for epoch in range(0,cfg.SOLVER.MAX_EPOCHS):
        epoch_list.append(epoch+1)
        losses = AverageMeter()
    # loss_maes = AverageMeter()
        accu_datas = AverageMeter()
        val_datas = AverageMeter()
        test_loss_datas = AverageMeter()
        model.train()
        for i, (img,label,maskid) in enumerate(train_loader):
            # print(label,maskid)
            # print(img.shape)
            target_data = label.long()
            if torch.cuda.is_available():
                img = img.cuda()
                # target_data = label.long()
                target_data = target_data.cuda()
            score = model(img)
            loss_data = loss_func(score,target_data)
            acc = get_accuracy(score,target_data,topk=(1,))
            optimizer.zero_grad()
            loss = loss_data
            losses.update(loss.item(),img.size(0))
            accu_datas.update(acc[0].item(),1)
            loss.backward()
            optimizer.step()
            # print accu_gender
            if i % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'LossSum {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Accu {accu.val:.4f}% ({accu.avg:.4f}%)\t'
                            'learning_rate {learning_rate:.7f}'.format(
                    epoch, i, len(train_loader),loss=losses,
                    accu=accu_datas, learning_rate = scheduler.get_lr()[0]))
        train_loss_list.append(losses.avg)
        train_accu_list.append(accu_datas.avg)
        scheduler.step()
        if (epoch+1) % cfg.SOLVER.EVAL_PERIOD==0:
            _,apcer,bpcer,acer = test_val(model,val_loader)
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
                torch.save(model,os.path.join(cfg.OUTPUT_DIR,'model_best.pth'))
        # test_accu_list.append(val_datas.avg)
        # test_loss_list.append(test_loss_datas.avg)
        # figure_name = cfg.MODEL.NAME
        # save_pth = '../results/%s'%figure_name
        # draw_figure(epoch_list,train_loss_list,train_accu_list,test_loss_list,test_accu_list,save_pth,figure_name)
        if (epoch+1) % cfg.SOLVER.CHECKPOINT_PERIOD==0:
            torch.save(model,os.path.join(cfg.OUTPUT_DIR,'checkpoint_%i.pth'%epoch))


def train(cfg,logger):
    # prepare dataset
    train_loader, val_loader,  num_classes = make_data_loader(cfg)
    # print(num_classes)
    # prepare model
    model = build_model(cfg, num_classes)

    # if cfg.MODEL.IF_WITH_CENTER == 'no':
    print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
    optimizer = make_optimizer(cfg, model)
    # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
    #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    loss_func = make_loss(cfg, num_classes)     # modified by gu

    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
        print('Start epoch:', start_epoch)
        path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
        print('Path to the checkpoint of optimizer:', path_to_optimizer)
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
        optimizer.load_state_dict(torch.load(path_to_optimizer))
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                        cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        start_epoch = 0
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                        cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    else:
        start_epoch = 0
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                        cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        # print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

    # arguments = {}

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,      # modify for using self trained model
        loss_func,
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