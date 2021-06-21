# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader, make_ccl_loader
# from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger
import torch.nn as nn
from eval_acer import eval_res

def test_val(model, val_loader, save_pth='./'):
    model.eval()
    # model_name = cfg.TEST.WEIGHT.split('/')[-2]+'_'+cfg.TEST.WEIGHT.split('/')[-1].split('.')[0]
    res_txt = open(os.path.join(save_pth,'./temp_res.txt'), 'w')
    res_dic = {}
    with torch.no_grad():
        for i,(img,_,_,img_pth) in enumerate(val_loader):
            print("%i / %i"%(i+1,len(val_loader)))
            # if i%400==0:
            #     print("%i / %i"%(i+1,len(val_loader)))
            if torch.cuda.is_available():
                img = img.cuda()
                model = model.cuda()
            if 'ccl' in cfg.MODEL.NAME:
                score = model(img,img)
            else:
                score = model(img)
            for sc,pth in zip(list(score),img_pth):
                pred_logit = nn.Softmax()(sc)
                # print(pred_logit)
                logit = str("{:.5f}".format(pred_logit.cpu().numpy()[0]))
                _, pred = score.topk(1, 1, True, True)
                pred = str(pred.cpu().numpy()[0][0])
                # print(pth)
                pic_name = pth.split('/')[-1]
                # print(pic_name)
                res_dic[pic_name] = logit
                res_txt.write(pic_name+' '+pred+' '+logit+'\n')
    res_txt.close()
    with open(os.path.join(save_pth,'temp_submit.txt'), 'w') as f:
        for j in range(4645):
            name = '%04d.png'%(j+1)
            f.write(name+' '+res_dic[name]+'\n')
    thresh,apcer,bpcer,acer = eval_res("./temp_submit.txt","./val.txt",eval_val='offical')
    return thresh,apcer,bpcer,acer
def test_self_val(model, val_loader, save_pth='./'):
    model.eval()
    # model_name = cfg.TEST.WEIGHT.split('/')[-2]+'_'+cfg.TEST.WEIGHT.split('/')[-1].split('.')[0]
    res_txt = open(os.path.join(save_pth,'./temp_res.txt'), 'w')
    # res_dic = {}
    with torch.no_grad():
        for i,(img,_,_,img_pth) in enumerate(val_loader):
            # print("%i / %i"%(i+1,len(val_loader)))
            if i%400==0:
                print("%i / %i"%(i+1,len(val_loader)))
            if torch.cuda.is_available():
                img = img.cuda()
                model = model.cuda()
            if 'ccl' in cfg.MODEL.NAME:
                score = model(img,img)
            else:
                score = model(img)
            for sc,pth in zip(list(score),img_pth):
                pred_logit = nn.Softmax()(sc)
                # print(pred_logit)
                logit = str("{:.5f}".format(pred_logit.cpu().numpy()[0]))
                _, pred = score.topk(1, 1, True, True)
                pred = str(pred.cpu().numpy()[0][0])
                # print(pth)
                pic_name = pth[0].split('/')[-3]+'/'+img_pth[0].split('/')[-2]+'/'+img_pth[0].split('/')[-1]
                # print(pic_name)
                # res_dic[pic_name] = logit
                res_txt.write(pic_name+' '+logit+'\n')
            # pic_name = img_pth[0].split('/')[-3]+'/'+img_pth[0].split('/')[-2]+'/'+img_pth[0].split('/')[-1]
            # res_dic[pic_name] = logit
            # res_txt.write(pic_name+' '+logit+'\n')
    res_txt.close()
    thresh,apcer,bpcer,acer = eval_res("./temp_res.txt","./labels/valid_label.txt")
    return thresh,apcer,bpcer,acer
def inference(cfg, model, val_loader, save_pth='./'):
    # 0代表是真人，1代表是mask attack
    # model = nn.DataParallel(model)
    model.eval()
    model_name = cfg.TEST.WEIGHT.split('/')[-2]+'_'+cfg.TEST.WEIGHT.split('/')[-1].split('.')[0]
    res_txt = open(os.path.join(save_pth,'./%s_res.txt'%model_name), 'w')
    res_dic = {}
    with torch.no_grad():
        for i,(img,_,_,img_pth) in enumerate(val_loader):
            print("%i / %i"%(i+1,len(val_loader)))
            if torch.cuda.is_available():
                img = img.cuda()
                model = model.cuda()
            if 'ccl' in cfg.MODEL.NAME:
                score = model(img,img)
            else:
                score = model(img)
            pred_logit = nn.Softmax()(score)
            logit = str("{:.5f}".format(pred_logit.cpu().numpy()[0][0]))
            _, pred = score.topk(1, 1, True, True)
            pred = str(pred.cpu().numpy()[0][0])
            pic_name = img_pth[0].split('/')[-1]
            res_dic[pic_name] = logit
            res_txt.write(pic_name+' '+pred+' '+logit+'\n')
    res_txt.close()
    submit_txt = os.path.join(save_pth,'%s_submit.txt'%model_name)
    with open(submit_txt, 'w') as f:
        for j in range(len(val_loader)):
            name = '%04d.png'%(j+1)
            f.write(name+' '+res_dic[name]+'\n')
    eval_res(submit_txt,'./val.txt',isShow=True)


def inference_parallel(cfg, model, val_loader, save_pth='./'):
    # 0代表是真人，1代表是mask attack
    # model = nn.DataParallel(model)
    model.eval()
    model_name = cfg.TEST.WEIGHT.split('/')[-2]+'_'+cfg.TEST.WEIGHT.split('/')[-1].split('.')[0]
    res_txt = open(os.path.join(save_pth,'./%s_res.txt'%model_name), 'w')
    res_dic = {}
    with torch.no_grad():
        for i,(img,_,_,img_pth) in enumerate(val_loader):
            print("%i / %i"%(i+1,len(val_loader)))
            if torch.cuda.is_available():
                img = img.cuda()
                model = model.cuda()
            if 'ccl' in cfg.MODEL.NAME:
                score = model(img,img)
            else:
                score = model(img)
            for sc,pth in zip(list(score),img_pth):
                pred_logit = nn.Softmax()(sc)
                # print(pred_logit)
                logit = str("{:.5f}".format(pred_logit.cpu().numpy()[0]))
                _, pred = score.topk(1, 1, True, True)
                pred = str(pred.cpu().numpy()[0][0])
                # print(pth)
                pic_name = pth.split('/')[-1]
                # print(pic_name)
                res_dic[pic_name] = logit
                res_txt.write(pic_name+' '+pred+' '+logit+'\n')
    res_txt.close()
    submit_txt = os.path.join(save_pth,'%s_submit.txt'%model_name)
    with open(submit_txt, 'w') as f:
        for j in range(173620):# 4645
            name = '%04d.png'%(j+1)
            f.write(name+' '+res_dic[name]+'\n')
    # eval_res(submit_txt,'./val.txt',isShow=True, eval_val='offical')
    
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

    train_loader, val_loader, num_classes = make_ccl_loader(cfg)
    if 'ccl' in cfg.MODEL.NAME:
        model,_ = build_model(cfg, num_classes)
    else:
        model = build_model(cfg,num_classes)
    model.load_param(cfg.TEST.WEIGHT)

    # inference(cfg, model, val_loader,'../results')
    inference_parallel(cfg, model, val_loader, '../results')

if __name__ == '__main__':
    main()
