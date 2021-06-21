
import os, numpy, random, time, math
import torch
import torch.nn.functional as F
from tqdm import tqdm
# from ssl_lib.algs import utils as alg_utils
# from ssl_lib.models import utils as model_utils
# from ssl_lib.consistency.builder import gen_consistency
from layers.regulaizer import Distribution_Loss
# from ssl_lib.param_scheduler import scheduler
from solver import lr_scheduler as scheduler
# from ssl_lib.utils import  Bar, AverageMeter
from utils.meter import AverageMeter
# from process.bar import Bar as Bar
# from .supervised import supervised_train
from layers.cross_entropy import CrossEntropy
from ssl_alg.builder import gen_ssl_alg
from modeling import build_model
from ssl_alg.imprint import imprint
from solver import make_optimizer,WarmupMultiStepLR
from test import test_val,test_self_val
import argparse
import os
import sys
import torch

from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_ccl_loader, make_ult_loader
from utils.logger import setup_logger
from layers import make_ccl_loss
from eval_acer import eval_res
import torch.optim as optim


LABELED_FEAT_TABLES=None
UNLABELED_FEAT_TABLES=None

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # alpha = 1-(1-alpha)*
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

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

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def infer_interleave(forward_fn, inputs_train,cfg,bs):
    merge_one_batch = 0
    interleave_flag = 1
    if interleave_flag:
        inputs_interleave = list(torch.split(inputs_train, bs))
        inputs_interleave = interleave(inputs_interleave, bs)
        if merge_one_batch:
            inputs_interleave = [torch.cat(inputs_interleave)] ####
    else:
        inputs_interleave = [inputs_train]

    out_lists_inter = [forward_fn(inputs_interleave[0],return_fmap=True)]
    for inputs in inputs_interleave[1:]:
        out_lists_inter.append(forward_fn(inputs,return_fmap=True))

    for ret_id in [-1,-3]:
        ret_list=[]
        for o_list in out_lists_inter:
            ret_list.append(o_list[ret_id])
        # put interleaved samples back
        if interleave_flag:
            if merge_one_batch:
                ret_list = list(torch.split(ret_list[0], bs))
            ret_list = interleave(ret_list, bs)
            feat_l = ret_list[0]
            feat_u_w, feat_u_s = torch.cat(ret_list[1:],dim=0).chunk(2)
            #feat_l,feat_u_w, feat_u_s = ret_list
        else:
            feat_l = ret_list[0][:bs]
            feat_u_w, feat_u_s = ret_list[0][bs:].chunk(2)
        if ret_id==-1:
            logits_l,logits_u_w, logits_u_s = feat_l,feat_u_w, feat_u_s
        else:
            cur_feat_l = feat_l
            cur_feat_u = feat_u_w
            cur_feat_s = feat_u_s
            feat_target = torch.cat((feat_l, feat_u_w), dim=0)
    return  logits_l,logits_u_w, logits_u_s,cur_feat_l,cur_feat_u,cur_feat_s,feat_target

def train(epoch,train_loader , model,source_model,ema_teacher,optimizer,lr_scheduler, ssl_alg, cfg,device,loss_func,loss_constra, logger):
    # if cfg.coef==0 and cfg.lambda_mmd==0 and cfg.lambda_kd==0:
    #     loss, acc = supervised_train(epoch,train_loader, model,optimizer,lr_scheduler, cfg,device)
    #     return (loss, loss, 0, 0, 0,acc, 0,0,0,0)
    model.train()
    ema_teacher.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ce = AverageMeter()
    losses_ssl = AverageMeter()
    losses_mmd = AverageMeter()
    losses_kd = AverageMeter()
    masks_ssl = AverageMeter()
    masks_mmd = AverageMeter()
    masks_kd = AverageMeter()
    labeled_acc = AverageMeter()
    unlabeled_acc = AverageMeter()

    mmd_criterion = Distribution_Loss(loss='mmd').to(device)
    kd_criterion = Distribution_Loss(loss='mse').to(device)
    consistency_criterion = CrossEntropy()
    # n_iter =   24839 // 192## done

    n_iter =   33767 // cfg.SOLVER.IMS_PER_BATCH## done
    # n_iter = cfg.n_imgs_per_epoch // cfg.l_batch_size
    # bar = Bar('Training', max=n_iter)
    end = time.time()
    for batch_idx, (data_l, data_u) in enumerate(train_loader):
        # img0,img1,pair_label,label0,label1,maskid0,maskid1 = data_l
        inputs_l0, inputs_l1, pair_label, labels0, labels1, maskid0, maskid1 = data_l
        inputs_l0, inputs_l1, pair_label, labels0, labels1, maskid0, maskid1 = inputs_l0.to(device), inputs_l1.to(device), pair_label.to(device), labels0.to(device),labels1.to(device),maskid0.to(device),maskid1.to(device)
        inputs_u_w, inputs_u_s, labels_u = data_u
        inputs_u_w, inputs_u_s, labels_u = inputs_u_w.to(device), inputs_u_s.to(device), labels_u.to(device)
        inputs_train = torch.cat((inputs_l0, inputs_u_w, inputs_u_s), dim=0)
        # inputs_train_0 = torch.cat((inputs_l1, inputs_u_w, inputs_u_s), dim=0)
        data_time.update(time.time() - end)

        bs = inputs_l0.size(0)
        cur_iteration = epoch * n_iter + batch_idx
        # cur_iteration = epoch*cfg.per_epoch_steps+batch_idx

        forward_fn = model.forward
        '''
        ret_list = forward_fn(inputs_train, return_fmap=True)
        logits_l = ret_list[-1][:bs]
        logits_u_w, logits_u_s = ret_list[-1][bs:].chunk(2)
        feat_l = ret_list[-3][:bs]
        feat_u_w, feat_u_s = ret_list[-3][bs:].chunk(2)
        feat_target = torch.cat((feat_l, feat_u_w), dim=0)
        '''

        ## 
        score0,score_mask0,feat0,score1,score_mask1,feat1 = model(inputs_l0,inputs_l1)
        score0_ema,_,feat0_ema,score1_ema,_,feat1_ema = ema_teacher(inputs_l0,inputs_l1,flag=True)
        loss_mse = torch.nn.MSELoss()

        loss_c = loss_mse(feat0,feat0_ema)+loss_mse(feat1, feat1_ema)

        loss_ccl = 0.5*loss_c+loss_func(score0,labels0,score1,labels1)+loss_func(score_mask0,maskid0,score_mask1,maskid1)+(loss_constra(feat0,feat1_ema,pair_label)+loss_constra(feat1,feat0_ema,pair_label))*0.5*0.7
        logits_l,logits_u_w, logits_u_s,feat_l,feat_u_w,feat_u_s,feat_target  = infer_interleave(forward_fn, inputs_train,cfg, bs)
        L_supervised = F.cross_entropy(logits_l, labels0)

        # calc total loss
        coef = scheduler.linear_warmup(0, 120, cur_iteration+1)
        L_consistency = torch.zeros_like(L_supervised) 
        mask = torch.zeros_like(L_supervised)
        coef_flag = 0  
        if coef_flag > 0:
            # get target values
            if ema_teacher is not None: # get target values from teacher model
                ema_forward_fn = ema_teacher.forward
                ema_logits = ema_forward_fn(inputs_train,return_fmap=True)[-1]
                ema_logits_u_w, _ = ema_logits[bs:].chunk(2)
            else:
                ema_forward_fn = forward_fn
                ema_logits_u_w = logits_u_w

            # calc consistency loss
            model.module.update_batch_stats(False)
            y, targets, mask = ssl_alg(
                stu_preds = logits_u_s,
                tea_logits = ema_logits_u_w.detach(),
                w_data = inputs_u_w,
                s_data = inputs_u_s,
                stu_forward = forward_fn,
                tea_forward = ema_forward_fn
            )
            model.module.update_batch_stats(True)
            L_consistency = consistency_criterion(y, targets, mask)
        L_mmd = torch.zeros_like(L_supervised)  
        mmd_mask_u = torch.zeros_like(L_supervised)
        lambda_mmd = 50   
        if lambda_mmd>0:
            mmd_mask_l = get_mask(logits_l,0.7,  num_class=2)
            mmd_mask_u = get_mask(logits_u_w,0.7,  num_class=2)
            if mmd_mask_l.sum()>0 and mmd_mask_u.sum()>0:
                cur_feat_l, cur_feat_u = update_feat_table(feat_l,feat_u_w,128,128, mask_l=mmd_mask_l, mask_u=mmd_mask_u)

                if cur_iteration>10 and len(cur_feat_l)>20:
                    L_mmd = mmd_criterion(cur_feat_l, cur_feat_u)
        L_kd = torch.zeros_like(L_supervised)
        kd_mask = torch.zeros_like(L_supervised)
        lambda_kd_flag = 0  
        if lambda_kd_flag>0:
            src_inputs = torch.cat((inputs_l0, inputs_u_w), dim=0)
           
            with torch.no_grad():
                src_out_list = source_model(src_inputs,return_fmap=True)
            src_logits = src_out_list[-1]
            kd_mask = get_mask(src_logits,0.7, num_class=2)
            feat_source = src_out_list[-3].detach()
            L_kd = kd_criterion(feat_target, feat_source,mask=kd_mask, reduction='none')
            
            del src_out_list,  src_inputs
    
        lambda_mmd = scheduler.linear_warmup(lambda_mmd, 120, cur_iteration+1)
        lambda_kd = scheduler.linear_warmup(0, 120, cur_iteration+1)
        loss = L_supervised + coef * L_consistency + lambda_mmd * L_mmd + lambda_kd * L_kd + loss_ccl

        # update parameters
        cur_lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=5)
        optimizer.step()
        lr_scheduler.step()

        ## update ema model
        update_ema_variables(model,ema_teacher,0.996,epoch*n_iter+batch_idx)
        # if cfg.ema_teacher:
        #     model_utils.ema_update(
        #         ema_teacher, model, cfg.ema_teacher_factor,
        #         cfg.weight_decay * cur_lr if cfg.ema_apply_wd else None, 
        #         cur_iteration if cfg.ema_teacher_warmup else None,
        #         ema_train=cfg.ema_teacher_train)
        # calculate accuracy for labeled data
        acc_l = (logits_l.max(1)[1] == labels0).float().mean()
        acc_ul = (logits_u_w.max(1)[1] == labels_u).float().mean()

        losses.update(loss.item())
        losses_ce.update(L_supervised.item())
        losses_ssl.update(L_consistency.item())
        losses_mmd.update(L_mmd.item())
        losses_kd.update(L_kd.item())
        labeled_acc.update(acc_l.item())
        unlabeled_acc.update(acc_ul.item())
        batch_time.update(time.time() - end)
        masks_ssl.update(mask.mean())
        masks_mmd.update(mmd_mask_u.mean())
        masks_kd.update(kd_mask.mean())
        end = time.time()
        
        if (batch_idx+1) % 100==0:
            logger.info("{:3}/{batch:4}/{iter:4}. LR:{lr:.6f}. Data:{dt:.3f}s. Batch:{bt:.3f}s. Loss:{loss:.4f}. Loss_CE:{loss_ce:.4f}. Loss_SSL:{loss_ssl:.4f}. Loss_MMD:{loss_mmd:.4f}. Loss_KD:{loss_kd:.4f}. Acc_L:{acc_l:.4f}.  Acc_U:{acc_u:.4f}. Loss_CE:{loss_ce:.4f}. Mask_SSL:{m_ssl:.4f}. Mask_MMD:{m_mmd:.4f}. Mask_KD:{m_kd:.4f}.".format(
                    epoch,
                    batch=batch_idx+1,
                    iter=n_iter,
                    lr=cur_lr,
                    dt=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_ce=losses_ce.avg,
                    loss_ssl=losses_ssl.avg,
                    loss_mmd=losses_mmd.avg,
                    loss_kd=losses_kd.avg,
                    acc_l=labeled_acc.avg,
                    acc_u=unlabeled_acc.avg,
                    m_ssl=masks_ssl.avg,m_mmd=masks_mmd.avg, m_kd=masks_kd.avg))
    return (losses.avg, losses_ce.avg, losses_ssl.avg, losses_mmd.avg, losses_kd.avg,labeled_acc.avg, unlabeled_acc.avg, masks_ssl.avg, masks_mmd.avg,masks_kd.avg)


def val_ssl(model, val_loader, save_pth='./', label_path="./"):
    model.eval()
    # model_name = cfg.TEST.WEIGHT.split('/')[-2]+'_'+cfg.TEST.WEIGHT.split('/')[-1].split('.')[0]
    res_txt = open(os.path.join(save_pth,'./temp_res_val.txt'), 'w')
    # res_dic = {}
    with torch.no_grad():
        for i,(img,_,_,img_pth) in tqdm(enumerate(val_loader)):
            if torch.cuda.is_available():
                img = img.cuda()
                model = model.cuda()
            score = model(img,x2=img,return_fmap=True)[-1]
            pred_logit = torch.nn.Softmax(dim=1)(score)
            for pth, logit in zip(img_pth, pred_logit.cpu().numpy()):
                logit = str("{:.5f}".format(logit[0]))
                pic_name = pth.split('/')[-1]
                res_txt.write(pic_name+' '+logit+'\n')
    res_txt.close()
    thresh,apcer,bpcer,acer = eval_res("./temp_res_val.txt", label_path, eval_val="heihei")
    return thresh,apcer,bpcer,acer

def test_ssl(model, test_loader, thresh=0.5, save_pth='./', label_path="./"):
    model.eval()
    # model_name = cfg.TEST.WEIGHT.split('/')[-2]+'_'+cfg.TEST.WEIGHT.split('/')[-1].split('.')[0]
    res_txt = open(os.path.join(save_pth,'./temp_res_test.txt'), 'w')
    # res_dic = {}
    with torch.no_grad():
        for i,(img,_,_,img_pth) in tqdm(enumerate(test_loader)):
            if torch.cuda.is_available():
                img = img.cuda()
                model = model.cuda()
            score = model(img,x2=img,return_fmap=True)[-1]
            pred_logit = torch.nn.Softmax(dim=1)(score)
            for pth, logit in zip(img_pth, pred_logit.cpu().numpy()):
                logit = str("{:.5f}".format(logit[0]))
                pic_name = pth.split('/')[-1]
                res_txt.write(pic_name+' '+logit+'\n')
    res_txt.close()


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

    # if cfg.MODEL.DEVICE == "cuda":
        # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True

    ##prepare model,dataloader,optimizer,schedler
    train_loader_lt, val_loader_lt, test_loader, num_classes = make_ccl_loader(cfg)
    train_loader_ult = make_ult_loader(cfg)
    # train_loader = zip(train_loader_lt,train_loader_ult)
    ssl_alg = gen_ssl_alg("cr",cfg)
    model0, model1 = build_model(cfg, num_classes)
    if torch.cuda.device_count()>1:
        model0 = model0.cuda()
        model1 = model1.cuda()
    model0.train()
    model1.train()
    # if cfg.imprint:
    # model0 = imprint(model0,train_loader_lt,num_classes,-1,"cuda")
    # model1 = imprint(model1,train_loader_lt,num_classes,-1,"cuda")
    model0 = torch.nn.DataParallel(model0)
    model1 = torch.nn.DataParallel(model1)
    source_model = None
    # source_model,_ = build_model(cfg, num_classes)
    # source_model.load_param('../logs/ssl_50_120epo_sgd_a/model_best.pth')
    # source_model.cuda()
     # build optimizer
    wd_params, non_wd_params = [], []
    for name, param in model0.named_parameters():
        if 'bn' in name or 'bias' in name:
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params, 'weight_decay': 0.0001}, {'params': non_wd_params, 'weight_decay': 0}]

    # optimizer = optim.SGD(param_list, lr=cfg.lr, momentum=cfg.momentum, weight_decay=0, nesterov=True)
    # optimizer = make_optimizer(cfg, model0)
    optimizer = optim.SGD(param_list, lr=0.01, momentum=0.9, weight_decay=0, nesterov=True)
    loss_func,loss_constra = make_ccl_loss(cfg, num_classes)
    # set lr scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    

    best_acer = 100.0
    print("start train ...")
    for epoch in range(0,cfg.SOLVER.MAX_EPOCHS):
        print(epoch)
        train_loader = zip(train_loader_lt,train_loader_ult)
        train_logs = train(epoch, train_loader, model0, source_model, model1, optimizer, lr_scheduler, ssl_alg, cfg,"cuda",loss_func,loss_constra,logger)
        if (epoch+1) % cfg.SOLVER.EVAL_PERIOD==0:
            thresh, apcer,bpcer,acer = val_ssl(model1, val_loader_lt, save_pth=cfg.OUTPUT_DIR, label_path = "../extra_data/labels/val.txt")
            logger.info("VALID APCER:{:.4f} BPCER:{:.4f} ACER:{:.4f}".format( apcer*100, bpcer*100, acer*100))
            test_ssl(model1, test_loader, thresh=thresh, save_pth=cfg.OUTPUT_DIR, label_path = None)

        if (epoch+1) % cfg.SOLVER.CHECKPOINT_PERIOD==0:
            torch.save(model1,os.path.join(cfg.OUTPUT_DIR,'checkpoint_%i.pth'%epoch))
    logger.info("best acer is {:.4f}".format(best_acer))

    ## train
    # train(cfg,logger)
if __name__=="__main__":
    main()
