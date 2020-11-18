import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import cv2
from utils.meter import AverageMeter
from utils.metrics import R1_mAP,R1_mAP_eval,R1_mAP_Pseudo,R1_mAP_query_mining
from tqdm import tqdm
from .grid import GridMask


# grid = GridMask(96,320, 360, 0.6,1,0.8)

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import DistributedDataParallel
    from apex.parallel import convert_syncbn_model
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from tqdm import tqdm

from datasets import make_dataloader
import pickle


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')

    print("torch.cuda.device_count()",torch.cuda.device_count())
    if device:
        model.to(device)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            print("多卡训练")
            # model = DDP(model, delay_allreduce=True)  # 必须在initialze之后
            # model = nn.DataParallel(model)
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")  # 字母小写o,不是零。
            torch.distributed.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

            # model = convert_syncbn_model(model)
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
            # model = DistributedDataParallel(model, delay_allreduce=True)
            # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model = nn.DataParallel(model)
            # model = convert_syncbn_model(model)
        else:
            print("单卡训练")
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        model.to(device=0)


    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    # model.base._freeze_stages()
    logger.info('Freezing the stages number:{}'.format(cfg.MODEL.FROZEN))


    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    for epoch in range(1, epochs + 1):
        if epoch==5:
            print("balance 数据训练")
            # cfg.DATASETS.ROOT_DIR = '/home/lab3/bi/0716/Veri/ai_city/tools/mix_train_balance_flip.pkl'
            cfg.DATASETS.ROOT_DIR = 'datasets/mix_train_balance.pkl'
            train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)

        # model.base._freeze_stages()
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        # evaluator.reset()
        scheduler.step()
        model.train()
        # print(scheduler.get_lr()[0])
        for n_iter, (img, vid) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)

            #grid mask
            # img = grid(img)

            # score, feat,score_f1,score_f2,score_f3,f4,f4_score = model(img, target)
            # score, feat,score_f1,score_f2,feat1,score_layer2 = model(img, target)
            score, feat,score_f1,score_f2,feat1 = model(img, target)

            # print(feat.shape)
            loss = loss_fn(score, feat, target,score_f1,score_f2,feat1)
            # loss = loss_fn(score, feat, target,score_f1,score_f2,feat1,score_layer2)

            if cfg.SOLVER.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    # scaled_loss.backward(retain_graph=True)
            else:
                loss.backward()

            # loss.backward()
            optimizer.step()
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            acc = (score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            # print(loss_meter.val)
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_epoch{}.pth'.format(epoch)))
        if epoch==10:

            reduce_model_dict = model.half().state_dict()
            del_keys = []
            for key in reduce_model_dict.keys():
                if 'class' in key or 'sub1' in key or 'sub2' in key or 'base.fc' in key:
                    del_keys.append(key)
            for key in del_keys:
                del reduce_model_dict[key]

            torch.save(reduce_model_dict, os.path.join(cfg.OUTPUT_DIR,  cfg.MODEL.NAME+str(cfg.INPUT.SIZE_TRAIN[0])+'half.pth'))

            # torch.save(model.half().state_dict(), os.path.join(cfg.OUTPUT_DIR,  cfg.MODEL.NAME+str(cfg.INPUT.SIZE_TRAIN[0])+'half.pth'))
        # if epoch % eval_period == 0:
        #     model.eval()
        #     for n_iter, (img, vid, camid, _,_) in enumerate(val_loader):
        #         with torch.no_grad():
        #             img = img.to(device)
        #             feat = model(img)
        #             evaluator.update((feat, vid, camid))
        #
        #     cmc, mAP, _, _, _, _, _ = evaluator.compute()
        #     logger.info("Validation Results - Epoch: {}".format(epoch))
        #     logger.info("mAP: {:.1%}".format(mAP))
        #     for r in [1, 5, 10]:
        #         logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    if cfg.TEST.EVAL:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                       reranking=cfg.TEST.RE_RANKING,reranking_track=cfg.TEST.RE_RANKING_TRACK)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    if cfg.TEST.FLIP_FEATS == 'on':
        print("use flip test................")
    else:
        print("Not use flip test................")

    # print(val_loader)
    for n_iter, (img1,img2, pid, camid, imgpath) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            scr_img,v_img = img1,img2
            scr_img = scr_img.to(device)
            v_img = v_img.to(device)

            if cfg.TEST.FLIP_FEATS == 'on':
                # print('flip_aug')
                feat = torch.FloatTensor(scr_img.size(0), 3072).zero_().cuda()
                # feat = torch.FloatTensor(img.size(0), 768).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(scr_img.size(3) - 1, -1, -1).long().cuda()
                        scr_img = scr_img.index_select(3, inv_idx)
                    f = model(scr_img)
                    feat = feat + f
                #verticle img
                f = model(v_img)
                feat = feat + f


            else:
                feat = model(img)
                # print(feat.shape)

            # print(feat)

            if cfg.TEST.EVAL:
                evaluator.update((feat, pid, camid))
            else:
                evaluator.update((feat, pid, camid, trackid, imgpath))
            img_path_list.extend(imgpath)
        # print(img, pid, camid, imgpath)
        # with torch.no_grad():
        #
        #
        #
        #
        #     img = img.to(device)
        #
        #
        #     # if cfg.TEST.FLIP_FEATS == 'on':
        #     #     # print('flip_aug')
        #     #     feat = torch.FloatTensor(img.size(0), 3072).zero_().cuda()
        #     #     for i in range(2):
        #     #         if i == 1:
        #     #             inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
        #     #             img = img.index_select(3, inv_idx)
        #     #             f = model(img)
        #     #             feat = torch.cat((f1,f),1)
        #     #             # print(feat.shape)
        #     #         # feat = feat + f
        #     #         if i==0:
        #     #             f1 = model(img)
        #     # print(feat.shape)
        #
        #     if cfg.TEST.FLIP_FEATS == 'on':
        #         # print('flip_aug')
        #         feat = torch.FloatTensor(img.size(0), 3072).zero_().cuda()
        #         # feat = torch.FloatTensor(img.size(0), 768).zero_().cuda()
        #         for i in range(2):
        #             if i == 1:
        #                 inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
        #                 img = img.index_select(3, inv_idx)
        #             f = model(img)
        #             feat = feat + f
        #     else:
        #         feat = model(img)
        #         # print(feat.shape)
        #
        #
        #
        #
        #
        #     # print(feat)
        #
        #
        #
        #
        #
        #
        #
        #     if cfg.TEST.EVAL:
        #         evaluator.update((feat, pid, camid))
        #     else:
        #         evaluator.update((feat, pid, camid, trackid, imgpath))
        #     img_path_list.extend(imgpath)

    if cfg.TEST.EVAL:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    else:
        distmat, img_name_q, img_name_g, qfeats, gfeats = evaluator.compute(cfg.OUTPUT_DIR)
        np.save(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT) , distmat)
        print('over')


def do_inference_train_query(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    if cfg.TEST.EVAL:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                           reranking=cfg.TEST.RE_RANKING, reranking_track=cfg.TEST.RE_RANKING_TRACK)
    evaluator.reset()
    if device:
        # if torch.cuda.device_count() > 1:
        #     print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
        #     model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    if cfg.TEST.FLIP_FEATS == 'on':
        print("use flip test................")
    else:
        print("Not use flip test................")

    # print(val_loader)
    train = pickle.load(open('/home/lab3/bi/0814/dmt/query_with_high_score.pkl', 'rb'))
    img_dict = {}

    # # print(feat)

    for i in train:
        img_dict[i[0]] = i[1]
    print(img_dict)
    train = torch.load('/home/lab3/bi/0731/dmt/train.pth')
    for n_iter, (img, pid, camid, imgpath) in enumerate(tqdm(val_loader)):
        # print(img, pid, camid, imgpath)
        with torch.no_grad():
            img = img.to(device)

            if cfg.TEST.FLIP_FEATS == 'on':
                # print('flip_aug')
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            for num, i in enumerate(imgpath):
                if i in img_dict:
                    print("get",i)
                    feat[num] = (0.8*feat[num] + 0.2*train[img_dict[i]])


            if cfg.TEST.EVAL:
                evaluator.update((feat, pid, camid))
            else:
                evaluator.update((feat, pid, camid, trackid, imgpath))





            #
            # for num,i in enumerate(pid):
            #     # print(num,len(pid))
            #     if i!=temp_pid  :
            #         train.append(train_feat / cnt_pid)
            #         print("get!", cnt_pid,"pid",pid[num-1])
            #         cnt_pid = 1
            #         train_feat = feat[num]
            #
            #     else:
            #         train_feat+=feat[num]
            #         cnt_pid +=1
            #     temp_pid = i
            #     if n_iter == len(val_loader)-1 and  num == len(pid)-1:
            #         train.append(train_feat / cnt_pid)







            img_path_list.extend(imgpath)

    # 保存训练集权重
    # torch.save(train, "train.pth")
    if cfg.TEST.EVAL:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    else:
        distmat, img_name_q, img_name_g, qfeats, gfeats = evaluator.compute(cfg.OUTPUT_DIR)
        np.save(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT), distmat)
        print('over')




def do_inference_train(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    if cfg.TEST.EVAL:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                           reranking=cfg.TEST.RE_RANKING, reranking_track=cfg.TEST.RE_RANKING_TRACK)
    evaluator.reset()
    if device:
        # if torch.cuda.device_count() > 1:
        #     print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
        #     model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    if cfg.TEST.FLIP_FEATS == 'on':
        print("use flip test................")
    else:
        print("Not use flip test................")

    # print(val_loader)
    train = []
    train_feat = torch.FloatTensor(1, 2048).zero_().cuda()
    temp_pid = 0
    cnt_pid = 0
    for n_iter, (img, pid, camid, imgpath) in enumerate(tqdm(val_loader)):
        # print(img, pid, camid, imgpath)
        with torch.no_grad():
            img = img.to(device)

            if cfg.TEST.FLIP_FEATS == 'on':
                # print('flip_aug')
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            if cfg.TEST.EVAL:
                evaluator.update((feat, pid, camid))
            else:
                evaluator.update((feat, pid, camid, trackid, imgpath))

            for num,i in enumerate(pid):
                # print(num,len(pid))
                if i!=temp_pid  :
                    train.append(train_feat / cnt_pid)
                    print("get!", cnt_pid,"pid",pid[num-1])
                    cnt_pid = 1
                    train_feat = feat[num]

                else:
                    train_feat+=feat[num]
                    cnt_pid +=1
                temp_pid = i
                if n_iter == len(val_loader)-1 and  num == len(pid)-1:
                    train.append(train_feat / cnt_pid)







            img_path_list.extend(imgpath)

    # 保存训练集权重
    torch.save(train, "train.pth")
    if cfg.TEST.EVAL:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    else:
        distmat, img_name_q, img_name_g, qfeats, gfeats = evaluator.compute(cfg.OUTPUT_DIR)
        np.save(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT), distmat)
        print('over')

def do_inference_query_mining(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    evaluator = R1_mAP_query_mining(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                       reranking=cfg.TEST.RE_RANKING,reranking_track=cfg.TEST.RE_RANKING_TRACK)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, pid, camid, trackid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)

            if cfg.TEST.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            evaluator.update((feat, pid, camid, trackid, imgpath))
            img_path_list.extend(imgpath)

    distmat, img_name_q, img_name_g, qfeats, gfeats = evaluator.compute(cfg.OUTPUT_DIR)

    print('The shape of distmat is: {}'.format(distmat.shape))
    np.save(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT) , distmat)

    return distmat


def do_inference_Pseudo_track_rerank(cfg,
                 model,
                val_loader,
                num_query
                 ):
    device = "cuda"

    evaluator = R1_mAP_Pseudo(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, pid, camid, trackid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)

            if cfg.TEST.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            evaluator.update((feat, pid, camid, trackid, imgpath))
            img_path_list.extend(imgpath)

    distmat, img_name_q, img_name_g, qfeats, gfeats = evaluator.compute(cfg.OUTPUT_DIR)

    return distmat, img_name_q, img_name_g