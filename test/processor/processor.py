import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as tF
from loss.triplet_loss import TripletLoss
from utils.meter import AverageMeter
from utils.metrics import R1_mAP, R1_mAP_eval, R1_mAP_Pseudo, R1_mAP_query_mining
# from apex import amp


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             train_loader_b,
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

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model.to(device)
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
            model = nn.DataParallel(model)
        else:
            model.to(device)
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        # model.to(device)
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    # model.base._freeze_stages()
    logger.info('Freezing the stages number:{}'.format(cfg.MODEL.FROZEN))
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step()
        model.train()
        if epoch <= 80:
            loader = train_loader
        else:
            loader = train_loader_b
        for n_iter, (img, vid) in enumerate(loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if 'bdb' in cfg.MODEL.NAME:
                score, score2, feat1, feat2 = model(img, target)
                loss = loss_fn([score, score2], [feat1, feat2], target)
            else:
                score, feat = model(img, target)
                if cfg.DATALOADER.SAMPLER == 'softmax':
                    loss = F.cross_entropy(score, target)
                else:
                    loss = loss_fn(score, feat, target, model)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            acc = (score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

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


def do_inference_d4(cfg,
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
    from ttach.aliases import d4_transform
    trans = d4_transform()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    a = tqdm(total=(55168 + 29758) / cfg.TEST.IMS_PER_BATCH)
    for n_iter, (img, pid, camid, trackid, imgpath) in enumerate(val_loader):
        a.update(1)
        with torch.no_grad():
            img = img.to(device)
            imgs = []
            for t in trans:
                imgs.append(t.augment_image(img))
            feat = model(imgs[0])
            for im in imgs[1:]:
                feat += model(im)
            feat = feat / 8
            if cfg.TEST.EVAL:
                evaluator.update((feat, pid, imgpath))
            else:
                evaluator.update((feat, pid, camid, trackid, imgpath))
            img_path_list.extend(imgpath)
    if cfg.TEST.EVAL:
        evaluator.compute(cfg.MODEL.ID)
        # cmc, mAP, _, _, _, _, _ = evaluator.compute()
        # logger.info("Validation Results ")
        # logger.info("mAP: {:.1%}".format(mAP))
        # for r in [1, 5, 10]:
        #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    else:
        distmat, img_name_q, img_name_g, qfeats, gfeats = evaluator.compute(cfg.OUTPUT_DIR)
        np.save(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT), distmat)
        print('over')


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
                           reranking=cfg.TEST.RE_RANKING, reranking_track=cfg.TEST.RE_RANKING_TRACK)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    a = tqdm(total=(55168 + 29758) / cfg.TEST.IMS_PER_BATCH)
    for n_iter, (img, pid, camid, trackid, imgpath) in enumerate(val_loader):
        a.update(1)
        with torch.no_grad():
            img = img.to(device)
            feat1 = model(img)
            img_vflip = tF.vflip(img)
            img_hflip = tF.hflip(img)
            img_90 = tF.vflip(torch.transpose(img, 2, 3))
            # img_180 = tF.vflip(tF.hflip(img))
            # img_270 = tF.hflip(torch.transpose(img, 2, 3))
            feat2 = model(img_vflip)
            feat3 = model(img_hflip)
            feat4 = model(img_90)
            # feat4 = model(img_90)
            # feat5 = model(img_180)
            # feat6 = model(img_270)
            feat = (feat1 + feat2 + feat3 + feat4) / 4
            if cfg.TEST.EVAL:
                evaluator.update((feat, pid, imgpath))
            else:
                evaluator.update((feat, pid, camid, trackid, imgpath))
            img_path_list.extend(imgpath)
    if cfg.TEST.EVAL:
        evaluator.compute(cfg.MODEL.NAME)
        # cmc, mAP, _, _, _, _, _ = evaluator.compute()
        # logger.info("Validation Results ")
        # logger.info("mAP: {:.1%}".format(mAP))
        # for r in [1, 5, 10]:
        #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
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
                                    reranking=cfg.TEST.RE_RANKING, reranking_track=cfg.TEST.RE_RANKING_TRACK)
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
    np.save(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT), distmat)

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
