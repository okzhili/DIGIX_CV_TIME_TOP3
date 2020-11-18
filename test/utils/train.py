import os
from torch.backends import cudnn
from utils.logger import setup_logger
from shuju import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
from flyai.train_helper import upload_data, download, sava_train_model





if __name__ == '__main__':

    # download("dianshang/mix_train.pkl")
    download("dianshang/huafen_all_crop.pkl")
    download("dianshang/mix_train_balance.pkl")
    download("dianshang/r50_a.yml")
    download("model/resnest50_44880.pth")
    download("dianshang/crop1.zip",decompression=True)


    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="dianshang/r50_a.yml", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if True:
        cfg.merge_from_file("dianshang/r50_a.yml")
    cfg.merge_from_list(args.opts)
    cfg.freeze()









    # print("config file",args.config_file)




    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("reid_baseline", output_dir, if_train=True)
    logger.info("Running with config:\n{}".format(cfg))
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    if True:
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open("dianshang/r50_a.yml", 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)

    if cfg.MODEL.PRETRAIN_CHOICE == 'finetune':
        model = make_model(cfg, num_class=num_classes)
        model.load_param_finetune(cfg.MODEL.PRETRAIN_PATH)
        print('Loading pretrained model for finetuning......')
    else:
        model = make_model(cfg, num_class=num_classes)


    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                  cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)


    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,  # modify for using self trained model
        loss_func,
        num_query
    )
