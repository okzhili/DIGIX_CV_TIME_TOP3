import os

import torch

from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from model.make_model2 import make_model_layer34

from processor import do_inference
from processor.processor import do_inference_d4
from utils.logger import setup_logger
from utils.vis import vis_actmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--test_data_path", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--weight", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.test_data_path = args.test_data_path
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader,_,val_loader, num_query, num_classes = make_dataloader(cfg)
    if cfg.MODEL.NAME=='resnest' or cfg.MODEL.NAME=='resnest101' \
            or cfg.MODEL.NAME=='res2net101_normal' or cfg.MODEL.NAME=='resnest_ibn' or cfg.MODEL.NAME=='resnest101_ibn':
        model = make_model(cfg, num_class=num_classes)

    elif cfg.MODEL.NAME=='res2net101':
        model = make_model_layer34(cfg, num_class=num_classes)

    # elif cfg.MODEL.NAME=='resnest_512':
    #     model = make_model_(cfg, num_class=num_classes)
    # elif cfg.MODEL.NAME == 'res2net_320':
    #     model = make_model_(cfg, num_class=num_classes)
    # elif cfg.MODEL.NAME == 'res2net_512':
    #     model = make_model_layer34(cfg,num_classes)
    # elif 'resnet34' in cfg.MODEL.NAME:
    #     model = make_model_peng.make_model_resnet34(cfg, num_classes)
    # elif 'fpn' in cfg.MODEL.NAME:
    #     model = make_model_peng.make_model_fpn(cfg,num_classes)
    # elif 'ssd' in cfg.MODEL.NAME:
    #     model = make_model_peng.make_model_layer34(cfg,num_classes)
    model.load_param(args.weight)
    # model = torch.nn.DataParallel(model)
    # vis_actmap(model, (320, 320), val_loader, '/data-tmp/crop_data')
    do_inference_d4(cfg,
                 model,
                 val_loader,
                 num_query)
