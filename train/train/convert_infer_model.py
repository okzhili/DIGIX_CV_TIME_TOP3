import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model_infer,make_model
from processor import do_inference,do_inference_train,do_inference_train_query
from utils.logger import setup_logger
import torch
from collections import OrderedDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="/home/lab3/bi/0816_MGN/dmt/configs/submit.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
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

    train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)

    # model = make_model(cfg, num_class=num_classes)
    model = make_model_infer(cfg, num_class=num_classes)
    # model.load_param(cfg.TEST.WEIGHT)
    print("load pretained weight",cfg.TEST.WEIGHT)
    try:
        print("加载单卡训练权重")
        model.load_param(cfg.TEST.WEIGHT)
    except:

        print("加载多卡训练权重...............")

        # state_dictBA = torch.load(cfg.TEST.WEIGHT)
        # # create new OrderedDict that does not contain `module.`
        # new_state_dictBA = OrderedDict()
        # for k, v in state_dictBA.items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dictBA[name] = v
        # model.load_state_dict(new_state_dictBA)

    torch.save(model.half().state_dict(),  'infer.pth')


